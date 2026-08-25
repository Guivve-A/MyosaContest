/* Inferencia a bordo con TensorFlow Lite Micro.
 * ============================================
 *
 * Cadena completa, y el orden importa porque tiene que ser IDENTICO al del
 * entrenamiento:
 *
 *     29 caracteristicas
 *       -> z = (x - ref_dia) / (ref_tor - ref_dia)     calibracion de sesion
 *       -> (z - PC_FEATURE_MEAN) / PC_FEATURE_SCALE    estandarizado
 *       -> cuantizacion a int8 con la escala del tensor de entrada
 *       -> interprete
 *       -> descuantizacion a probabilidades
 *
 * El modelo se entreno sobre z, no sobre caracteristicas absolutas. Sin la
 * calibracion de sesion, o con la matriz de montaje sin cargar, el interprete
 * recibe numeros de otro espacio y devuelve un veredicto seguro y equivocado.
 * Por eso pc_inferencia_lista() exige las dos cosas antes de dejar clasificar.
 */

#ifndef PC_INFERENCIA_H
#define PC_INFERENCIA_H

#include "sesion.h"
#include "../../include/pneumocoach_model.h"
#include "../../include/pneumocoach_golden.h"

#ifdef PC_STANDARDISER_PLACEHOLDER
#error "El contrato se genero sin estandarizador entrenado. Normalizar con \
identidad no falla: produce veredictos seguros y equivocados. Entrena primero \
con ml/scripts/entrenar_real.py y vuelve a emitir."
#endif

#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

/* Arena del interprete.
 *
 * El modelo son 29 entradas, dos capas ocultas de 32 y 16, y 3 salidas: los
 * tensores intermedios caben de sobra en 8 kB. Se reserva estatico y no en el
 * monton, porque una asignacion fallida a mitad de sesion es un fallo que
 * aparece tarde y no se reproduce. */
#define PC_ARENA_BYTES 8192
static uint8_t pc_arena[PC_ARENA_BYTES];

typedef struct {
  bool listo;
  const tflite::Model *modelo;
  tflite::MicroInterpreter *interprete;
  TfLiteTensor *entrada;
  TfLiteTensor *salida;
  size_t arena_usada;
} pc_inferencia_t;

static pc_inferencia_t g_inf;

/* Resolutor con DOS operadores y no uno mas.
 *
 * AllOpsResolver arrastraria decenas de kernels que este modelo no usa. Fijar
 * los dos que hacen falta tiene un efecto util aparte del tamano: si alguien
 * anade una capa nueva al modelo -una BatchNormalization, un Dropout que no se
 * fusiono-, el interprete falla al arrancar en vez de comportarse de forma rara
 * en el dispositivo. */
static bool pc_inferencia_init() {
  memset(&g_inf, 0, sizeof(g_inf));
  g_inf.modelo = tflite::GetModel(pc_model_tflite);
  if (g_inf.modelo->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("# TFLM: version de esquema %lu, esperada %d\n",
                  (unsigned long)g_inf.modelo->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }

  static tflite::MicroMutableOpResolver<2> resolutor;
  if (resolutor.AddFullyConnected() != kTfLiteOk) return false;
  if (resolutor.AddSoftmax() != kTfLiteOk) return false;

  static tflite::MicroInterpreter interprete(
      g_inf.modelo, resolutor, pc_arena, PC_ARENA_BYTES);
  g_inf.interprete = &interprete;

  if (interprete.AllocateTensors() != kTfLiteOk) {
    Serial.println(F("# TFLM: AllocateTensors fallo; arena insuficiente"));
    return false;
  }
  g_inf.entrada = interprete.input(0);
  g_inf.salida = interprete.output(0);
  g_inf.arena_usada = interprete.arena_used_bytes();

  /* El contrato se comprueba al arrancar, no se supone. Un modelo emitido con
   * otro numero de caracteristicas o de clases entraria aqui sin quejarse y
   * clasificaria leyendo columnas equivocadas. */
  if (g_inf.entrada->dims->data[g_inf.entrada->dims->size - 1] != PC_N_FEATURES) {
    Serial.printf("# TFLM: el modelo espera %d entradas, el contrato dice %d\n",
                  g_inf.entrada->dims->data[g_inf.entrada->dims->size - 1],
                  PC_N_FEATURES);
    return false;
  }
  if (g_inf.salida->dims->data[g_inf.salida->dims->size - 1] != PC_N_CLASSES) {
    Serial.printf("# TFLM: el modelo da %d clases, el contrato dice %d\n",
                  g_inf.salida->dims->data[g_inf.salida->dims->size - 1],
                  PC_N_CLASSES);
    return false;
  }
  if (g_inf.entrada->type != kTfLiteInt8 || g_inf.salida->type != kTfLiteInt8) {
    Serial.println(F("# TFLM: se esperaba entrada y salida int8"));
    return false;
  }

  g_inf.listo = true;
  Serial.printf("# TFLM listo: %u B de modelo, arena %u de %u B\n",
                (unsigned)PC_MODEL_LEN, (unsigned)g_inf.arena_usada,
                (unsigned)PC_ARENA_BYTES);
  return true;
}

typedef struct {
  int clase;                    // -1 si no hay veredicto
  float confianza;
  float prob[PC_N_CLASSES];
  uint32_t us;
} pc_veredicto_t;

/* Clasifica un vector YA normalizado por la calibracion de sesion. */
static bool pc_inferir(const float *z, pc_veredicto_t *out) {
  memset(out, 0, sizeof(*out));
  out->clase = -1;
  if (!g_inf.listo) return false;

  uint32_t t0 = micros();
  const float esc_in = g_inf.entrada->params.scale;
  const int zp_in = g_inf.entrada->params.zero_point;

  for (int j = 0; j < PC_N_FEATURES; j++) {
    float s = (z[j] - PC_FEATURE_MEAN[j]) / PC_FEATURE_SCALE[j];
    int q = (int)lroundf(s / esc_in) + zp_in;
    /* La saturacion es deliberada y silenciosa: una caracteristica fuera de
     * rango se recorta al extremo del int8 en vez de dar la vuelta. Envolver
     * convertiria un valor extremo en uno moderado del signo contrario, que es
     * el peor error posible aqui. */
    g_inf.entrada->data.int8[j] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
  }

  if (g_inf.interprete->Invoke() != kTfLiteOk) return false;

  const float esc_out = g_inf.salida->params.scale;
  const int zp_out = g_inf.salida->params.zero_point;
  int mejor = 0;
  for (int c = 0; c < PC_N_CLASSES; c++) {
    out->prob[c] = (g_inf.salida->data.int8[c] - zp_out) * esc_out;
    if (out->prob[c] > out->prob[mejor]) mejor = c;
  }
  out->confianza = out->prob[mejor];
  out->us = micros() - t0;

  /* Por debajo del piso no se emite veredicto. Negarse a adivinar es mejor que
   * coachear mal al paciente. */
  out->clase = (out->confianza >= PC_CONFIDENCE_FLOOR) ? mejor : -1;
  return true;
}

/* El veredicto solo tiene sentido con la sesion calibrada y el montaje cargado.
 * Son las dos condiciones que hacen que z signifique lo mismo aqui que en el
 * entrenamiento. */
static bool pc_inferencia_lista(const pc_sesion_t *s, bool montaje_ok) {
  return g_inf.listo && montaje_ok && pc_ref_lista(&s->ref);
}

/* ------------------------------------------------------------------ */
/* Prueba de los vectores dorados                                      */
/* ------------------------------------------------------------------ */
/* Verifica el tramo que va de la caracteristica al veredicto:
 *
 *     29 caracteristicas -> z -> estandarizado -> INT8 -> probabilidades
 *
 * NO verifica el DSP. De cuentas crudas a caracteristicas se encarga
 * tools/paridad.py, que inyecta grabaciones enteras y filtra de forma continua,
 * que es como funciona esto de verdad.
 *
 * La primera version metia cuentas crudas y comparaba caracteristicas. No podia
 * salir: eran ventanas SUELTAS y Python las calculo sobre el filtrado continuo
 * de la grabacion entera, con estado acumulado de minutos antes. Salian 190 de
 * 348 fuera de tolerancia y aun asi los 12 veredictos coincidian, que es el tipo
 * de coincidencia que no hay que confundir con una verificacion.
 *
 * No hace falta llevar el sensor puesto. */
static void pc_golden_test() {
  if (!g_inf.listo) {
    Serial.println(F("# DORADOS: el interprete no arranco"));
    return;
  }
  static float z[PC_N_FEATURES];
  int mal_z = 0, mal_p = 0, mal_v = 0;
  float peor_z = 0.0f, peor_p = 0.0f;
  uint32_t us_total = 0;

  Serial.printf("# DORADOS: %d ventanas, caracteristica -> veredicto\n",
                PC_GOLDEN_N);
  for (int w = 0; w < PC_GOLDEN_N; w++) {
    /* z con la referencia que emitio Python, no con la de la sesion en curso:
     * asi la prueba mide la proyeccion y no depende de haber calibrado. */
    for (int j = 0; j < PC_N_FEATURES; j++) {
      float dia = PC_GOLDEN_REF_DIA[j], tor = PC_GOLDEN_REF_TOR[j];
      float eje = tor - dia;
      float escala = fmaxf(fabsf(dia), fabsf(tor)) + 1e-12f;
      z[j] = (fabsf(eje) / escala > PC_CONTRASTE_MINIMO)
             ? (PC_GOLDEN_FEATURES[w][j] - dia) / eje : 0.0f;
      float e = fabsf(z[j] - PC_GOLDEN_Z[w][j]);
      float t = PC_GOLDEN_ATOL + PC_GOLDEN_RTOL * fabsf(PC_GOLDEN_Z[w][j]);
      if (e > t) { mal_z++; if (e / t > peor_z) peor_z = e / t; }
    }

    pc_veredicto_t v;
    if (!pc_inferir(z, &v)) {
      Serial.printf("#   v%d inferencia fallo\n", w);
      mal_v++;
      continue;
    }
    us_total += v.us;

    for (int c = 0; c < PC_N_CLASSES; c++) {
      float e = fabsf(v.prob[c] - PC_GOLDEN_PROB[w][c]);
      /* Las probabilidades salen de int8: su resolucion es el paso de
       * cuantizacion de la salida, asi que la tolerancia es absoluta y de ese
       * orden, no relativa. */
      float t = 2.0f * g_inf.salida->params.scale + PC_GOLDEN_ATOL;
      if (e > t) { mal_p++; if (e / t > peor_p) peor_p = e / t; }
    }

    int mejor = 0;
    for (int c = 1; c < PC_N_CLASSES; c++)
      if (v.prob[c] > v.prob[mejor]) mejor = c;
    if (mejor != PC_GOLDEN_VERDICT[w]) {
      mal_v++;
      Serial.printf("#   v%-2d veredicto %d, esperado %d\n",
                    w, mejor, PC_GOLDEN_VERDICT[w]);
    }
  }

  Serial.printf("# DORADOS proyeccion z  : %d de %d fuera (peor x%.2f)\n",
                mal_z, PC_GOLDEN_N * PC_N_FEATURES, peor_z);
  Serial.printf("# DORADOS probabilidades: %d de %d fuera (peor x%.2f)\n",
                mal_p, PC_GOLDEN_N * PC_N_CLASSES, peor_p);
  Serial.printf("# DORADOS veredicto     : %d de %d discrepan\n",
                mal_v, PC_GOLDEN_N);
  Serial.printf("# DORADOS inferencia    : %lu us por ventana\n",
                (unsigned long)(us_total / PC_GOLDEN_N));
  Serial.println((mal_z || mal_p || mal_v)
                 ? F("# DORADOS: FALLA. El dispositivo no reproduce a Python.")
                 : F("# DORADOS: OK. Inferencia identica a la de Python."));
}

#endif /* PC_INFERENCIA_H */
