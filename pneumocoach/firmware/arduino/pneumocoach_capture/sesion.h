/* Sesion clinica a bordo: calentamiento, calibracion y coaching.
 * =============================================================
 *
 * Puerto de `ml/pneumocoach/calibracion.py`. La pregunta "esto es toracico en
 * terminos absolutos?" no tiene respuesta estable -tres sesiones del mismo
 * sujeto dieron amplitudes que varian un 50 % y caracteristicas que cambian de
 * signo-. La que si la tiene es relativa: se parece mas a TU toracica o a TU
 * diafragmatica de hoy?
 *
 * Por eso el paciente ejecuta dos maniobras de referencia al empezar. De cada
 * una sale un vector de caracteristicas promedio, y esos dos vectores definen
 * el eje sobre el que se proyecta todo lo demas:
 *
 *     z = (x - ref_dia) / (ref_tor - ref_dia)
 *
 * Su diafragmatica cae en 0 y su toracica en 1, sea cual sea su contextura, su
 * montaje o su esfuerzo de ese dia.
 *
 * El dispositivo tiene que calcularlo igual que Python, porque el modelo se
 * entrena sobre z. Las constantes -PC_REF_SEGUNDOS, PC_CONTRASTE_MINIMO- vienen
 * del contrato compartido; no hay copias aqui.
 */

#ifndef PC_SESION_H
#define PC_SESION_H

#include "features.h"

/* Duracion de cada maniobra, en muestras. Con 30 s a 50 Hz caben exactamente
 * PC_REF_VENTANAS ventanas completas dentro de la fase. */
#define PC_REF_MUESTRAS ((uint32_t)(PC_REF_SEGUNDOS * PC_FS_HZ))
#define PC_PREPARA_MUESTRAS ((uint32_t)(PC_PREPARA_SEGUNDOS * PC_FS_HZ))
#define PC_RESULTADO_MUESTRAS ((uint32_t)(PC_RESULTADO_SEGUNDOS * PC_FS_HZ))

/* ------------------------------------------------------------------ */
/* Cadena continua: muestras crudas -> vector de 29                    */
/* ------------------------------------------------------------------ */
/* Los filtros guardan estado entre llamadas y nunca se reinician dentro de una
 * sesion. Reiniciarlos al cambiar de fase volveria a meter el transitorio de
 * asentamiento dentro del primer vector de cada maniobra. */

typedef struct {
  pc_front_t front;
  float ring[5][PC_WINDOW_N];
  uint32_t n;
} pc_pipeline_t;

/* Una sola ventana de trabajo para todo el firmware: el modo de paridad y la
 * sesion no corren nunca a la vez, y son 12 kB que no conviene duplicar. */
static pc_ventana_t pc_scratch;

static void pc_pipeline_reset(pc_pipeline_t *p) {
  pc_front_init(&p->front);
  memset(p->ring, 0, sizeof(p->ring));
  p->n = 0;
}

/* Devuelve true cuando se completa una ventana; f_out recibe las 29
 * caracteristicas e ini_out el indice global de la primera muestra. */
static bool pc_pipeline_push(pc_pipeline_t *p, const int16_t raw6[6],
                             float *f_out, uint32_t *ini_out) {
  pc_muestra_t m;
  pc_front_step(&p->front, raw6, &m);

  uint32_t k = p->n % PC_WINDOW_N;
  p->ring[0][k] = m.tilt;
  p->ring[1][k] = m.axial;
  p->ring[2][k] = m.hf;
  p->ring[3][k] = m.accel_mag;
  p->ring[4][k] = m.gyro_mag;
  p->n++;

  if (p->n < PC_WINDOW_N) return false;
  if ((p->n - PC_WINDOW_N) % PC_HOP_N != 0) return false;

  uint32_t ini = p->n % PC_WINDOW_N;
  for (int i = 0; i < PC_WINDOW_N; i++) {
    uint32_t j = (ini + i) % PC_WINDOW_N;
    pc_scratch.tilt[i] = p->ring[0][j];
    pc_scratch.axial[i] = p->ring[1][j];
    pc_scratch.hf[i] = p->ring[2][j];
    pc_scratch.accel_mag[i] = p->ring[3][j];
    pc_scratch.gyro_mag[i] = p->ring[4][j];
  }
  pc_extraer(&pc_scratch, f_out);
  *ini_out = p->n - PC_WINDOW_N;
  return true;
}

/* ------------------------------------------------------------------ */
/* Referencia de la sesion                                             */
/* ------------------------------------------------------------------ */

typedef struct {
  float suma_dia[PC_N_FEATURES];
  float suma_tor[PC_N_FEATURES];
  uint16_t n_dia, n_tor;
} pc_ref_t;

static void pc_ref_reset(pc_ref_t *r) { memset(r, 0, sizeof(*r)); }

static void pc_ref_acumula(pc_ref_t *r, const float *f, bool toracica) {
  float *s = toracica ? r->suma_tor : r->suma_dia;
  for (int j = 0; j < PC_N_FEATURES; j++) s[j] += f[j];
  if (toracica) r->n_tor++; else r->n_dia++;
}

static bool pc_ref_lista(const pc_ref_t *r) { return r->n_dia > 0 && r->n_tor > 0; }

/* Proyecta x sobre el eje del paciente. Las caracteristicas cuyo eje se
 * degenera -las dos maniobras salieron casi iguales- se dejan en 0 en vez de
 * propagar un cociente inestable: aportan nada, pero no envenenan el resto del
 * vector. Es la misma regla que ReferenciaSesion.normaliza. */
static int pc_ref_normaliza(const pc_ref_t *r, const float *x, float *z) {
  if (!pc_ref_lista(r)) return 0;
  int informativas = 0;
  for (int j = 0; j < PC_N_FEATURES; j++) {
    float dia = r->suma_dia[j] / r->n_dia;
    float tor = r->suma_tor[j] / r->n_tor;
    float eje = tor - dia;
    float escala = fmaxf(fabsf(dia), fabsf(tor)) + 1e-12f;
    if (fabsf(eje) / escala > PC_CONTRASTE_MINIMO) {
      z[j] = (x[j] - dia) / eje;
      informativas++;
    } else {
      z[j] = 0.0f;
    }
  }
  return informativas;
}

/* Contraste mediano entre las dos maniobras. DIAGNOSTICO, NO AVAL.
 *
 * Seria muy util que el dispositivo detectara solo una calibracion inservible
 * -el paciente ejecuto las dos maniobras igual, el sensor se movio, no entendio
 * la consigna- y pidiera repetir. tools/medir_calidad_calibracion.py comprueba
 * si se puede, y con los datos de hoy no se puede:
 *
 *   contraste relativo   senal 0.115-0.571   nulo 0.072-0.379
 *   d de Cohen           senal 0.545-4.179   nulo 0.599-1.601
 *
 * donde el nulo son dos mitades del MISMO bloque, o sea el paciente haciendo lo
 * mismo dos veces. Los rangos se solapan en ambos casos: no existe umbral que
 * acepte toda calibracion buena y rechace alguna mala. Poner uno de todos modos
 * daria una tranquilidad falsa, que es peor que no dar ninguna.
 *
 * No es un fallo del estadistico sino la limitacion de ADR-0007: la deriva
 * dentro de una maniobra es del mismo orden que la diferencia entre maniobras.
 * Asi que este numero se ensena y se registra, y no bloquea nada. Cuando exista
 * una referencia independiente que etiquete las maniobras, se vuelve a medir.
 *
 * La mediana, y no la media, porque unas pocas caracteristicas con eje enorme
 * bastarian para tapar que las demas no separan nada. */
static float pc_ref_contraste_mediano(const pc_ref_t *r) {
  if (!pc_ref_lista(r)) return 0.0f;
  float c[PC_N_FEATURES];
  for (int j = 0; j < PC_N_FEATURES; j++) {
    float dia = r->suma_dia[j] / r->n_dia;
    float tor = r->suma_tor[j] / r->n_tor;
    float escala = fmaxf(fabsf(dia), fabsf(tor)) + 1e-12f;
    c[j] = fabsf(tor - dia) / escala;
  }
  for (int i = 1; i < PC_N_FEATURES; i++) {  // insercion: 29 elementos
    float v = c[i];
    int k = i - 1;
    while (k >= 0 && c[k] > v) { c[k + 1] = c[k]; k--; }
    c[k + 1] = v;
  }
  return (PC_N_FEATURES % 2) ? c[PC_N_FEATURES / 2]
                             : 0.5f * (c[PC_N_FEATURES / 2 - 1] + c[PC_N_FEATURES / 2]);
}

/* ------------------------------------------------------------------ */
/* Maquina de estados                                                  */
/* ------------------------------------------------------------------ */

typedef enum {
  SES_CALENTANDO,  // los filtros se asientan; sirve tambien para colocar
  SES_LISTO,       // asentado, esperando que empiece la calibracion
  SES_CAL_DIA,     // maniobra de referencia diafragmatica
  SES_CAL_TOR,     // maniobra de referencia toracica
  SES_CALIBRADO,   // eje calculado, mostrando su calidad
  SES_ACTIVA,      // coaching
} pc_sesion_est_t;

typedef struct {
  pc_sesion_est_t est;
  pc_pipeline_t pipe;
  pc_ref_t ref;
  uint32_t ini_fase;          // indice global en que empezo la fase
  float f[PC_N_FEATURES];     // ultimo vector crudo
  float z[PC_N_FEATURES];     // ultimo vector normalizado
  bool hay_vector;
  int informativas;
  float contraste;            // mediana; diagnostico, no criterio de validez
  uint16_t descartadas;       // ventanas que cruzaban el limite de fase
} pc_sesion_t;

static void pc_sesion_reset(pc_sesion_t *s) {
  /* El montaje se salva a mano por encima del memset.
   *
   * pc_front_init tambien lo preserva, pero eso no basta: este memset borra la
   * estructura ENTERA, front incluido, antes de que pc_front_init llegue a
   * verlo. Es un nivel mas arriba y hay que atajarlo aqui.
   *
   * Sin esto, cancelar una sesion o recalibrar devolvia el DSP a los ejes
   * crudos del sensor mientras el analisis seguia usando el marco anatomico:
   * el dispositivo no fallaba, calculaba sobre otro sistema de coordenadas. */
  float R[9], sesgo[3];
  bool tenia = s->pipe.front.usar_R;
  if (tenia) {
    memcpy(R, s->pipe.front.R, sizeof(R));
    memcpy(sesgo, s->pipe.front.sesgo_gyro, sizeof(sesgo));
  }
  memset(s, 0, sizeof(*s));
  pc_pipeline_reset(&s->pipe);
  pc_ref_reset(&s->ref);
  if (tenia) pc_front_set_mount(&s->pipe.front, R, sesgo);
  s->est = SES_CALENTANDO;
}

static void pc_sesion_fase(pc_sesion_t *s, pc_sesion_est_t e) {
  s->est = e;
  s->ini_fase = s->pipe.n;
}

/* Segundos que le quedan a la fase en curso. 0 si la fase no es temporizada. */
static float pc_sesion_restante(const pc_sesion_t *s) {
  uint32_t total = 0;
  if (s->est == SES_CALENTANDO) {
    if (s->pipe.n >= PC_WARMUP_N) return 0.0f;
    return (PC_WARMUP_N - s->pipe.n) / PC_FS_HZ;
  }
  switch (s->est) {
    case SES_CAL_DIA:
    case SES_CAL_TOR:   total = PC_REF_MUESTRAS; break;
    case SES_LISTO:     total = PC_PREPARA_MUESTRAS; break;
    case SES_CALIBRADO: total = PC_RESULTADO_MUESTRAS; break;
    default:            return 0.0f;
  }
  uint32_t hechas = s->pipe.n - s->ini_fase;
  if (hechas >= total) return 0.0f;
  return (total - hechas) / PC_FS_HZ;
}

/* Una muestra. Avanza la maquina y devuelve true si hubo vector nuevo. */
static bool pc_sesion_muestra(pc_sesion_t *s, const int16_t raw6[6]) {
  uint32_t ini = 0;
  bool nuevo = pc_pipeline_push(&s->pipe, raw6, s->f, &ini);

  switch (s->est) {
    /* La sesion avanza SOLA. No hay boton que pulsar ni gesto que hacer.
     *
     * El paciente esta a mitad de un ejercicio respiratorio con las dos manos
     * sobre el cuerpo -la consigna lo exige-, asi que cualquier cosa que tenga
     * que accionar es un movimiento que acaba en el canal que medimos. Y el
     * unico dispositivo de entrada del kit, el APDS9960, quedo fuera de alcance
     * (ADR-0010).
     *
     * Los comandos 'n' y 'k' siguen existiendo como mando manual para ensayar y
     * para la demo, pero no hacen falta para completar una sesion. */
    case SES_CALENTANDO:
      if (s->pipe.n >= PC_WARMUP_N) pc_sesion_fase(s, SES_LISTO);
      break;

    case SES_LISTO:
      if (s->pipe.n - s->ini_fase >= PC_PREPARA_MUESTRAS) {
        pc_ref_reset(&s->ref);
        pc_sesion_fase(s, SES_CAL_DIA);
      }
      break;

    case SES_CALIBRADO:
      if (s->pipe.n - s->ini_fase >= PC_RESULTADO_MUESTRAS)
        pc_sesion_fase(s, SES_ACTIVA);
      break;

    case SES_CAL_DIA:
    case SES_CAL_TOR: {
      /* Solo cuentan las ventanas ENTERAS dentro de la maniobra. La primera
       * ventana tras el cambio de fase arrastra 12 s de la fase anterior, y
       * meterla en la referencia contamina el eje con la maniobra equivocada.
       * Por eso la fase dura PC_REF_SEGUNDOS y no menos: los primeros 12 s se
       * van en llenar la primera ventana valida. */
      if (nuevo) {
        bool dentro = ini >= s->ini_fase &&
                      (ini + PC_WINDOW_N) <= (s->ini_fase + PC_REF_MUESTRAS);
        if (dentro) pc_ref_acumula(&s->ref, s->f, s->est == SES_CAL_TOR);
        else s->descartadas++;
      }
      if (s->pipe.n - s->ini_fase >= PC_REF_MUESTRAS) {
        if (s->est == SES_CAL_DIA) {
          pc_sesion_fase(s, SES_CAL_TOR);
        } else {
          s->informativas = pc_ref_normaliza(&s->ref, s->f, s->z);
          s->contraste = pc_ref_contraste_mediano(&s->ref);
          pc_sesion_fase(s, SES_CALIBRADO);
        }
      }
      break;
    }

    case SES_ACTIVA:
      if (nuevo) s->informativas = pc_ref_normaliza(&s->ref, s->f, s->z);
      break;

    default:
      break;
  }

  if (nuevo) s->hay_vector = true;
  return nuevo;
}

/* Eventos de usuario. Un gesto o una tecla; la maquina no distingue. */
static void pc_sesion_cancelar(pc_sesion_t *s);

static void pc_sesion_siguiente(pc_sesion_t *s) {
  switch (s->est) {
    case SES_LISTO:     pc_ref_reset(&s->ref); pc_sesion_fase(s, SES_CAL_DIA); break;
    case SES_CALIBRADO: pc_sesion_fase(s, SES_ACTIVA);       break;
    default: break;  // durante calentamiento o maniobra no hay nada que saltar
  }
}

static void pc_sesion_cancelar(pc_sesion_t *s) {
  /* Vuelve a LISTO sin tocar el pipeline: los filtros siguen asentados, asi que
   * cancelar no cuesta otro calentamiento. */
  if (s->est == SES_CALENTANDO) return;
  pc_ref_reset(&s->ref);
  s->hay_vector = false;
  s->informativas = 0;
  s->contraste = 0.0f;
  s->descartadas = 0;
  pc_sesion_fase(s, SES_LISTO);
}

static const char *pc_sesion_nombre(pc_sesion_est_t e) {
  switch (e) {
    case SES_CALENTANDO: return "ASENTANDO";
    case SES_LISTO:      return "LISTO";
    case SES_CAL_DIA:    return "CAL DIAFRAG";
    case SES_CAL_TOR:    return "CAL TORACICA";
    case SES_CALIBRADO:  return "CALIBRADO";
    case SES_ACTIVA:     return "SESION";
  }
  return "?";
}

#endif /* PC_SESION_H */
