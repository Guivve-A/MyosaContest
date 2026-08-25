/* Cadena de DSP respiratorio en C.
 * ================================
 *
 * Puerto de `ml/pneumocoach/dsp.py`. No es una reimplementacion "equivalente":
 * tiene que producir los mismos numeros que Python dentro de PC_GOLDEN_RTOL,
 * porque el modelo se entreno sobre la salida de Python. Un DSP que se
 * "parece" produce un dispositivo que arranca, corre, y clasifica mal — el
 * fallo silencioso que este proyecto lleva evitando desde el principio.
 *
 * Decisiones que hay que respetar al tocar este archivo:
 *
 *   FILTRADO CONTINUO, NO POR VENTANA. Los biquads guardan estado entre
 *   llamadas y procesan la muestra segun llega. Filtrar cada ventana por
 *   separado meteria el transitorio de asentamiento del paso alto de 0.1 Hz
 *   dentro de cada vector de caracteristicas.
 *
 *   CAUSAL, UNA SOLA PASADA. Python usa sosfilt, no sosfiltfilt, precisamente
 *   para que esto sea reproducible aqui.
 *
 *   DIEZMADO POR SUBMUESTREO SIMPLE. El paso banda ya limita a 1 Hz, muy por
 *   debajo del Nyquist de 2.5 Hz de la tasa diezmada, asi que un filtro
 *   antialias adicional seria codigo muerto — y una diferencia silenciosa si
 *   Python no lo tuviera.
 *
 * Todo en float32: el Xtensa LX6 tiene FPU de precision simple por hardware.
 * NumPy calcula en float64, asi que la coincidencia no sera bit a bit; de ahi
 * la tolerancia relativa de 2e-2, que atrapa diferencias de algoritmo sin
 * disparar por redondeo.
 */

#ifndef PC_DSP_H
#define PC_DSP_H

#include <math.h>
#include <string.h>

/* Ruta relativa a firmware/include/, que es donde emit_c_artifacts.py escribe.
 * No se copia al sketch a proposito: dos copias del contrato compartido es
 * exactamente como se desincronizan Python y C. */
#include "../../include/pneumocoach_config.h"

#define PC_EPS 1e-12f

/* ------------------------------------------------------------------ */
/* Biquads en cascada, Direct Form II transpuesta                      */
/* ------------------------------------------------------------------ */
/* Misma forma que usa scipy.signal.sosfilt. El estado son dos acumuladores
 * por seccion; procesar muestra a muestra da exactamente la misma secuencia
 * que sosfilt sobre el vector completo. */

typedef struct {
  const float *sos;  // {b0,b1,b2,a1,a2} por seccion
  uint8_t n_sec;
  float z1[4], z2[4];  // holgura para hasta 4 secciones
} pc_biquad_t;

static void pc_biquad_init(pc_biquad_t *f, const float *sos, uint8_t n_sec) {
  f->sos = sos;
  f->n_sec = n_sec;
  memset(f->z1, 0, sizeof(f->z1));
  memset(f->z2, 0, sizeof(f->z2));
}

static float pc_biquad_step(pc_biquad_t *f, float x) {
  for (uint8_t s = 0; s < f->n_sec; s++) {
    const float *c = f->sos + s * 5;
    float y = c[0] * x + f->z1[s];
    f->z1[s] = c[1] * x - c[3] * y + f->z2[s];
    f->z2[s] = c[2] * x - c[4] * y;
    x = y;
  }
  return x;
}

/* ------------------------------------------------------------------ */
/* Front end: cuentas crudas -> canales tilt y axial                   */
/* ------------------------------------------------------------------ */

typedef struct {
  float pitch;      // estado del filtro complementario, grados
  bool sembrado;    // el primer valor se siembra con el acelerometro
  pc_biquad_t bp_tilt, bp_axial, hp_hf;

  /* Matriz de montaje medida por tools/orientacion.py. Filas: superior,
   * lateral, anterior. Sin ella el DSP asume que el sensor quedo alineado con
   * el marco anatomico, cosa que no pasa nunca sobre un torso real. */
  float R[9];
  bool usar_R;
  float sesgo_gyro[3];

  /* Valor de la primera muestra de cada canal. Se resta antes de filtrar. */
  float ped_tilt, ped_axial, ped_hf;
} pc_front_t;

static void pc_front_init(pc_front_t *f) {
  /* La matriz de montaje SOBREVIVE al reinicio.
   *
   * Es una propiedad del dispositivo y de como quedo pegado hoy, no del estado
   * de la sesion. Borrarla en cada reset hacia que cualquier recalibracion, o
   * el simple hecho de cancelar, devolviera el DSP a los ejes crudos del
   * sensor sin que nada lo dijera. */
  float R[9];
  float sesgo[3];
  bool tenia = f->usar_R;
  if (tenia) {
    memcpy(R, f->R, sizeof(R));
    memcpy(sesgo, f->sesgo_gyro, sizeof(sesgo));
  }
  memset(f, 0, sizeof(*f));
  if (tenia) {
    memcpy(f->R, R, sizeof(R));
    memcpy(f->sesgo_gyro, sesgo, sizeof(sesgo));
    f->usar_R = true;
  }
  pc_biquad_init(&f->bp_tilt, PC_BP_SOS, PC_BP_SECTIONS);
  pc_biquad_init(&f->bp_axial, PC_BP_SOS, PC_BP_SECTIONS);
  pc_biquad_init(&f->hp_hf, PC_HF_SOS, PC_HF_SECTIONS);
}

static void pc_front_set_mount(pc_front_t *f, const float R[9], const float sesgo[3]) {
  memcpy(f->R, R, sizeof(f->R));
  memcpy(f->sesgo_gyro, sesgo, sizeof(f->sesgo_gyro));
  f->usar_R = true;
}

/* Salidas de una muestra. `hf` y `mag` alimentan las caracteristicas de
 * artefacto, que se calculan sobre la senal SIN filtrar. */
typedef struct {
  float tilt, axial, hf, accel_mag, gyro_mag, pitch;
} pc_muestra_t;

static void pc_front_step(pc_front_t *f, const int16_t raw[6], pc_muestra_t *out) {
  float a[3], g[3];
  for (int i = 0; i < 3; i++) {
    a[i] = raw[i] / PC_ACCEL_LSB_PER_G;
    g[i] = raw[3 + i] / PC_GYRO_LSB_PER_DPS - f->sesgo_gyro[i];
  }

  if (f->usar_R) {  // v_cuerpo = R * v_sensor
    float ab[3], gb[3];
    for (int r = 0; r < 3; r++) {
      ab[r] = f->R[r * 3] * a[0] + f->R[r * 3 + 1] * a[1] + f->R[r * 3 + 2] * a[2];
      gb[r] = f->R[r * 3] * g[0] + f->R[r * 3 + 1] * g[1] + f->R[r * 3 + 2] * g[2];
    }
    memcpy(a, ab, sizeof(a));
    memcpy(g, gb, sizeof(g));
  }

  float lat = sqrtf(a[0] * a[0] + a[1] * a[1]) + PC_EPS;
  float pitch_acc = atan2f(a[2], lat) * 57.29577951308232f;

  /* El estado del complementario arranca en el angulo que mide el acelerometro,
   * no en cero, para que la estimacion empiece sobre el cuerpo en vez de subir
   * desde la horizontal. Python siembra igual en complementary_pitch. */
  if (!f->sembrado) f->pitch = pitch_acc;

  const float dt = 1.0f / PC_FS_HZ;
  f->pitch = PC_COMP_ALPHA * (f->pitch + g[1] * dt) +
             (1.0f - PC_COMP_ALPHA) * pitch_acc;

  /* Quitar la proyeccion de la gravedad deja traslacion antero-posterior pura
   * en vez de la inclinacion que ya medimos aparte. */
  float az_trans = a[2] - sinf(f->pitch * 0.017453292519943295f);
  float mag = sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);

  /* Los pedestales se toman DESPUES del paso del complementario, porque lo que
   * Python resta es pitch_deg[0] -la primera muestra que sale del filtro- y no
   * el angulo crudo del acelerometro. Capturarlos antes deja un desfase de un
   * paso entre las dos implementaciones. */
  if (!f->sembrado) {
    f->ped_tilt = f->pitch;
    f->ped_axial = az_trans;
    f->ped_hf = mag;
    f->sembrado = true;
  }

  out->pitch = f->pitch;
  out->accel_mag = mag;
  out->gyro_mag = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);

  /* Se filtra la desviacion respecto de la primera muestra, no el valor
   * absoluto. La entrada del paso banda es el angulo de montaje -unos 45 grados
   * sobre un torso real- y la senal que interesa son decimas de grado encima.
   * Cancelar 45 contra 45 en float32 para recuperar 0.36 consume casi toda la
   * precision disponible: medido en el dispositivo, el canal tilt se apartaba
   * 1.3e-3 de la referencia float64; restando el pedestal baja a 1.4e-4.
   *
   * No cuesta exactitud. Un paso alto tiene ganancia nula en continua, asi que
   * quitar una constante de su entrada no cambia su salida en regimen
   * permanente -las dos formas coinciden a 1.5e-12 en float64-. Lo unico que
   * cambia es cuanta precision le queda a float32. Python hace lo mismo.
   *
   * La magnitud tambien entra relativa a su primera muestra. Antes se le
   * restaba la media de la grabacion entera, que es no causal -la media de
   * datos que aun no han llegado- e irreproducible aqui. */
  out->tilt = pc_biquad_step(&f->bp_tilt, f->pitch - f->ped_tilt);
  out->axial = pc_biquad_step(&f->bp_axial, az_trans - f->ped_axial);
  out->hf = pc_biquad_step(&f->hp_hf, mag - f->ped_hf);
}

/* ------------------------------------------------------------------ */
/* FFT real de 128 puntos, radix-2                                     */
/* ------------------------------------------------------------------ */
/* Se implementa aqui en vez de usar esp-dsp porque esp-dsp es un componente de
 * ESP-IDF y este sketch se compila en Arduino sin dependencias. Son 128 puntos
 * y ~900 mariposas: el coste es despreciable frente al presupuesto de 20 ms. */

static void pc_fft128(float *re, float *im) {
  const int N = PC_NFFT;
  // Inversion de bits
  for (int i = 1, j = 0; i < N; i++) {
    int bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int len = 2; len <= N; len <<= 1) {
    float ang = -6.283185307179586f / len;
    float wr = cosf(ang), wi = sinf(ang);
    for (int i = 0; i < N; i += len) {
      float cr = 1.0f, ci = 0.0f;
      for (int k = 0; k < len / 2; k++) {
        int a = i + k, b = a + len / 2;
        float tr = re[b] * cr - im[b] * ci;
        float ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr;        im[a] += ti;
        float ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/* Espectro de potencia de una ventana: diezmar x10, Hann, rellenar con ceros,
 * rFFT. Devuelve PC_NFFT/2+1 bins. */
static void pc_espectro(const float *x, float *pot) {
  static float re[PC_NFFT], im[PC_NFFT];
  memset(re, 0, sizeof(re));
  memset(im, 0, sizeof(im));

  float media = 0.0f;
  for (int i = 0; i < PC_DEC_N; i++) media += x[i * PC_DECIM];
  media /= PC_DEC_N;

  for (int i = 0; i < PC_DEC_N; i++)
    re[i] = (x[i * PC_DECIM] - media) * PC_HANN_DEC[i];

  pc_fft128(re, im);
  for (int k = 0; k <= PC_NFFT / 2; k++) pot[k] = re[k] * re[k] + im[k] * im[k];
}

#endif /* PC_DSP_H */
