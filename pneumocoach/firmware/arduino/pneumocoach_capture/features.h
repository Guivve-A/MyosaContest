/* Extraccion de las 29 caracteristicas.
 * =====================================
 *
 * Puerto de `dsp.extract_features()`. El ORDEN de escritura en el vector es
 * load-bearing: el modelo se entreno sobre ese orden exacto y PC_FEATURE_NAME
 * lo documenta. Cambiar una posicion sin regenerar el modelo produce un
 * dispositivo que clasifica con confianza usando las columnas equivocadas.
 *
 * Cada bloque lleva el indice de destino escrito al lado para que un cambio
 * accidental salte a la vista en la revision.
 */

#ifndef PC_FEATURES_H
#define PC_FEATURES_H

#include "dsp.h"

/* Ventana ya filtrada, lista para extraer. */
typedef struct {
  float tilt[PC_WINDOW_N];
  float axial[PC_WINDOW_N];
  float hf[PC_WINDOW_N];
  float accel_mag[PC_WINDOW_N];
  float gyro_mag[PC_WINDOW_N];
} pc_ventana_t;

/* --------------------------------------------------------------------- */

static float pc_media(const float *x, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += x[i];
  return s / n;
}

static float pc_rms(const float *x, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += x[i] * x[i];
  return sqrtf(s / n);
}

/* Coeficiente de variacion. Devuelve 0 con menos de dos elementos, igual que
 * la version de Python: sin al menos dos periodos no hay variabilidad que
 * medir y un valor inventado seria peor que un cero explicito. */
static float pc_cv(const float *v, int n) {
  if (n < 2) return 0.0f;
  float m = pc_media(v, n);
  if (fabsf(m) < PC_EPS) return 0.0f;
  float s = 0.0f;
  for (int i = 0; i < n; i++) s += (v[i] - m) * (v[i] - m);
  return sqrtf(s / n) / m;
}

/* Caracteristicas espectrales de un canal. Escribe 7 valores consecutivos:
 * dom_freq, dom_ratio, centroid, bp_slow, bp_normal, bp_fast, spec_entropy. */
static void pc_espectrales(const float *x, float *dst) {
  static float pot[PC_NFFT / 2 + 1];
  pc_espectro(x, pot);

  const float df = PC_FS_DEC_HZ / PC_NFFT;
  /* Se ignoran los bins por debajo de 0.05 Hz: el paso banda ya quito el
   * contenido real de ahi, asi que lo que queda es campaneo del filtro. */
  int k0 = (int)ceilf(0.05f / df);

  float total = PC_EPS;
  for (int k = k0; k <= PC_NFFT / 2; k++) total += pot[k];

  int kmax = k0;
  float pmax = -1.0f, centro = 0.0f, ent = 0.0f;
  float slow = 0.0f, normal = 0.0f, fast = 0.0f;
  int nbins = 0;

  for (int k = k0; k <= PC_NFFT / 2; k++) {
    float f = k * df, p = pot[k];
    if (p > pmax) { pmax = p; kmax = k; }
    centro += f * p;
    float pn = p / total;
    ent += -pn * logf(pn + PC_EPS);
    nbins++;
  }
  /* Las bandas se acumulan sobre TODOS los bins, no solo desde k0, igual que
   * la version de Python: alli _band_power usa el espectro completo. */
  for (int k = 0; k <= PC_NFFT / 2; k++) {
    float f = k * df, p = pot[k];
    if (f >= PC_BAND_SLOW_LO && f < PC_BAND_SLOW_HI) slow += p;
    if (f >= PC_BAND_NORMAL_LO && f < PC_BAND_NORMAL_HI) normal += p;
    if (f >= PC_BAND_FAST_LO && f < PC_BAND_FAST_HI) fast += p;
  }

  dst[0] = kmax * df;                       // dom_freq
  dst[1] = pmax / total;                    // dom_ratio
  dst[2] = centro / total;                  // centroid
  dst[3] = slow / total;                    // bp_slow
  dst[4] = normal / total;                  // bp_normal
  dst[5] = fast / total;                    // bp_fast
  dst[6] = ent / logf((float)nbins);        // spec_entropy
}

/* Instante, en muestras, en que x cruza thr entre k-1 y k.
 *
 * Sin interpolar, el instante del cruce se cuantiza a la muestra entera y una
 * muestra que cae a filo del umbral se decide por diferencias de 1e-4. Medido
 * sobre este mismo dispositivo: en una ventana la senal pasaba a 9.8e-5 del
 * umbral mientras la diferencia float32/float64 frente a Python era 1.06e-4
 * -mayor que la distancia al umbral-, asi que el cruce caia en muestras
 * distintas y las cuatro caracteristicas de temporizacion se separaban un 10 %.
 *
 * No es solo paridad: a 50 Hz una muestra son 20 ms sobre respiraciones de 4-5
 * s, y con dos o tres respiraciones por ventana ese redondeo entra entero en el
 * coeficiente de variacion. */
static float pc_instante_cruce(const float *x, int k, float thr) {
  if (k == 0) return 0.0f;
  float x0 = x[k - 1], x1 = x[k];
  if (x1 == x0) return (float)k;
  float frac = (thr - x0) / (x1 - x0);
  if (frac < 0.0f || frac > 1.0f) return (float)k;  // defensivo
  return (float)(k - 1) + frac;
}

/* Segmentacion de respiraciones con comparador de histeresis a +/-0.25 sigma.
 * La histeresis, en vez de un cruce por cero pelado, es lo que impide que el
 * ruido en los puntos de giro parta una respiracion en cinco. */
static int pc_segmentar(const float *x, int n, float *periodos, float *ie, int max) {
  float m = pc_media(x, n);
  float sd = 0.0f;
  for (int i = 0; i < n; i++) sd += (x[i] - m) * (x[i] - m);
  sd = sqrtf(sd / n);
  if (sd < PC_EPS) return 0;

  const float hi = 0.25f * sd, lo = -0.25f * sd;
  int estado = 0, n_tr = 0;
  static float tr_pos[128];
  static int tr_dir[128];

  /* El umbral se compara contra el valor CRUDO, no contra el valor menos su
   * media. La media solo entra en el calculo de sigma.
   *
   * Es una diferencia de un pelo -la senal viene de un paso banda y su media
   * ronda cero- pero desplaza los cruces lo justo para que se detecten menos
   * respiraciones, y eso descuadraba las cuatro caracteristicas de
   * temporizacion mientras las otras veinticinco coincidian. La version de
   * Python es la referencia porque el modelo se entreno sobre ella. */
  for (int i = 0; i < n && n_tr < 128; i++) {
    float v = x[i];
    if (estado <= 0 && v > hi) {
      estado = 1; tr_pos[n_tr] = pc_instante_cruce(x, i, hi); tr_dir[n_tr++] = 1;
    } else if (estado >= 0 && v < lo) {
      estado = -1; tr_pos[n_tr] = pc_instante_cruce(x, i, lo); tr_dir[n_tr++] = -1;
    }
  }

  int cnt = 0;
  for (int i = 0; i + 2 < n_tr && cnt < max; i++) {
    if (tr_dir[i] != 1) continue;
    float t_insp = (tr_pos[i + 1] - tr_pos[i]) / PC_FS_HZ;
    float t_exp = (tr_pos[i + 2] - tr_pos[i + 1]) / PC_FS_HZ;
    if (t_insp <= 0.0f || t_exp <= 0.0f) continue;
    periodos[cnt] = t_insp + t_exp;
    ie[cnt] = t_insp / t_exp;
    cnt++;
  }
  return cnt;
}

/* --------------------------------------------------------------------- */

static void pc_extraer(const pc_ventana_t *w, float *f) {
  const int N = PC_WINDOW_N;

  /* --- canal tilt: indices 0..9 --- */
  f[0] = pc_rms(w->tilt, N);                                   // tilt_rms
  float mn = w->tilt[0], mx = w->tilt[0];
  for (int i = 1; i < N; i++) {
    if (w->tilt[i] < mn) mn = w->tilt[i];
    if (w->tilt[i] > mx) mx = w->tilt[i];
  }
  f[1] = mx - mn;                                              // tilt_p2p
  int cruces = 0;
  for (int i = 1; i < N; i++)
    if ((w->tilt[i - 1] < 0.0f) != (w->tilt[i] < 0.0f)) cruces++;
  f[2] = cruces / (N / PC_FS_HZ);                              // tilt_zcr
  pc_espectrales(w->tilt, &f[3]);                              // 3..9

  /* --- canal axial: indices 10..19 --- */
  f[10] = pc_rms(w->axial, N);                                 // axial_rms
  mn = w->axial[0]; mx = w->axial[0];
  for (int i = 1; i < N; i++) {
    if (w->axial[i] < mn) mn = w->axial[i];
    if (w->axial[i] > mx) mx = w->axial[i];
  }
  f[11] = mx - mn;                                             // axial_p2p
  cruces = 0;
  for (int i = 1; i < N; i++)
    if ((w->axial[i - 1] < 0.0f) != (w->axial[i] < 0.0f)) cruces++;
  f[12] = cruces / (N / PC_FS_HZ);                             // axial_zcr
  pc_espectrales(w->axial, &f[13]);                            // 13..19

  /* --- cruzadas: 20, 21 --- */
  f[20] = log10f((f[0] + PC_EPS) / (f[10] + PC_EPS));          // log_tilt_axial_ratio

  float mt = pc_media(w->tilt, N), ma = pc_media(w->axial, N);
  float num = 0.0f, dt = 0.0f, da = 0.0f;
  for (int i = 0; i < N; i++) {
    float a = w->tilt[i] - mt, b = w->axial[i] - ma;
    num += a * b; dt += a * a; da += b * b;
  }
  f[21] = num / (sqrtf(dt * da) + PC_EPS);                     // tilt_axial_xcorr

  /* --- temporizacion de respiraciones: 22..25 ---
   * Se segmenta sobre tilt, que es el canal de mayor SNR en todas las clases
   * (ver la nota de fisica en synth.py). */
  static float per[32], ie[32];
  int nb = pc_segmentar(w->tilt, N, per, ie, 32);
  f[22] = nb ? 60.0f / pc_media(per, nb) : 0.0f;               // breath_rate_bpm
  f[23] = pc_cv(per, nb);                                      // breath_period_cv
  f[24] = nb ? pc_media(ie, nb) : 0.0f;                        // ie_ratio_mean
  f[25] = pc_cv(ie, nb);                                       // ie_ratio_cv

  /* --- artefacto: 26..28, sobre la senal SIN paso banda --- */
  float mmag = pc_media(w->accel_mag, N);
  float rms_ac = 0.0f;
  for (int i = 0; i < N; i++) {
    float d = w->accel_mag[i] - mmag;
    rms_ac += d * d;
  }
  rms_ac = sqrtf(rms_ac / N);
  f[26] = pc_rms(w->hf, N) / (rms_ac + PC_EPS);                // hf_energy_ratio

  float jmax = 0.0f;
  for (int i = 1; i < N; i++) {
    float d = fabsf(w->accel_mag[i] - w->accel_mag[i - 1]);
    if (d > jmax) jmax = d;
  }
  f[27] = jmax * PC_FS_HZ;                                     // jerk_max

  float mg = pc_media(w->gyro_mag, N);
  float sg = 0.0f;
  for (int i = 0; i < N; i++) {
    float d = w->gyro_mag[i] - mg;
    sg += d * d;
  }
  f[28] = sqrtf(sg / N);                                       // gyro_rms
}

#endif /* PC_FEATURES_H */
