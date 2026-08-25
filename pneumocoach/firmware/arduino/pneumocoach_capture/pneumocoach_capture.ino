/* PneumoCoach - firmware de adquisicion y volcado CSV
 * ===================================================
 * MYOSA Mini Kit / ESP32-WROOM-32E
 *
 * PROPOSITO
 *   Streamear el MPU6050 a exactamente 50 Hz por el puerto serial en CSV, para
 *   construir el primer dataset real. No hay inferencia aqui a proposito: la
 *   regla es no tocar la IA hasta que la adquisicion sea impecable.
 *
 * POR QUE NO USA millis()
 *   Un lazo con millis() no da muestreo determinista: el jitter depende de todo
 *   lo demas que pase en el lazo, y el jitter en el dominio del tiempo se
 *   convierte en ensuciamiento del espectro justo en la banda de 0.1-1 Hz que
 *   nos importa. En su lugar, una tarea de FreeRTOS anclada al nucleo 1 con
 *   vTaskDelayUntil() despierta en instantes absolutos, asi que el error de
 *   periodo no se acumula. El nucleo 0 queda libre para el serial (y despues
 *   para BLE y la OLED, que es donde el stack de radio de Espressif vive por
 *   defecto).
 *
 * DEPENDENCIAS
 *   Ninguna. Solo Wire.h. Se compila en el Arduino IDE recien instalado, sin
 *   bajar una sola libreria. Eso es deliberado: hoy queremos datos, no pelear
 *   con gestores de paquetes.
 *
 * SALIDA
 *   Cabecera con '#', luego lineas CSV:
 *     seq,t_us,ax,ay,az,gx,gy,gz,mark
 *   ax..az y gx..gz son CUENTAS CRUDAS int16, no unidades fisicas. El escalado
 *   se hace en el PC, para que el archivo guarde exactamente lo que el registro
 *   0x3B entrego y cualquier conversion sea auditable despues.
 *
 * COMANDOS (un caracter por el serial)
 *   s  iniciar streaming        x  detener
 *   m  marcar evento (la columna `mark` se pone en 1 esa muestra)
 *   i  reimprimir cabecera e info del bus
 *   ?  ayuda
 */

#include <Preferences.h>
#include <Wire.h>

#include "apds.h"
#include "oled.h"
#include "sesion.h"
#include "inferencia.h"
#include "ble.h"

// ---------------------------------------------------------------------------
// Configuracion. Estos valores DEBEN coincidir con ml/pneumocoach/config.py.
// ---------------------------------------------------------------------------
#define PIN_SDA 21
#define PIN_SCL 22
#define I2C_HZ 400000L  // si el bus falla con la correa larga, bajar a 100000

// MYOSA amarra AD0 en alto. La direccion habitual 0x68 NO responde ACK aqui.
#define ADDR_MPU6050 0x69
#define ADDR_APDS9960 0x39
#define ADDR_BMP180 0x77
#define ADDR_SSD1306 0x3C

#define FS_HZ 50
#define PERIOD_MS (1000 / FS_HZ)  // 20 ms

// Registros. Comunes a toda la familia MPU-6xxx salvo ACCEL_CONFIG2, que solo
// existe en el MPU6500 y derivados.
#define REG_SMPLRT_DIV 0x19
#define REG_CONFIG 0x1A
#define REG_GYRO_CONFIG 0x1B
#define REG_ACCEL_CONFIG 0x1C
#define REG_ACCEL_CONFIG2 0x1D  // solo MPU6500/9250
#define REG_ACCEL_XOUT_H 0x3B
#define REG_PWR_MGMT_1 0x6B
#define REG_WHO_AM_I 0x75

/* Variantes que hemos visto en la practica.
 *
 * El modulo del kit dice "MPU-6050" impreso, pero su WHO_AM_I devuelve 0x70,
 * o sea silicio MPU6500. Es habitual en los clones GY-521: el encapsulado y la
 * serigrafia son de 6050 y el dado es 6500. Para nuestros fines son
 * compatibles a nivel de registros de datos (la rafaga desde 0x3B es
 * identica), pero la configuracion NO lo es:
 *
 *   - En el MPU6050, CONFIG(0x1A).DLPF_CFG fija el ancho de banda de AMBOS,
 *     acelerometro y giroscopo.
 *   - En el MPU6500 ese registro solo afecta al giroscopo. El acelerometro
 *     tiene su propio filtro en ACCEL_CONFIG2(0x1D), que en el 6050 ni
 *     siquiera existe.
 *
 * Si no se escribe ACCEL_CONFIG2, el acelerometro del 6500 queda con 460 Hz de
 * ancho de banda. Muestreando a 50 Hz (Nyquist 25 Hz) eso mete aliasing de
 * banda ancha justo encima de la senal respiratoria, y el efecto es
 * traicionero: la senal se ve razonable pero el espectro esta contaminado.
 */
typedef enum { IMU_DESCONOCIDO, IMU_MPU6050, IMU_MPU6500 } imu_tipo_t;
static imu_tipo_t imu_tipo = IMU_DESCONOCIDO;
static uint8_t imu_whoami = 0;

static const uint32_t BAUD = 115200;

// ---------------------------------------------------------------------------
// Estado compartido
// ---------------------------------------------------------------------------
typedef struct {
  uint32_t seq;
  uint32_t t_us;
  int16_t a[3];
  int16_t g[3];
  uint8_t mark;
} sample_t;

static QueueHandle_t q_samples;
static SemaphoreHandle_t mtx_i2c;
static volatile bool streaming = false;
static volatile bool mark_pending = false;
static volatile uint32_t n_dropped = 0;  // cola llena = muestra perdida
static volatile uint32_t n_i2c_err = 0;

/* Lo que la pantalla necesita saber. Se escribe desde la tarea de adquisicion
 * y se lee desde la de pantalla. Son int32/float alineados y de escritura
 * atomica en Xtensa, asi que no hace falta mutex: un valor rasgado en un
 * indicador visual no tiene consecuencias, y un mutex aqui podria bloquear la
 * adquisicion, que si las tiene. */
static volatile float ui_accel_mag = 0.0f;
static volatile float ui_tilt_inst = 0.0f;
static volatile uint32_t ui_muestras = 0;
static volatile bool ui_imu_ok = false;
static volatile uint8_t ui_dispositivos = 0;

/* Cola dedicada del DSP.
 *
 * No se reutiliza q_samples porque esa la vacia la tarea de CSV y una cola no
 * puede tener dos consumidores. Y sobre todo: el DSP tiene que correr SIEMPRE,
 * tambien sin grabar, porque los filtros necesitan asentarse y la calibracion
 * ocurre antes de que exista ningun CSV.
 *
 * La adquisicion solo encola; el trabajo -1 ms por ventana cada 3 s- se hace en
 * el nucleo 0, para no meterlo en la tarea que tiene que despertar cada 20 ms
 * sin fallar. */
typedef struct { int16_t v[6]; } raw6_t;
static QueueHandle_t q_dsp;
static TaskHandle_t h_dsp = NULL;
static volatile uint32_t n_dsp_perdidas = 0;
static pc_sesion_t g_sesion;
/* Ultimo veredicto, para la pantalla. -1 = por debajo del piso de confianza,
 * -2 = todavia no se puede clasificar. */
static volatile int ui_veredicto = -2;
static volatile float ui_confianza = 0.0f;
static volatile uint32_t ui_us_inferencia = 0;
static volatile int ui_ultimo_gesto = 0;
static volatile uint32_t ui_t_gesto = 0;
/* Modo prueba: los gestos se detectan y se informan, pero no avanzan la
 * sesion. Sin el, el primer gesto hacia arriba arrancaria la calibracion y
 * los siguientes caerian en una fase que los ignora, y la prueba mediria
 * el gating en vez del decodificador. */
static volatile bool g_gestos_prueba = false;

/* Matriz de montaje, en almacenamiento no volatil.
 *
 * No puede ir en el header generado: no es una constante de compilacion sino
 * una medida del sujeto y de COMO quedo pegada la placa hoy. Hornearla en el
 * firmware obligaria a reflashear cada vez que alguien se recoloca el sensor.
 *
 * Sin ella el dispositivo procesa sobre los ejes crudos del sensor mientras que
 * el entrenamiento usa el marco anatomico corregido, y el modelo recibe
 * caracteristicas de un sistema de coordenadas distinto al suyo. No falla:
 * clasifica con confianza sobre el eje equivocado. */
static Preferences g_prefs;
static float g_R[9];
static float g_sesgo[3];
static bool g_montaje_ok = false;

static bool montaje_cargar() {
  g_prefs.begin("pneumo", true);
  size_t nR = g_prefs.getBytes("R", g_R, sizeof(g_R));
  size_t nb = g_prefs.getBytes("bias", g_sesgo, sizeof(g_sesgo));
  g_prefs.end();
  g_montaje_ok = (nR == sizeof(g_R) && nb == sizeof(g_sesgo));
  if (g_montaje_ok) pc_front_set_mount(&g_sesion.pipe.front, g_R, g_sesgo);
  return g_montaje_ok;
}

static void montaje_guardar() {
  g_prefs.begin("pneumo", false);
  g_prefs.putBytes("R", g_R, sizeof(g_R));
  g_prefs.putBytes("bias", g_sesgo, sizeof(g_sesgo));
  g_prefs.end();
  g_montaje_ok = true;
  pc_front_set_mount(&g_sesion.pipe.front, g_R, g_sesgo);
}

// ---------------------------------------------------------------------------
// I2C de bajo nivel
// ---------------------------------------------------------------------------
static bool reg_write(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool reg_read(uint8_t addr, uint8_t reg, uint8_t *buf, size_t n) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;  // repeated start
  if (Wire.requestFrom((int)addr, (int)n, (int)true) != (int)n) return false;
  for (size_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

static bool probe(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

// ---------------------------------------------------------------------------
// Arranque del sensor
// ---------------------------------------------------------------------------
static bool mpu_init() {
  if (!reg_read(ADDR_MPU6050, REG_WHO_AM_I, &imu_whoami, 1)) {
    Serial.println(F("# no se pudo leer WHO_AM_I"));
    return false;
  }

  switch (imu_whoami) {
    // El MPU6050 devuelve la direccion con AD0 enmascarado, o sea 0x68 aunque
    // el dispositivo conteste en 0x69. No es un error.
    case 0x68: imu_tipo = IMU_MPU6050; break;
    case 0x70: imu_tipo = IMU_MPU6500; break;  // MPU6500
    case 0x71:                                 // MPU9250
    case 0x73:                                 // MPU9255
      imu_tipo = IMU_MPU6500;
      break;
    default:
      Serial.printf("# WHO_AM_I no reconocido: 0x%02X\n", imu_whoami);
      return false;
  }
  Serial.printf("# IMU detectada: %s (WHO_AM_I 0x%02X)\n",
                imu_tipo == IMU_MPU6050 ? "MPU6050" : "MPU6500/9250", imu_whoami);

  // PWR_MGMT_1 = 0x01 -> SLEEP=0 y CLKSEL=1 (PLL con referencia del giroscopo).
  // Escribir 0x00 tambien despierta el chip, pero deja el oscilador interno de
  // 8 MHz, que deriva con la temperatura. La libreria del fabricante usa
  // MPU6050_CLOCK_PLL_XGYRO por la misma razon, y la hoja de datos de InvenSense
  // lo recomienda. Con una banda de interes de 0.1-1 Hz, la deriva del reloj se
  // traduce en corrimiento de la frecuencia respiratoria medida.
  if (!reg_write(ADDR_MPU6050, REG_PWR_MGMT_1, 0x01)) return false;
  delay(100);

  // Anti-aliasing. Muestreamos a 50 Hz, o sea Nyquist = 25 Hz: el ancho de
  // banda analogico tiene que quedar por debajo.
  //   MPU6050 -> CONFIG.DLPF_CFG=4 cubre accel (21 Hz) y gyro (20 Hz).
  //   MPU6500 -> CONFIG.DLPF_CFG=4 solo cubre el gyro (20 Hz); el accel
  //              necesita ademas ACCEL_CONFIG2.A_DLPF_CFG=4 (21.2 Hz).
  if (!reg_write(ADDR_MPU6050, REG_CONFIG, 0x04)) return false;
  if (imu_tipo == IMU_MPU6500) {
    if (!reg_write(ADDR_MPU6050, REG_ACCEL_CONFIG2, 0x04)) return false;
  }

  // Con DLPF activo la tasa base es 1 kHz. Divisor 19 -> 50 Hz, asi cada
  // lectura nuestra cae sobre una conversion nueva y no repetimos muestras.
  if (!reg_write(ADDR_MPU6050, REG_SMPLRT_DIV, 19)) return false;

  if (!reg_write(ADDR_MPU6050, REG_GYRO_CONFIG, 0x00)) return false;   // +/-250 dps
  if (!reg_write(ADDR_MPU6050, REG_ACCEL_CONFIG, 0x00)) return false;  // +/-2 g
  delay(50);

  // Releer lo escrito. Un ACK en el bus no garantiza que el registro haya
  // tomado el valor, y una configuracion silenciosamente ignorada es
  // exactamente el fallo que no se nota hasta que el modelo no funciona.
  uint8_t v[4];
  bool ok = true;
  ok &= reg_read(ADDR_MPU6050, REG_SMPLRT_DIV, &v[0], 1);
  ok &= reg_read(ADDR_MPU6050, REG_CONFIG, &v[1], 1);
  ok &= reg_read(ADDR_MPU6050, REG_GYRO_CONFIG, &v[2], 1);
  ok &= reg_read(ADDR_MPU6050, REG_ACCEL_CONFIG, &v[3], 1);
  if (!ok) return false;
  Serial.printf("# verificacion: SMPLRT=%u CONFIG=%u GYRO=%u ACCEL=%u\n",
                v[0], v[1], v[2], v[3]);
  if (v[0] != 19 || v[1] != 4) {
    Serial.println(F("# ATENCION: la configuracion no quedo escrita"));
    return false;
  }
  if (imu_tipo == IMU_MPU6500) {
    uint8_t a2 = 0;
    if (reg_read(ADDR_MPU6050, REG_ACCEL_CONFIG2, &a2, 1))
      Serial.printf("# ACCEL_CONFIG2=%u (anti-aliasing del acelerometro)\n", a2);
  }
  return true;
}

/* Comprueba que el sensor entrega fisica creible, no solo que responde.
 * Un modulo dormido o mal configurado contesta el bus perfectamente y
 * devuelve ceros o un valor congelado. */
static void mpu_autoprueba() {
  uint8_t raw[14];
  double suma = 0;
  int n = 0;
  int16_t primero = 0;
  bool vario = false;

  for (int i = 0; i < 50; i++) {
    if (!reg_read(ADDR_MPU6050, REG_ACCEL_XOUT_H, raw, 14)) continue;
    int16_t ax = (int16_t)((raw[0] << 8) | raw[1]);
    int16_t ay = (int16_t)((raw[2] << 8) | raw[3]);
    int16_t az = (int16_t)((raw[4] << 8) | raw[5]);
    if (n == 0) primero = ax;
    else if (ax != primero) vario = true;
    double gx = ax / 16384.0, gy = ay / 16384.0, gz = az / 16384.0;
    suma += sqrt(gx * gx + gy * gy + gz * gz);
    n++;
    delay(20);
  }
  if (n == 0) {
    Serial.println(F("# AUTOPRUEBA: sin lecturas"));
    return;
  }
  double mag = suma / n;
  Serial.printf("# autoprueba: |accel| = %.3f g sobre %d muestras\n", mag, n);
  if (mag < 0.85 || mag > 1.15)
    Serial.println(F("# ATENCION: la magnitud no ronda 1 g. Escala o montaje mal."));
  else
    Serial.println(F("# gravedad OK (~1 g)"));
  if (!vario)
    Serial.println(F("# ATENCION: la lectura no cambia. Sensor dormido?"));
}

// ---------------------------------------------------------------------------
// Tarea de adquisicion  -- nucleo 1, prioridad alta
// ---------------------------------------------------------------------------
static void task_acquire(void *arg) {
  (void)arg;
  uint8_t raw[14];
  sample_t s;
  s.seq = 0;

  TickType_t last = xTaskGetTickCount();
  const TickType_t period = pdMS_TO_TICKS(PERIOD_MS);

  for (;;) {
    // Despertar en instantes absolutos. El error de periodo no se acumula,
    // que es justo lo que un lazo con delay() no puede garantizar.
    vTaskDelayUntil(&last, period);

    /* La lectura ocurre SIEMPRE, se este grabando o no.
     *
     * Antes esta guarda estaba aqui, y dejaba la vista de colocacion de la
     * pantalla congelada en cero: mostraba el angulo solo mientras se
     * grababa, que es justo cuando ya no hace falta. La pantalla sirve para
     * colocar el sensor ANTES de empezar.
     *
     * El coste es leer 14 bytes cada 20 ms tambien en reposo, y en reposo
     * no hay nada mas compitiendo por el bus. */

    bool ok = false;
    if (xSemaphoreTake(mtx_i2c, pdMS_TO_TICKS(5)) == pdTRUE) {
      ok = reg_read(ADDR_MPU6050, REG_ACCEL_XOUT_H, raw, 14);
      xSemaphoreGive(mtx_i2c);
    }
    if (!ok) {
      n_i2c_err++;
      continue;
    }

    s.t_us = micros();
    s.a[0] = (int16_t)((raw[0] << 8) | raw[1]);
    s.a[1] = (int16_t)((raw[2] << 8) | raw[3]);
    s.a[2] = (int16_t)((raw[4] << 8) | raw[5]);
    // raw[6..7] es temperatura; se omite, no la usamos
    s.g[0] = (int16_t)((raw[8] << 8) | raw[9]);
    s.g[1] = (int16_t)((raw[10] << 8) | raw[11]);
    s.g[2] = (int16_t)((raw[12] << 8) | raw[13]);
    s.mark = mark_pending ? 1 : 0;
    mark_pending = false;

    /* Indicadores para la pantalla. El angulo es el crudo del acelerometro,
     * no el del filtro complementario: la pantalla solo tiene que mostrar que
     * el sensor responde al movimiento, y calcular aqui el filtro gastaria
     * tiempo en la tarea que menos puede permitirselo. */
    float ax = s.a[0] / 16384.0f, ay = s.a[1] / 16384.0f, az = s.a[2] / 16384.0f;
    ui_accel_mag = sqrtf(ax * ax + ay * ay + az * az);
    ui_tilt_inst = atan2f(az, sqrtf(ax * ax + ay * ay)) * 57.2958f;

    /* El DSP recibe la muestra pase lo que pase. Los filtros tienen que
     * asentarse y la calibracion sucede antes de que haya nada que grabar. */
    raw6_t r6 = {{s.a[0], s.a[1], s.a[2], s.g[0], s.g[1], s.g[2]}};
    if (xQueueSend(q_dsp, &r6, 0) != pdTRUE) n_dsp_perdidas++;

    /* A partir de aqui solo si se graba: el contador y la cola alimentan el
     * CSV, y en reposo no debe avanzar ninguno de los dos. */
    if (!streaming) continue;
    s.seq++;
    ui_muestras = s.seq;

    if (xQueueSend(q_samples, &s, 0) != pdTRUE) n_dropped++;
  }
}

// ---------------------------------------------------------------------------
// Tarea de salida  -- nucleo 0, prioridad baja
// ---------------------------------------------------------------------------
static void task_print(void *arg) {
  (void)arg;
  sample_t s;
  char line[96];
  for (;;) {
    if (xQueueReceive(q_samples, &s, portMAX_DELAY) == pdTRUE) {
      int n = snprintf(line, sizeof(line), "%lu,%lu,%d,%d,%d,%d,%d,%d,%u\n",
                       (unsigned long)s.seq, (unsigned long)s.t_us,
                       s.a[0], s.a[1], s.a[2], s.g[0], s.g[1], s.g[2], s.mark);
      Serial.write((const uint8_t *)line, n);
    }
  }
}

// ---------------------------------------------------------------------------
// Tarea de pantalla  -- nucleo 0, prioridad baja, 5 Hz
// ---------------------------------------------------------------------------

/* Cinco hercios y no mas. Cada refresco completo son 1024 bytes por el mismo
 * bus I2C que usa el IMU; a 20 Hz la pantalla se comeria una fraccion
 * apreciable del ancho de banda y empezarian a perderse muestras. Para un
 * indicador que mira un humano, 5 Hz es de sobra. */
// ---------------------------------------------------------------------------
// Tarea de pantalla  -- nucleo 0, prioridad baja, 5 Hz
// ---------------------------------------------------------------------------

/* Cinco hercios y no mas. Cada refresco completo son 1024 bytes por el mismo
 * bus I2C que usa el IMU; a 20 Hz la pantalla se comeria una fraccion
 * apreciable del ancho de banda y empezarian a perderse muestras. Para un
 * indicador que mira un humano, 5 Hz es de sobra. */
/* Gestos. Nucleo 0, prioridad 1: por debajo de todo lo demas.
 *
 * Un gesto que llega tarde es un incordio; una muestra de la IMU que llega
 * tarde es un agujero en la senal. El orden de prioridades lo dice.
 *
 * El mutex se toma UNA vez por sondeo y se suelta enseguida: dentro caben como
 * mucho 8 conjuntos de la FIFO, unos 32 bytes, menos de 1 ms de bus. */
static void task_gestos(void *arg) {
  (void)arg;
  TickType_t last = xTaskGetTickCount();
  for (;;) {
    vTaskDelayUntil(&last, pdMS_TO_TICKS(20));
    if (!g_apds.presente) continue;

    pc_gesto_t g = PC_GESTO_NINGUNO;
    if (xSemaphoreTake(mtx_i2c, pdMS_TO_TICKS(4)) == pdTRUE) {
      g = pc_apds_poll();
      xSemaphoreGive(mtx_i2c);
    }
    if (g == PC_GESTO_NINGUNO) continue;

    /* Antirrebote. Un solo paso de la mano deja varios conjuntos en la FIFO y
     * puede resolverse como dos gestos seguidos; el segundo desharia lo que
     * hizo el primero. */
    uint32_t ahora = millis();
    if (ahora - ui_t_gesto < 1500) continue;

    /* Linea cruda, siempre y antes de cualquier filtro por estado: es la que
     * lee tools/probar_gestos.py, y tiene que reflejar lo que DETECTO el
     * sensor, no lo que el firmware decidio hacer con ello. */
    Serial.printf("G,%s\n", pc_gesto_nombre(g));

    /* DONDE se aceptan gestos, y por que no en todas partes.
     *
     * Durante las maniobras de referencia el paciente tiene las DOS MANOS sobre
     * el pecho y el abdomen -la consigna de dos manos lo exige- y es justo ahi
     * donde mira este sensor. Un movimiento de la mano del pecho se leeria como
     * un gesto y cancelaria la calibracion a los veinte segundos de empezarla.
     *
     * Asi que los gestos solo valen donde de verdad hacen falta: arrancar la
     * calibracion, arrancar el coaching, y terminarlo. En el resto se registran
     * para diagnostico y no hacen nada. */
    pc_sesion_est_t e = g_sesion.est;
    if (g_gestos_prueba) {
      ui_ultimo_gesto = (int)g;
      ui_t_gesto = ahora;
      continue;                       // detectado e informado, sin actuar
    }
    bool acepta = (e == SES_LISTO || e == SES_CALIBRADO || e == SES_ACTIVA);
    ui_ultimo_gesto = (int)g;
    ui_t_gesto = ahora;

    if (!acepta) {
      Serial.printf("# GESTO %s ignorado en %s\n",
                    pc_gesto_nombre(g), pc_sesion_nombre(e));
      continue;
    }
    if (g == PC_GESTO_ARRIBA && (e == SES_LISTO || e == SES_CALIBRADO)) {
      pc_sesion_siguiente(&g_sesion);
      Serial.println(F("# GESTO arriba -> siguiente"));
    } else if (g == PC_GESTO_ABAJO && e == SES_ACTIVA) {
      pc_sesion_cancelar(&g_sesion);
      Serial.println(F("# GESTO abajo -> terminar sesion"));
    } else {
      Serial.printf("# GESTO %s sin accion en %s\n",
                    pc_gesto_nombre(g), pc_sesion_nombre(e));
    }
  }
}

static void task_dsp(void *arg) {
  (void)arg;
  raw6_t r;
  for (;;) {
    if (xQueueReceive(q_dsp, &r, portMAX_DELAY) != pdTRUE) continue;
    if (!pc_sesion_muestra(&g_sesion, r.v)) continue;
    if (g_sesion.est != SES_ACTIVA) continue;

    /* Solo se clasifica con la sesion calibrada y el montaje cargado: son las
     * dos condiciones que hacen que z signifique aqui lo mismo que significaba
     * durante el entrenamiento. */
    if (!pc_inferencia_lista(&g_sesion, g_montaje_ok)) {
      ui_veredicto = -2;   // falta calibrar
      continue;
    }
    pc_veredicto_t v;
    if (!pc_inferir(g_sesion.z, &v)) { ui_veredicto = -2; continue; }
    ui_veredicto = v.clase;
    ui_confianza = v.confianza;
    ui_us_inferencia = v.us;
    pc_ble_veredicto(v.clase, v.confianza, g_sesion.f, g_sesion.f[20]);
  }
}

// ---------------------------------------------------------------------------
// Tarea de pantalla  -- nucleo 0, prioridad 2
// ---------------------------------------------------------------------------
static void task_pantalla(void *arg) {
  (void)arg;
  char linea[24];
  uint32_t ultimas = 0;
  uint32_t t_ultimo = millis();

  for (;;) {
    vTaskDelay(pdMS_TO_TICKS(200));
    if (!oled_presente) continue;

    /* Tasa efectiva medida contra el reloj del sistema. Es la misma cifra que
     * verifica el PC, pero visible sin cable: si cae, se ve en la pantalla. */
    uint32_t ahora = millis();
    uint32_t n = ui_muestras;
    float hz = (ahora > t_ultimo) ? (n - ultimas) * 1000.0f / (ahora - t_ultimo) : 0.0f;
    ultimas = n;
    t_ultimo = ahora;

    oled_clear();
    oled_text(0, 0, "PNEUMOCOACH", 1);
    snprintf(linea, sizeof(linea), "%s%s",
             pc_sesion_nombre(g_sesion.est), streaming ? " REC" : "");
    oled_text(OLED_W - 6 * (int)strlen(linea), 0, linea, 1);
    oled_hline(9);

    if (!ui_imu_ok) {
      oled_text_center(24, "SIN IMU", 2);
      oled_text_center(46, "REVISAR 0X69", 1);
      oled_flush();
      continue;
    }

    /* La vista la manda la fase de la sesion, no el hecho de estar grabando.
     * El CSV es una herramienta de laboratorio; el paciente solo ve en que
     * punto de su sesion esta. */
    switch (g_sesion.est) {
      case SES_CALENTANDO: {
        /* El asentamiento del filtro se ensena en vez de esconderse: coincide
         * con el rato en que el paciente se esta colocando la placa, asi que
         * no es tiempo muerto si se ve para que sirve. */
        oled_text(0, 13, "COLOCA EL SENSOR", 1);
        snprintf(linea, sizeof(linea), "%+.1f", ui_tilt_inst);
        oled_text_center(24, linea, 2);
        oled_text_center(42, "GRADOS", 1);
        float r = pc_sesion_restante(&g_sesion);
        snprintf(linea, sizeof(linea), "LISTO EN %2.0f s", r);
        oled_text(0, 54, linea, 1);
        oled_barra(78, 54, 50, 8, 1.0f - r / (PC_WARMUP_N / PC_FS_HZ));
        break;
      }

      case SES_LISTO: {
        float r = pc_sesion_restante(&g_sesion);
        oled_text_center(12, "PREPARATE", 1);
        snprintf(linea, sizeof(linea), "%2.0f", r);
        oled_text_center(22, linea, 2);
        oled_text_center(42, "MANO DERECHA AL PECHO", 1);
        oled_text_center(52, "IZQUIERDA AL ABDOMEN", 1);
        oled_barra(0, 60, OLED_W, 4, 1.0f - r / PC_PREPARA_SEGUNDOS);
        break;
      }

      case SES_CAL_DIA:
      case SES_CAL_TOR: {
        bool tor = g_sesion.est == SES_CAL_TOR;
        oled_text_center(12, tor ? "RESPIRA CON" : "RESPIRA CON", 1);
        oled_text_center(22, tor ? "EL PECHO" : "EL ABDOMEN", 2);
        float r = pc_sesion_restante(&g_sesion);
        snprintf(linea, sizeof(linea), "%2.0f", r);
        oled_text(0, 42, linea, 2);
        oled_text(26, 48, "s", 1);
        oled_barra(38, 44, 90, 10, 1.0f - r / PC_REF_SEGUNDOS);
        snprintf(linea, sizeof(linea), "%s  %u ventanas",
                 tor ? "2 de 2" : "1 de 2",
                 (unsigned)(tor ? g_sesion.ref.n_tor : g_sesion.ref.n_dia));
        oled_text(0, 57, linea, 1);
        break;
      }

      case SES_CALIBRADO:
        /* El contraste se ensena como dato, sin semaforo. Ningun estadistico
         * probado separa una calibracion buena de una en la que el paciente
         * repitio la misma maniobra (tools/medir_calidad_calibracion.py), asi
         * que un "CALIBRACION OK" en pantalla seria una garantia inventada. */
        oled_text_center(10, "CALIBRADO", 2);
        snprintf(linea, sizeof(linea), "%.2f", g_sesion.contraste);
        oled_text_center(28, linea, 2);
        oled_text_center(46, "CONTRASTE (SIN UMBRAL)", 1);
        {
          float r = pc_sesion_restante(&g_sesion);
          snprintf(linea, sizeof(linea), "EMPIEZA EN %2.0f s", r);
          oled_text_center(57, linea, 1);
        }
        break;

      case SES_ACTIVA: {
        /* El veredicto manda; las medidas quedan de apoyo.
         *
         * Si esta por debajo del piso de confianza NO se ensena una clase: se
         * ensena que no hay veredicto. Negarse a adivinar es mejor que coachear
         * mal, y en pantalla la diferencia tiene que verse. */
        int v = ui_veredicto;
        if (v == -2) {
          oled_text_center(14, "SIN CALIBRAR", 1);
          oled_text_center(30, "---", 2);
          oled_text_center(50, "FALTA REFERENCIA", 1);
        } else if (v < 0) {
          oled_text_center(16, "---", 2);
          oled_text_center(38, "SIN VEREDICTO", 1);
          snprintf(linea, sizeof(linea), "conf %.2f < %.2f",
                   ui_confianza, PC_CONFIDENCE_FLOOR);
          oled_text_center(50, linea, 1);
        } else {
          oled_text_center(13, PC_CLASS_OLED[v], 2);
          oled_barra(0, 32, OLED_W, 8, ui_confianza);
          snprintf(linea, sizeof(linea), "conf %.0f%%", ui_confianza * 100.0f);
          oled_text(0, 43, linea, 1);
        }
        snprintf(linea, sizeof(linea), "%.1f rpm  I:E %.2f",
                 g_sesion.f[22], g_sesion.f[24]);
        oled_text(0, 56, linea, 1);
        break;
      }
    }

    oled_flush();

    /* La telemetria sale desde aqui y no desde una tarea propia: esta rutina ya
     * despierta a 5 Hz y ya tiene delante el estado de la sesion. Una tarea mas
     * seria otro consumidor del mismo dato y otro sitio donde desincronizarse
     * de lo que muestra la pantalla. */
    pc_ble_sensores(ui_tilt_inst, g_sesion.z[10], ui_accel_mag, 0.0f);
    pc_ble_estado(&g_sesion, g_montaje_ok, streaming);
  }
}

// ---------------------------------------------------------------------------
// Diagnostico
// ---------------------------------------------------------------------------
static void scan_bus() {
  Serial.printf("# montaje: %s\n",
                g_montaje_ok ? "cargado de NVS"
                             : "SIN CALIBRAR - ejes crudos del sensor");
  Serial.println(F("# --- escaneo I2C ---"));
  int found = 0;
  for (uint8_t a = 1; a < 127; a++) {
    if (!probe(a)) continue;
    found++;
    const char *name = "desconocido";
    if (a == ADDR_MPU6050) name = "MPU6050 (esperado)";
    else if (a == ADDR_APDS9960) name = "APDS9960";
    else if (a == ADDR_BMP180) name = "BMP180";
    else if (a == ADDR_SSD1306) name = "SSD1306 OLED";
    else if (a == 0x68) name = "MPU6050 en 0x68: AD0 esta BAJO, revisar la placa";
    Serial.printf("#   0x%02X  %s\n", a, name);
  }
  ui_dispositivos = (uint8_t)found;
  Serial.printf("# %d dispositivos\n", found);
  if (!probe(ADDR_MPU6050))
    Serial.println(F("# ATENCION: nada responde en 0x69. Sin IMU no hay captura."));
}

static void print_header() {
  Serial.println();
  Serial.println(F("# PneumoCoach captura v1"));
  Serial.printf("# fs=%d Hz  accel=+/-2g (16384 LSB/g)  gyro=+/-250dps (131 LSB/dps)\n", FS_HZ);
  Serial.printf("# DLPF=21Hz  I2C=%ld Hz  addr=0x%02X\n", I2C_HZ, ADDR_MPU6050);
  Serial.printf("# perdidas=%lu  errores_i2c=%lu\n",
                (unsigned long)n_dropped, (unsigned long)n_i2c_err);
  Serial.println(F("# columnas: seq,t_us,ax,ay,az,gx,gy,gz,mark"));
  Serial.println(F("# valores CRUDOS int16 tal como salen del registro 0x3B"));
}

static void handle_cmd(char c) {
  switch (c) {
    case 's':
      n_dropped = n_i2c_err = 0;
      print_header();
      streaming = true;
      Serial.println(F("# STREAMING"));
      break;
    case 'x':
      streaming = false;
      Serial.printf("# DETENIDO  perdidas=%lu  errores_i2c=%lu\n",
                    (unsigned long)n_dropped, (unsigned long)n_i2c_err);
      break;
    case 'm': mark_pending = true; break;
    case 'T': modo_paridad(); break;
    case 'i': print_header(); scan_bus(); break;
    /* Los mismos eventos que produciran los gestos del APDS9960. Existir por
     * serie primero no es un apano: es lo que permite probar y demostrar el
     * flujo completo sin depender de un modulo que aun no esta en el bus. */
    case 'n': pc_sesion_siguiente(&g_sesion); break;
    case 'k': pc_sesion_cancelar(&g_sesion); break;
    /* Vuelca los dos vectores de referencia. Sin esto, la unica prueba de que
     * la calibracion a bordo funciona seria que el dispositivo dice que ha
     * terminado, que no prueba nada sobre los numeros. */
    /* M,<r0..r8>,<b0..b2>  fija la matriz de montaje y la guarda. */
    case 'M': {
      char linea[240];
      size_t n = Serial.readBytesUntil('\n', linea, sizeof(linea) - 1);
      linea[n] = 0;
      float v[12];
      char *tok = strtok(linea, ",");
      int k = 0;
      while (tok && k < 12) { v[k++] = atof(tok); tok = strtok(NULL, ","); }
      if (k != 12) {
        Serial.printf("# MONTAJE ERROR: %d valores, hacen falta 12\n", k);
        break;
      }
      memcpy(g_R, v, sizeof(g_R));
      memcpy(g_sesgo, v + 9, sizeof(g_sesgo));
      montaje_guardar();
      pc_sesion_reset(&g_sesion);
      pc_front_set_mount(&g_sesion.pipe.front, g_R, g_sesgo);
      Serial.println(F("# MONTAJE guardado; la sesion se reinicia"));
      break;
    }
    /* Vectores dorados: mete cuentas crudas y comprueba las tres etapas
     * contra Python. No hace falta llevar el sensor puesto. */
    /* Proximidad en crudo durante 25 s. Distingue 'el sensor no ve la
     * mano' de 'la ve y el decodificador no saca direccion'. */
    /* Alterna el modo prueba de gestos. */
    /* Volcado de los registros del APDS tal como estaban al arrancar. */
    case 'q':
      if (xSemaphoreTake(mtx_i2c, pdMS_TO_TICKS(20)) == pdTRUE) {
        pc_apds_volcado();
        xSemaphoreGive(mtx_i2c);
      }
      break;
    case 'y':
      g_gestos_prueba = !g_gestos_prueba;
      Serial.printf("# GESTOS modo prueba %s: se detectan y se informan, "
                    "%s" "\n",
                    g_gestos_prueba ? "ON" : "OFF",
                    g_gestos_prueba ? "sin avanzar la sesion"
                                    : "y vuelven a actuar");
      break;
    case 'p': {
      Serial.println(F("# PROX: acerca y aleja la mano. prox / GVALID / FIFO / GMODE"));
      uint32_t t0 = millis();
      uint8_t pmax = 0;
      while (millis() - t0 < 25000) {
        uint8_t pr, st, nv, gc;
        if (xSemaphoreTake(mtx_i2c, pdMS_TO_TICKS(5)) == pdTRUE) {
          pc_apds_diag(&pr, &st, &nv, &gc);
          xSemaphoreGive(mtx_i2c);
        }
        if (pr > pmax) pmax = pr;
        Serial.printf("P,%3u,%u,%2u,%u\n", pr, st & 1, nv, gc & 1);
        vTaskDelay(pdMS_TO_TICKS(100));
      }
      Serial.printf("# PROX fin. maximo %u (umbral de entrada %u)\n",
                    pmax, 40);
      break;
    }
    case 'G':
      pc_golden_test();
      break;
    case 'R': {
      Serial.printf("RD,%u", (unsigned)g_sesion.ref.n_dia);
      for (int j = 0; j < PC_N_FEATURES; j++)
        Serial.printf(",%.7g", g_sesion.ref.n_dia
                      ? g_sesion.ref.suma_dia[j] / g_sesion.ref.n_dia : 0.0f);
      Serial.println();
      Serial.printf("RT,%u", (unsigned)g_sesion.ref.n_tor);
      for (int j = 0; j < PC_N_FEATURES; j++)
        Serial.printf(",%.7g", g_sesion.ref.n_tor
                      ? g_sesion.ref.suma_tor[j] / g_sesion.ref.n_tor : 0.0f);
      Serial.println();
      break;
    }
    /* Inyeccion por la maquina de SESION, no por la de paridad. El PC manda las
     * mismas muestras que grabo en vivo y el dispositivo rehace su calibracion
     * sobre ellas; asi se comparan los vectores de referencia contra Python sin
     * ningun problema de alineamiento entre contadores. */
    case 'S': modo_sesion(); break;
    case 'c':
      Serial.printf("# SESION %s  n=%lu  ref dia=%u tor=%u  utiles=%d"
                    "  contraste=%.3f  descartadas=%u  cola_perdidas=%lu\n",
                    pc_sesion_nombre(g_sesion.est),
                    (unsigned long)g_sesion.pipe.n,
                    (unsigned)g_sesion.ref.n_dia, (unsigned)g_sesion.ref.n_tor,
                    g_sesion.informativas, g_sesion.contraste,
                    (unsigned)g_sesion.descartadas,
                    (unsigned long)n_dsp_perdidas);
      break;
    case '?':
      Serial.println(F("# s=iniciar x=detener m=marcar i=info "
                       "n=siguiente k=cancelar c=sesion R=referencia "
                       "M=montaje G=dorados p=proximidad y=prueba-gestos q=registros-apds S=inyectar T=paridad ?=ayuda"));
      break;
    default: break;
  }
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Modo de paridad: el PC inyecta una grabacion, el dispositivo devuelve
// las caracteristicas que calcula.
// ---------------------------------------------------------------------------

/* Por que en el dispositivo y no en el PC.
 *
 * El unico compilador de C disponible aqui es el cruzado de Xtensa, asi que no
 * hay forma comoda de correr este DSP en el host. Resulta ser mejor: la prueba
 * corre sobre el hardware real, con su FPU de precision simple, que es
 * exactamente donde tiene que coincidir. Un DSP que casa en el PC y no en el
 * ESP32 no sirve de nada.
 *
 * Protocolo: tras 'T', el PC envia tramas binarias de 12 bytes -seis int16 en
 * little endian-. El dispositivo filtra de forma continua y, cada PC_HOP_N
 * muestras a partir de la primera ventana completa, imprime una linea
 *   F,<indice_inicio>,<f0>,...,<f28>
 * El PC compara contra Python. Terminar con la trama 0xFFFF x6.
 */

/* La paridad usa la misma cadena que la sesion. Antes tenia su propia copia del
 * anillo y de la ventana: 12 kB duplicados y, peor, dos implementaciones que
 * podian separarse sin que ninguna prueba lo notara. */
static pc_pipeline_t g_pipe;

static void paridad_reset() {
  pc_pipeline_reset(&g_pipe);
  /* El montaje tambien aqui. La prueba de paridad corria sobre los ejes crudos
   * mientras el analisis usaba el marco anatomico, y como los dos lados
   * estaban igual de mal, coincidian: la prueba pasaba sin ejercitar nunca la
   * rotacion. Es como el firmware llego a no cargar la matriz sin que nada lo
   * dijera. */
  if (g_montaje_ok) pc_front_set_mount(&g_pipe.front, g_R, g_sesgo);
}

static void paridad_muestra(const int16_t raw6[6]) {
  static float f[PC_N_FEATURES];
  uint32_t ini = 0;
  uint32_t t0 = micros();
  bool nuevo = pc_pipeline_push(&g_pipe, raw6, f, &ini);
  uint32_t us = micros() - t0;
  if (!nuevo) return;

  Serial.printf("F,%lu,%lu", (unsigned long)ini, (unsigned long)us);
  for (int i = 0; i < PC_N_FEATURES; i++) Serial.printf(",%.7g", f[i]);
  Serial.println();
}

/* Reproduce una grabacion por la maquina de sesion.
 *
 * Protocolo: tras 'S', el PC manda tramas binarias de 12 bytes. Cuatro valores
 * de trama estan reservados y no llevan datos:
 *
 *     -1 x6   fin de la inyeccion
 *     -2 x6   a partir de aqui, maniobra DIAFRAGMATICA
 *     -3 x6   a partir de aqui, maniobra TORACICA
 *     -4 x6   a partir de aqui, ninguna maniobra
 *
 * Las marcas van dentro de la propia trama y no como un byte suelto. La primera
 * version las mandaba sueltas y miraba el primer byte disponible para
 * detectarlas, lo que rompe: 0x64 es 'd' y aparece constantemente dentro de una
 * muestra int16, asi que el dispositivo se comia medio dato como si fuera una
 * marca y perdia el sincronismo del resto de la inyeccion.
 *
 * Ninguna de las cuatro tramas reservadas puede confundirse con una muestra: el
 * acelerometro siempre lleva ~16384 cuentas de gravedad en algun eje, asi que
 * los seis valores no pueden valer -1 ni -4 a la vez.
 *
 * A diferencia del modo de paridad, aqui NO se reinician los filtros entre
 * fases: es lo que hace la sesion real, y reiniciarlos meteria el transitorio
 * de asentamiento dentro del primer vector de cada maniobra.
 */
static void modo_sesion() {
  if (h_dsp) vTaskSuspend(h_dsp);
  pc_sesion_reset(&g_sesion);
  Serial.println(F("# SESION-INYECCION lista"));

  uint8_t buf[12];
  uint32_t sin_datos = millis();
  for (;;) {
    if (Serial.available() >= 12) {
      Serial.readBytes(buf, 12);
      int16_t s6[6];
      for (int i = 0; i < 6; i++)
        s6[i] = (int16_t)((uint16_t)buf[2 * i] | ((uint16_t)buf[2 * i + 1] << 8));

      bool iguales = true;
      for (int i = 1; i < 6; i++) if (s6[i] != s6[0]) { iguales = false; break; }
      if (iguales && s6[0] >= -4 && s6[0] <= -1) {
        if (s6[0] == -1) break;                       // fin
        g_sesion.est = (s6[0] == -2) ? SES_CAL_DIA
                     : (s6[0] == -3) ? SES_CAL_TOR : SES_LISTO;
        g_sesion.ini_fase = g_sesion.pipe.n;
        sin_datos = millis();
        continue;
      }

      /* Se empuja por el pipeline sin dejar que la maquina avance de fase
       * sola: las fronteras las manda el PC. */
      uint32_t ini = 0;
      static float f[PC_N_FEATURES];
      bool nuevo = pc_pipeline_push(&g_sesion.pipe, s6, f, &ini);
      if (nuevo && (g_sesion.est == SES_CAL_DIA || g_sesion.est == SES_CAL_TOR)) {
        if (ini >= g_sesion.ini_fase)
        {
          pc_ref_acumula(&g_sesion.ref, f, g_sesion.est == SES_CAL_TOR);
          /* Que ventanas entraron, para poder compararlas una a una con
           * las que uso Python en vez de suponer que son las mismas. */
          Serial.printf("W,%c,%lu\n",
                        g_sesion.est == SES_CAL_TOR ? 't' : 'd',
                        (unsigned long)ini);
        }
        else
          g_sesion.descartadas++;
      }
      sin_datos = millis();
    } else if (millis() - sin_datos > 8000) {
      break;
    } else {
      vTaskDelay(1);
    }
  }
  g_sesion.contraste = pc_ref_contraste_mediano(&g_sesion.ref);
  g_sesion.informativas = pc_ref_normaliza(&g_sesion.ref, g_sesion.f, g_sesion.z);
  Serial.printf("# SESION-INYECCION fin, %lu muestras, dia=%u tor=%u"
                "\n", (unsigned long)g_sesion.pipe.n,
                (unsigned)g_sesion.ref.n_dia, (unsigned)g_sesion.ref.n_tor);
  if (h_dsp) vTaskResume(h_dsp);
}

static void modo_paridad() {
  /* La sesion se suspende mientras dura la prueba. pc_scratch es una sola
   * ventana compartida por las dos cadenas -12 kB no se duplican a la ligera- y
   * si la tarea de DSP la pisara a medias, la paridad compararia basura contra
   * Python y el fallo pareceria del DSP. */
  if (h_dsp) vTaskSuspend(h_dsp);
  paridad_reset();
  Serial.println(F("# PARIDAD lista, enviando tramas de 12 bytes"));
  uint8_t buf[12];
  uint32_t sin_datos = millis();

  for (;;) {
    if (Serial.available() >= 12) {
      Serial.readBytes(buf, 12);
      int16_t s6[6];
      for (int i = 0; i < 6; i++)
        s6[i] = (int16_t)((uint16_t)buf[2 * i] | ((uint16_t)buf[2 * i + 1] << 8));
      bool fin = true;
      for (int i = 0; i < 6; i++) if (s6[i] != -1) { fin = false; break; }
      if (fin) break;
      paridad_muestra(s6);
      sin_datos = millis();
    } else if (millis() - sin_datos > 8000) {
      break;  // el PC se fue; no quedarse colgado
    } else {
      vTaskDelay(1);
    }
  }
  Serial.printf("# PARIDAD fin, %lu muestras\n", (unsigned long)g_pipe.n);
  if (h_dsp) vTaskResume(h_dsp);

  /* Volcado de la ultima ventana. Cuando una caracteristica se sale de
   * tolerancia hay que saber si el culpable es la cadena de filtros o el
   * calculo de la caracteristica, y para eso hacen falta los canales, no
   * mas escalares. El PC lo pide con 'D' y compara muestra a muestra. */
  uint32_t t_espera = millis();
  while (millis() - t_espera < 3000) {
    if (Serial.available()) {
      if (Serial.read() != 'D') continue;
      uint32_t ini = ((g_pipe.n - PC_WINDOW_N) / PC_HOP_N) * PC_HOP_N;
      Serial.printf("W,%lu,%d\n", (unsigned long)ini, PC_WINDOW_N);
      for (int i = 0; i < PC_WINDOW_N; i++)
        Serial.printf("C,%.9g,%.9g\n", pc_scratch.tilt[i], pc_scratch.axial[i]);
      Serial.println(F("# VOLCADO fin"));
      break;
    }
    vTaskDelay(1);
  }
}

void setup() {
  Serial.begin(BAUD);
  delay(400);

  Wire.begin(PIN_SDA, PIN_SCL, I2C_HZ);
  Wire.setTimeOut(50);

  Serial.println();
  Serial.println(F("# ==================================="));
  Serial.println(F("# PneumoCoach - adquisicion MPU6050"));
  Serial.println(F("# ==================================="));
  /* El montaje se carga ANTES del banner. Si no, el arranque anuncia
   * 'SIN CALIBRAR' aunque la matriz este guardada en NVS, que es justo el
   * mensaje que hace dudar de algo que funciona. */
  pc_sesion_reset(&g_sesion);
  montaje_cargar();
  scan_bus();

  if (!mpu_init()) {
    Serial.println(F("# FALLO al inicializar la IMU."));
    Serial.println(F("# Revisar: modulo conectado, AD0 en alto, cables SDA=21 SCL=22."));
  } else {
    mpu_autoprueba();
    ui_imu_ok = true;
    Serial.println(F("# IMU lista."));
  }

  if (oled_init()) {
    oled_clear();
    oled_text_center(20, "PNEUMOCOACH", 1);
    oled_text_center(36, "INICIANDO", 1);
    oled_flush();
    Serial.println(F("# OLED lista en 0x3C."));
  } else {
    Serial.println(F("# OLED no responde en 0x3C (opcional, se continua)."));
  }

  q_samples = xQueueCreate(256, sizeof(sample_t));
  /* 128 muestras = 2.5 s de holgura. El DSP consume una muestra cada 20 ms y
   * solo hace trabajo real -1 ms- cada 150; la cola absorbe esa rafaga. */
  q_dsp = xQueueCreate(128, sizeof(raw6_t));
  pc_inferencia_init();
  pc_ble_init();
  /* El APDS9960 no se inicializa: los gestos quedaron fuera de alcance
   * (ADR-0010) y la sesion avanza sola. Encenderle el motor de gestos
   * seria pulsar su LED infrarrojo para nada. El escaneo I2C sigue
   * reportando que el modulo esta en el bus. */
  mtx_i2c = xSemaphoreCreateMutex();

  // La adquisicion va al nucleo 1 (APP_CPU) porque el nucleo 0 (PRO_CPU) es
  // donde Espressif corre el stack de WiFi/BT. Cuando agreguemos BLE, esta
  // separacion es lo que evita que la radio desplace el muestreo.
  xTaskCreatePinnedToCore(task_acquire, "acquire", 4096, NULL, 10, NULL, 1);
  xTaskCreatePinnedToCore(task_print, "print", 4096, NULL, 3, NULL, 0);
  /* La pantalla va al nucleo 0 y con prioridad menor que el volcado serial:
   * si algo tiene que ceder, que ceda el indicador visual. */
  /* Por encima de la tarea de CSV: su trabajo esta acotado y es corto, y si el
   * volcado por serie la dejara sin turno la cola se llenaria. */
  xTaskCreatePinnedToCore(task_dsp, "dsp", 4096, NULL, 4, &h_dsp, 0);
  xTaskCreatePinnedToCore(task_pantalla, "pantalla", 4096, NULL, 2, NULL, 0);

  Serial.println(F("# Listo. Enviar 's' para iniciar, '?' para ayuda."));
}

void loop() {
  while (Serial.available()) handle_cmd((char)Serial.read());
  vTaskDelay(pdMS_TO_TICKS(20));
}
