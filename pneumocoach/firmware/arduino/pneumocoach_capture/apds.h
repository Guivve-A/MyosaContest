/* Gestos sin contacto con el APDS9960.
 * ===================================
 *
 * Por que existe
 * --------------
 * Es el UNICO dispositivo de entrada del kit. Sin el, la sesion no se puede
 * avanzar sin un PC conectado, y eso echa abajo justo lo que el resto del
 * firmware consiguio: que la sesion clinica corra sola.
 *
 * Ademas es lo correcto clinicamente. El paciente esta a mitad de un ejercicio
 * respiratorio con el sensor pegado al esternon; alargar el brazo para pulsar
 * algo mueve el torso, y ese movimiento entra en el mismo canal que medimos.
 *
 * La restriccion que manda: NO ROBAR MUESTRAS
 * -------------------------------------------
 * El bus I2C lo comparten la IMU -que tiene que leerse cada 20 ms sin fallar
 * uno-, la pantalla y este sensor. La FIFO de gestos puede llegar a 32 conjuntos
 * de 4 bytes; leerla entera de una vez son 128 bytes que a 400 kHz bloquean el
 * bus unos 3 ms, y encadenando lecturas se acumula.
 *
 * Por eso cada sondeo lee COMO MUCHO PC_APDS_MAX_LEER conjuntos. Es la misma
 * leccion que la pantalla: alli el framebuffer de 1 kB se envia pagina a pagina
 * por esto mismo.
 *
 * Este fichero NO toca el mutex del bus. El que llama es quien lo tiene, porque
 * asi todas las lecturas de un sondeo caen dentro de una sola seccion critica
 * acotada, en vez de pelearse por el bus una a una.
 */

#ifndef PC_APDS_H
#define PC_APDS_H

#include <stdint.h>

/* Definidas mas abajo en el sketch; declaradas aqui porque este cabecero se
 * incluye antes. Es la misma unidad de compilacion, asi que vale. */
static bool reg_write(uint8_t addr, uint8_t reg, uint8_t val);
static bool reg_read(uint8_t addr, uint8_t reg, uint8_t *buf, size_t n);

#define PC_APDS_ADDR      0x39

/* Mapa de registros del APDS9960, tomado de la libreria del fabricante
 * (Resources_docs/arduino-libraries-main/LightProximityAndGesture). */
#define APDS_ENABLE       0x80
#define APDS_ATIME        0x81
#define APDS_WTIME        0x83
#define APDS_PPULSE       0x8E
#define APDS_CONTROL      0x8F
#define APDS_ID           0x92
#define APDS_PDATA        0x9C
#define APDS_POFFSET_UR   0x9D
#define APDS_POFFSET_DL   0x9E
#define APDS_CONFIG1      0x8D
#define APDS_GPENTH       0xA0
#define APDS_GEXTH        0xA1
#define APDS_GCONF1       0xA2
#define APDS_GCONF2       0xA3
#define APDS_GPULSE       0xA6
#define APDS_GCONF3       0xAA
#define APDS_GCONF4       0xAB
#define APDS_GFLVL        0xAE
#define APDS_GSTATUS      0xAF
#define APDS_GFIFO_U      0xFC

/* Conjuntos leidos por sondeo. Con 8 a 50 Hz salen 400 por segundo, muy por
 * encima de los ~180 que produce el motor con GWTIME de 5.6 ms, asi que la FIFO
 * -de 32- no se llena. Y 8 conjuntos son 32 bytes: menos de 1 ms de bus. */
#define PC_APDS_MAX_LEER  8

/* Un gesto son unas decenas de conjuntos. 64 sobran y acotan la memoria. */
#define PC_APDS_MAX_CONJ  64

typedef enum {
  PC_GESTO_NINGUNO = 0,
  PC_GESTO_ARRIBA,
  PC_GESTO_ABAJO,
  PC_GESTO_IZQUIERDA,
  PC_GESTO_DERECHA,
} pc_gesto_t;

typedef struct {
  bool presente;
  uint8_t id;
  /* Acumulacion del gesto en curso. */
  uint8_t u[PC_APDS_MAX_CONJ], d[PC_APDS_MAX_CONJ];
  uint8_t l[PC_APDS_MAX_CONJ], r[PC_APDS_MAX_CONJ];
  uint8_t n;
  bool en_gesto;
  uint32_t descartados;   // conjuntos perdidos por FIFO llena
  uint32_t detectados;
} pc_apds_t;

/* Comprobacion de que la configuracion quedo escrita.
 *
 * La primera version comparaba contra los valores de RESET de la hoja de datos
 * y concluyo "probablemente no es un APDS9960". Era falso, y el error de
 * razonamiento vale la pena dejarlo escrito: este chip tiene alimentacion
 * propia y ninguna linea de reset, asi que NO se reinicia cuando lo hace el
 * ESP32. Lo que se leia al arrancar no eran valores de reset sino los que este
 * mismo firmware habia escrito en la ejecucion anterior.
 *
 * Que coincidieran con lo escrito era, de hecho, la prueba de que el chip
 * acepta y retiene la configuracion. La tabla compara ahora contra eso.
 *
 * ID, STATUS y PDATA no se escriben; su columna "esperado" no significa nada y
 * van al final solo como informacion. */
typedef struct { uint8_t reg; uint8_t esperado; const char *nombre; } pc_apds_reset_t;

static const pc_apds_reset_t PC_APDS_RESET[] = {
  {0x80, 0x45, "ENABLE"},  {0x81, 0xDB, "ATIME"},   {0x83, 0xF6, "WTIME"},
  {0x8D, 0x60, "CONFIG1"}, {0x8E, 0x4F, "PPULSE"},  {0x8F, 0x08, "CONTROL"},
  {0x9D, 0x00, "POFFSET_UR"}, {0x9E, 0x00, "POFFSET_DL"},
  {0xA2, 0x40, "GCONF1"},  {0xA3, 0x4A, "GCONF2"},  {0xA6, 0xC9, "GPULSE"},
  {0x92, 0x00, "ID"},      {0x93, 0x00, "STATUS"},  {0x9C, 0x00, "PDATA"},
};
#define PC_APDS_N_RESET (sizeof(PC_APDS_RESET) / sizeof(PC_APDS_RESET[0]))

static uint8_t g_apds_al_arrancar[PC_APDS_N_RESET];

static pc_apds_t g_apds;

/* Vuelca lo leido al arrancar contra lo que la hoja de datos dice que deberia
 * haber. El que llama tiene el mutex del bus. */
static void pc_apds_volcado() {
  int distintos = 0;
  Serial.println(F("# APDS registros leidos ANTES de configurar en este arranque."));
  Serial.println(F("#   Deben coincidir con lo escrito en el arranque anterior:"));
  Serial.println(F("#   el chip no se reinicia con el ESP32."));
  for (unsigned i = 0; i < PC_APDS_N_RESET; i++) {
    /* Los tres ultimos son de solo lectura: se muestran, no se juzgan. */
    bool juzgar = i < PC_APDS_N_RESET - 3;
    bool ok = !juzgar || g_apds_al_arrancar[i] == PC_APDS_RESET[i].esperado;
    if (!ok) distintos++;
    Serial.printf("#   0x%02X %-11s 0x%02X  %s\n",
                  PC_APDS_RESET[i].reg, PC_APDS_RESET[i].nombre,
                  g_apds_al_arrancar[i],
                  !juzgar ? "(solo lectura)" : (ok ? "ok" : "<- NO quedo escrito"));
  }
  Serial.printf("# APDS %d de %u registros de configuracion NO quedaron escritos\n",
                distintos, (unsigned)(PC_APDS_N_RESET - 3));
  Serial.println(distintos == 0
      ? F("#   La configuracion se aplica y el chip la retiene.")
      : F("#   El chip ignora escrituras: ahi esta el problema."));
}

static bool pc_apds_init() {
  memset(&g_apds, 0, sizeof(g_apds));
  /* Foto de los registros ANTES de tocar nada. Despues de configurar ya no se
   * puede distinguir un valor de reset de uno que escribimos nosotros. */
  for (unsigned i = 0; i < PC_APDS_N_RESET; i++)
    reg_read(PC_APDS_ADDR, PC_APDS_RESET[i].reg, &g_apds_al_arrancar[i], 1);
  /* Un fallo aqui tiene que decir QUE fallo. "Ausente" cuando el dispositivo
   * responde en el escaneo del bus manda a buscar el problema al sitio
   * equivocado. */
  if (!reg_read(PC_APDS_ADDR, APDS_ID, &g_apds.id, 1)) {
    Serial.println(F("# APDS9960: no responde al leer el registro ID"));
    return false;
  }
  /* Identificadores aceptados.
   *
   * La hoja de datos documenta 0xAB, 0x9C y 0xA8 segun la tirada. El ejemplar
   * de este kit devuelve 0x9E, que no esta en esa lista y que algunas
   * librerias rechazan. Es el mismo patron que la IMU: la serigrafia dice
   * MPU6050 y el silicio responde 0x70, o sea MPU6500.
   *
   * Se acepta, pero el identificador NO es la validacion. Que un chip responda
   * un numero plausible no prueba que su motor de gestos funcione; eso lo
   * decide la prueba funcional -pasar la mano y ver si sale la direccion
   * correcta-, y hasta que esa pase, los gestos no son una funcion del
   * sistema. */
  if (g_apds.id != 0xAB && g_apds.id != 0x9C && g_apds.id != 0xA8
      && g_apds.id != 0x9E) {
    pc_apds_volcado();
    Serial.printf("# APDS9960: ID inesperado 0x%02X"
                  " (se esperaba AB, 9C o A8)\n", g_apds.id);
    return false;
  }

  reg_write(PC_APDS_ADDR, APDS_ENABLE, 0x00);   // todo apagado para configurar
  reg_write(PC_APDS_ADDR, APDS_ATIME, 219);
  reg_write(PC_APDS_ADDR, APDS_WTIME, 246);
  /* Mas pulsos y mas largos: 32 us x 16 en vez de 16 us x 8. La lectura de
   * proximidad es el reflejo integrado de esos pulsos, asi que multiplicarlos
   * sube la senal sin tocar el LED. */
  reg_write(PC_APDS_ADDR, APDS_PPULSE, 0x4F);

  /* CONTROL: ganancia de proximidad 4x y LED a 100 mA.
   *
   * Este registro no se escribia, asi que se quedaba en su valor por defecto:
   * ganancia 1x, la mas baja de las cuatro. Medido en reposo la proximidad daba
   * 4 contra un umbral de entrada de 40, y con esa ganancia una mano a un palmo
   * puede no llegar a cruzarlo. El motor de gestos ni siquiera arranca, y desde
   * fuera se ve como "no detecta gestos".
   *
   * PGAIN esta en los bits [3:2] y LDRIVE en los [7:6], donde 0 son 100 mA.
   * El LED de proximidad solo pulsa al medir, asi que su consumo medio es
   * bajo; el que se dejo a 50 mA es el de gestos, que pulsa mucho mas. */
  reg_write(PC_APDS_ADDR, APDS_CONTROL, (0 << 6) | (2 << 2) | 0);

  /* Offsets de proximidad a cero, y CONFIG1 al valor del fabricante.
   *
   * Estos dos registros se me habian pasado, y el de los offsets explica el
   * sintoma: lo que guardan se RESTA de la medida de proximidad, asi que un
   * valor alto aplana la lectura y la mano deja de verse. El fabricante los
   * pone a cero explicitamente en su begin().
   *
   * Y no se ponen solos: el APDS tiene alimentacion propia y ninguna linea de
   * reset, asi que conserva su configuracion cuando el ESP32 reinicia. Lo que
   * no escribamos nosotros se queda como estuviera. */
  reg_write(PC_APDS_ADDR, APDS_POFFSET_UR, 0);
  reg_write(PC_APDS_ADDR, APDS_POFFSET_DL, 0);
  reg_write(PC_APDS_ADDR, APDS_CONFIG1, 0x60);

  /* Umbrales de entrada y salida del motor de gestos. La histeresis entre los
   * dos evita que una mano que se queda quieta cerca lo arranque y lo pare en
   * bucle. */
  reg_write(PC_APDS_ADDR, APDS_GPENTH, 40);
  reg_write(PC_APDS_ADDR, APDS_GEXTH, 30);

  /* GCONF1: la FIFO marca GVALID con 4 conjuntos dentro. Suficiente para leer
   * a tiempo sin despertar por cada muestra. */
  reg_write(PC_APDS_ADDR, APDS_GCONF1, 0x40);

  /* GCONF2: ganancia 4x, LED a 50 mA, espera de 5.6 ms entre conjuntos.
   *
   * El LED a 100 mA -el maximo- daria mas alcance, pero son picos de corriente
   * sobre el mismo rail que alimenta la IMU y el ESP32, y este dispositivo va a
   * ir con bateria. 50 mA alcanza de sobra para una mano a un palmo. */
  reg_write(PC_APDS_ADDR, APDS_GCONF2, (2 << 5) | (1 << 3) | 2);
  reg_write(PC_APDS_ADDR, APDS_GPULSE, 0xC9);   // 32 us, 10 pulsos

  /* PON + PEN + GEN. El motor de gestos entra solo cuando la proximidad supera
   * GPENTH; no hay que activarlo a mano. */
  reg_write(PC_APDS_ADDR, APDS_ENABLE, 0x01 | 0x04 | 0x40);

  /* Se lee de vuelta: si el chip no acepto la configuracion, los gestos no
   * funcionarian y el arranque diria que todo esta bien. */
  uint8_t en = 0, gc2 = 0, ctl = 0, pp = 0;
  reg_read(PC_APDS_ADDR, APDS_ENABLE, &en, 1);
  reg_read(PC_APDS_ADDR, APDS_GCONF2, &gc2, 1);
  reg_read(PC_APDS_ADDR, APDS_CONTROL, &ctl, 1);
  reg_read(PC_APDS_ADDR, APDS_PPULSE, &pp, 1);
  if (en != (0x01 | 0x04 | 0x40)) {
    Serial.printf("# APDS9960: ENABLE quedo en 0x%02X, se escribio 0x45\n", en);
    return false;
  }
  /* Se releen TODOS los registros de configuracion, no solo ENABLE.
   *
   * Al subir la ganancia de proximidad la linea base no se movio -4 antes, 4
   * despues-, y sin relectura no habia forma de saber si el registro se estaba
   * aplicando o si el efecto era otro. Comprobar solo uno de cuatro deja tres
   * sitios donde el fallo pasa desapercibido. */
  Serial.printf("# APDS9960 listo: ID 0x%02X%s\n",
                g_apds.id, (g_apds.id == 0x9E) ? " (fuera de la hoja de datos)" : "");
  Serial.printf("#   ENABLE  0x%02X (esc 0x45)   GCONF2 0x%02X (esc 0x4A)\n",
                en, gc2);
  Serial.printf("#   CONTROL 0x%02X (esc 0x08)   PPULSE 0x%02X (esc 0x4F)\n",
                ctl, pp);
  if (ctl != 0x08 || pp != 0x4F)
    Serial.println(F("#   AVISO: la configuracion no quedo como se escribio"));
  g_apds.presente = true;
  return true;
}

/* Decide la direccion comparando el PRINCIPIO del gesto con el FINAL.
 *
 * Los cuatro fotodiodos estan en cruz. Una mano que cruza de abajo arriba tapa
 * primero el de abajo y al final el de arriba, asi que lo que identifica al
 * gesto no es el valor absoluto -que depende de la distancia y del reflejo de
 * la piel- sino COMO CAMBIA el reparto entre los dos extremos. Por eso se
 * normaliza a un cociente y se mira la diferencia entre el primer y el ultimo
 * tramo. */
static pc_gesto_t pc_apds_decidir(const pc_apds_t *a) {
  if (a->n < 4) return PC_GESTO_NINGUNO;

  const int m = (a->n < 8) ? 1 : 2;   // conjuntos promediados en cada extremo
  int u0 = 0, d0 = 0, l0 = 0, r0 = 0, u1 = 0, d1 = 0, l1 = 0, r1 = 0;
  for (int i = 0; i < m; i++) {
    u0 += a->u[i];      d0 += a->d[i];      l0 += a->l[i];      r0 += a->r[i];
    u1 += a->u[a->n - 1 - i]; d1 += a->d[a->n - 1 - i];
    l1 += a->l[a->n - 1 - i]; r1 += a->r[a->n - 1 - i];
  }
  if (u0 + d0 == 0 || u1 + d1 == 0 || l0 + r0 == 0 || l1 + r1 == 0)
    return PC_GESTO_NINGUNO;

  int ud = ((u1 - d1) * 100) / (u1 + d1) - ((u0 - d0) * 100) / (u0 + d0);
  int lr = ((r1 - l1) * 100) / (r1 + l1) - ((r0 - l0) * 100) / (r0 + l0);

  /* Umbral de 25 sobre un cociente que va de -100 a 100. Por debajo de eso la
   * mano no cruzo: se quedo encima. Preferimos no reconocer un gesto antes que
   * inventarse uno, porque aqui un falso positivo cancela una calibracion a
   * mitad. */
  const int UMBRAL = 25;
  int aud = ud < 0 ? -ud : ud;
  int alr = lr < 0 ? -lr : lr;
  if (aud < UMBRAL && alr < UMBRAL) return PC_GESTO_NINGUNO;
  if (aud >= alr) return ud > 0 ? PC_GESTO_ARRIBA : PC_GESTO_ABAJO;
  return lr > 0 ? PC_GESTO_DERECHA : PC_GESTO_IZQUIERDA;
}

/* Un sondeo. Devuelve el gesto cuando termina, o NINGUNO.
 *
 * El que llama tiene que tener el mutex del bus: asi las lecturas de un sondeo
 * caen dentro de una sola seccion critica acotada. */
static pc_gesto_t pc_apds_poll() {
  if (!g_apds.presente) return PC_GESTO_NINGUNO;

  uint8_t st = 0;
  if (!reg_read(PC_APDS_ADDR, APDS_GSTATUS, &st, 1)) return PC_GESTO_NINGUNO;
  const bool valido = st & 0x01;   // GVALID
  const bool desbordo = st & 0x02; // GFOV

  uint8_t gconf4 = 0;
  reg_read(PC_APDS_ADDR, APDS_GCONF4, &gconf4, 1);
  const bool motor_activo = gconf4 & 0x01;  // GMODE

  if (valido) {
    uint8_t nivel = 0;
    reg_read(PC_APDS_ADDR, APDS_GFLVL, &nivel, 1);
    if (nivel > PC_APDS_MAX_LEER) nivel = PC_APDS_MAX_LEER;
    if (nivel) {
      uint8_t buf[PC_APDS_MAX_LEER * 4];
      if (reg_read(PC_APDS_ADDR, APDS_GFIFO_U, buf, nivel * 4)) {
        g_apds.en_gesto = true;
        for (int i = 0; i < nivel; i++) {
          if (g_apds.n >= PC_APDS_MAX_CONJ) { g_apds.descartados++; break; }
          g_apds.u[g_apds.n] = buf[i * 4 + 0];
          g_apds.d[g_apds.n] = buf[i * 4 + 1];
          g_apds.l[g_apds.n] = buf[i * 4 + 2];
          g_apds.r[g_apds.n] = buf[i * 4 + 3];
          g_apds.n++;
        }
      }
    }
    if (desbordo) g_apds.descartados++;
    return PC_GESTO_NINGUNO;
  }

  /* Sin datos validos y con el motor apagado: la mano se fue. Ahi se decide. */
  if (g_apds.en_gesto && !motor_activo) {
    pc_gesto_t g = pc_apds_decidir(&g_apds);
    g_apds.n = 0;
    g_apds.en_gesto = false;
    if (g != PC_GESTO_NINGUNO) g_apds.detectados++;
    return g;
  }
  return PC_GESTO_NINGUNO;
}

/* Diagnostico en crudo: proximidad, estado y nivel de FIFO.
 *
 * Sirve para separar dos fallos que desde fuera se ven igual: que el sensor no
 * vea la mano, o que la vea y el decodificador no saque direccion. Sin esto,
 * "no detecta gestos" es un sintoma sin causa.
 *
 * El que llama tiene el mutex del bus. */
static void pc_apds_diag(uint8_t *prox, uint8_t *estado, uint8_t *nivel,
                         uint8_t *gconf4) {
  *prox = *estado = *nivel = *gconf4 = 0;
  reg_read(PC_APDS_ADDR, APDS_PDATA, prox, 1);
  reg_read(PC_APDS_ADDR, APDS_GSTATUS, estado, 1);
  reg_read(PC_APDS_ADDR, APDS_GFLVL, nivel, 1);
  reg_read(PC_APDS_ADDR, APDS_GCONF4, gconf4, 1);
}

static const char *pc_gesto_nombre(pc_gesto_t g) {
  switch (g) {
    case PC_GESTO_ARRIBA:    return "ARRIBA";
    case PC_GESTO_ABAJO:     return "ABAJO";
    case PC_GESTO_IZQUIERDA: return "IZQUIERDA";
    case PC_GESTO_DERECHA:   return "DERECHA";
    default:                 return "NINGUNO";
  }
}

#endif /* PC_APDS_H */
