/* Driver minimo para SSD1306 128x64 por I2C.
 * ==========================================
 *
 * Sin Adafruit_GFX ni Adafruit_SSD1306, a proposito. El sketch de captura
 * compila hoy en un Arduino IDE recien instalado, sin bajar una sola libreria,
 * y esa propiedad vale mas de lo que cuesta este archivo: cualquiera del equipo
 * puede flashear sin pelear con el gestor de paquetes.
 *
 * Ademas da control sobre el trafico I2C, que aqui importa de verdad. La OLED
 * es el consumidor de ancho de banda mas pesado del bus: 1024 bytes por
 * refresco completo frente a los 14 del MPU6500. Si el volcado bloquea la
 * lectura del IMU, se pierden muestras y el espectro se ensucia. Por eso el
 * framebuffer se envia por PAGINAS -128 bytes- soltando el bus entre cada una,
 * en vez de en una unica transaccion de 1 KB.
 *
 * Consumo: 1024 bytes de framebuffer en SRAM. Con 320 KB disponibles y el
 * tensor arena presupuestado en 8 KB, cabe de sobra.
 */

#ifndef PC_OLED_H
#define PC_OLED_H

#include <Wire.h>

#define OLED_ADDR 0x3C
#define OLED_W 128
#define OLED_H 64
#define OLED_PAGES (OLED_H / 8)

static uint8_t oled_fb[OLED_W * OLED_PAGES];

/* Fuente 5x7, ASCII 32..90 (espacio, simbolos, digitos, mayusculas).
 * Solo mayusculas: en una pantalla de 21 caracteres por linea, las minusculas
 * no aportan legibilidad y duplicarian la tabla. */
static const uint8_t OLED_FONT[] PROGMEM = {
  0x00,0x00,0x00,0x00,0x00, /*   */ 0x00,0x00,0x5F,0x00,0x00, /* ! */
  0x00,0x07,0x00,0x07,0x00, /* " */ 0x14,0x7F,0x14,0x7F,0x14, /* # */
  0x24,0x2A,0x7F,0x2A,0x12, /* $ */ 0x23,0x13,0x08,0x64,0x62, /* % */
  0x36,0x49,0x55,0x22,0x50, /* & */ 0x00,0x05,0x03,0x00,0x00, /* ' */
  0x00,0x1C,0x22,0x41,0x00, /* ( */ 0x00,0x41,0x22,0x1C,0x00, /* ) */
  0x14,0x08,0x3E,0x08,0x14, /* * */ 0x08,0x08,0x3E,0x08,0x08, /* + */
  0x00,0x50,0x30,0x00,0x00, /* , */ 0x08,0x08,0x08,0x08,0x08, /* - */
  0x00,0x60,0x60,0x00,0x00, /* . */ 0x20,0x10,0x08,0x04,0x02, /* / */
  0x3E,0x51,0x49,0x45,0x3E, /* 0 */ 0x00,0x42,0x7F,0x40,0x00, /* 1 */
  0x42,0x61,0x51,0x49,0x46, /* 2 */ 0x21,0x41,0x45,0x4B,0x31, /* 3 */
  0x18,0x14,0x12,0x7F,0x10, /* 4 */ 0x27,0x45,0x45,0x45,0x39, /* 5 */
  0x3C,0x4A,0x49,0x49,0x30, /* 6 */ 0x01,0x71,0x09,0x05,0x03, /* 7 */
  0x36,0x49,0x49,0x49,0x36, /* 8 */ 0x06,0x49,0x49,0x29,0x1E, /* 9 */
  0x00,0x36,0x36,0x00,0x00, /* : */ 0x00,0x56,0x36,0x00,0x00, /* ; */
  0x00,0x08,0x14,0x22,0x41, /* < */ 0x14,0x14,0x14,0x14,0x14, /* = */
  0x41,0x22,0x14,0x08,0x00, /* > */ 0x02,0x01,0x51,0x09,0x06, /* ? */
  0x32,0x49,0x79,0x41,0x3E, /* @ */ 0x7E,0x11,0x11,0x11,0x7E, /* A */
  0x7F,0x49,0x49,0x49,0x36, /* B */ 0x3E,0x41,0x41,0x41,0x22, /* C */
  0x7F,0x41,0x41,0x22,0x1C, /* D */ 0x7F,0x49,0x49,0x49,0x41, /* E */
  0x7F,0x09,0x09,0x09,0x01, /* F */ 0x3E,0x41,0x49,0x49,0x7A, /* G */
  0x7F,0x08,0x08,0x08,0x7F, /* H */ 0x00,0x41,0x7F,0x41,0x00, /* I */
  0x20,0x40,0x41,0x3F,0x01, /* J */ 0x7F,0x08,0x14,0x22,0x41, /* K */
  0x7F,0x40,0x40,0x40,0x40, /* L */ 0x7F,0x02,0x0C,0x02,0x7F, /* M */
  0x7F,0x04,0x08,0x10,0x7F, /* N */ 0x3E,0x41,0x41,0x41,0x3E, /* O */
  0x7F,0x09,0x09,0x09,0x06, /* P */ 0x3E,0x41,0x51,0x21,0x5E, /* Q */
  0x7F,0x09,0x19,0x29,0x46, /* R */ 0x46,0x49,0x49,0x49,0x31, /* S */
  0x01,0x01,0x7F,0x01,0x01, /* T */ 0x3F,0x40,0x40,0x40,0x3F, /* U */
  0x1F,0x20,0x40,0x20,0x1F, /* V */ 0x3F,0x40,0x38,0x40,0x3F, /* W */
  0x63,0x14,0x08,0x14,0x63, /* X */ 0x07,0x08,0x70,0x08,0x07, /* Y */
  0x61,0x51,0x49,0x45,0x43, /* Z */
};

static bool oled_presente = false;

static void oled_cmd(uint8_t c) {
  Wire.beginTransmission(OLED_ADDR);
  Wire.write(0x00);  // Co=0, D/C=0 -> el byte siguiente es comando
  Wire.write(c);
  Wire.endTransmission();
}

static bool oled_init() {
  Wire.beginTransmission(OLED_ADDR);
  if (Wire.endTransmission() != 0) {
    oled_presente = false;
    return false;
  }
  static const uint8_t seq[] = {
      0xAE,              // apagar mientras se configura
      0xD5, 0x80,        // divisor de reloj
      0xA8, 0x3F,        // multiplex = 64-1
      0xD3, 0x00,        // sin desplazamiento vertical
      0x40,              // linea de inicio 0
      0x8D, 0x14,        // charge pump interno ON (sin esto no ilumina)
      0x20, 0x00,        // direccionamiento horizontal
      0xA1, 0xC8,        // remapeo: origen arriba-izquierda
      0xDA, 0x12,        // configuracion de pines COM para 128x64
      0x81, 0xCF,        // contraste
      0xD9, 0xF1, 0xDB, 0x40,
      0xA4,              // mostrar contenido de RAM, no todo encendido
      0xA6,              // video normal, no invertido
      0xAF,              // encender
  };
  for (uint8_t i = 0; i < sizeof(seq); i++) oled_cmd(seq[i]);
  memset(oled_fb, 0, sizeof(oled_fb));
  oled_presente = true;
  return true;
}

static void oled_clear() { memset(oled_fb, 0, sizeof(oled_fb)); }

/* Vuelca el framebuffer PAGINA A PAGINA, soltando el bus entre cada una.
 * Una transaccion unica de 1 KB a 400 kHz ocupa el bus ~26 ms, mas de un tick
 * de adquisicion: la tarea del IMU perderia su ventana. Troceado en ocho, cada
 * bloqueo es de ~3 ms y cabe entre lecturas. */
static void oled_flush() {
  if (!oled_presente) return;
  for (uint8_t page = 0; page < OLED_PAGES; page++) {
    oled_cmd(0xB0 + page);
    oled_cmd(0x00);
    oled_cmd(0x10);
    const uint8_t *src = &oled_fb[page * OLED_W];
    for (uint8_t chunk = 0; chunk < OLED_W; chunk += 32) {
      Wire.beginTransmission(OLED_ADDR);
      Wire.write(0x40);  // los bytes siguientes son datos
      Wire.write(src + chunk, 32);
      Wire.endTransmission();
    }
  }
}

static void oled_pixel(int x, int y, bool on) {
  if (x < 0 || x >= OLED_W || y < 0 || y >= OLED_H) return;
  uint8_t *p = &oled_fb[(y / 8) * OLED_W + x];
  if (on) *p |= (1 << (y & 7));
  else *p &= ~(1 << (y & 7));
}

/* Dibuja un caracter con escala entera. size=1 -> 5x7, size=2 -> 10x14. */
static void oled_char(int x, int y, char c, uint8_t size) {
  if (c >= 'a' && c <= 'z') c -= 32;  // solo hay mayusculas en la tabla
  if (c < 32 || c > 'Z') c = ' ';
  const uint8_t *g = OLED_FONT + (c - 32) * 5;
  for (uint8_t col = 0; col < 5; col++) {
    uint8_t bits = pgm_read_byte(g + col);
    for (uint8_t row = 0; row < 7; row++) {
      if (!(bits & (1 << row))) continue;
      for (uint8_t sx = 0; sx < size; sx++)
        for (uint8_t sy = 0; sy < size; sy++)
          oled_pixel(x + col * size + sx, y + row * size + sy, true);
    }
  }
}

static void oled_text(int x, int y, const char *s, uint8_t size = 1) {
  while (*s) {
    oled_char(x, y, *s++, size);
    x += 6 * size;
    if (x > OLED_W - 5 * size) return;  // no envolver: truncar
  }
}

static void oled_text_center(int y, const char *s, uint8_t size = 1) {
  int w = strlen(s) * 6 * size - size;
  oled_text((OLED_W - w) / 2, y, s, size);
}

static void oled_hline(int y, int x0 = 0, int x1 = OLED_W - 1) {
  for (int x = x0; x <= x1; x++) oled_pixel(x, y, true);
}

static void oled_rect(int x, int y, int w, int h, bool relleno) {
  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++)
      if (relleno || i == 0 || j == 0 || i == w - 1 || j == h - 1)
        oled_pixel(x + i, y + j, true);
}

/* Barra de progreso. `frac` se recorta a [0,1]. */
static void oled_barra(int x, int y, int w, int h, float frac) {
  oled_rect(x, y, w, h, false);
  if (frac < 0) frac = 0;
  if (frac > 1) frac = 1;
  int relleno = (int)((w - 4) * frac);
  if (relleno > 0) oled_rect(x + 2, y + 2, relleno, h - 4, true);
}

#endif /* PC_OLED_H */
