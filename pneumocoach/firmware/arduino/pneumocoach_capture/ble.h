/* Servidor BLE: el dispositivo publica su sesion para la app.
 * =========================================================
 *
 * Los UUID no se eligen aqui: ya estaban acordados en companion/src/lib/ble.ts
 * desde que se escribio la app. Este fichero es la otra mitad de ese contrato.
 *
 * Formato de los datos
 * --------------------
 * Binario empaquetado, little-endian, que es el orden nativo tanto del ESP32
 * como de DataView en el navegador cuando se le pide littleEndian=true. Nada de
 * JSON: son 22 bytes contra unos 200, y con notificaciones cada 3 s sobre una
 * conexion que tiene que sobrevivir a una demo en una sala llena de gente, el
 * tamano importa mas que la comodidad de leerlo.
 *
 *   VERDICT  (22 B, notifica al cerrar cada ventana)
 *     u8    clase          0..N-1, o 255 si no hay veredicto
 *     u8    confianza      0..255, equivale a 0..1
 *     f32   bpm
 *     f32   ie
 *     f32   tilt_grados
 *     f32   axial_mg
 *     f32   log_cociente
 *
 *   SENSORS  (16 B, notifica a 10 Hz)
 *     f32   tilt, axial, accel_mag, gyro_mag
 *
 *   STATUS   (14 B, notifica al cambiar de fase)
 *     u8    fase
 *     u8    banderas       bit0 montaje, bit1 calibrado, bit2 grabando
 *     f32   restante_s
 *     u16   ref_dia, ref_tor
 *     f32   contraste
 *
 * Por que el veredicto se envia aunque no lo haya
 * -----------------------------------------------
 * La clase 255 no es un error: es "el dispositivo se nego a adivinar". La app
 * tiene que poder distinguir eso de "no ha llegado nada todavia", porque son
 * dos estados clinicamente distintos y uno de ellos es el que protege al
 * paciente de un consejo inventado.
 */

#ifndef PC_BLE_H
#define PC_BLE_H

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

#include "sesion.h"

#define PC_BLE_NOMBRE   "PneumoCoach"
#define PC_BLE_SERVICIO "4fafc201-1fb5-459e-8fcc-c5c9c33191b0"
#define PC_BLE_VEREDICTO "beb5483e-36e1-4688-b7f5-ea07361b2b00"
#define PC_BLE_SENSORES  "beb5483e-36e1-4688-b7f5-ea07361b2b01"
#define PC_BLE_ESTADO    "beb5483e-36e1-4688-b7f5-ea07361b2b02"

static BLECharacteristic *g_ch_ver = nullptr;
static BLECharacteristic *g_ch_sen = nullptr;
static BLECharacteristic *g_ch_est = nullptr;
static volatile bool g_ble_conectado = false;

class PcServidorCB : public BLEServerCallbacks {
  void onConnect(BLEServer *s) override {
    g_ble_conectado = true;
    Serial.println(F("# BLE conectado"));
  }
  void onDisconnect(BLEServer *s) override {
    g_ble_conectado = false;
    Serial.println(F("# BLE desconectado, volviendo a anunciar"));
    /* Reanunciar de inmediato. Sin esto, cerrar la app deja el dispositivo
     * invisible hasta el siguiente reinicio, que en mitad de una demo es
     * indistinguible de que se haya colgado. */
    BLEDevice::startAdvertising();
  }
};

static void pc_ble_init() {
  BLEDevice::init(PC_BLE_NOMBRE);
  BLEServer *srv = BLEDevice::createServer();
  srv->setCallbacks(new PcServidorCB());

  BLEService *svc = srv->createService(PC_BLE_SERVICIO);

  auto crear = [&](const char *uuid) {
    BLECharacteristic *c = svc->createCharacteristic(
        uuid, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    /* El descriptor 0x2902 es lo que permite al cliente activar las
     * notificaciones. Sin el, la app se conecta, ve las caracteristicas y no
     * recibe nada: un fallo que parece de la app y es del firmware. */
    c->addDescriptor(new BLE2902());
    return c;
  };
  g_ch_ver = crear(PC_BLE_VEREDICTO);
  g_ch_sen = crear(PC_BLE_SENSORES);
  g_ch_est = crear(PC_BLE_ESTADO);

  svc->start();
  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(PC_BLE_SERVICIO);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();
  Serial.printf("# BLE anunciando como \"%s\"\n", PC_BLE_NOMBRE);
}

/* Escritores little-endian. Se hace a mano en vez de volcar un struct porque el
 * empaquetado de un struct depende del compilador, y el otro extremo es un
 * DataView en un navegador que no sabe nada de eso. */
static inline void pc_pon_f32(uint8_t *b, size_t off, float v) {
  memcpy(b + off, &v, 4);
}
static inline void pc_pon_u16(uint8_t *b, size_t off, uint16_t v) {
  b[off] = v & 0xFF; b[off + 1] = v >> 8;
}

static void pc_ble_veredicto(int clase, float conf, const float *f, float log_ratio) {
  if (!g_ble_conectado || !g_ch_ver) return;
  uint8_t b[22];
  b[0] = (clase < 0) ? 255 : (uint8_t)clase;
  b[1] = (uint8_t)(conf < 0 ? 0 : (conf > 1 ? 255 : conf * 255.0f));
  pc_pon_f32(b, 2, f[22]);            // bpm
  pc_pon_f32(b, 6, f[24]);            // relacion I:E
  pc_pon_f32(b, 10, f[0]);            // tilt_rms, grados
  pc_pon_f32(b, 14, f[10] * 1000.0f); // axial_rms, mg
  pc_pon_f32(b, 18, log_ratio);
  g_ch_ver->setValue(b, sizeof(b));
  g_ch_ver->notify();
}

static void pc_ble_sensores(float tilt, float axial, float amag, float gmag) {
  if (!g_ble_conectado || !g_ch_sen) return;
  uint8_t b[16];
  pc_pon_f32(b, 0, tilt);
  pc_pon_f32(b, 4, axial);
  pc_pon_f32(b, 8, amag);
  pc_pon_f32(b, 12, gmag);
  g_ch_sen->setValue(b, sizeof(b));
  g_ch_sen->notify();
}

static void pc_ble_estado(const pc_sesion_t *s, bool montaje, bool grabando) {
  if (!g_ble_conectado || !g_ch_est) return;
  uint8_t b[14];
  b[0] = (uint8_t)s->est;
  b[1] = (montaje ? 1 : 0) | (pc_ref_lista(&s->ref) ? 2 : 0) | (grabando ? 4 : 0);
  pc_pon_f32(b, 2, pc_sesion_restante(s));
  pc_pon_u16(b, 6, s->ref.n_dia);
  pc_pon_u16(b, 8, s->ref.n_tor);
  pc_pon_f32(b, 10, s->contraste);
  g_ch_est->setValue(b, sizeof(b));
  g_ch_est->notify();
}

#endif /* PC_BLE_H */
