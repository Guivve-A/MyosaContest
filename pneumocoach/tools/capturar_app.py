"""Capturas reales de la app companion, para blog y presentacion.

    python tools/capturar_app.py --url http://localhost:3000

Por que no vale una maqueta
---------------------------
Una diapositiva con un dibujo de la interfaz afirma algo que no se ha medido:
que la interfaz existe y se ve asi. Este script arranca Chrome sin ventana,
carga la app DE VERDAD desde el servidor que este corriendo, y guarda lo que el
navegador pinta. Si la app esta rota, la captura sale rota, que es exactamente
lo que uno quiere de una prueba.

Como se conduce el navegador
----------------------------
Por CDP -el protocolo de DevTools- sobre WebSocket. Chrome expone la lista de
objetivos en HTTP y cada uno acepta ordenes por WS. No hace falta Playwright ni
Selenium: son tres llamadas.

El tutorial se recuerda en localStorage bajo `pc-tutorial-visto`. Se captura
primero SIN esa marca -que es lo que ve un paciente la primera vez- y despues
se pone la marca para llegar al panel. Ese orden importa: al reves habria que
borrar el perfil entre capturas.

Salida en `data/capturas/`, que esta ignorado por git igual que el resto de
`data/`.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
import websocket

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import _consola  # noqa: E402,F401

CHROME = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

# ancho, alto, escala. La escala 2 es la de un telefono moderno: sin ella las
# capturas se ven borrosas proyectadas.
MOVIL = (390, 844, 2)
ESCRITORIO = (1280, 900, 2)


def busca_chrome() -> str:
    for c in CHROME:
        if Path(c).exists():
            return c
    raise SystemExit("No se encontro Chrome ni Edge. Instala uno o pasa --chrome.")


class Sesion:
    """Un objetivo de Chrome conducido por CDP."""

    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self.n = 0

    def manda(self, metodo: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": metodo,
                                 "params": params}))
        # Chrome intercala eventos con las respuestas; hay que descartar todo lo
        # que no lleve nuestro id o la espera se queda con el evento equivocado.
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{metodo}: {msg['error']}")
                return msg.get("result", {})

    def cerrar(self):
        try:
            self.ws.close()
        except Exception:
            pass


def arranca(chrome: str, puerto: int, perfil: Path) -> subprocess.Popen:
    proc = subprocess.Popen(
        [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--no-first-run", "--no-default-browser-check",
         f"--remote-debugging-port={puerto}",
         # Chrome rechaza el WebSocket de CDP si el Origin no esta permitido.
         # El navegador vive segundos, en un perfil temporal y sin red externa.
         "--remote-allow-origins=*",
         f"--user-data-dir={perfil}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            r = requests.get(f"http://127.0.0.1:{puerto}/json/version", timeout=1)
            if r.ok:
                return proc
        except Exception:
            pass
        time.sleep(0.25)
    proc.kill()
    raise SystemExit("Chrome no abrio el puerto de depuracion.")


def objetivo(puerto: int) -> str:
    for t in requests.get(f"http://127.0.0.1:{puerto}/json", timeout=5).json():
        if t.get("type") == "page":
            return t["webSocketDebuggerUrl"]
    raise SystemExit("Chrome no expone ninguna pestana.")


"""CSS que se inyecta despues de cada carga.

El indicador de herramientas de Next.js se pinta sobre la esquina inferior y no
forma parte de la app: en una diapositiva parece un boton nuestro. Se oculta
para la captura, no en el codigo, porque en desarrollo si se quiere.
"""
OCULTAR = "nextjs-portal,[data-next-badge-root],#__next-build-watcher" \
          "{display:none !important}"


def carga(s: Sesion, url: str, medidas, espera: float = 2.5):
    w, h, escala = medidas
    s.manda("Emulation.setDeviceMetricsOverride", width=w, height=h,
            deviceScaleFactor=escala, mobile=escala > 1 and w < 768)
    s.manda("Page.navigate", url=url)
    time.sleep(espera)
    js(s, f"""(() => {{
        const e = document.createElement('style');
        e.textContent = {json.dumps(OCULTAR)};
        document.head.appendChild(e);
      }})()""")


def js(s: Sesion, expr: str):
    r = s.manda("Runtime.evaluate", expression=expr, awaitPromise=True,
                returnByValue=True)
    return r.get("result", {}).get("value")


def dispara(s: Sesion, texto: str) -> bool:
    """Pulsa el primer elemento cuyo texto visible contenga `texto`."""
    hecho = js(s, f"""
      (() => {{
        const t = {json.dumps(texto)}.toLowerCase();
        const el = [...document.querySelectorAll('button,[role=tab],a')]
          .find(e => (e.innerText||'').toLowerCase().includes(t));
        if (!el) return false;
        el.click();
        return true;
      }})()
    """)
    if hecho:
        time.sleep(1.2)
    return bool(hecho)


def guarda(s: Sesion, destino: Path):
    r = s.manda("Page.captureScreenshot", format="png", captureBeyondViewport=False)
    destino.write_bytes(base64.b64decode(r["data"]))
    kb = destino.stat().st_size / 1024
    print(f"  {destino.name:<34} {kb:7.0f} KB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:3000")
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--salida", default="data/capturas")
    ap.add_argument("--puerto", type=int, default=9333)
    args = ap.parse_args()

    chrome = args.chrome or busca_chrome()
    salida = REPO / args.salida
    salida.mkdir(parents=True, exist_ok=True)

    try:
        requests.get(args.url, timeout=3)
    except Exception:
        raise SystemExit(f"No responde {args.url}. Levanta el servidor antes.")

    # Un Chrome de una ejecucion anterior que siguiera vivo se queda con el
    # puerto, y el nuestro -el que si lleva los flags- muere en silencio al no
    # poder abrirlo. Entonces nos conectariamos al VIEJO y el fallo parece del
    # protocolo. Comprobarlo antes cuesta una llamada.
    try:
        v = requests.get(f"http://127.0.0.1:{args.puerto}/json/version", timeout=1)
        if v.ok:
            raise SystemExit(
                f"El puerto {args.puerto} ya lo tiene otro Chrome "
                f"({v.json().get('Browser', '?')}). Cierralo "
                f"(taskkill /F /IM chrome.exe) o pasa otro --puerto.")
    except requests.RequestException:
        pass

    perfil = Path(tempfile.mkdtemp(prefix="pc_chrome_"))
    proc = arranca(chrome, args.puerto, perfil)
    try:
        s = Sesion(objetivo(args.puerto))
    except Exception:
        proc.kill()
        shutil.rmtree(perfil, ignore_errors=True)
        raise
    try:
        s.manda("Page.enable")
        s.manda("Runtime.enable")

        print("Capturando (movil 390x844 @2x):")

        # 1-3. El tutorial, tal como lo ve alguien que abre la app por primera
        # vez. Se captura ANTES de marcar `pc-tutorial-visto`.
        carga(s, args.url, MOVIL)
        js(s, "localStorage.removeItem('pc-tutorial-visto')")
        carga(s, args.url, MOVIL)
        guarda(s, salida / "app-tutorial-1-conexion.png")
        for i, etiqueta in ((2, "Siguiente"), (3, "Siguiente")):
            if dispara(s, etiqueta):
                guarda(s, salida / f"app-tutorial-{i}-paso.png")

        # 4-6. El panel. La marca en localStorage salta el tutorial.
        #
        # Sin conectar, las tarjetas salen vacias y la captura no ensena nada.
        # Se pulsa Conectar, que en un navegador sin BLE cae al SIMULADOR: la
        # barra superior lo declara con la insignia "SIMULADO" y esa insignia
        # tiene que quedar VISIBLE en la captura. Un panel lleno presentado como
        # si fueran datos de un paciente seria exactamente la clase de
        # afirmacion que este proyecto no hace.
        js(s, "localStorage.setItem('pc-tutorial-visto','1')")
        carga(s, args.url, MOVIL)
        dispara(s, "Conectar")
        time.sleep(9)   # que la traza de las graficas tenga historia
        guarda(s, salida / "app-resumen.png")
        for pestana, nombre in (("Sensores", "app-sensores.png"),
                                ("Progreso", "app-progreso.png")):
            if dispara(s, pestana):
                time.sleep(2)
                guarda(s, salida / nombre)

        print("Capturando (escritorio 1280x900 @2x):")
        carga(s, args.url, ESCRITORIO)
        dispara(s, "Conectar")
        time.sleep(9)
        guarda(s, salida / "app-escritorio-resumen.png")
    finally:
        s.cerrar()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        shutil.rmtree(perfil, ignore_errors=True)

    print(f"\nEn {salida.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
