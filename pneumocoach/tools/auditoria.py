"""Auditoría del proyecto: qué está medido, qué está supuesto, qué está mal.

    python tools/auditoria.py

Cuatro bloques, en orden de gravedad:

  A. ¿Es sólida la cifra que reportamos? Un número mal medido contamina todo
     lo que se construya encima, y es el error más caro porque nadie lo ve.
  B. ¿Están íntegras las grabaciones sobre las que concluimos?
  C. ¿Queda algún artefacto desplegable entrenado con física refutada?
  D. ¿Queda alguna afirmación viva en el repositorio que ya sepamos falsa?

Se ejecuta entero y no se detiene en el primer fallo: la idea es ver el estado
completo, no el primer síntoma.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ml"))
sys.path.insert(0, str(REPO / "tools"))


import _consola  # noqa: E402,F401  UTF-8 en Windows
from pneumocoach import config as C  # noqa: E402
from pneumocoach.calibracion import ReferenciaSesion  # noqa: E402
from analizar_captura import bloques, cargar, cargar_calibracion  # noqa: E402
from medir_calibracion import GRUPO, SESIONES, cargar_sesion  # noqa: E402
from scipy import stats  # noqa: E402

HALLAZGOS: list[tuple[str, str]] = []


def grave(msg):
    HALLAZGOS.append(("GRAVE", msg))
    print(f"  [GRAVE]  {msg}")


def aviso(msg):
    HALLAZGOS.append(("AVISO", msg))
    print(f"  [aviso]  {msg}")


def ok(msg):
    print(f"  [ok]     {msg}")


# ==========================================================================
def bloque_a_solidez_de_la_cifra(D):
    print("\n" + "=" * 74)
    print("  A · ¿ES SÓLIDA LA CIFRA?")
    print("=" * 74)

    # A1. Balance de clases en cada conjunto de prueba.
    print("\n  A1. Balance de clases por pliegue")
    for k, (X, y, ref) in D.items():
        yte = y[~ref]
        p = yte.mean()
        marca = "" if 0.35 <= p <= 0.65 else "  <- desbalanceado"
        print(f"      {k:<16}{len(yte):>4} ventanas   {(1-p)*100:>5.1f}% dia / "
              f"{p*100:.1f}% tor{marca}")
        if not 0.3 <= p <= 0.7:
            aviso(f"{k}: clases desbalanceadas, la accuracy es engañosa ahí")

    # A2. Independencia efectiva: las ventanas se solapan al 75 %.
    solape = 1 - C.HOP_N / C.WINDOW_N
    print(f"\n  A2. Solape de ventanas: {solape:.0%}")
    print(f"      {'pliegue':<16}{'ventanas':>10}{'independientes~':>18}")
    for k, (X, y, ref) in D.items():
        n = int((~ref).sum())
        indep = max(1, int(n * (1 - solape)))
        print(f"      {k:<16}{n:>10}{indep:>18}")
    aviso("Con 75 % de solape, el n efectivo es ~1/4 del nominal. Los "
          "intervalos de confianza reales son mucho más anchos que los "
          "que sugiere el número de ventanas.")

    # A3. Sesiones del mismo protocolo se predicen entre si.
    print("\nA3. Similitud entre sesiones")
    print("      Dos sesiones del mismo grupo comparten consignas y estructura:")
    print("      entrenar con una y probar en la otra mide parecido, no")
    print("      generalizacion.")
    for g in sorted({GRUPO[k] for k in D}):
        print(f"        grupo {g}: {[k for k in D if GRUPO[k] == g]}")
    Xs = {k: D[k][0] for k in D}
    cen = {k: Xs[k].mean(axis=0) for k in Xs}
    ks = list(D)
    print()
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = ks[i], ks[j]
            dist = np.linalg.norm(
                (cen[a] - cen[b]) / (np.abs(cen[a]) + np.abs(cen[b]) + 1e-9))
            mismo = "  <- mismo protocolo" if GRUPO[a] == GRUPO[b] else ""
            print(f"      distancia {a} <-> {b}: {dist:.2f}{mismo}")

    # A4. La cifra, con validacion por grupo y su intervalo.
    from medir_calibracion import evaluar
    _, sin_f, grupos, _ = evaluar(False, D)
    _, con_f, _, crudo = evaluar(True, D)
    print("\nA4. Dejar-un-GRUPO-de-protocolo-fuera")
    for g, a, b in zip(grupos, sin_f, con_f):
        print(f"      probando en {g:<12} sin {a:.3f}   con {b:.3f}")
    ac = sum(c[0] for c in crudo)
    nv = sum(c[1] for c in crudo)
    p_hat = ac / nv
    n_eff = nv / (C.WINDOW_N / C.HOP_N)
    lo, hi = stats.beta.ppf([0.025, 0.975],
                            p_hat * n_eff + 0.5, (1 - p_hat) * n_eff + 0.5)
    pv = stats.binomtest(round(p_hat * n_eff), round(n_eff), 0.5,
                         alternative="greater").pvalue
    print(f"\nagregado {ac}/{nv} = {p_hat:.3f}   n efectivo {n_eff:.0f}")
    print(f"      IC 95 % [{lo:.3f}, {hi:.3f}]   p = {pv:.4f}")
    if lo > 0.5:
        print("      SIGNIFICATIVO frente al azar de 0.500")
    else:
        grave(f"La cifra {p_hat:.3f} NO es significativa: el IC 95 % "
              f"[{lo:.3f}, {hi:.3f}] incluye el azar.")


# ==========================================================================
def bloque_b_integridad(R, sesgo):
    print("\n" + "=" * 74)
    print("  B · INTEGRIDAD DE LAS GRABACIONES")
    print("=" * 74)
    print(f"\n  {'sesión':<16}{'muestras':>9}{'fs real':>9}{'satur.':>8}"
          f"{'ceros':>7}{'|g|':>7}{'ruido az':>10}")
    for k, path in SESIONES.items():
        counts, et = cargar(REPO / path)
        n = len(counts)
        sat = int(np.count_nonzero(np.abs(counts[:, :3]) >= 32700))
        cer = int(np.count_nonzero(np.all(counts == 0, axis=1)))
        g = np.linalg.norm(counts[:, :3] / C.ACCEL_LSB_PER_G, axis=1).mean()
        ruido = float(np.std(np.diff(counts[:, 2] / C.ACCEL_LSB_PER_G))) * 1000
        # La tasa real se estima del propio conteo contra la duración declarada
        # de los bloques, que es lo único independiente que tenemos.
        dur = sum(b - a for _, a, b in bloques(et)) / C.FS_HZ
        fs = n / dur if dur else 0
        print(f"  {k:<16}{n:>9}{fs:>9.2f}{sat:>8}{cer:>7}{g:>7.3f}{ruido:>9.1f}m")
        if sat:
            grave(f"{k}: {sat} muestras saturadas")
        if cer:
            grave(f"{k}: {cer} muestras en cero (bus caído)")
        if abs(g - 1.0) > 0.15:
            grave(f"{k}: |g| = {g:.3f}, fuera de rango")
        elif abs(g - 1.0) > 0.06:
            aviso(f"{k}: |g| = {g:.3f}, error de escala del {abs(g-1)*100:.1f} % sin calibrar")
        if ruido > 20:
            aviso(f"{k}: ruido {ruido:.1f} mg, por encima del umbral de 20")


# ==========================================================================
def bloque_c_artefactos():
    """De donde salio el modelo que se puede flashear.

    La primera version daba por hecho que cualquier modelo desplegable era el
    sintetico, porque cuando se escribio solo existia ese. Comprobar la
    EXISTENCIA en vez de la PROCEDENCIA convierte la auditoria en un detector
    que se equivoca en cuanto el problema se arregla, y una alarma que suena
    siempre deja de mirarse.
    """
    import json
    print("\n" + "=" * 74)
    print("  C · ARTEFACTOS DESPLEGABLES")
    print("=" * 74)
    modelo = REPO / "firmware" / "src" / "pneumocoach_model.c"
    informe = REPO / "ml" / "artifacts" / "report.json"

    if not modelo.exists():
        ok("No hay modelo desplegable en el firmware.")
        return

    if not informe.exists():
        grave("Hay un modelo en firmware/src/ pero no hay report.json que diga "
              "de donde salio. Un modelo sin procedencia no se flashea.")
        return

    rep = json.loads(informe.read_text(encoding="utf-8"))
    fuente = str(rep.get("fuente", ""))
    if "real" not in fuente.lower():
        grave(f"El modelo desplegable procede de: {fuente or 'sin declarar'}. "
              "Si no viene de grabaciones reales, el dispositivo dara "
              "veredictos confiados sobre una fisica que ya se refuto.")
        return

    print(f"\nprocedencia : {fuente}")
    acc = rep.get("binaria_cruzando_protocolos")
    ic = rep.get("ic95") or [float('nan')] * 2
    if acc is not None:
        print(f"  cifra       : {acc:.3f}  IC 95 % [{ic[0]:.3f}, {ic[1]:.3f}]"
              f"  n efectivo {rep.get('n_efectivo', 0):.0f}")
        if ic[0] <= 0.5:
            grave(f"La cifra del modelo desplegable ({acc:.3f}) no es "
                  f"significativa: el IC incluye el azar.")
        else:
            ok(f"El modelo desplegable viene de datos reales y su cifra es "
               f"significativa.")
    print(f"  tamano      : {rep.get('bytes_modelo', 0)} bytes")


# ==========================================================================
def bloque_d_afirmaciones():
    print("\n" + "=" * 74)
    print("  D · AFIRMACIONES VIVAS QUE YA SABEMOS FALSAS")
    print("=" * 74)
    # Las salvedades se buscan en una ventana a ambos lados del hallazgo. Van en
    # LOS DOS IDIOMAS del repositorio: el codigo y los comentarios estan en
    # espanol, la documentacion en ingles, y una lista con solo raices espanolas
    # marcaba como afirmacion cada linea inglesa que negaba justamente eso.
    SALVEDADES_COCIENTE = (
        # espanol
        "falso", "refutad", "no separa", "no separaba", "no es", "no lo es",
        "invertid", "al reves", "cancela", "adr-0006",
        # ingles
        "false", "refuted", "does not separate", "no separation", "inverted",
        "cancels", "later measured", "not the discriminator", "no longer",
    )

    patrones = [
        # Antes esto era un lookahead `(?!.{0,120}(retirad|Retracted|...))`, y
        # tenia dos fallos que se tapaban entre si. `.` no casa salto de linea
        # sin re.S, asi que una salvedad en la linea siguiente no contaba y
        # obligaba a escribir la prosa para contentar al comprobador. Y el
        # lookahead solo mira HACIA DELANTE: "Retracted: ... el 89.7 %" no lo
        # satisfacia. Ahora usa la misma ventana a ambos lados que el 0.785, que
        # es el mecanismo que este mismo fichero ya habia elegido por lo mismo.
        (r"89\.7\s*%", "cita el 89.7 % retirado",
         ("retirad", "retractad", "retracted", "earlier version", "no sobrevivio",
          "did not survive", "sintetic", "synthetic", "simulation is not")),
        (r"0\.8968|0\.894", "cita métricas del modelo sintético", None),
        (r"primary discriminator", "llama al cociente discriminador principal",
         SALVEDADES_COCIENTE),
        (r"discriminador (principal|del proyecto)",
         "llama al cociente discriminador principal",
         SALVEDADES_COCIENTE),
        # Este patron nacio SIN salvedades, y con la documentacion en espanol
        # eso no se notaba porque la frase inglesa no aparecia. Al traducir los
        # documentos empezo a marcar precisamente las lineas que DECLARAN que el
        # cociente no separa. Un patron sin salvedad no se puede satisfacer:
        # obliga a no escribir la frase ni siquiera para negarla.
        (r"tilt/axial ratio.{0,40}separat", "afirma que el cociente separa",
         SALVEDADES_COCIENTE),
        # Los tres patrones de arriba solo pillan ingles tecnico o la frase
        # literal "discriminador principal". La prosa normal se colaba entera:
        # la pestana Sensores de la app decia "Su cociente es lo que separa una
        # tecnica correcta de una toracica" y este bloque salia limpio. Un
        # criterio de verificacion que pasa con el texto malo no es un criterio.
        # `[^.]` mantiene la busqueda DENTRO de una frase. La primera version
        # usaba `.` y cruzaba puntos, asi que marcaba un comentario que
        # advertia CONTRA una separacion espuria como si la afirmara. Un
        # comprobador con falsos positivos se acaba desactivando, y entonces
        # deja de atrapar los verdaderos.
        (r"cociente[^.]{0,80}separa[^.]{0,40}"
         r"(t[eé]cnica|tor[aá]cica|diafragm[aá]tica|clase)",
         "afirma en prosa que el cociente separa las tecnicas",
         ("falso", "refutad", "no separa", "no es", "adr-0006", "no lo es",
          "invertid", "al reves", "cancela")),
        (r"firma de la respiraci[oó]n tor[aá]cica",
         "llama al cociente la firma de la respiracion toracica",
         ("falso", "refutad", "adr-0006", "no lo es", "invertid", "al reves")),
        # El promedio de los tres pliegues. Es citable, pero solo junto a la
        # razon por la que esta inflado. La salvedad puede ir antes o despues
        # del numero y en cualquiera de los dos idiomas del repo, asi que se
        # busca en una ventana a ambos lados en vez de con un lookahead.
        (r"78\.5\s*%|0\.785", "cita el promedio inflado de 0.785 sin la salvedad",
         ("inflad", "pliegue s1", "mismo protocolo", "no usarlo",
          "comparten protocolo", "not the", "same protocol", "near-duplicate",
          "similarity, not generalis", "we report is 0.606",
          # formas que aparecen en la documentacion en ingles
          "inflated", "superseded", "share a protocol", "predict each other",
          "resemble each other", "average", "not a performance estimate")),
    ]
    exts = {".md", ".py", ".ts", ".tsx", ".h", ".c", ".ino"}
    # Copias de trabajo aisladas y dependencias: auditarlas duplica cada aviso
    # y ademas senala un fichero que no existe para quien lee el informe.
    saltar = {"node_modules", ".next", ".git", ".worktrees", ".venv",
              "artifacts", "respaldo", "bin"}
    hits = 0
    for f in REPO.rglob("*"):
        if f.suffix not in exts or any(s in f.parts for s in saltar):
            continue
        # Los ADR documentan lo refutado a propósito: ahí las citas son válidas.
        if "adr" in f.parts or f.name == "auditoria.py":
            continue
        # Los ficheros GENERADOS son tablas de numeros, no afirmaciones. Una
        # secuencia de digitos dentro de un vector de floats casaba con "0.785"
        # y disparaba una alarma sobre datos que nadie escribio ni afirma.
        if f.name.startswith("pneumocoach_") and f.suffix in {".h", ".c"}:
            continue
        try:
            txt = f.read_text(encoding="utf-8")
        except Exception:
            continue
        for pat, desc, salvedades in patrones:
            for m in re.finditer(pat, txt, re.I):
                if salvedades:
                    ctx = txt[max(0, m.start() - 200):m.end() + 200].lower()
                    if any(s in ctx for s in salvedades):
                        continue
                linea = txt[:m.start()].count("\n") + 1
                rel = f.relative_to(REPO)
                aviso(f"{rel}:{linea} {desc}")
                hits += 1
                hits += 1
    if not hits:
        ok("Ninguna afirmación refutada viva fuera de los ADR.")


# ==========================================================================
def hay_grabaciones() -> list[str]:
    """Sesiones declaradas que NO están en disco.

    `data/raw/` está en `.gitignore` porque son datos biométricos de personas
    identificables por contexto. Quien clone el repositorio no las tiene, y los
    bloques A y B no se pueden ejecutar sin ellas. Antes eso reventaba con un
    `FileNotFoundError` a mitad de la salida, que parece un fallo del código y
    es simplemente que faltan los datos.
    """
    return [k for k, v in SESIONES.items() if not (REPO / v).exists()]


def main():
    faltan = hay_grabaciones()
    if faltan:
        print("=" * 74)
        print("  SIN GRABACIONES — se auditan solo los bloques C y D")
        print("=" * 74)
        print()
        print("  No están en disco:")
        for k in faltan:
            print(f"    {k:<16} {SESIONES[k]}")
        print()
        print("  `data/raw/` está en .gitignore: son datos biométricos de")
        print("  personas identificables por contexto y no entran al")
        print("  repositorio. Los bloques A (solidez de la cifra) y B")
        print("  (integridad de las grabaciones) necesitan los datos crudos.")
        print()
        print("  Para grabar una sesión: docs/DUAL_SENSOR_PROTOCOL.md")
    else:
        R, sesgo = cargar_calibracion("s01")
        D = {k: cargar_sesion(v) for k, v in SESIONES.items()}
        bloque_a_solidez_de_la_cifra(D)
        bloque_b_integridad(R, sesgo)

    bloque_c_artefactos()
    bloque_d_afirmaciones()

    g = sum(1 for s, _ in HALLAZGOS if s == "GRAVE")
    a = sum(1 for s, _ in HALLAZGOS if s == "AVISO")
    print("\n" + "=" * 74)
    print(f"  RESUMEN: {g} graves, {a} avisos")
    print("=" * 74)
    return 1 if g else 0


if __name__ == "__main__":
    sys.exit(main())
