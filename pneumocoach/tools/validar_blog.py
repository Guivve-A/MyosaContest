"""Valida el blog contra las reglas de envío de MYOSA.

    python tools/validar_blog.py

La guía dice que un envío fuera de formato puede resultar en descalificación
inmediata, así que la conformidad se comprueba mecánicamente y no a ojo.
Cada regla del documento oficial es una comprobación aquí.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "tools"))
import _consola  # noqa: E402,F401  UTF-8 en Windows
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BLOG = REPO / "blog" / "pneumocoach.md"
# La guia exige que el markdown, las imagenes y el video esten en la MISMA
# carpeta, asi que la carpeta de recursos ES la carpeta del markdown. Antes
# habia un subdirectorio `assets/` y rutas absolutas de sitio: las dos cosas
# incumplen y son motivo de reenvio.
ENVIO = BLOG.parent
ASSETS = ENVIO

# Orden exacto del documento de guías. Las marcadas opcionales pueden faltar,
# pero si aparecen deben ir en esta posición.
SECCIONES = [
    ("## Acknowledgements", True),
    ("## Overview", True),
    ("## Demo / Examples", True),
    ("### **Images**", True),
    ("### **Videos**", True),
    ("## Features (Detailed)", True),
    ("## Usage Instructions", True),
    ("## Tech Stack", True),
    ("## Requirements / Installation", True),
    ("## File Structure", False),
    ("## License", False),
    ("## Contribution Notes", False),
]

FRONTMATTER = ["publishDate", "title", "excerpt", "image", "tags"]

fallos: list[str] = []
avisos: list[str] = []


def falla(msg):
    fallos.append(msg)


def avisa(msg):
    avisos.append(msg)


def main() -> int:
    if not BLOG.exists():
        print(f"No existe {BLOG}")
        return 1
    texto = BLOG.read_text(encoding="utf-8")
    lineas = texto.splitlines()

    print("=" * 68)
    print(f"  Validando {BLOG.name} contra las guías de MYOSA")
    print("=" * 68)

    # ---- 1. Frontmatter -------------------------------------------------
    if not texto.startswith("---\n"):
        falla("El archivo debe empezar con '---' en la primera línea")
    else:
        fin = texto.index("\n---\n", 4)
        fm = texto[4:fin]
        for campo in FRONTMATTER:
            if not re.search(rf"^{campo}:", fm, re.M):
                falla(f"Falta el campo '{campo}' en el frontmatter")
        if m := re.search(r"^publishDate:\s*(\S+)", fm, re.M):
            if not re.match(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}Z)?$", m.group(1)):
                falla(f"publishDate mal formado: {m.group(1)}")
        tags = re.findall(r"^\s+-\s+(\S+)", fm, re.M)
        if len(tags) < 3:
            avisa(f"Solo {len(tags)} tags; el ejemplo muestra 3 o más")
        print(f"  frontmatter        OK  ({len(tags)} tags)")

    # ---- 2. Tagline ------------------------------------------------------
    if not re.search(r"^>\s+\S", texto, re.M):
        falla("Falta la línea de tagline con '>' después del frontmatter")
    else:
        print("  tagline            OK")

    # ---- 3. Secciones en orden ------------------------------------------
    pos = -1
    for encabezado, obligatoria in SECCIONES:
        try:
            i = texto.index(encabezado)
        except ValueError:
            if obligatoria:
                falla(f"Falta la sección obligatoria: {encabezado}")
            continue
        if i < pos:
            falla(f"Sección fuera de orden: {encabezado}")
        pos = i
    print(f"  secciones          OK  ({len(SECCIONES)} comprobadas, en orden)")

    # ---- 4. Imágenes -----------------------------------------------------
    # Formato exigido: <p align="center"> con <img ... width> y <i>caption</i>
    bloques = re.findall(
        r'<p align="center">\s*\n\s*<img src="([^"]+)"[^>]*width="\d+"[^>]*>\s*<br/>\s*\n\s*<i>([^<]+)</i>\s*\n\s*</p>',
        texto,
    )
    imgs_sueltas = re.findall(r"!\[[^\]]*\]\([^)]+\)", texto)
    if imgs_sueltas:
        falla(f"{len(imgs_sueltas)} imagen(es) en markdown; la guía exige el bloque HTML")
    if not bloques:
        falla("No se encontró ninguna imagen en el formato exigido")
    else:
        print(f"  imágenes           OK  ({len(bloques)} en formato correcto)")

    for src, cap in bloques:
        nombre = src.rsplit("/", 1)[-1]
        # "All files (Markdown, images, and video) must be placed in the same
        # folder". Una ruta con barras apunta fuera de la carpeta de envio, y
        # una que empiece por "/" ademas asume una raiz de sitio que no
        # controlamos. Las dos son motivo de reenvio.
        if "/" in src:
            falla(f"Ruta no relativa: {src!r}. La guia exige el markdown, las "
                  f"imagenes y el video en la MISMA carpeta, asi que el src "
                  f"debe ser solo {nombre!r}.")
        if " " in nombre:
            falla(f"Nombre de imagen con espacio: {nombre}")
        if nombre != nombre.lower():
            falla(f"Nombre de imagen con mayúsculas: {nombre}")
        if not nombre.lower().endswith((".jpg", ".jpeg", ".png")):
            falla(f"Formato de imagen no permitido: {nombre}")
        if not (ASSETS / nombre).exists():
            falla(f"Imagen referenciada pero ausente en blog/: {nombre}")
        if not cap.strip():
            falla(f"Imagen sin caption: {nombre}")

    # ---- 5. Vídeo --------------------------------------------------------
    if re.search(r"youtu\.?be|youtube\.com", texto, re.I):
        falla("Hay un enlace de YouTube. La guía lo prohíbe explícitamente.")
    vids = re.findall(
        r'<video controls width="100%">\s*\n\s*<source src="([^"]+)" type="video/mp4">\s*\n\s*</video>',
        texto,
    )
    if not vids:
        falla("Falta el vídeo. La guía lo declara obligatorio.")
    else:
        print(f"  vídeo              OK  ({len(vids)} en formato correcto)")
        for v in vids:
            nombre = v.rsplit("/", 1)[-1]
            if not nombre.lower().endswith(".mp4"):
                falla(f"El vídeo debe ser .mp4: {nombre}")
            if " " in nombre or nombre != nombre.lower():
                falla(f"Nombre de vídeo inválido: {nombre}")
            if "/" in v:
                falla(f"Ruta de video no relativa: {v!r}. Debe ser solo "
                      f"{nombre!r}.")
            if not (ASSETS / nombre).exists():
                avisa(f"PENDIENTE: falta grabar y colocar {nombre} en blog/")

    # ---- 6. Portada ------------------------------------------------------
    if m := re.search(r"^image:\s*(\S+)", texto, re.M):
        crudo = m.group(1)
        portada = crudo.rsplit("/", 1)[-1]
        if "/" in crudo:
            falla(f"La portada del frontmatter no es relativa: {crudo!r}. "
                  f"Debe ser solo {portada!r}.")
        if not (ASSETS / portada).exists():
            falla(f"La imagen de portada no existe: {portada}")
        else:
            print(f"  portada            OK  ({portada})")
    else:
        falla("Falta la imagen de portada en el frontmatter")

    # ---- 7. Contenido mínimo --------------------------------------------
    if "**Key features:**" not in texto:
        avisa("El ejemplo incluye '**Key features:**' bajo Overview")
    subsecciones = re.findall(r"^### \*\*\d+\.", texto, re.M)
    if len(subsecciones) < 3:
        avisa(f"Features (Detailed) tiene {len(subsecciones)} subsecciones; el ejemplo muestra 3+")
    else:
        print(f"  features           OK  ({len(subsecciones)} subsecciones numeradas)")

    bloques_codigo = texto.count("```")
    if bloques_codigo % 2:
        falla("Hay un bloque de código sin cerrar")
    print(f"  bloques de código  OK  ({bloques_codigo // 2})")

    # ---- 8. La carpeta de envío -----------------------------------------
    # Todo en minúsculas y sin espacios, y nada que no forme parte del envío:
    # un fichero de más en la carpeta es un fichero que el jurado abre.
    PERMITIDOS = {".md", ".png", ".jpg", ".jpeg", ".mp4"}
    for f in sorted(ENVIO.iterdir()) if ENVIO.exists() else []:
        if f.is_dir():
            falla(f"Subcarpeta en el envío: {f.name}/. La guía exige que el "
                  f"markdown, las imágenes y el vídeo estén en la MISMA carpeta.")
            continue
        if " " in f.name or f.name != f.name.lower():
            falla(f"Nombre inválido en el envío: {f.name}")
        if f.suffix.lower() not in PERMITIDOS:
            falla(f"Archivo que no pertenece al envío: {f.name}")

    referenciados = {s.rsplit("/", 1)[-1] for s, _ in bloques}
    referenciados |= {v.rsplit("/", 1)[-1] for v in vids}
    if m := re.search(r"^image:\s*(\S+)", texto, re.M):
        referenciados.add(m.group(1).rsplit("/", 1)[-1])
    for f in sorted(ENVIO.iterdir()) if ENVIO.exists() else []:
        if f.is_file() and f.suffix.lower() != ".md" and f.name not in referenciados:
            avisa(f"{f.name} está en la carpeta pero el markdown no lo usa")

    palabras = len(texto.split())
    print(f"  extensión          {palabras:,} palabras")

    # ---- Resultado -------------------------------------------------------
    print()
    if avisos:
        print("AVISOS")
        for a in avisos:
            print(f"  · {a}")
        print()
    if fallos:
        print("FALLOS DE FORMATO — corregir antes de enviar")
        for f in fallos:
            print(f"  ! {f}")
        print()
        return 1
    print("CONFORME con las guías de envío de MYOSA.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
