"""Salida de consola en UTF-8, sin depender del entorno.

Importar antes de imprimir nada:

    import _consola  # noqa: F401

Por que existe
--------------
En Windows, Python hereda la pagina de codigos de la consola -cp1252 en un
sistema en espanol- y revienta con UnicodeEncodeError al imprimir un caracter
que no este en ella. En estas herramientas eso pasa con `sigma`, con el punto
medio y con las flechas de los informes.

El apano habitual es exportar PYTHONIOENCODING=utf-8 antes de cada comando. No
sirve: se olvida, no funciona igual en PowerShell que en bash, y convierte un
fallo del programa en deberes del usuario. Una herramienta tiene que poder
imprimir su propia salida.

`reconfigure` existe desde Python 3.7 y no hace nada en plataformas donde la
consola ya es UTF-8, asi que es seguro llamarlo siempre.
"""

from __future__ import annotations

import sys

for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError, OSError):
        # Redirigido a algo que no admite reconfiguracion. No es motivo para
        # abortar: lo peor que pasa es que vuelva el comportamiento anterior.
        pass
