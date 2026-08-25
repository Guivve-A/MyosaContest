"""Calibración por sesión: el eje de referencia del propio paciente.

Por qué existe
--------------
Tres sesiones del mismo sujeto, con el mismo montaje y separadas por minutos,
dieron amplitudes que varían un 50 % y características cuyo signo cambia entre
sesiones. Un clasificador entrenado en dos sesiones y probado en la tercera
alcanza 0.66 contra un azar de 0.50 — y eso es la generalización MÁS FÁCIL
posible, porque ni siquiera cambia de persona.

La causa no es el sensor ni el modelo: es que la relación entre «técnica» y
mecánica del esternón depende del sujeto, del montaje y de cómo ejecute ese día.
Preguntar «¿esto es torácico en términos absolutos?» no tiene respuesta estable.

La pregunta que sí la tiene es relativa: **¿esto se parece más a TU torácica o a
TU diafragmática de hoy?**

Cómo funciona
-------------
Al inicio de la sesión el paciente ejecuta dos maniobras de referencia, guiado
por el dispositivo. De cada una se extrae un vector de características promedio.
Esos dos vectores definen, para cada característica, un eje:

    z = (x - ref_dia) / (ref_tor - ref_dia)

Con eso la diafragmática del paciente cae en 0 y su torácica en 1, sea cual sea
su contextura, su montaje o su esfuerzo de ese día. El modelo global se entrena
y evalúa sobre `z`, no sobre valores absolutos.

Es una transformación afín por característica, así que corrige a la vez el
desplazamiento y la ganancia — las dos formas en que la señal derivaba.

Coste clínico
-------------
Unos cuarenta segundos al inicio de cada sesión. Es un coste real y hay que
declararlo, pero no es ajeno a la práctica: un fisioterapeuta también observa al
paciente antes de corregirlo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C

# Ambas viven en config.py: el firmware corre esta misma calibración a bordo y
# las dos implementaciones tienen que compartir los números. Se reexportan aquí
# porque este es el módulo donde se usan y donde se documentan.
REF_SEGUNDOS = C.REF_SEGUNDOS
CONTRASTE_MINIMO = C.CONTRASTE_MINIMO


@dataclass
class ReferenciaSesion:
    """Los dos vectores que definen el eje de este paciente en esta sesión."""

    dia: np.ndarray  # vector de características de la maniobra diafragmática
    tor: np.ndarray  # vector de características de la maniobra torácica
    n_dia: int = 0
    n_tor: int = 0

    @classmethod
    def desde_ventanas(
        cls, X_dia: np.ndarray, X_tor: np.ndarray
    ) -> "ReferenciaSesion":
        if len(X_dia) == 0 or len(X_tor) == 0:
            raise ValueError("hacen falta ventanas de ambas maniobras de referencia")
        return cls(
            dia=np.asarray(X_dia, dtype=np.float64).mean(axis=0),
            tor=np.asarray(X_tor, dtype=np.float64).mean(axis=0),
            n_dia=len(X_dia),
            n_tor=len(X_tor),
        )

    @property
    def eje(self) -> np.ndarray:
        """Diferencia entre referencias. Es el denominador de la proyección."""
        return self.tor - self.dia

    @property
    def informativas(self) -> np.ndarray:
        """Máscara de características con contraste suficiente en esta sesión.

        Una característica que da casi el mismo valor en las dos maniobras no
        distingue nada para este paciente hoy, y dividir por ese eje amplifica
        ruido hasta hacerlo dominante.
        """
        escala = np.maximum(np.abs(self.dia), np.abs(self.tor)) + 1e-12
        return np.abs(self.eje) / escala > CONTRASTE_MINIMO

    def normaliza(self, X: np.ndarray) -> np.ndarray:
        """Proyecta sobre el eje del paciente: su diafragmática=0, torácica=1."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        eje = self.eje.copy()
        malas = ~self.informativas
        # Las no informativas se dejan en 0 en vez de propagar un cociente
        # inestable: aportan nada, pero no envenenan el resto del vector.
        eje[malas] = 1.0
        z = (X - self.dia) / eje
        z[:, malas] = 0.0
        return z

    def calidad(self) -> dict[str, float]:
        """Diagnóstico de la calibración, para avisar antes de coachear."""
        inf = self.informativas
        return {
            "caracteristicas_informativas": int(inf.sum()),
            "fraccion_informativa": float(inf.mean()),
            "contraste_mediano": float(
                np.median(
                    np.abs(self.eje[inf])
                    / (np.maximum(np.abs(self.dia[inf]), np.abs(self.tor[inf])) + 1e-12)
                )
            ) if inf.any() else 0.0,
            "n_ventanas_dia": self.n_dia,
            "n_ventanas_tor": self.n_tor,
        }

    def to_dict(self) -> dict:
        return {
            "ref_dia": self.dia.tolist(),
            "ref_tor": self.tor.tolist(),
            "n_dia": self.n_dia,
            "n_tor": self.n_tor,
            "feature_names": list(C.FEATURE_NAMES),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenciaSesion":
        if list(d.get("feature_names", C.FEATURE_NAMES)) != list(C.FEATURE_NAMES):
            raise ValueError(
                "la referencia se calculó con otro conjunto de características; "
                "hay que recalibrar"
            )
        return cls(
            dia=np.asarray(d["ref_dia"], dtype=np.float64),
            tor=np.asarray(d["ref_tor"], dtype=np.float64),
            n_dia=int(d.get("n_dia", 0)),
            n_tor=int(d.get("n_tor", 0)),
        )
