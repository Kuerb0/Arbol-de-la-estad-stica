"""Rama SELECCION / forward stepwise por criterio de información (AIC o BIC).

Origen: `forward_stepwise_glm` (notebook de consultoría). Equivale a MODEL ... / SELECTION=FORWARD
de PROC LOGISTIC, pero usando AIC/BIC como criterio de entrada en lugar del p-valor de SLENTRY.

CORRECCIÓN respecto al original: `smf.glm(formula, data)` sin `family` ajusta un modelo GAUSSIANO.
Aquí la familia es Binomial si el objetivo es binario 0/1 y Gaussiana en otro caso, o la que indiques.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _familia_por_defecto(serie: pd.Series):
    valores = set(serie.dropna().unique())
    return sm.families.Binomial() if valores <= {0, 1} and len(valores) == 2 else sm.families.Gaussian()


def seleccion_forward(
    df: pd.DataFrame,
    objetivo: str,
    candidatas: Iterable[str],
    categoricas: Iterable[str] = (),
    familia=None,
    criterio: str = "aic",
    base: Iterable[str] = (),
    max_pasos: int | None = None,
    verbose: bool = False,
) -> dict:
    """Añade en cada paso la variable que más reduce `criterio` y para cuando ninguna mejora.

    Parámetros
    ----------
    candidatas : variables entre las que elegir (las categóricas se envuelven en C()).
    base       : variables que entran siempre (se parte de ellas).
    criterio   : 'aic' o 'bic'.
    Devuelve dict: formula, seleccionadas, criterio_final, modelo y historial (paso, variable, criterio).

    Coste: en cada paso ajusta un GLM por candidata restante (O(p^2) ajustes); con muchas
    candidatas o muchas filas, pre-filtrar con `cribar_variables` y/o muestrear.
    """
    if criterio not in ("aic", "bic"):
        raise ValueError("criterio debe ser 'aic' o 'bic'")
    categoricas = set(categoricas)
    familia = familia or _familia_por_defecto(df[objetivo])
    medir = (lambda m: m.aic) if criterio == "aic" else (lambda m: m.bic_llf)

    def _term(v):
        return f"C({v})" if v in categoricas else v

    def _formula(vars_):
        return f"{objetivo} ~ " + (" + ".join(_term(v) for v in vars_) if vars_ else "1")

    seleccionadas = list(base)
    restantes = [c for c in candidatas if c not in seleccionadas]
    usadas = [objetivo] + seleccionadas + restantes
    datos = df[list(dict.fromkeys(usadas))].dropna()  # mismo conjunto de filas en todos los pasos

    mejor = medir(smf.glm(_formula(seleccionadas), data=datos, family=familia).fit())
    historial = [{"paso": 0, "variable": None, "criterio": float(mejor)}]

    while restantes and (max_pasos is None or len(seleccionadas) - len(list(base)) < max_pasos):
        intentos = []
        for v in restantes:
            try:
                m = smf.glm(_formula(seleccionadas + [v]), data=datos, family=familia).fit()
                intentos.append((float(medir(m)), v))
            except Exception:  # noqa: BLE001  (separación perfecta, matriz singular...)
                continue
        if not intentos:
            break
        nuevo, v_mejor = min(intentos)
        if nuevo < mejor:
            mejor = nuevo
            seleccionadas.append(v_mejor)
            restantes.remove(v_mejor)
            historial.append({"paso": len(historial), "variable": v_mejor, "criterio": nuevo})
            if verbose:
                print(f"paso {len(historial) - 1}: +{v_mejor}  {criterio.upper()}={nuevo:.2f}")
        else:
            break

    modelo = smf.glm(_formula(seleccionadas), data=datos, family=familia).fit()
    return {"formula": _formula(seleccionadas), "seleccionadas": seleccionadas,
            "criterio_final": float(mejor), "modelo": modelo,
            "historial": pd.DataFrame(historial)}
