"""Rama DIAGNOSTICO / multicolinealidad (VIF).

Orígenes: bloque VIF de `notebook de consultoría`, `multinomial_logit_sas_like`
(VIF sobre matriz de diseño) y `filtrar_vif` de notebook de consultoría.
Equivale a la opción VIF de PROC REG en SAS.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _clasificar(v: float) -> str:
    if v < 5:
        return "OK (< 5)"
    if v < 10:
        return "Moderado (5-10)"
    return "ALTO (>= 10)"


def calcular_vif(X: pd.DataFrame, ignorar: Iterable[str] = ("const", "Intercept")) -> pd.DataFrame:
    """VIF de cada columna de la matriz de diseño `X` (puede incluir la constante).

    La constante se incluye en el cálculo (es lo correcto para que el VIF sea el estándar)
    pero se omite del resultado. VIF no finito se devuelve como inf.
    """
    cols = list(X.columns)
    valores = X.to_numpy(dtype=float)
    filas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, c in enumerate(cols):
            if c in set(ignorar):
                continue
            try:
                v = float(variance_inflation_factor(valores, i))
            except Exception:
                v = np.inf
            filas.append({"variable": c, "VIF": v if np.isfinite(v) else np.inf})
    out = pd.DataFrame(filas)
    if out.empty:
        return out.assign(diagnostico=pd.Series(dtype=str))
    out["diagnostico"] = out["VIF"].apply(_clasificar)
    return out.sort_values("VIF", ascending=False).reset_index(drop=True)


def filtrar_vif_iterativo(
    X: pd.DataFrame,
    umbral: float = 8.0,
    protegidas: Iterable[str] = (),
    verbose: bool = False,
) -> list[str]:
    """Elimina una a una la columna con mayor VIF (> umbral) hasta que ninguna lo supere.

    `protegidas` nunca se eliminan (p. ej. las variables de negocio obligatorias).
    'const' siempre queda protegida. Devuelve la lista de columnas que se conservan.
    Coste: recalcula todos los VIF en cada vuelta (O(p^2) regresiones); para p grande,
    muestrear filas antes.
    """
    protegidas = set(protegidas) | {"const"}
    cols = list(X.columns)
    while True:
        vif = calcular_vif(X[cols], ignorar=())
        candidatas = vif[~vif["variable"].isin(protegidas)]
        if candidatas.empty:
            break
        peor = candidatas.iloc[0]
        if peor["VIF"] > umbral:
            if verbose:
                print(f"VIF {peor['VIF']:.1f} -> eliminando '{peor['variable']}'")
            cols.remove(peor["variable"])
        else:
            break
    return cols
