"""Rama SELECCION / cribado univariante de variables (chi-cuadrado contra el objetivo).

Orígenes (notebook de consultoría): `evaluar_significancia_categoria`, `valorar_variables_df_orig`,
`es_posible_fuga_o_modelo`. Es un filtro previo al modelo: qué variables merece la pena probar.
Equivale a PROC FREQ ... / CHISQ por cada variable (tras PROC RANK para las numéricas).
Gauss: con muestras grandes casi todo sale "significativo"; usar V de Cramér / diff_max_prob
para decidir si la diferencia es relevante, no solo p < alpha.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

PREFIJOS_FUGA = ("y_", "prob_", "renewed_pred")
PATRONES_FUGA = ("pred", "prob")


def es_posible_fuga(col: str, excluir: Iterable[str] = (), prefijos: Iterable[str] = PREFIJOS_FUGA,
                    patrones: Iterable[str] = PATRONES_FUGA) -> bool:
    """True si la columna parece derivada del objetivo o de un modelo (fuga de información)."""
    c = col.lower()
    return (col in set(excluir)
            or any(c.startswith(p) for p in prefijos)
            or any(p in c for p in patrones))


def _discretizar(serie: pd.Series) -> pd.Series | None:
    """Numérica -> categórica por cuantiles (deciles si >=10 valores distintos, 4 si >=4, si no ya es discreta)."""
    nun = serie.dropna().nunique()
    q = 10 if nun >= 10 else 4 if nun >= 4 else nun
    if q < 2:
        return None
    try:
        cat = pd.qcut(serie, q=q, duplicates="drop")
    except ValueError:
        return None
    return cat if cat.dropna().nunique() >= 2 else None


def contraste_chi2_variable(df: pd.DataFrame, variable: str, objetivo: str, clase_positiva=1) -> dict | None:
    """Chi-cuadrado de independencia entre una variable categórica y un objetivo binario.

    Devuelve None si no es evaluable (< 2 niveles en alguno de los dos). Incluye V de Cramér y
    `diff_max_prob`: diferencia entre la mayor y la menor tasa de `clase_positiva` por nivel.
    """
    d = df[[variable, objetivo]].dropna()
    if d[variable].nunique() < 2 or d[objetivo].nunique() < 2:
        return None
    tabla = pd.crosstab(d[variable], d[objetivo])
    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return None
    chi2, p, _, _ = chi2_contingency(tabla)
    n = int(tabla.to_numpy().sum())
    v = float(np.sqrt(chi2 / (n * (min(tabla.shape) - 1))))
    tasa = (tabla[clase_positiva] / tabla.sum(axis=1)) if clase_positiva in tabla.columns \
        else pd.Series(0.0, index=tabla.index)
    return {"variable": variable, "chi2": float(chi2), "p_valor": float(p), "cramer_v": v,
            "n_categorias": int(tabla.shape[0]), "n_muestra": n,
            "diff_max_prob": float(tasa.max() - tasa.min())}


def cribar_variables(df: pd.DataFrame, objetivo: str, alpha: float = 0.05, max_niveles: int = 30,
                     excluir: Iterable[str] = (), redundantes: Iterable[str] = (),
                     clase_positiva=1) -> pd.DataFrame:
    """Evalúa TODAS las columnas contra el objetivo y marca cuáles se pueden incluir.

    Motivos posibles: target, posible_fuga_o_variable_modelo, sin_variabilidad, no_categorizable,
    alta_cardinalidad, no_evaluable, redundante, no_significativa, incluible_significativa.
    No modifica `df` (las numéricas se discretizan internamente). Ordenado: incluibles primero,
    por p-valor ascendente.
    """
    excluir, redundantes = set(excluir) | {objetivo}, set(redundantes)
    filas = []
    for col in df.columns:
        fila = {"variable": col, "dtype": str(df[col].dtype), "n_unicos": int(df[col].dropna().nunique()),
                "incluir": False, "motivo": "", "chi2": np.nan, "p_valor": np.nan, "cramer_v": np.nan,
                "n_categorias": np.nan, "diff_max_prob": np.nan}
        if col == objetivo:
            fila["motivo"] = "target"
        elif es_posible_fuga(col, excluir):
            fila["motivo"] = "posible_fuga_o_variable_modelo"
        elif fila["n_unicos"] < 2:
            fila["motivo"] = "sin_variabilidad"
        elif col in redundantes:
            fila["motivo"] = "redundante"
        else:
            serie = df[col]
            if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):
                serie = _discretizar(serie) if fila["n_unicos"] >= 4 else serie
            if serie is None:
                fila["motivo"] = "no_categorizable"
            elif serie.dropna().nunique() > max_niveles:
                fila["motivo"] = "alta_cardinalidad"
            else:
                tmp = pd.DataFrame({col: serie.astype(str).where(serie.notna()), objetivo: df[objetivo]})
                r = contraste_chi2_variable(tmp, col, objetivo, clase_positiva)
                if r is None:
                    fila["motivo"] = "no_evaluable"
                else:
                    fila.update({k: r[k] for k in ("chi2", "p_valor", "cramer_v", "n_categorias", "diff_max_prob")})
                    fila["incluir"] = r["p_valor"] < alpha
                    fila["motivo"] = "incluible_significativa" if fila["incluir"] else "no_significativa"
        filas.append(fila)
    out = pd.DataFrame(filas)
    return out.sort_values(["incluir", "p_valor", "variable"], ascending=[False, True, True],
                           na_position="last").reset_index(drop=True)
