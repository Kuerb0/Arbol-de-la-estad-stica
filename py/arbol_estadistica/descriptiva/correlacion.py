"""Correlaciones: Pearson, Spearman y Kendall con IC; matriz; parcial y semiparcial (PROC CORR)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .._util import _columnas


def correlacion_con_ic(x, y, metodo: str = "pearson", nivel: float = 0.95) -> dict:
    """Coeficiente de correlación, p-valor e IC.

    IC con la transformación z de Fisher: Pearson se(z) = 1/√(n−3); Spearman con la corrección de
    Bonett-Wright se(z) = √((1 + r²/2)/(n−3)); Kendall se(z) = √(0.437/(n−4)) (Fieller). Quita pares con NaN.
    Equivale a PROC CORR PEARSON / SPEARMAN / KENDALL con FISHER.
    Gauss: Pearson mide relación LINEAL y es sensible a atípicos; Spearman y Kendall miden relación monótona.
    """
    d = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)}).dropna()
    n = len(d)
    if n < 5:
        raise ValueError(f"Hacen falta al menos 5 pares completos (hay {n}).")
    if metodo == "pearson":
        r, p = stats.pearsonr(d.x, d.y); se = 1 / np.sqrt(n - 3)
    elif metodo == "spearman":
        r, p = stats.spearmanr(d.x, d.y); se = np.sqrt((1 + r ** 2 / 2) / (n - 3))
    elif metodo == "kendall":
        r, p = stats.kendalltau(d.x, d.y); se = np.sqrt(0.437 / (n - 4))
    else:
        raise ValueError("metodo: 'pearson', 'spearman' o 'kendall'")
    z = np.arctanh(np.clip(r, -0.999999, 0.999999)); c = stats.norm.ppf(0.5 + nivel / 2)
    return {"metodo": metodo, "r": float(r), "p_valor": float(p), "n": n,
            "ic_inf": float(np.tanh(z - c * se)), "ic_sup": float(np.tanh(z + c * se))}


def matriz_correlaciones(df: pd.DataFrame, columnas=None, metodo: str = "pearson") -> dict:
    """Matriz de correlaciones y de p-valores (pares completos en cada celda). Devuelve {'r', 'p_valor', 'n'}."""
    cols = list(columnas) if columnas is not None else list(df.select_dtypes("number").columns)
    _columnas(df, cols)
    k = len(cols)
    r, p, n = (pd.DataFrame(np.eye(k) if i == 0 else np.zeros((k, k)), index=cols, columns=cols) for i in range(3))
    for i in range(k):
        for j in range(i + 1, k):
            try:
                res = correlacion_con_ic(df[cols[i]], df[cols[j]], metodo)
                r.iloc[i, j] = r.iloc[j, i] = res["r"]; p.iloc[i, j] = p.iloc[j, i] = res["p_valor"]
                n.iloc[i, j] = n.iloc[j, i] = res["n"]
            except ValueError:
                r.iloc[i, j] = r.iloc[j, i] = p.iloc[i, j] = p.iloc[j, i] = np.nan
    for i, c in enumerate(cols):
        n.iloc[i, i] = df[c].notna().sum()
    return {"r": r, "p_valor": p, "n": n.astype(int)}


def correlacion_parcial(df: pd.DataFrame, x: str, y: str, controles, metodo: str = "pearson") -> dict:
    """Correlación PARCIAL (x e y libres de los controles) y SEMIPARCIAL (solo x libre de los controles).

    Se calcula con residuos de regresiones lineales (Spearman: sobre rangos). Equivale a PROC CORR PARTIAL.
    p-valor de la parcial con t = r·√((n−2−k)/(1−r²)), gl = n − 2 − k.
    """
    controles = list(controles)
    _columnas(df, [x, y] + controles)
    d = df[[x, y] + controles].apply(pd.to_numeric, errors="coerce").dropna()
    if metodo == "spearman":
        d = d.rank()
    n, k = len(d), len(controles)
    if n - 2 - k < 3:
        raise ValueError("Muy pocas filas para tantos controles.")
    Z = np.column_stack([np.ones(n), d[controles].to_numpy()])

    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    ex, ey = resid(d[x].to_numpy()), resid(d[y].to_numpy())
    r = float(np.corrcoef(ex, ey)[0, 1])
    sp = float(np.corrcoef(ex, d[y].to_numpy())[0, 1])
    gl = n - 2 - k
    t = r * np.sqrt(gl / max(1 - r ** 2, 1e-15))
    return {"parcial": r, "semiparcial": sp, "p_valor": float(2 * stats.t.sf(abs(t), gl)), "gl": gl, "n": n,
            "bruta": float(np.corrcoef(d[x], d[y])[0, 1])}
