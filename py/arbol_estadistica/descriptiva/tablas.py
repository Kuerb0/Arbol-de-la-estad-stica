"""Tablas de contingencia: chi², residuos, V de Cramér y medidas de riesgo 2×2 (PROC FREQ)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .._util import _columnas


def tabla_contingencia(df: pd.DataFrame, fila: str, columna: str) -> dict:
    """Tabla cruzada con todo lo de PROC FREQ / CHISQ: conteos, % por fila, esperados, chi² de Pearson y de
    razón de verosimilitudes (G²), Fisher exacto si es 2×2, V de Cramér y residuos ajustados (|r| > 2 = celda que
    se aparta de la independencia). Avisa si hay esperados < 5.
    """
    _columnas(df, [fila, columna])
    t = pd.crosstab(df[fila], df[columna])
    if t.shape[0] < 2 or t.shape[1] < 2:
        raise ValueError("Cada variable necesita al menos 2 niveles con datos.")
    chi2, p, gl, esp = stats.chi2_contingency(t, correction=False)
    g2, p_g2, _, _ = stats.chi2_contingency(t, correction=False, lambda_="log-likelihood")
    n = t.to_numpy().sum()
    filas_p, cols_p = t.sum(axis=1).to_numpy() / n, t.sum(axis=0).to_numpy() / n
    ajust = (t.to_numpy() - esp) / np.sqrt(esp * np.outer(1 - filas_p, 1 - cols_p))
    v = float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))
    out = {"conteos": t, "porcentaje_fila": t.div(t.sum(axis=1), axis=0) * 100,
           "esperados": pd.DataFrame(esp, index=t.index, columns=t.columns),
           "residuos_ajustados": pd.DataFrame(ajust, index=t.index, columns=t.columns),
           "chi2": float(chi2), "gl": int(gl), "p_valor": float(p), "g2": float(g2), "p_valor_g2": float(p_g2),
           "cramer_v": v, "avisos": []}
    if (esp < 5).mean() > 0.2:
        out["avisos"].append("Más del 20 % de las celdas tienen esperado < 5: la aproximación χ² no es fiable (Fisher o agrupar niveles).")
    if t.shape == (2, 2):
        out["p_fisher"] = float(stats.fisher_exact(t.to_numpy())[1])
    return out


def medidas_riesgo_2x2(expuestos_evento: int, expuestos_total: int, no_expuestos_evento: int, no_expuestos_total: int,
                       nivel: float = 0.95) -> pd.DataFrame:
    """Riesgo relativo (RR), diferencia de riesgos (DR), odds ratio (OR) y NNT con IC (Katz para RR, Woolf para OR,
    Wald para DR). Equivale a PROC FREQ / RELRISK RISKDIFF.

    Gauss: el OR se parece al RR solo si el evento es raro (< 10 %); con eventos frecuentes el OR exagera el efecto.
    """
    a, n1, c, n0 = expuestos_evento, expuestos_total, no_expuestos_evento, no_expuestos_total
    if not (0 <= a <= n1 and 0 <= c <= n0 and n1 > 0 and n0 > 0):
        raise ValueError("Conteos incoherentes.")
    b, d = n1 - a, n0 - c
    z = stats.norm.ppf(0.5 + nivel / 2)
    aa, bb, cc, dd = (v + 0.5 for v in (a, b, c, d)) if 0 in (a, b, c, d) else (a, b, c, d)   # Haldane si hay ceros
    p1, p0 = a / n1, c / n0
    rr = (aa / (aa + bb)) / (cc / (cc + dd)); se_rr = np.sqrt(1 / aa - 1 / (aa + bb) + 1 / cc - 1 / (cc + dd))
    orr = aa * dd / (bb * cc); se_or = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    dr = p1 - p0; se_dr = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    filas = [("riesgo_expuestos", p1, np.nan, np.nan), ("riesgo_no_expuestos", p0, np.nan, np.nan),
             ("riesgo_relativo", rr, rr * np.exp(-z * se_rr), rr * np.exp(z * se_rr)),
             ("diferencia_riesgos", dr, dr - z * se_dr, dr + z * se_dr),
             ("odds_ratio", orr, orr * np.exp(-z * se_or), orr * np.exp(z * se_or)),
             ("nnt", 1 / dr if dr else np.inf, np.nan, np.nan)]
    return pd.DataFrame(filas, columns=["medida", "valor", "ic_inf", "ic_sup"]).set_index("medida")
