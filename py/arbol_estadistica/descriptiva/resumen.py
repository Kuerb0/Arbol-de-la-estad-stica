"""Resumen numérico: posición, variabilidad, forma y atípicos (PROC MEANS / UNIVARIATE)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .._util import _columnas, _numerico


def resumen_descriptivo(df: pd.DataFrame, columnas=None, recorte: float = 0.10) -> pd.DataFrame:
    """Una fila por variable numérica: n, faltantes, media, media recortada, mediana, desviación, MAD, IQR,
    percentiles, asimetría, curtosis (exceso) y coeficiente de variación.

    Equivale a PROC MEANS / PROC UNIVARIATE. `recorte` = fracción que se quita por CADA cola en la media recortada.
    Gauss: con colas pesadas o atípicos, describe con mediana, MAD e IQR (robustos); la media y la desviación
    pueden moverse mucho por un solo valor.
    """
    if not 0 <= recorte < 0.5:
        raise ValueError("recorte debe estar en [0, 0.5)")
    cols = list(columnas) if columnas is not None else list(df.select_dtypes("number").columns)
    _columnas(df, cols)
    filas = []
    for c in cols:
        bruto = pd.to_numeric(df[c], errors="coerce")
        x = bruto.dropna().to_numpy(float)
        if len(x) == 0:
            filas.append({"variable": c, "n": 0, "faltantes": int(bruto.isna().sum())})
            continue
        q = np.percentile(x, [5, 25, 50, 75, 95])
        media, sd = float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else np.nan
        filas.append({
            "variable": c, "n": len(x), "faltantes": int(bruto.isna().sum()),
            "media": media, "media_recortada": float(stats.trim_mean(x, recorte)), "mediana": float(q[2]),
            "desviacion": sd, "mad": float(stats.median_abs_deviation(x, scale="normal")),
            "iqr": float(q[3] - q[1]), "min": float(x.min()), "p5": float(q[0]), "p25": float(q[1]),
            "p75": float(q[3]), "p95": float(q[4]), "max": float(x.max()),
            "asimetria": float(stats.skew(x, bias=False)) if len(x) > 2 else np.nan,
            "curtosis_exceso": float(stats.kurtosis(x, bias=False)) if len(x) > 3 else np.nan,
            "cv": sd / abs(media) if media else np.nan,
        })
    return pd.DataFrame(filas).set_index("variable")


def detectar_atipicos(x, metodo: str = "iqr", k: float | None = None) -> pd.DataFrame:
    """Marca valores atípicos de una variable numérica.

    metodo:
      'iqr' -> fuera de [Q1 − k·IQR, Q3 + k·IQR] (Tukey; k = 1.5 por defecto, 3 = «extremos»)
      'mad' -> |x − mediana| / MAD_normalizada > k (z robusto; k = 3.5 por defecto, Iglewicz-Hoaglin)
      'z'   -> |x − media| / desviación > k (k = 3; NO robusto: los atípicos inflan la desviación y se esconden)
    Devuelve DataFrame con valor, puntuación, atipico y los límites en `attrs['limites']`. Los NaN no se marcan.
    """
    s = pd.Series(np.asarray(x, dtype=float))
    v = _numerico(s, "x", 3)
    if metodo == "iqr":
        k = 1.5 if k is None else k
        q1, q3 = np.percentile(v, [25, 75])
        lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
        punt = (s - np.median(v)) / ((q3 - q1) or 1.0)
    elif metodo == "mad":
        k = 3.5 if k is None else k
        mad = stats.median_abs_deviation(v, scale="normal") or 1e-12
        med = np.median(v)
        lo, hi = med - k * mad, med + k * mad
        punt = (s - med) / mad
    elif metodo == "z":
        k = 3.0 if k is None else k
        m, sd = v.mean(), v.std(ddof=1) or 1e-12
        lo, hi = m - k * sd, m + k * sd
        punt = (s - m) / sd
    else:
        raise ValueError("metodo debe ser 'iqr', 'mad' o 'z'")
    out = pd.DataFrame({"valor": s, "puntuacion": punt, "atipico": (s < lo) | (s > hi)})
    out.attrs["limites"] = (float(lo), float(hi))
    return out
