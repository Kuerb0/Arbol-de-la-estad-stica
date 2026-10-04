"""Comprobar supuestos: normalidad y homogeneidad de varianzas (PROC UNIVARIATE NORMAL, GLM HOVTEST)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .._util import _columnas, _numerico


def contraste_normalidad(x, alpha: float = 0.05) -> dict:
    """Varios contrastes de normalidad a la vez: Shapiro-Wilk (n ≤ 5000), D'Agostino-Pearson K²,
    Jarque-Bera, Anderson-Darling y Kolmogorov-Smirnov con corrección de Lilliefors.

    Devuelve {'tabla', 'asimetria', 'curtosis_exceso', 'n', 'avisos'}. Equivale a PROC UNIVARIATE NORMAL.
    Gauss: con n grande cualquier desviación mínima sale significativa (y con n pequeño no se detecta nada):
    acompáñalo SIEMPRE del gráfico Q-Q (graficos.grafico_qq) y decide por la forma, no solo por el p-valor.
    """
    v = _numerico(x, "x", 8)
    n = len(v)
    filas = []
    if n <= 5000:
        r = stats.shapiro(v); filas.append(("Shapiro-Wilk", r.statistic, r.pvalue))
    if n >= 20:
        r = stats.normaltest(v); filas.append(("D'Agostino K²", r.statistic, r.pvalue))
    r = stats.jarque_bera(v); filas.append(("Jarque-Bera", r.statistic, r.pvalue))
    try:                                   # scipy >= 1.17: p-valor interpolado en las tablas
        ad = stats.anderson(v, "norm", method="interpolate")
        filas.append(("Anderson-Darling", ad.statistic, float(ad.pvalue)))
    except TypeError:                      # scipy antiguo: sin p-valor; se compara con el valor crítico del 5 %
        ad = stats.anderson(v, "norm")
        crit = dict(zip(ad.significance_level, ad.critical_values))
        filas.append(("Anderson-Darling", ad.statistic, 0.01 if ad.statistic > crit.get(5.0, np.inf) else 0.5))
    from statsmodels.stats.diagnostic import lilliefors
    est, p = lilliefors(v, "norm"); filas.append(("Kolmogorov-Smirnov (Lilliefors)", est, p))
    t = pd.DataFrame(filas, columns=["contraste", "estadistico", "p_valor"]).set_index("contraste")
    t["rechaza_normalidad"] = t["p_valor"] < alpha
    avisos = []
    if n > 2000:
        avisos.append(f"n = {n}: los contrastes detectan desviaciones irrelevantes; mira el Q-Q y la asimetría/curtosis.")
    if n < 30:
        avisos.append(f"n = {n}: poca potencia; no rechazar NO prueba que sea normal.")
    return {"tabla": t, "asimetria": float(stats.skew(v, bias=False)), "curtosis_exceso": float(stats.kurtosis(v, bias=False)),
            "n": n, "avisos": avisos}


def homogeneidad_varianzas(df: pd.DataFrame, valor: str, grupo: str, alpha: float = 0.05) -> pd.DataFrame:
    """Levene (centrado en la media), Brown-Forsythe (en la mediana, robusto), Bartlett (exige normalidad)
    y Fligner-Killeen (no paramétrico). Equivale a PROC GLM / MEANS HOVTEST=LEVENE|BF|BARTLETT.

    Gauss: Bartlett es muy sensible a la no normalidad; con datos asimétricos usa Brown-Forsythe o Fligner.
    Si las varianzas difieren, compara medias con Welch (elegir_contraste lo hace) en lugar de t clásico/ANOVA.
    """
    _columnas(df, [valor, grupo])
    d = df[[valor, grupo]].dropna()
    muestras = [g[valor].to_numpy(float) for _, g in d.groupby(grupo)]
    if len(muestras) < 2 or min(len(m) for m in muestras) < 2:
        raise ValueError("Hacen falta al menos 2 grupos con 2 o más observaciones.")
    filas = [("Levene (media)", *stats.levene(*muestras, center="mean")),
             ("Brown-Forsythe (mediana)", *stats.levene(*muestras, center="median")),
             ("Bartlett", *stats.bartlett(*muestras)),
             ("Fligner-Killeen", *stats.fligner(*muestras))]
    t = pd.DataFrame(filas, columns=["contraste", "estadistico", "p_valor"]).set_index("contraste")
    t["rechaza_igualdad"] = t["p_valor"] < alpha
    t.attrs["desviaciones"] = d.groupby(grupo)[valor].std().to_dict()
    return t
