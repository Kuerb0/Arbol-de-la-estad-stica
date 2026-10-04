"""Rama GRAFICOS / regresión lineal y GLM: recta con bandas y diagnóstico de residuos.

Origen: nuevo (en consultoría los diagnósticos se dibujaban a mano en cada notebook).
Equivale a PROC REG con ODS GRAPHICS (FITPLOT y PLOTS=DIAGNOSTICS) y PROC GENMOD PLOTS=.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib.pyplot as plt

from ._base import APAGADO, AZUL, FONDO, NARANJA, TINTA2, _ejes, _figura, _leyenda


def grafico_regresion_simple(df: pd.DataFrame, x: str, y: str, nivel: float = 0.95, ax=None):
    """Nube de puntos + recta MCO + banda de confianza de la media + banda de predicción.

    La banda estrecha (confianza) dice dónde está la recta verdadera; la ancha (predicción) dónde
    caerá UNA observación nueva. Confundirlas es el error más típico al comunicar una regresión.
    Devuelve la Figure (título: ecuación, R² y p-valor de la pendiente).
    """
    d = df[[x, y]].dropna()
    if len(d) < 3:
        raise ValueError("Hacen falta al menos 3 observaciones.")
    mod = sm.OLS(d[y].to_numpy(float), sm.add_constant(d[x].to_numpy(float))).fit()
    xs = np.linspace(d[x].min(), d[x].max(), 200)
    pr = mod.get_prediction(sm.add_constant(xs)).summary_frame(alpha=1 - nivel)
    fig, ax = _figura(ax)
    ax.scatter(d[x], d[y], s=18, color=AZUL, alpha=0.45, linewidths=0, label="observaciones")
    ax.fill_between(xs, pr["obs_ci_lower"], pr["obs_ci_upper"], color=AZUL, alpha=0.07,
                    label=f"predicción {nivel:.0%}")
    ax.fill_between(xs, pr["mean_ci_lower"], pr["mean_ci_upper"], color=AZUL, alpha=0.22,
                    label=f"confianza de la media {nivel:.0%}")
    ax.plot(xs, pr["mean"], color=AZUL, lw=2, label="recta MCO")
    b0, b1 = mod.params
    _ejes(ax, f"{y} = {b0:.3g} {'+' if b1 >= 0 else '−'} {abs(b1):.3g}·{x}   ·   R² = {mod.rsquared:.3f}   ·   "
              f"p(pendiente) = {mod.pvalues[1]:.2g}", x, y, rejilla="both")
    _leyenda(ax, loc="best")
    return fig


def _influencia(modelo):
    if not hasattr(modelo, "get_influence"):
        raise TypeError("Este objeto no da influencia (get_influence). Para logísticas usa "
                        "grafico_residuos_agrupados(y, p) o ajusta con sm.GLM(..., Binomial()).")
    inf = modelo.get_influence()
    res = getattr(inf, "resid_studentized_internal", None)
    if res is None:
        res = inf.resid_studentized
    return np.asarray(res, float), np.asarray(inf.hat_matrix_diag, float), np.asarray(inf.cooks_distance[0], float)


def grafico_diagnostico_residuos(modelo, n_etiquetas: int = 3):
    """Los 4 gráficos clásicos de diagnóstico de un modelo de statsmodels (OLS, GLM; acepta cualquiera con get_influence).

    1) Residuos vs ajustados: debería ser una nube sin forma (curva = falta un término no lineal).
    2) Q-Q normal de residuos estandarizados: puntos sobre la diagonal (solo exigible en OLS).
    3) Escala-localización: tendencia creciente = heterocedasticidad -> usar errores robustos (HC3).
    4) Residuos vs apalancamiento con distancia de Cook: puntos con Cook > 0.5 influyen mucho.
    Se etiquetan las `n_etiquetas` observaciones con mayor Cook (por su índice).
    Gauss: en GLM los residuos no tienen por qué ser normales; mira sobre todo 1, 3 y 4. Con respuesta 0/1
    los paneles 1-3 salen en dos bandas y dicen poco: usa grafico_residuos_agrupados.
    """
    res, lev, cook = _influencia(modelo)
    aj = np.asarray(modelo.fittedvalues, float)
    etiquetas = getattr(modelo.model.data, "row_labels", None)
    idx = np.asarray(etiquetas) if etiquetas is not None else np.arange(len(res))
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 7.2), layout="constrained")
    fig.patch.set_facecolor(FONDO)
    top = np.argsort(cook)[::-1][:n_etiquetas]
    resid_crudos = np.asarray(modelo.resid_response if hasattr(modelo, "resid_response") else modelo.resid, float)

    def _suave(ax, xx, yy):
        if len(xx) > 5:
            s = lowess(yy, xx, frac=0.6, return_sorted=True)
            ax.plot(s[:, 0], s[:, 1], color=NARANJA, lw=2)

    ax = axs[0, 0]
    ax.scatter(aj, resid_crudos, s=12, color=AZUL, alpha=0.45, linewidths=0)
    ax.axhline(0, color=APAGADO, lw=1)
    _suave(ax, aj, resid_crudos)
    _ejes(ax, "Residuos vs ajustados", "valor ajustado", "residuo")

    ax = axs[0, 1]
    (teo, obs), _ = stats.probplot(res, dist="norm")
    ax.scatter(teo, obs, s=12, color=AZUL, alpha=0.6, linewidths=0)
    lim = [min(teo.min(), obs.min()), max(teo.max(), obs.max())]
    ax.plot(lim, lim, color=APAGADO, lw=1)
    _ejes(ax, "Q-Q normal", "cuantil teórico", "residuo estandarizado", rejilla="both")

    ax = axs[1, 0]
    raiz = np.sqrt(np.abs(res))
    ax.scatter(aj, raiz, s=12, color=AZUL, alpha=0.45, linewidths=0)
    _suave(ax, aj, raiz)
    _ejes(ax, "Escala-localización", "valor ajustado", "√|residuo estandarizado|")

    ax = axs[1, 1]
    ax.scatter(lev, res, s=12 + 120 * cook / max(cook.max(), 1e-12), color=AZUL, alpha=0.45, linewidths=0)
    p = len(np.asarray(modelo.params))
    hs = np.linspace(max(lev.min(), 1e-3), max(lev.max(), 2 * p / len(lev)) * 1.05, 100)
    y_lim = (min(res.min(), -3) * 1.1, max(res.max(), 3) * 1.1)
    for c in (0.5, 1.0):
        lim_r = np.sqrt(c * p * (1 - hs) / hs)
        for signo in (1, -1):
            ax.plot(hs, signo * lim_r, color=APAGADO, lw=0.8)
        if lim_r[-1] < y_lim[1]:
            ax.text(hs[-1], lim_r[-1], f" Cook {c}", color=TINTA2, fontsize=8, va="center", clip_on=True)
    ax.set_ylim(*y_lim)
    if cook.max() < 0.5:
        ax.text(0.98, 0.04, f"Cook máx. {cook.max():.2f} (< 0.5: sin puntos influyentes)", transform=ax.transAxes,
                ha="right", fontsize=8, color=TINTA2)
    for i in top:
        ax.annotate(str(idx[i]), (lev[i], res[i]), fontsize=8, color=TINTA2, xytext=(3, 3), textcoords="offset points")
    _ejes(ax, "Residuos vs apalancamiento", "apalancamiento (h)", "residuo estandarizado", rejilla="both")
    return fig
