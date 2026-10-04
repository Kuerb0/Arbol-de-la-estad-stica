"""Rama GRAFICOS / descriptiva: forma de una variable, ajuste de distribuciones y matriz de correlaciones.

Equivale a PROC UNIVARIATE HISTOGRAM / PPPLOT y PROC CORR PLOTS=MATRIX (heatmap).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .._util import _numerico
from ..descriptiva.distribuciones import ajustar_distribuciones, estimar_densidad
from ._base import APAGADO, AZUL, FONDO, NARANJA, TINTA, TINTA2, _color, _ejes, _leyenda


def grafico_distribucion(x, titulo: str = ""):
    """Histograma + densidad (KDE) arriba y diagrama de caja debajo, con media y mediana marcadas.
    Para ver de un vistazo asimetría, colas, multimodalidad y atípicos."""
    v = _numerico(x, "x", 3)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.6), layout="constrained", sharex=True,
                                 gridspec_kw={"height_ratios": [4, 1]})
    fig.patch.set_facecolor(FONDO)
    a1.hist(v, bins="auto", density=True, color=AZUL, alpha=0.35, edgecolor="white")
    d = estimar_densidad(v)
    a1.plot(d.x, d.densidad, color=AZUL, lw=2, label="densidad (KDE)")
    a1.axvline(v.mean(), color=NARANJA, lw=1.5, label=f"media {v.mean():.3g}")
    a1.axvline(np.median(v), color=TINTA, lw=1.5, ls="--", label=f"mediana {np.median(v):.3g}")
    _ejes(a1, titulo or f"Distribución (n = {len(v)}, asimetría {stats.skew(v):.2f})", y="densidad")
    _leyenda(a1)
    a2.boxplot(v, vert=False, widths=0.6, patch_artist=True,
               boxprops={"facecolor": "#dbe7f8", "edgecolor": AZUL}, medianprops={"color": TINTA},
               flierprops={"marker": "o", "markersize": 3, "markerfacecolor": NARANJA, "markeredgecolor": "none"})
    _ejes(a2, rejilla="x"); a2.set_yticks([])
    return fig


def grafico_ajuste_distribuciones(x, tabla=None, mejores: int = 3):
    """Histograma con las `mejores` distribuciones ajustadas (por AIC) superpuestas y su gráfico P-P.

    `tabla` = salida de descriptiva.ajustar_distribuciones (si no se da, se calcula). En el P-P una buena
    candidata sigue la diagonal; las desviaciones en los extremos indican colas mal modeladas."""
    v = np.sort(_numerico(x, "x", 10))
    t = ajustar_distribuciones(v) if tabla is None else tabla
    t = t.head(mejores)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    fig.patch.set_facecolor(FONDO)
    a1.hist(v, bins="auto", density=True, color=APAGADO, alpha=0.3, edgecolor="white")
    xs = np.linspace(v.min(), v.max(), 300)
    emp = (np.arange(1, len(v) + 1) - 0.5) / len(v)
    for i, fila in enumerate(t.itertuples()):
        dist = getattr(stats, fila.scipy if fila.scipy != "loglogistic" else "fisk")
        a1.plot(xs, dist.pdf(xs, *fila.parametros), color=_color(i), lw=2, label=f"{fila.distribucion} (ΔAIC {fila.delta_AIC:.1f})")
        a2.plot(emp, dist.cdf(v, *fila.parametros), color=_color(i), lw=1.6)
    a2.plot([0, 1], [0, 1], color=TINTA2, lw=1, ls=":")
    _ejes(a1, "Ajuste por máxima verosimilitud", "x", "densidad"); _leyenda(a1)
    _ejes(a2, "Gráfico P-P", "probabilidad empírica", "probabilidad teórica", rejilla="both")
    return fig


def grafico_matriz_correlaciones(r, p_valores=None, alpha: float = 0.05, ax=None):
    """Mapa de calor de una matriz de correlaciones (escala divergente −1…1) con el valor en cada celda;
    las celdas no significativas (si se pasan `p_valores`) salen en gris."""
    import pandas as pd
    r = pd.DataFrame(r)
    if ax is None:
        lado = 1.0 + 0.55 * len(r)
        fig, ax = plt.subplots(figsize=(lado + 1.2, lado), layout="constrained")
        fig.patch.set_facecolor(FONDO)
    else:
        fig = ax.figure
    im = ax.imshow(r.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(r)), r.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(r)), r.index, fontsize=9)
    for i in range(len(r)):
        for j in range(len(r)):
            val = r.iloc[i, j]
            gris = p_valores is not None and i != j and pd.DataFrame(p_valores).iloc[i, j] >= alpha
            ax.text(j, i, "" if np.isnan(val) else f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color=APAGADO if gris else (TINTA if abs(val) < 0.6 else "white"))
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Matriz de correlaciones", loc="left", fontsize=11, color=TINTA, fontweight="bold")
    return fig
