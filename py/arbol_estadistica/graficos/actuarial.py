"""Rama GRAFICOS / actuarial: tabla de mortalidad (qx en escala log y supervivientes) y distribución de la reserva."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._base import AZUL, FONDO, NARANJA, TINTA2, _color, _ejes, _leyenda


def grafico_tabla_mortalidad(*tablas, nombres=None):
    """qx en escala logarítmica (la forma de Gompertz es casi una recta a partir de los 30) y supervivientes lx
    de una o varias tablas de actuarial.tabla_mortalidad (p. ej. hombres frente a mujeres, o antes/después de un choque)."""
    nombres = nombres or [f"tabla {i + 1}" for i in range(len(tablas))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    fig.patch.set_facecolor(FONDO)
    for i, (t, n) in enumerate(zip(tablas, nombres)):
        q = t.qx.iloc[:-1]
        a1.semilogy(q.index, q.clip(lower=1e-6), color=_color(i), lw=2, label=n)
        a2.plot(t.index, t.lx / t.lx.iloc[0], color=_color(i), lw=2, label=n)
    _ejes(a1, "Probabilidad de fallecimiento qx (log)", "edad", "qx", rejilla="both"); _leyenda(a1)
    _ejes(a2, "Supervivientes lx / l0", "edad", "proporción viva"); _leyenda(a2)
    return fig


def grafico_reserva_bootstrap(resultado: dict, ax=None):
    """Histograma de la reserva total simulada (actuarial.bootstrap_chain_ladder) con el chain ladder, la media y el
    percentil 99.5 % (VaR de Solvencia II)."""
    from ._base import _figura
    s = resultado["simulaciones"]
    fig, ax = _figura(ax)
    ax.hist(s, bins=60, color=AZUL, alpha=0.35, edgecolor="white")
    ax.axvline(resultado["reserva_chain_ladder"], color=TINTA2, lw=1.5, label=f"chain ladder {resultado['reserva_chain_ladder']:,.0f}")
    ax.axvline(resultado["percentiles"][99.5], color=NARANJA, lw=1.5, ls="--", label=f"percentil 99.5 % {resultado['percentiles'][99.5]:,.0f}")
    _ejes(ax, "Distribución de la reserva (bootstrap ODP)", "reserva total", "frecuencia"); _leyenda(ax)
    return fig
