"""Rama GRAFICOS / simulación: trazas de MCMC, arrepentimiento de bandidos y trayectorias de procesos estocásticos."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ._base import APAGADO, AZUL, NARANJA, _color, _ejes, _figura, _leyenda


def grafico_trazas_mcmc(resultado: dict, parametros=None):
    """Traza (por cadena) y densidad posterior de cada parámetro de simulacion.metropolis, con R-hat y ESS en el título.
    Cadenas bien mezcladas = «oruga peluda» sin tendencia y densidades superpuestas."""
    M = resultado["cadenas"]; res = resultado["resumen"]
    nombres = list(res.index) if parametros is None else list(parametros)
    fig, axs = plt.subplots(len(nombres), 2, figsize=(10, 2.4 * len(nombres)), squeeze=False, gridspec_kw={"width_ratios": [3, 1]})
    for f, nom in enumerate(nombres):
        j = list(res.index).index(nom)
        for c in range(M.shape[0]):
            axs[f, 0].plot(M[c, :, j], lw=0.5, alpha=0.8, color=_color(c))
            axs[f, 1].hist(M[c, :, j], bins=40, histtype="step", density=True, color=_color(c))
        _ejes(axs[f, 0], f"{nom}: R-hat {res.loc[nom, 'r_hat']:.3f} · ESS {res.loc[nom, 'ess']:.0f}", "iteración", "")
        _ejes(axs[f, 1], "posterior", "", "")
    fig.tight_layout()
    return fig


def grafico_bandido(resultados: dict, ax=None):
    """Arrepentimiento acumulado de varias estrategias (dict nombre → resultado de simular_bandido): cuanto más plano,
    antes ha aprendido cuál es el mejor brazo. El reparto uniforme (A/B clásico) crece en línea recta."""
    fig, ax = _figura(ax)
    for i, (nom, r) in enumerate(resultados.items()):
        ax.plot(r["arrepentimiento"].index, r["arrepentimiento"].to_numpy(), color=_color(i), lw=2, label=nom)
    _ejes(ax, "Bandido multibrazo: arrepentimiento acumulado", "ronda", "éxitos perdidos frente al mejor brazo"); _leyenda(ax)
    return fig


def grafico_trayectorias(trayectorias, n_mostrar: int = 30, bandas: bool = True, ax=None, titulo: str = "Trayectorias simuladas"):
    """Trayectorias de un proceso (DataFrame filas = tiempo, columnas = trayectoria: simular_browniano, paseo_aleatorio,
    simular_cadena_markov numérica) con la media y la banda 5-95 % de todas ellas."""
    T = trayectorias
    fig, ax = _figura(ax)
    idx = np.asarray(T.index, float)
    ax.plot(idx, T.iloc[:, :n_mostrar].to_numpy(), color=APAGADO, lw=0.6, alpha=0.6)
    if bandas and T.shape[1] > 10:
        q = np.quantile(T.to_numpy(), [0.05, 0.95], axis=1)
        ax.fill_between(idx, q[0], q[1], color=AZUL, alpha=0.15, label="5-95 %")
        ax.plot(idx, T.mean(axis=1), color=NARANJA, lw=2, label="media")
        _leyenda(ax)
    _ejes(ax, titulo, T.index.name or "t", "")
    return fig
