"""Rama GRAFICOS / finanzas: frontera eficiente y nube de una cópula."""
from __future__ import annotations

import numpy as np

from ._base import APAGADO, AZUL, NARANJA, TINTA, _ejes, _figura, _leyenda


def grafico_frontera_eficiente(resultado: dict, medias=None, covarianza=None, ax=None):
    """Frontera de finanzas.frontera_eficiente en el plano volatilidad-rentabilidad, con la cartera de mínima varianza,
    la tangente (si se calculó) y, si se pasan medias y covarianza, los activos individuales."""
    f = resultado["frontera"]
    fig, ax = _figura(ax)
    ax.plot(f.volatilidad, f.rentabilidad, color=AZUL, lw=2.2, label="frontera eficiente")
    r, v = resultado["minima_varianza_rent_vol"]
    ax.plot(v, r, "o", color=TINTA, label="mínima varianza")
    if "tangente" in resultado and covarianza is not None and medias is not None:
        w = resultado["tangente"].to_numpy(); mu = np.asarray(medias, float); S = np.asarray(covarianza, float)
        ax.plot(np.sqrt(w @ S @ w), w @ mu, "*", ms=14, color=NARANJA, label=f"tangente (Sharpe {resultado['sharpe_tangente']:.2f})")
    if medias is not None and covarianza is not None:
        sd = np.sqrt(np.diag(np.asarray(covarianza, float)))
        ax.scatter(sd, np.asarray(medias, float), color=APAGADO, s=30, zorder=3)
        for nom, x, y in zip(getattr(medias, "index", range(len(sd))), sd, np.asarray(medias, float)):
            ax.annotate(str(nom), (x, y), xytext=(4, 2), textcoords="offset points", fontsize=8)
    _ejes(ax, "Frontera de Markowitz", "volatilidad", "rentabilidad esperada", rejilla="both"); _leyenda(ax)
    return fig


def grafico_copula(datos, ax=None, muestra: int = 4000):
    """Nube de (u, v) de finanzas.simular_copula: se ve dónde se concentra la dependencia (colas inferior o superior)."""
    d = datos.sample(min(muestra, len(datos)), random_state=1)
    fig, ax = _figura(ax, (5, 5))
    ax.scatter(d.u, d.v, s=3, alpha=0.35, color=AZUL)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _ejes(ax, f"Cópula (τ de Kendall = {datos.attrs.get('tau_kendall', float('nan')):.2f})", "u", "v", rejilla="")
    return fig
