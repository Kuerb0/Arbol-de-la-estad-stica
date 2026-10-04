"""Rama GRAFICOS / supervivencia: curvas de Kaplan-Meier con censuras, IC y log-rank.

Origen: nuevo; acompaña a modelos.kaplan_meier / contraste_log_rank.
Equivale a PROC LIFETEST PLOTS=SURVIVAL(CL ATRISK) con STRATA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..modelos.supervivencia import contraste_log_rank, kaplan_meier
from ._base import APAGADO, TINTA2, _color, _ejes, _figura, _leyenda


def grafico_kaplan_meier(df: pd.DataFrame, duracion: str, evento: str, grupo: str | None = None,
                         ic: bool = True, ax=None):
    """Curva(s) de supervivencia escalonadas, banda IC 95 %, marcas «|» en los censurados y la
    mediana de cada grupo en la leyenda. Con `grupo`, el título lleva el p-valor del log-rank.

    Leer: la altura a tiempo t es la proporción que sigue «viva» (sin evento). Curvas que se
    cruzan = riesgos NO proporcionales (cuidado con Cox y con el log-rank).
    """
    km = kaplan_meier(df, duracion, evento, grupo)
    tabla, meds = km["tabla"], km["medianas"].set_index("grupo")
    if len(meds) > 8:
        raise ValueError("Más de 8 grupos: agrupa niveles antes de dibujar.")
    fig, ax = _figura(ax)
    d = df[[duracion, evento] + ([grupo] if grupo else [])].dropna()
    ax.axhline(0.5, color=APAGADO, lw=0.8)
    for i, (g, t) in enumerate(tabla.groupby("grupo", sort=False)):
        c = _color(i)
        med = meds.loc[g, "mediana"]
        etiqueta = (f"{g} · " if grupo else "") + (f"mediana {med:.3g}" if np.isfinite(med) else "mediana no alcanzada")
        sub = d if grupo is None else d[d[grupo] == g]
        fin = float(sub[duracion].max())
        if fin > t["tiempo"].iloc[-1]:                      # la curva sigue plana hasta el último seguimiento
            t = pd.concat([t, t.iloc[[-1]].assign(tiempo=fin)], ignore_index=True)
        ax.step(t["tiempo"], t["supervivencia"], where="post", color=c, lw=2, label=etiqueta)
        if ic:
            ax.fill_between(t["tiempo"], t["ic_inf"], t["ic_sup"], step="post", color=c, alpha=0.12, lw=0)
        cens = np.sort(sub.loc[sub[evento] == 0, duracion].to_numpy(float))
        if len(cens):
            idx = np.searchsorted(t["tiempo"].to_numpy(), cens, side="right") - 1
            ax.scatter(cens, t["supervivencia"].to_numpy()[idx], marker="|", s=40, color=c, linewidths=1.2)
    ax.set_ylim(0, 1.02); ax.set_xlim(left=0)
    titulo = "Supervivencia de Kaplan-Meier"
    if grupo:
        titulo += f" · log-rank p = {contraste_log_rank(df, duracion, evento, grupo)['p_valor']:.3g}"
    _ejes(ax, titulo, duracion, "proporción sin evento S(t)")
    _leyenda(ax, loc="upper right")
    ax.text(0.0, -0.16, "| = censurado", transform=ax.transAxes, fontsize=8, color=TINTA2)
    return fig
