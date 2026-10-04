"""Rama GRAFICOS / ML: importancias y dependencia parcial (con curvas ICE)."""
from __future__ import annotations

import numpy as np

from ._base import APAGADO, AZUL, TINTA2, _ejes, _figura


def grafico_importancias(tabla, maximo: int = 15, ax=None):
    """Barras horizontales de importancia (ml.importancia_permutacion o la serie `importancias` de un modelo), con ± sd si la hay."""
    import pandas as pd
    t = tabla if isinstance(tabla, pd.DataFrame) else tabla.to_frame("importancia")
    t = t.sort_values("importancia").tail(maximo)
    fig, ax = _figura(ax, (6.4, 0.35 * len(t) + 1.2))
    ax.barh(t.index.astype(str), t["importancia"], xerr=t["sd"] if "sd" in t else None, color=AZUL, alpha=0.85,
            error_kw={"ecolor": TINTA2, "lw": 1})
    _ejes(ax, "Importancia de variables" + (f" ({t.attrs.get('metrica')})" if t.attrs.get("metrica") else ""), "importancia", "", rejilla="x")
    return fig


def grafico_dependencia_parcial(resultado: dict, ax=None):
    """Curva de dependencia parcial de ml.dependencia_parcial y, si se pidieron, las curvas ICE en gris."""
    t, v = resultado["tabla"], resultado["variable"]
    fig, ax = _figura(ax)
    if "ice" in resultado:
        for _, fila in resultado["ice"].iterrows():
            ax.plot(fila.index.astype(float), fila.to_numpy(), color=APAGADO, lw=0.6, alpha=0.4)
    ax.plot(t[v], t["prediccion_media"], color=AZUL, lw=2.5)
    _ejes(ax, f"Dependencia parcial de {v}", v, "predicción media")
    return fig
