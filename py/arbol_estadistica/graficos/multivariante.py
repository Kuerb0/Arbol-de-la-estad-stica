"""Rama GRAFICOS / multivariante: dendrograma y biplot de componentes principales."""
from __future__ import annotations

import numpy as np

from ._base import APAGADO, AZUL, NARANJA, TINTA, _ejes, _figura


def grafico_dendrograma(resultado: dict, k: int | None = None, max_hojas: int = 40, ax=None):
    """Dendrograma de multivariante.clustering_jerarquico (truncado a `max_hojas`), con la línea de corte para `k` grupos."""
    from scipy.cluster.hierarchy import dendrogram
    L = resultado["enlace"]
    fig, ax = _figura(ax, (8, 4.2))
    dendrogram(L, truncate_mode="lastp", p=max_hojas, ax=ax, color_threshold=None, above_threshold_color=AZUL,
               link_color_func=lambda _: AZUL, no_labels=True)
    if k and 1 < k <= len(L):
        h = (L[-k, 2] + L[-k + 1, 2]) / 2
        ax.axhline(h, color=NARANJA, ls="--", lw=1.2)
    _ejes(ax, f"Dendrograma (cofenética {resultado['cofenetica']:.2f})", "", "distancia de fusión")
    return fig


def grafico_biplot(resultado_pca: dict, ejes=("PC1", "PC2"), muestra: int = 1500, semilla: int = 42, ax=None):
    """Biplot de multivariante.pca_completo: puntos (observaciones) y flechas (cargas de las variables)."""
    p, c, v = resultado_pca["puntuaciones"], resultado_pca["cargas"], resultado_pca["varianza"]
    a, b = ejes
    if len(p) > muestra:
        p = p.sample(muestra, random_state=semilla)
    fig, ax = _figura(ax, (6.4, 5.2))
    esc = np.abs(p[[a, b]]).to_numpy().max() / max(np.abs(c[[a, b]]).to_numpy().max(), 1e-9) * 0.8
    ax.scatter(p[a], p[b], s=6, color=APAGADO, alpha=0.4)
    for var, fila in c.iterrows():
        ax.annotate("", xy=(fila[a] * esc, fila[b] * esc), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": AZUL, "lw": 1.4})
        ax.text(fila[a] * esc * 1.08, fila[b] * esc * 1.08, str(var), color=TINTA, fontsize=8.5, ha="center")
    ax.axhline(0, color=APAGADO, lw=.6); ax.axvline(0, color=APAGADO, lw=.6)
    _ejes(ax, "Biplot", f"{a} ({v.loc[a, 'pct_varianza']:.1f} %)", f"{b} ({v.loc[b, 'pct_varianza']:.1f} %)", rejilla="")
    return fig
