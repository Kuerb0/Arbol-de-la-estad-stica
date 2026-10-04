"""Estilo común de los gráficos del árbol (paleta validada para daltonismo, tinta y rejilla discretas)."""
from __future__ import annotations

import matplotlib.pyplot as plt

# Paleta categórica en orden fijo (los 3 primeros validados todos-contra-todos para daltonismo).
PALETA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
AZUL, NARANJA, AQUA = PALETA[0], PALETA[1], PALETA[2]
TINTA, TINTA2, APAGADO = "#0b0b0b", "#52514e", "#898781"
REJILLA, EJE, FONDO = "#e1e0d9", "#c3c2b7", "#fcfcfb"
ESTADO = {"bien": "#0ca30c", "aviso": "#fab219", "serio": "#ec835a", "critico": "#d03b3b"}


def _figura(ax=None, tam=(6.4, 4.0)):
    """(fig, ax): usa el `ax` que te pasen o crea uno nuevo."""
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=tam, layout="constrained")
    fig.patch.set_facecolor(FONDO)
    return fig, ax


def _ejes(ax, titulo: str = "", x: str = "", y: str = "", rejilla: str = "y") -> None:
    ax.set_facecolor(FONDO)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(EJE)
        ax.spines[lado].set_linewidth(1)
    ax.tick_params(colors=APAGADO, labelcolor=TINTA2, labelsize=9, length=3)
    if rejilla:
        ax.grid(axis=rejilla, color=REJILLA, linewidth=0.8)
        ax.set_axisbelow(True)
    if titulo:
        ax.set_title(titulo, loc="left", fontsize=11, color=TINTA, fontweight="bold")
    if x:
        ax.set_xlabel(x, color=TINTA2, fontsize=9.5)
    if y:
        ax.set_ylabel(y, color=TINTA2, fontsize=9.5)


def _leyenda(ax, **kw) -> None:
    ley = ax.legend(frameon=False, fontsize=8.5, labelcolor=TINTA2, **kw)
    return ley


def _color(i: int) -> str:
    if i >= len(PALETA):
        raise ValueError("Más de 8 series: agrupa en «Otros» o usa paneles separados.")
    return PALETA[i]
