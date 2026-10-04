"""Rama GRAFICOS / exploración multivariante y regresión: hexbin, violín, coordenadas paralelas, curvas de Andrews,
caras de Chernoff, gráfico de variable añadida, bandas simultáneas de regresión y control T² de Hotelling."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Ellipse

from ._base import APAGADO, AZUL, NARANJA, TINTA, _color, _ejes, _figura, _leyenda


def _escala01(D):
    D = D.astype(float)
    return (D - D.min()) / (D.max() - D.min()).replace(0, 1)


def grafico_hexbin(df: pd.DataFrame, x: str, y: str, tam: int = 40, log: bool = True, ax=None):
    """Hexbin: densidad de una nube con muchos puntos (cuando el scatter se satura), escala logarítmica de conteos."""
    fig, ax = _figura(ax, (6, 5))
    hb = ax.hexbin(df[x], df[y], gridsize=tam, bins="log" if log else None, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="conteo (log)" if log else "conteo")
    _ejes(ax, f"{y} frente a {x}", x, y, rejilla="")
    return fig


def grafico_violin(df: pd.DataFrame, variable: str, grupo: str, ax=None):
    """Violín por grupo: la densidad completa (no solo los cuartiles de la caja), con mediana y cuartiles marcados."""
    grupos = [g for g, _ in df.groupby(grupo, observed=True)]
    datos = [df.loc[df[grupo] == g, variable].dropna().to_numpy() for g in grupos]
    fig, ax = _figura(ax)
    p = ax.violinplot(datos, showmedians=True, showextrema=False, quantiles=[[0.25, 0.75]] * len(datos))
    for i, b in enumerate(p["bodies"]):
        b.set_facecolor(_color(i)); b.set_alpha(0.55)
    ax.set_xticks(range(1, len(grupos) + 1), [str(g) for g in grupos])
    _ejes(ax, f"{variable} por {grupo}", grupo, variable)
    return fig


def grafico_coordenadas_paralelas(df: pd.DataFrame, variables, clase: str | None = None, muestra: int = 500, ax=None):
    """Coordenadas paralelas: cada observación es una línea que recorre los ejes (variables reescaladas a [0, 1]).
    Grupos que se separan en alguna variable se ven como haces de líneas distintos."""
    variables = list(variables)
    d = df.sample(min(muestra, len(df)), random_state=1)
    Z = _escala01(d[variables])
    fig, ax = _figura(ax, (max(6, 1.2 * len(variables)), 4))
    clases = d[clase].astype(str) if clase else pd.Series("todos", index=d.index)
    for i, c in enumerate(sorted(clases.unique())):
        m = (clases == c).to_numpy()
        ax.plot(range(len(variables)), Z[m].to_numpy().T, color=_color(i), alpha=0.25, lw=0.8)
        ax.plot([], [], color=_color(i), label=c)
    ax.set_xticks(range(len(variables)), variables, rotation=30, ha="right")
    _ejes(ax, "Coordenadas paralelas", "", "valor reescalado", rejilla="x")
    if clase:
        _leyenda(ax)
    return fig


def grafico_curvas_andrews(df: pd.DataFrame, variables, clase: str | None = None, muestra: int = 300, ax=None):
    """Curvas de Andrews: cada observación x se dibuja como f(t) = x1/√2 + x2 sin t + x3 cos t + x4 sin 2t + …;
    observaciones parecidas dan curvas parecidas y la distancia entre curvas refleja la euclídea (variables
    estandarizadas). Alternativa compacta a las coordenadas paralelas."""
    variables = list(variables)
    d = df.sample(min(muestra, len(df)), random_state=1)
    Z = ((d[variables] - d[variables].mean()) / d[variables].std(ddof=1).replace(0, 1)).to_numpy()
    t = np.linspace(-np.pi, np.pi, 200)
    B = [np.full_like(t, 1 / np.sqrt(2))]
    for k in range(1, len(variables)):
        h = (k + 1) // 2
        B.append(np.sin(h * t) if k % 2 else np.cos(h * t))
    F = Z @ np.vstack(B)
    fig, ax = _figura(ax)
    clases = d[clase].astype(str).to_numpy() if clase else np.array(["todos"] * len(d))
    for i, c in enumerate(sorted(set(clases))):
        ax.plot(t, F[clases == c].T, color=_color(i), alpha=0.3, lw=0.8)
        ax.plot([], [], color=_color(i), label=c)
    _ejes(ax, "Curvas de Andrews", "t", "f(t)")
    if clase:
        _leyenda(ax)
    return fig


def grafico_caras_chernoff(df: pd.DataFrame, variables, etiquetas=None, columnas: int = 5, max_caras: int = 20):
    """Caras de Chernoff: hasta 8 variables (reescaladas a [0, 1]) se asignan a rasgos de una cara esquemática
    (ancho y alto de la cara, tamaño y separación de ojos, tamaño de pupila, inclinación de cejas, sonrisa, nariz).
    Curiosidad histórica: el cerebro compara caras muy bien, pero la percepción depende de qué variable va a qué
    rasgo; úsalas para pocos casos (perfiles de segmentos), no para inferir."""
    variables = list(variables)[:8]
    Z = _escala01(df[variables]).iloc[:max_caras].to_numpy()
    Z = np.c_[Z, np.full((len(Z), 8 - Z.shape[1]), 0.5)]
    filas = int(np.ceil(len(Z) / columnas))
    fig, axs = plt.subplots(filas, columnas, figsize=(1.8 * columnas, 2.0 * filas), squeeze=False)
    for i, ax in enumerate(axs.ravel()):
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axis("off")
        if i >= len(Z):
            continue
        a, h, ojo, sep, pup, ceja, son, nariz = Z[i]
        ax.add_patch(Ellipse((0, 0), 1.4 + 0.6 * a, 1.8 + 0.6 * h, fill=False, lw=1.5, color=TINTA))
        for s in (-1, 1):
            cx = s * (0.25 + 0.2 * sep)
            ax.add_patch(Ellipse((cx, 0.3), 0.18 + 0.2 * ojo, 0.12 + 0.12 * ojo, fill=False, color=TINTA))
            ax.add_patch(Ellipse((cx, 0.3), 0.04 + 0.08 * pup, 0.04 + 0.08 * pup, color=TINTA))
            ang = (ceja - 0.5) * 40 * s
            ax.plot([cx - 0.15, cx + 0.15], [0.55 + np.tan(np.radians(ang)) * -0.15, 0.55 + np.tan(np.radians(ang)) * 0.15], color=TINTA)
        ax.plot([0, 0], [0.15, -0.1 - 0.2 * nariz], color=TINTA)
        curva = (son - 0.5) * 0.5
        xx = np.linspace(-0.35, 0.35, 30)
        ax.plot(xx, -0.5 - curva * (1 - (xx / 0.35) ** 2), color=NARANJA, lw=1.5)
        ax.set_title(str(etiquetas[i]) if etiquetas is not None else str(df.index[i]), fontsize=8)
    fig.suptitle("Caras de Chernoff: " + ", ".join(variables), fontsize=9)
    return fig


def grafico_variable_anadida(modelo, variable: str, ax=None):
    """Gráfico de variable añadida (regresión parcial) para un modelo OLS de statsmodels con fórmula: residuos de y
    sin `variable` frente a residuos de `variable` sobre las demás X. La pendiente es exactamente el coeficiente de
    `variable` en el modelo completo; se ven su efecto «limpio», puntos influyentes y no linealidades."""
    import statsmodels.api as sm
    X = pd.DataFrame(modelo.model.exog, columns=modelo.model.exog_names)
    y = modelo.model.endog
    col = [c for c in X.columns if c == variable or c == f'Q("{variable}")']
    if not col:
        raise KeyError(f"{variable} no está entre {list(X.columns)}")
    c = col[0]
    otras = X.drop(columns=c)
    ry = sm.OLS(y, otras).fit().resid; rx = sm.OLS(X[c], otras).fit().resid
    fig, ax = _figura(ax)
    ax.scatter(rx, ry, s=10, alpha=0.5, color=AZUL)
    b = modelo.params[c]
    xs = np.linspace(rx.min(), rx.max(), 2)
    ax.plot(xs, b * xs, color=NARANJA, lw=2, label=f"pendiente = {b:.4g}")
    _ejes(ax, f"Variable añadida: {variable}", f"{variable} | resto", "y | resto", rejilla="both"); _leyenda(ax)
    return fig


def grafico_bandas_regresion(bandas: pd.DataFrame, x=None, y=None, ax=None):
    """Recta ajustada con la banda simultánea (de modelos.bandas_confianza_regresion) y el intervalo de predicción."""
    fig, ax = _figura(ax)
    if x is not None and y is not None:
        ax.scatter(x, y, s=10, alpha=0.5, color=APAGADO)
    ax.plot(bandas.x, bandas.ajuste, color=AZUL, lw=2, label="ajuste")
    ax.fill_between(bandas.x, bandas.banda_inf, bandas.banda_sup, color=AZUL, alpha=0.2, label="banda de confianza")
    ax.plot(bandas.x, bandas.prediccion_inf, color=NARANJA, ls="--", lw=1, label="predicción")
    ax.plot(bandas.x, bandas.prediccion_sup, color=NARANJA, ls="--", lw=1)
    _ejes(ax, "Regresión con bandas", "x", "y"); _leyenda(ax)
    return fig


def grafico_control_t2(resultado: dict, ax=None):
    """Gráfico de control T² de multivariante.control_t2_multivariante (fase I y, si existe, fase II) con sus límites."""
    fig, ax = _figura(ax, (8, 3.6))
    f1 = resultado["fase1"]
    ax.plot(np.arange(len(f1)), f1.T2, marker="o", ms=3, color=AZUL, lw=1, label="fase I")
    ax.hlines(f1.limite.iloc[0], 0, len(f1) - 1, color=AZUL, ls="--", lw=1)
    fuera = f1.fuera.to_numpy()
    ax.scatter(np.where(fuera)[0], f1.T2[fuera], color="#d03b3b", zorder=3)
    if "fase2" in resultado:
        f2 = resultado["fase2"]; x2 = np.arange(len(f1), len(f1) + len(f2))
        ax.plot(x2, f2.T2, marker="o", ms=3, color=NARANJA, lw=1, label="fase II")
        ax.hlines(f2.limite.iloc[0], x2[0], x2[-1], color=NARANJA, ls="--", lw=1)
        f = f2.fuera.to_numpy(); ax.scatter(x2[f], f2.T2[f], color="#d03b3b", zorder=3)
    _ejes(ax, "Control T² de Hotelling", "observación", "T²"); _leyenda(ax)
    return fig
