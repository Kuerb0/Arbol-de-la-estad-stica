"""Rama GRAFICOS / segmentación y colinealidad: elección de K, clusters en PCA y VIF.

Origen: los gráficos de `notebook de consultoría` y notebook de consultoría (silhouette por K y
dispersión PC1-PC2 coloreada por cluster), generalizados.
Equivale a PROC SGPLOT sobre la salida de PROC FASTCLUS / PROC PRINCOMP y a la opción VIF de PROC REG.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..clustering.kmeans_pca import proyeccion_pca
from ._base import APAGADO, AZUL, ESTADO, FONDO, TINTA, TINTA2, _color, _ejes, _figura


def grafico_seleccion_k(tabla: pd.DataFrame):
    """Dos paneles a partir de clustering.buscar_k_silhouette: silhouette por K (mayor es mejor)
    e inercia por K (buscar el «codo»). El K elegido se marca en ambos."""
    if not {"k", "silhouette", "inercia"} <= set(tabla.columns):
        raise KeyError("Usa la salida de buscar_k_silhouette.")
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), layout="constrained")
    fig.patch.set_facecolor(FONDO)
    k_el = int(tabla.loc[tabla["silhouette"].idxmax(), "k"])
    for ax, col, tit in ((axs[0], "silhouette", "Silhouette (mayor es mejor)"), (axs[1], "inercia", "Inercia (busca el codo)")):
        ax.plot(tabla["k"], tabla[col], color=AZUL, lw=2, marker="o", ms=6, mec="white", mew=1.5)
        fila = tabla[tabla["k"] == k_el].iloc[0]
        ax.scatter([k_el], [fila[col]], s=90, facecolor="none", edgecolor=TINTA, linewidth=1.5, zorder=3)
        ax.set_xticks(tabla["k"])
        _ejes(ax, tit, "K (nº de clusters)", col)
    axs[0].annotate(f"K = {k_el}", (k_el, tabla["silhouette"].max()), xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color=TINTA2, va="center")
    return fig


def grafico_clusters_pca(X, etiquetas, ax=None, muestra: int = 5000, semilla: int = 42):
    """Proyección PC1-PC2 coloreada por cluster, con el centro de cada uno etiquetado.
    Se dibuja una muestra de `muestra` puntos (Linus) y los ejes muestran la varianza explicada.
    OJO: la PCA es solo para ver; dos clusters solapados en 2D pueden estar separados en el espacio completo."""
    X = np.asarray(X, float)
    lab = np.asarray(etiquetas)
    niveles = sorted(pd.unique(lab), key=str)
    if len(niveles) > 8:
        raise ValueError("Más de 8 clusters: dibuja por partes o agrupa.")
    z, var = proyeccion_pca(X, 2, semilla)
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(X), min(muestra, len(X)), replace=False)
    fig, ax = _figura(ax, (6.0, 5.0))
    for i, g in enumerate(niveles):
        m = lab[idx] == g
        ax.scatter(z["PC1"].to_numpy()[idx][m], z["PC2"].to_numpy()[idx][m], s=12, color=_color(i), alpha=0.5,
                   linewidths=0, label=f"C{g}" if str(g).isdigit() else str(g))
        cx, cy = z["PC1"][lab == g].mean(), z["PC2"][lab == g].mean()
        ax.scatter([cx], [cy], s=110, color=_color(i), edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(f"C{g}" if str(g).isdigit() else str(g), (cx, cy), xytext=(7, 6), textcoords="offset points",
                    fontsize=9, color=TINTA, fontweight="bold")
    _ejes(ax, "Clusters proyectados en las 2 primeras componentes", f"PC1 ({var[0]:.0%} varianza)",
          f"PC2 ({var[1]:.0%} varianza)", rejilla="both")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TINTA2, markerscale=1.6, loc="best")
    return fig


def grafico_vif(tabla: pd.DataFrame, ax=None):
    """Barras horizontales del VIF (salida de diagnostico.calcular_vif) con los cortes 5 y 10.
    Color de estado con su etiqueta: verde < 5, ámbar 5-10, rojo >= 10."""
    if not {"variable", "VIF"} <= set(tabla.columns):
        raise KeyError("Usa la salida de calcular_vif.")
    t = tabla.replace(np.inf, np.nan).dropna(subset=["VIF"]).sort_values("VIF")
    fig, ax = _figura(ax, (6.4, max(2.4, 0.36 * len(t) + 1.2)))
    colores = [ESTADO["bien"] if v < 5 else ESTADO["aviso"] if v < 10 else ESTADO["critico"] for v in t["VIF"]]
    ax.barh(t["variable"].astype(str), t["VIF"], height=0.55, color=colores)
    for c in (5, 10):
        ax.axvline(c, color=APAGADO, lw=1)
    for y, v in enumerate(t["VIF"]):
        etiqueta = "OK" if v < 5 else "moderado" if v < 10 else "ALTO"
        ax.annotate(f"{v:.1f} · {etiqueta}", (v, y), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=TINTA2)
    _ejes(ax, "Factor de inflación de la varianza (VIF)", "VIF", "", rejilla="x")
    return fig
