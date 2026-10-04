"""Rama GRAFICOS / diseño: interacción, efectos de un 2^k (seminormal), superficie de respuesta, balance de la
propensión (love plot) y forest plot de un metaanálisis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._base import APAGADO, AZUL, NARANJA, TINTA, _color, _ejes, _figura, _leyenda


def grafico_interaccion(df: pd.DataFrame, respuesta: str, factor_x: str, factor_traza: str, ax=None):
    """Gráfico de interacción: media de la respuesta por nivel de `factor_x`, una línea por nivel de `factor_traza`, con
    ±1 EE. Líneas paralelas = sin interacción; cruzadas o con pendientes distintas = interacción."""
    g = df.groupby([factor_traza, factor_x], observed=True)[respuesta].agg(["mean", "sem"]).reset_index()
    fig, ax = _figura(ax)
    niveles = list(pd.unique(df[factor_x].dropna()))
    try:
        niveles = sorted(niveles)
    except TypeError:
        pass
    for i, (nivel, s) in enumerate(g.groupby(factor_traza, observed=True)):
        s = s.set_index(factor_x).reindex(niveles)
        ax.errorbar(range(len(niveles)), s["mean"], yerr=s["sem"], marker="o", capsize=3, color=_color(i), lw=2, label=str(nivel))
    ax.set_xticks(range(len(niveles)), [str(n) for n in niveles])
    _ejes(ax, f"Interacción {factor_x} × {factor_traza}", factor_x, f"media de {respuesta}"); _leyenda(ax, title=factor_traza)
    return fig


def grafico_efectos_2k(resultado: dict, ax=None):
    """Gráfico seminormal (half-normal, Daniel) de los |efectos| de efectos_factorial_2k: los efectos inertes se alinean
    en una recta que pasa por el origen; los activos se separan por arriba. Se marcan los que superan el ME de Lenth."""
    t = resultado["tabla"].assign(a=lambda x: x.estimacion.abs()).sort_values("a")
    m = len(t)
    q = stats.halfnorm.ppf((np.arange(1, m + 1) - 0.5) / m)
    fig, ax = _figura(ax)
    act = t.activo_me.to_numpy()
    ax.scatter(t.a[~act], q[~act], color=APAGADO, zorder=3)
    ax.scatter(t.a[act], q[act], color=NARANJA, zorder=3, label="activo (ME de Lenth)")
    for nom, x, y in zip(t.index[act], t.a[act], q[act]):
        ax.annotate(nom, (x, y), xytext=(5, -3), textcoords="offset points", fontsize=9)
    ax.plot([0, resultado["pse_lenth"] * q.max()], [0, q.max()], color=AZUL, lw=1, ls="--", label="pendiente PSE")
    _ejes(ax, "Efectos: gráfico seminormal", "|efecto|", "cuantil seminormal", rejilla="both"); _leyenda(ax)
    return fig


def grafico_superficie_respuesta(resultado: dict, factores=None, ax=None, malla: int = 60):
    """Curvas de nivel de la superficie de segundo orden de superficie_respuesta (dos factores; el resto en su punto
    estacionario), con el punto estacionario y los puntos del diseño."""
    m = resultado["modelo"]; xs = resultado["punto_estacionario"]
    f = list(factores or xs.index[:2])
    datos = m.model.data.frame
    rx = [datos[c] for c in f]
    lo = [min(r.min(), xs[c]) for r, c in zip(rx, f)]; hi = [max(r.max(), xs[c]) for r, c in zip(rx, f)]
    g1, g2 = np.meshgrid(np.linspace(lo[0], hi[0], malla), np.linspace(lo[1], hi[1], malla))
    rej = pd.DataFrame({c: xs[c] for c in xs.index}, index=range(g1.size))
    rej[f[0]], rej[f[1]] = g1.ravel(), g2.ravel()
    z = np.asarray(m.predict(rej)).reshape(g1.shape)
    fig, ax = _figura(ax, (6, 5))
    cs = ax.contourf(g1, g2, z, levels=15, cmap="viridis", alpha=0.85)
    fig.colorbar(cs, ax=ax)
    ax.scatter(datos[f[0]], datos[f[1]], color="white", edgecolor=TINTA, s=25, zorder=3, label="diseño")
    ax.plot(xs[f[0]], xs[f[1]], "*", ms=16, color=NARANJA, label=f"estacionario ({resultado['tipo']})")
    _ejes(ax, "Superficie de respuesta", f[0], f[1], rejilla=""); _leyenda(ax)
    return fig


def grafico_balance(resultado: dict, ax=None):
    """Love plot de puntuacion_propension: |diferencia de medias estandarizada| de cada covariable antes y después del
    ajuste, con la referencia 0.1."""
    b = resultado["balance"].abs().sort_values("smd_antes")
    fig, ax = _figura(ax, (6, max(3, 0.35 * len(b) + 1)))
    y = np.arange(len(b))
    ax.scatter(b.smd_antes, y, color=APAGADO, label="antes", zorder=3)
    ax.scatter(b.smd_despues, y, color=AZUL, label="después", zorder=3)
    ax.axvline(0.1, color=NARANJA, ls="--", lw=1)
    ax.set_yticks(y, b.index)
    _ejes(ax, f"Balance de covariables ({resultado['metodo']})", "|SMD|", "", rejilla="x"); _leyenda(ax)
    return fig


def grafico_metaanalisis(resultado: dict, ax=None):
    """Forest plot de metaanalisis: efecto e IC de cada estudio (tamaño ∝ peso), diamante del efecto combinado e
    intervalo de predicción."""
    t = resultado["tabla"]
    k = len(t)
    fig, ax = _figura(ax, (7, 0.4 * k + 2))
    y = np.arange(k, 0, -1)
    ax.hlines(y, t.ic_inf, t.ic_sup, color=TINTA, lw=1)
    ax.scatter(t.efecto, y, s=20 + 6 * t.peso, marker="s", color=AZUL, zorder=3)
    mu, (lo, hi) = resultado["efecto"], resultado["ic"]
    ax.fill([lo, mu, hi, mu], [-0.5, -0.2, -0.5, -0.8], color=NARANJA)
    pi = resultado["intervalo_prediccion"]
    if np.all(np.isfinite(pi)):
        ax.hlines(-1.3, pi[0], pi[1], color=NARANJA, lw=2, ls=":")
    ax.axvline(0, color=APAGADO, lw=1)
    ax.set_yticks(list(y) + [-0.5, -1.3], list(t.index) + ["combinado", "predicción"])
    _ejes(ax, f"Metaanálisis ({resultado['metodo']}) · I² = {resultado['I2']:.0%}", "efecto", "", rejilla="x")
    return fig
