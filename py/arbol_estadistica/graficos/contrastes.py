"""Rama GRAFICOS / contrastes de hipótesis: cómo funcionan y qué dicen tus datos.

Origen: nuevo; acompaña a `contrastes` (elegir_contraste, potencia, ajustar_p_valores).
Equivale a PROC TTEST PLOTS=, PROC UNIVARIATE QQPLOT, PROC POWER PLOT y PROC MULTTEST PLOTS=.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..contrastes.elegir_contraste import elegir_contraste
from ..contrastes.multiples import ajustar_p_valores
from ..contrastes.potencia import curva_potencia, potencia_contraste_medias, tamano_muestral_medias
from ._base import APAGADO, AZUL, NARANJA, TINTA, TINTA2, _color, _ejes, _figura, _leyenda


def _distribucion(nombre: str, gl):
    if nombre == "normal":
        return stats.norm()
    if nombre == "t":
        if gl is None:
            raise ValueError("La t necesita `gl`.")
        return stats.t(gl)
    if nombre == "chi2":
        if gl is None:
            raise ValueError("La chi2 necesita `gl`.")
        return stats.chi2(gl)
    if nombre == "f":
        if not isinstance(gl, (tuple, list)) or len(gl) != 2:
            raise ValueError("La F necesita `gl=(gl_num, gl_den)`.")
        return stats.f(*gl)
    raise ValueError("distribucion: 'normal', 't', 'chi2' o 'f'.")


def grafico_region_rechazo(estadistico: float, distribucion: str = "t", gl=None, alpha: float = 0.05,
                           alternativa: str = "two-sided", ax=None):
    """Dibuja la distribución del estadístico BAJO H0, la región de rechazo (naranja) y el área del
    p-valor (azul) a partir del estadístico observado.

    distribucion: 'normal', 't' (gl), 'chi2' (gl) o 'f' (gl=(num, den)); chi2 y F son siempre de cola derecha.
    alternativa : 'two-sided', 'larger' o 'smaller'.
    Lectura: si la línea del estadístico cae en naranja, p < alpha. El p-valor NO es la probabilidad
    de que H0 sea cierta: es lo raro que sería un estadístico así de extremo SI H0 fuera cierta.
    """
    dist = _distribucion(distribucion, gl)
    if distribucion in ("chi2", "f"):
        alternativa = "larger"
    if alternativa not in ("two-sided", "larger", "smaller"):
        raise ValueError("alternativa: 'two-sided', 'larger' o 'smaller'.")
    lo, hi = dist.ppf(0.0005), dist.ppf(0.9995)
    lo, hi = min(lo, estadistico - 0.5), max(hi, estadistico + 0.5)
    if distribucion in ("chi2", "f"):
        lo = 0
    xs = np.linspace(lo, hi, 600)
    ys = dist.pdf(xs)
    fig, ax = _figura(ax)
    ax.plot(xs, ys, color=TINTA2, lw=1.6, label="distribución bajo H0")
    if alternativa == "two-sided":
        c = dist.ppf(1 - alpha / 2)
        reg = [(xs <= -c), (xs >= c)] if distribucion in ("normal", "t") else [(xs >= c)]
        p = 2 * min(dist.cdf(estadistico), dist.sf(estadistico))
        lim = abs(estadistico)
        areas = [(xs <= -lim), (xs >= lim)]
        criticos = [-c, c]
    elif alternativa == "larger":
        c = dist.ppf(1 - alpha); reg = [(xs >= c)]; p = dist.sf(estadistico)
        areas = [(xs >= estadistico)]; criticos = [c]
    else:
        c = dist.ppf(alpha); reg = [(xs <= c)]; p = dist.cdf(estadistico)
        areas = [(xs <= estadistico)]; criticos = [c]
    for i, m in enumerate(reg):
        ax.fill_between(xs, 0, ys, where=m, color=NARANJA, alpha=0.35, lw=0,
                        label=f"región de rechazo (alpha = {alpha})" if i == 0 else None)
    for i, m in enumerate(areas):
        ax.fill_between(xs, 0, ys, where=m, color=AZUL, alpha=0.45, lw=0,
                        label=f"p-valor = {p:.3g}" if i == 0 else None)
    for cv in criticos:
        ax.annotate(f"crítico {cv:.2f}", (cv, dist.pdf(cv)), xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=8, color=TINTA2)
    ax.axvline(estadistico, color=AZUL, lw=2)
    centro = dist.median()
    derecha = estadistico >= centro
    ax.annotate(f"observado = {estadistico:.2f}", (estadistico, ys.max() * 0.62), xytext=(-6 if derecha else 6, 0),
                textcoords="offset points", fontsize=9, color=TINTA, ha="right" if derecha else "left")
    ax.set_ylim(0, ys.max() * 1.12)
    decision = "se rechaza H0" if p < alpha else "no se rechaza H0"
    _ejes(ax, f"{distribucion.upper() if distribucion != 'normal' else 'Normal'}"
              f"{'' if gl is None else f'({gl})'} · p = {p:.3g} → {decision}", "valor del estadístico", "densidad")
    _leyenda(ax, loc="upper left" if estadistico >= centro else "upper right")
    return fig


def grafico_potencia(efecto: float, n_grupo1: int, alpha: float = 0.05, ratio: float = 1.0, ax=None):
    """Las dos distribuciones del estadístico t: bajo H0 (sin efecto) y bajo H1 (con efecto d).

    Naranja = alpha (falso positivo), gris = beta (no detectar un efecto que existe),
    azul = potencia = 1 - beta. Subir n separa las curvas; subir alpha mueve el corte a la izquierda.
    Test bilateral de dos muestras independientes.
    """
    n2 = max(int(round(n_grupo1 * ratio)), 2)
    gl = n_grupo1 + n2 - 2
    nc = abs(efecto) * np.sqrt(n_grupo1 * n2 / (n_grupo1 + n2))
    h0, h1 = stats.t(gl), stats.nct(gl, nc)
    c = h0.ppf(1 - alpha / 2)
    xs = np.linspace(min(-4, -c - 1), max(4, nc + 4), 700)
    y0, y1 = h0.pdf(xs), h1.pdf(xs)
    pot = potencia_contraste_medias(n_grupo1, efecto, alpha, n2 / n_grupo1)
    fig, ax = _figura(ax)
    ax.fill_between(xs, 0, y1, where=xs < c, color=APAGADO, alpha=0.30, lw=0, label=f"beta = {1 - pot:.2f}")
    ax.fill_between(xs, 0, y1, where=xs >= c, color=AZUL, alpha=0.35, lw=0, label=f"potencia = {pot:.2f}")
    ax.fill_between(xs, 0, y0, where=np.abs(xs) >= c, color=NARANJA, alpha=0.55, lw=0, label=f"alpha = {alpha}")
    ax.plot(xs, y0, color=TINTA2, lw=1.6, label="H0: sin efecto")
    ax.plot(xs, y1, color=AZUL, lw=2, label=f"H1: d = {efecto}")
    ax.axvline(c, color=APAGADO, lw=1)
    ax.set_ylim(0, max(y0.max(), y1.max()) * 1.1)
    necesita = tamano_muestral_medias(efecto=efecto, alpha=alpha, ratio=ratio)["n_grupo1"] if efecto else None
    _ejes(ax, f"Potencia {pot:.0%} con n = {n_grupo1} por grupo"
              + (f" · para 80 % harían falta {necesita}" if necesita else ""), "estadístico t", "densidad")
    _leyenda(ax, loc="upper right")
    return fig


def grafico_curva_potencia(efectos, n_max: int = 300, alpha: float = 0.05, objetivo: float = 0.80, ax=None):
    """Potencia frente a n por grupo para uno o varios efectos (d de Cohen), con el n que alcanza
    el `objetivo` marcado. Útil para decidir el tamaño de un A/B test antes de lanzarlo."""
    efectos = [efectos] if np.isscalar(efectos) else list(efectos)
    fig, ax = _figura(ax)
    ns = np.unique(np.linspace(4, n_max, 120).astype(int))
    ax.axhline(objetivo, color=APAGADO, lw=1)
    for i, d in enumerate(efectos):
        cp = curva_potencia(d, ns, alpha)
        ax.plot(cp["n_grupo1"], cp["potencia"], color=_color(i), lw=2, label=f"d = {d}")
        n_obj = tamano_muestral_medias(efecto=d, alpha=alpha, potencia=objetivo)["n_grupo1"]
        if n_obj <= n_max:
            ax.scatter([n_obj], [objetivo], s=48, color=_color(i), edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(f"n = {n_obj}", (n_obj, objetivo), xytext=(4, -14), textcoords="offset points",
                        fontsize=8.5, color=TINTA2)
    ax.set_ylim(0, 1.02); ax.set_xlim(0, n_max)
    _ejes(ax, f"Curva de potencia (alpha = {alpha}, objetivo {objetivo:.0%})", "n por grupo", "potencia")
    _leyenda(ax, loc="lower right")
    return fig


def grafico_comparar_grupos(df: pd.DataFrame, valor: str, grupo: str, alpha: float = 0.05, ax=None):
    """Compara `valor` entre grupos y pone en el título el contraste que elige elegir_contraste
    (t / Welch / Mann-Whitney / ANOVA / Kruskal / chi2 / Fisher), su p-valor y el tamaño del efecto.

    Numérica: caja + puntos (con jitter) + media (rombo). Categórica: barras 100 % apiladas.
    """
    r = elegir_contraste(df, valor, grupo, alpha=alpha)
    d = df[[valor, grupo]].dropna()
    niveles = sorted(d[grupo].unique(), key=str)
    if len(niveles) > 8:
        raise ValueError("Más de 8 grupos: agrupa niveles antes de dibujar.")
    fig, ax = _figura(ax)
    titulo = f"{r['contraste']}: p = {r['p_valor']:.3g} · {r['efecto_nombre']} = {r['efecto']:.2f}"
    if r["contraste"] in ("Chi-cuadrado de independencia", "Fisher exacto"):
        tab = pd.crosstab(d[grupo], d[valor], normalize="index").reindex(niveles)
        abajo = np.zeros(len(tab))
        for j, col in enumerate(tab.columns):
            ax.bar([str(v) for v in tab.index], tab[col], bottom=abajo, width=0.6, color=_color(j),
                   edgecolor="white", linewidth=2, label=f"{valor} = {col}")
            abajo += tab[col].to_numpy()
        ax.set_ylim(0, 1)
        _ejes(ax, titulo, grupo, "proporción")
        _leyenda(ax, loc="upper left", bbox_to_anchor=(1, 1))
        return fig
    rng = np.random.default_rng(42)
    datos = [d.loc[d[grupo] == g, valor].to_numpy(float) for g in niveles]
    ax.boxplot(datos, positions=range(len(niveles)), widths=0.45, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="none", edgecolor=APAGADO), medianprops=dict(color=TINTA, lw=1.5),
               whiskerprops=dict(color=APAGADO), capprops=dict(color=APAGADO))
    for i, m in enumerate(datos):
        muestra = m if len(m) <= 400 else rng.choice(m, 400, replace=False)
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(muestra)), muestra, s=10, color=_color(i), alpha=0.45, linewidths=0)
        ax.scatter([i], [m.mean()], marker="D", s=46, color=_color(i), edgecolor="white", linewidth=1.5, zorder=3)
    ax.set_xticks(range(len(niveles)), [f"{g}\n(n={len(m)})" for g, m in zip(niveles, datos)])
    _ejes(ax, titulo, grupo, valor)
    return fig


def grafico_qq(x, ax=None):
    """Gráfico Q-Q normal con el p-valor de Shapiro-Wilk (n <= 5000) en el título.

    Curva en S = colas pesadas; curva en U = asimetría. Con n grande Shapiro rechaza por
    desviaciones mínimas: decide mirando el gráfico, no solo el p-valor.
    """
    v = pd.Series(np.asarray(x, float)).dropna().to_numpy()
    if len(v) < 3:
        raise ValueError("Hacen falta al menos 3 valores.")
    (teo, obs), (pend, ord_, _) = stats.probplot(v, dist="norm")
    fig, ax = _figura(ax, (5.0, 4.6))
    ax.scatter(teo, obs, s=14, color=AZUL, alpha=0.6, linewidths=0)
    ax.plot(teo, pend * teo + ord_, color=APAGADO, lw=1)
    sub = f"Shapiro-Wilk p = {stats.shapiro(v).pvalue:.3g}" if len(v) <= 5000 else f"n = {len(v):,} (Shapiro no aplica)"
    _ejes(ax, f"Q-Q normal · {sub}", "cuantil teórico N(0,1)", "cuantil observado", rejilla="both")
    return fig


def grafico_fdr(p_valores, alpha: float = 0.05, ax=None):
    """p-valores ordenados frente a su rango con la recta de Benjamini-Hochberg (i/m·alpha), el
    umbral de Bonferroni (alpha/m) y alpha sin corregir. Azul = descubrimientos con BH.
    El título compara cuántos «significativos» salen con cada criterio."""
    t = ajustar_p_valores(p_valores, "fdr_bh", alpha).sort_values("rango")
    m = len(t)
    bonf = int((t["p_valor"] <= alpha / m).sum())
    fig, ax = _figura(ax)
    r = t["rango"].to_numpy()
    ax.plot(r, r / m * alpha, color=AZUL, lw=2, label="recta BH: rango/m · alpha")
    ax.axhline(alpha / m, color=NARANJA, lw=1.5, label=f"Bonferroni: alpha/m = {alpha / m:.2g}")
    ax.axhline(alpha, color=APAGADO, lw=1, label=f"sin corregir: alpha = {alpha}")
    rech = t["rechaza"].to_numpy()
    ax.scatter(r[~rech], t["p_valor"].to_numpy()[~rech], s=14, color=APAGADO, alpha=0.7, linewidths=0, label="no rechazados")
    ax.scatter(r[rech], t["p_valor"].to_numpy()[rech], s=18, color=AZUL, linewidths=0, label="descubrimientos (BH)")
    ax.set_yscale("log")
    ax.set_ylim(max(t["p_valor"].min() * 0.5, 1e-12), 1.2)
    _ejes(ax, f"BH: {int(rech.sum())} descubrimientos · Bonferroni: {bonf} · sin corregir: {int((t['p_valor'] < alpha).sum())}",
          "rango del p-valor (1 = el menor)", "p-valor (escala log)")
    _leyenda(ax, loc="lower right")
    return fig
