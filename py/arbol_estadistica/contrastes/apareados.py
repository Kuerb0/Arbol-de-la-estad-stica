"""Muestras relacionadas: t apareado / Wilcoxon, Friedman y McNemar (PROC TTEST PAIRED, NPAR1WAY, FREQ AGREE)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .._util import _columnas


def contraste_apareado(antes, despues, alpha: float = 0.05, tipo: str = "auto") -> dict:
    """Compara dos medidas del MISMO individuo (antes/después, dos tasadores…) a través de las diferencias.

    tipo: 'auto' (t apareado si las diferencias son ~normales por Shapiro o n ≥ 30; si no, Wilcoxon de rangos
    con signo), 't' o 'wilcoxon'. Devuelve contraste, p, diferencia media con IC (t), mediana de diferencias,
    tamaño del efecto (d_z = media/desviación de las diferencias; r = Z/√n para Wilcoxon) e interpretación.
    Equivale a PROC TTEST PAIRED / PROC UNIVARIATE (signed rank) sobre las diferencias.
    Gauss: NO uses un t de grupos independientes con datos apareados: ignora la correlación y pierde potencia.
    OJO: t y Wilcoxon no contrastan lo mismo. t: media de las diferencias = 0. Wilcoxon: diferencias simétricas
    alrededor de 0 (pseudomediana). Con diferencias asimétricas y media 0, Wilcoxon rechaza a menudo: si lo que
    te importa es la MEDIA (p. ej. el coste total), fuerza tipo='t' (n ≥ 30) o usa bootstrap_ic de la media.
    """
    a, b = np.asarray(antes, float), np.asarray(despues, float)
    if a.shape != b.shape:
        raise ValueError("antes y despues deben tener la misma longitud (un par por individuo).")
    ok = np.isfinite(a) & np.isfinite(b)
    d = b[ok] - a[ok]
    n = len(d)
    if n < 3:
        raise ValueError("Hacen falta al menos 3 pares completos.")
    normal = n >= 30 or (stats.shapiro(d).pvalue > 0.05 if n <= 5000 else True)
    if tipo == "auto":
        tipo = "t" if normal else "wilcoxon"
    media, sd = float(d.mean()), float(d.std(ddof=1))
    tc = stats.t.ppf(1 - alpha / 2, n - 1)
    ic = (media - tc * sd / np.sqrt(n), media + tc * sd / np.sqrt(n))
    if tipo == "t":
        r = stats.ttest_rel(b[ok], a[ok]); nombre, est, p = "t apareado", float(r.statistic), float(r.pvalue)
        efecto_nombre, efecto = "d_z", media / sd if sd > 0 else 0.0
    elif tipo == "wilcoxon":
        if np.all(d == 0):
            raise ValueError("Todas las diferencias son 0.")
        r = stats.wilcoxon(d, zero_method="wilcox", method="auto")
        nombre, est, p = "Wilcoxon de rangos con signo", float(r.statistic), float(r.pvalue)
        z = stats.norm.isf(p / 2) * np.sign(np.median(d) or media)
        efecto_nombre, efecto = "r", float(z / np.sqrt(n))
    else:
        raise ValueError("tipo: 'auto', 't' o 'wilcoxon'")
    sig = p < alpha
    avisos = [] if normal else ["Diferencias no normales: Wilcoxon contrasta la pseudomediana, no la media."] if tipo == "wilcoxon" else \
        ["Diferencias no normales con n < 30: el t apareado puede no ser fiable."]
    return {"avisos": avisos, "contraste": nombre, "estadistico": est, "p_valor": p, "significativo": bool(sig), "n_pares": n,
            "diferencia_media": media, "ic_diferencia": ic, "mediana_diferencias": float(np.median(d)),
            "efecto_nombre": efecto_nombre, "efecto": float(efecto), "diferencias_normales": bool(normal),
            "interpretacion": f"{nombre}: p={p:.4g} -> {'hay' if sig else 'no hay evidencia de'} cambio (después − antes = {media:.4g})."}


def contraste_friedman(df: pd.DataFrame, sujeto: str, condicion: str, valor: str) -> dict:
    """Friedman: ¿difieren 3+ condiciones medidas en los MISMOS sujetos? (ANOVA de medidas repetidas no paramétrico).

    Formato largo (una fila por sujeto y condición). Devuelve χ², p y la W de Kendall (0 = nada, 1 = acuerdo total)
    como tamaño del efecto. Solo usa sujetos con todas las condiciones. Equivale a PROC FREQ CMH2 SCORES=RANK.
    """
    _columnas(df, [sujeto, condicion, valor])
    ancho = df.pivot_table(index=sujeto, columns=condicion, values=valor, aggfunc="mean").dropna()
    n, k = ancho.shape
    if k < 3 or n < 2:
        raise ValueError("Hacen falta ≥ 3 condiciones y ≥ 2 sujetos completos.")
    r = stats.friedmanchisquare(*[ancho[c].to_numpy() for c in ancho.columns])
    w = float(r.statistic / (n * (k - 1)))
    return {"chi2": float(r.statistic), "gl": k - 1, "p_valor": float(r.pvalue), "w_kendall": w, "n_sujetos": n,
            "rangos_medios": ancho.rank(axis=1).mean().to_dict()}


def contraste_mcnemar(antes, despues) -> dict:
    """McNemar para proporciones apareadas (0/1 antes y después en los mismos individuos).

    Solo cuentan los pares discordantes b (1→0) y c (0→1). Exacto (binomial) si b + c < 25; si no, χ² con
    corrección de continuidad. Equivale a PROC FREQ / AGREE (McNemar).
    """
    a, b_ = np.asarray(antes), np.asarray(despues)
    if a.shape != b_.shape:
        raise ValueError("Misma longitud.")
    ok = ~(pd.isna(a) | pd.isna(b_))
    a, b_ = a[ok].astype(int), b_[ok].astype(int)
    if not set(np.unique(np.r_[a, b_])) <= {0, 1}:
        raise ValueError("Los valores deben ser 0/1.")
    b = int(((a == 1) & (b_ == 0)).sum()); c = int(((a == 0) & (b_ == 1)).sum())
    if b + c == 0:
        return {"b": 0, "c": 0, "p_valor": 1.0, "metodo": "sin pares discordantes"}
    if b + c < 25:
        p = float(stats.binomtest(min(b, c), b + c, 0.5).pvalue); metodo, est = "exacto (binomial)", np.nan
    else:
        est = (abs(b - c) - 1) ** 2 / (b + c); p = float(stats.chi2.sf(est, 1)); metodo = "χ² con corrección"
    return {"b": b, "c": c, "estadistico": float(est), "p_valor": p, "metodo": metodo,
            "prop_antes": float(a.mean()), "prop_despues": float(b_.mean()), "odds_ratio_apareado": (c / b) if b else np.inf}
