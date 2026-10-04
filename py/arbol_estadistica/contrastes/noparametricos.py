"""Contrastes por permutación, tendencia ordenada (Jonckheere-Terpstra) y post-hoc robustos (Dunn, Games-Howell)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .._util import _columnas, _numerico
from .multiples import ajustar_p_valores

_ESTADISTICOS = {"media": lambda a, b: a.mean() - b.mean(), "mediana": lambda a, b: np.median(a) - np.median(b)}


def contraste_permutacion(a, b, estadistico="media", n_perm: int = 10000, alternativa: str = "two-sided",
                          semilla: int = 42) -> dict:
    """Contraste de permutación para dos grupos: reparte al azar las etiquetas y compara.

    `estadistico`: 'media', 'mediana' o una función f(a, b). El p-valor es (1 + #extremos)/(1 + n_perm) (nunca 0).
    Sin supuestos de distribución: solo intercambiabilidad bajo H0 (mismas distribuciones).
    Equivale a PROC NPAR1WAY SCORES=DATA / EXACT (Monte Carlo).
    """
    x, y = _numerico(a, "a", 2), _numerico(b, "b", 2)
    f = _ESTADISTICOS.get(estadistico, estadistico)
    if not callable(f):
        raise ValueError("estadistico: 'media', 'mediana' o una función f(a, b)")
    obs = float(f(x, y))
    rng = np.random.default_rng(semilla)
    todo, na = np.r_[x, y], len(x)
    dist = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(todo); dist[i] = f(p[:na], p[na:])
    if alternativa == "two-sided":
        ext = np.sum(np.abs(dist) >= abs(obs) - 1e-12)
    elif alternativa == "greater":
        ext = np.sum(dist >= obs - 1e-12)
    elif alternativa == "less":
        ext = np.sum(dist <= obs + 1e-12)
    else:
        raise ValueError("alternativa: 'two-sided', 'greater' o 'less'")
    return {"estadistico": obs, "p_valor": float((1 + ext) / (1 + n_perm)), "n_perm": n_perm,
            "distribucion_nula": dist}


def contraste_jonckheere(df: pd.DataFrame, valor: str, grupo: str, orden=None, alternativa: str = "increasing") -> dict:
    """Jonckheere-Terpstra: ¿la variable crece (o decrece) con el ORDEN de los grupos? Más potente que Kruskal-Wallis
    cuando la alternativa es una tendencia (dosis, tramos de edad, niveles de bonus).

    J = Σ_{i<j} #{(x en grupo i, y en grupo j): x < y} (+½ empates). Aproximación normal con corrección por empates.
    """
    _columnas(df, [valor, grupo])
    d = df[[valor, grupo]].dropna()
    niveles = list(orden) if orden is not None else sorted(d[grupo].unique())
    muestras = [d.loc[d[grupo] == g, valor].to_numpy(float) for g in niveles]
    if len(muestras) < 3 or any(len(m) == 0 for m in muestras):
        raise ValueError("Hacen falta ≥ 3 grupos con datos (en el orden de la tendencia).")
    J = 0.0
    for i, j in itertools.combinations(range(len(muestras)), 2):
        # #{x_i < x_j} + ½ empates = U de Mann-Whitney de la muestra j frente a la i (por rangos: O(n log n))
        J += float(stats.mannwhitneyu(muestras[j], muestras[i], method="asymptotic").statistic)
    ns = np.array([len(m) for m in muestras]); N = ns.sum()
    media = (N ** 2 - np.sum(ns ** 2)) / 4
    _, t = np.unique(np.concatenate(muestras), return_counts=True)
    var = (N ** 2 * (2 * N + 3) - np.sum(ns ** 2 * (2 * ns + 3)) - np.sum(t * (t - 1) * (2 * t + 5))) / 72
    z = (J - media) / np.sqrt(var)
    p = stats.norm.sf(z) if alternativa == "increasing" else stats.norm.cdf(z) if alternativa == "decreasing" else 2 * stats.norm.sf(abs(z))
    return {"J": float(J), "z": float(z), "p_valor": float(p), "orden": niveles, "alternativa": alternativa}


def posthoc_dunn(df: pd.DataFrame, valor: str, grupo: str, ajuste: str = "holm") -> pd.DataFrame:
    """Post-hoc de Dunn tras un Kruskal-Wallis: z por pares con los rangos conjuntos (corrección por empates)
    y p ajustados (Holm por defecto; 'fdr_bh', 'bonferroni'…)."""
    _columnas(df, [valor, grupo])
    d = df[[valor, grupo]].dropna().copy()
    d["_r"] = stats.rankdata(d[valor])
    N = len(d); _, t = np.unique(d[valor], return_counts=True)
    corr = np.sum(t ** 3 - t) / (12 * (N - 1))
    g = d.groupby(grupo)["_r"].agg(["mean", "size"])
    filas = []
    for a, b in itertools.combinations(g.index, 2):
        se = np.sqrt((N * (N + 1) / 12 - corr) * (1 / g.loc[a, "size"] + 1 / g.loc[b, "size"]))
        z = (g.loc[a, "mean"] - g.loc[b, "mean"]) / se
        filas.append({"grupo_1": a, "grupo_2": b, "dif_rango_medio": g.loc[a, "mean"] - g.loc[b, "mean"], "z": z,
                      "p_valor": 2 * stats.norm.sf(abs(z))})
    t = pd.DataFrame(filas)
    adj = ajustar_p_valores(t["p_valor"].to_numpy(), metodo=ajuste).sort_index()
    t["p_ajustado"] = adj["p_ajustado"].to_numpy(); t["rechaza"] = adj["rechaza"].to_numpy()
    return t


def games_howell(df: pd.DataFrame, valor: str, grupo: str, alpha: float = 0.05) -> pd.DataFrame:
    """Games-Howell: comparaciones por pares SIN suponer varianzas iguales (Welch + rango estudentizado).
    La alternativa a Tukey cuando Levene/Brown-Forsythe rechazan."""
    _columnas(df, [valor, grupo])
    g = df[[valor, grupo]].dropna().groupby(grupo)[valor].agg(["mean", "var", "size"])
    k = len(g)
    if k < 2:
        raise ValueError("Hacen falta ≥ 2 grupos.")
    filas = []
    for a, b in itertools.combinations(g.index, 2):
        va, vb = g.loc[a, "var"] / g.loc[a, "size"], g.loc[b, "var"] / g.loc[b, "size"]
        dif, se = g.loc[a, "mean"] - g.loc[b, "mean"], np.sqrt(va + vb)
        gl = (va + vb) ** 2 / (va ** 2 / (g.loc[a, "size"] - 1) + vb ** 2 / (g.loc[b, "size"] - 1))
        q = abs(dif) / se * np.sqrt(2)
        p = float(stats.studentized_range.sf(q, k, gl))
        qc = stats.studentized_range.ppf(1 - alpha, k, gl) / np.sqrt(2)
        filas.append({"grupo_1": a, "grupo_2": b, "diferencia_medias": dif, "ic_inf": dif - qc * se, "ic_sup": dif + qc * se,
                      "gl": gl, "p_ajustado": min(p, 1.0), "rechaza": p < alpha})
    return pd.DataFrame(filas)
