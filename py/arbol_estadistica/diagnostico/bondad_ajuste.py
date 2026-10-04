"""Rama DIAGNOSTICO / bondad de ajuste y asociación (modelos binarios).

Origen: `hosmer_lemeshow`, `_hl_silent` y `_asociacion` de notebook de consultoría
(re-estimación "PROC LOGISTIC": LACKFIT y Association of Predicted Probabilities).
Equivale a: LACKFIT (Hosmer-Lemeshow) y la tabla "Association of Predicted Probabilities
and Observed Responses" de PROC LOGISTIC.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def hosmer_lemeshow(y_real, y_prob, g: int = 10) -> dict:
    """Test de Hosmer-Lemeshow con `g` grupos por deciles de probabilidad predicha.

    H0: el modelo está bien calibrado. p > 0.05 -> no se rechaza (ajuste aceptable).
    Gauss: con n grande (decenas de miles) casi siempre rechaza; mirar también la tabla
    observado/esperado y el efecto práctico, no solo el p-valor.
    Devuelve dict con estadistico, gl (= grupos - 2), p_valor y tabla (O/E por grupo).
    """
    d = pd.DataFrame({"y": np.asarray(y_real), "p": np.asarray(y_prob, dtype=float)})
    d["grupo"] = pd.qcut(d["p"], g, duplicates="drop")
    t = d.groupby("grupo", observed=True).agg(O1=("y", "sum"), N=("y", "count"), E1=("p", "sum"))
    t["O0"] = t["N"] - t["O1"]
    t["E0"] = t["N"] - t["E1"]
    est = (
        (t["O1"] - t["E1"]) ** 2 / t["E1"].clip(lower=1e-3)
        + (t["O0"] - t["E0"]) ** 2 / t["E0"].clip(lower=1e-3)
    ).sum()
    gl = max(len(t) - 2, 1)
    p = float(chi2.sf(est, gl))
    return {"estadistico": float(est), "gl": gl, "p_valor": p, "ajuste_aceptable": p > 0.05, "tabla": t}


def estadisticos_asociacion(y_real, y_prob) -> dict:
    """c (= AUC), D de Somers, Gamma y Tau-a, EXACTOS y sin muestreo (O(n log n)).

    Compara cada par (positivo, negativo): concordante si p_pos > p_neg, discordante si
    p_pos < p_neg, empate si iguales. Se calcula agrupando por valores únicos de probabilidad,
    lo que evita la matriz n1 x n0 del original (que había que submuestrear).
        c = (conc + 0.5*empates) / (n1*n0)       D = (conc - disc) / (n1*n0)
        Gamma = (conc - disc) / (conc + disc)    Tau-a = (conc - disc) / (n(n-1)/2)
    """
    y = np.asarray(y_real).astype(int)
    p = np.asarray(y_prob, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("Se necesitan observaciones de ambas clases.")

    df = pd.DataFrame({"p": p, "y": y})
    por_valor = df.groupby("p")["y"].agg(pos="sum", n="count").sort_index()
    pos = por_valor["pos"].to_numpy(dtype=np.int64)
    neg = por_valor["n"].to_numpy(dtype=np.int64) - pos
    neg_menores = np.cumsum(neg) - neg  # negativos con probabilidad estrictamente menor

    conc = int(np.sum(pos * neg_menores))
    emp = int(np.sum(pos * neg))
    disc = n1 * n0 - conc - emp
    n = len(y)
    return {
        "c": (conc + 0.5 * emp) / (n1 * n0),
        "Somers_D": (conc - disc) / (n1 * n0),
        "Gamma": (conc - disc) / max(conc + disc, 1),
        "Tau_a": (conc - disc) / max(0.5 * n * (n - 1), 1),
        "concordantes": conc,
        "discordantes": disc,
        "empates": emp,
    }
