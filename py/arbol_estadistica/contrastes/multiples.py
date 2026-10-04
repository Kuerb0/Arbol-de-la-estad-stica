"""Rama CONTRASTES / comparaciones múltiples: FDR (Benjamini-Hochberg), Bonferroni y Holm.

Origen: hueco del catálogo (Very Normal, «Statistics' Million Dollar Idea (False Discovery Rate)»);
no venía de consultoría. Equivale a PROC MULTTEST (opciones FDR, BONFERRONI, HOLM) de SAS.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

_METODOS = {"fdr_bh": "fdr_bh", "bh": "fdr_bh", "fdr_by": "fdr_by", "bonferroni": "bonferroni", "holm": "holm"}


def ajustar_p_valores(p_valores, metodo: str = "fdr_bh", alpha: float = 0.05,
                      nombres: Sequence[str] | None = None) -> pd.DataFrame:
    """Ajusta una lista de p-valores por comparaciones múltiples y dice cuáles se rechazan.

    metodo : 'fdr_bh' (Benjamini-Hochberg, controla la proporción esperada de falsos
             descubrimientos), 'fdr_by' (BH válido con cualquier dependencia, más conservador),
             'holm' o 'bonferroni' (controlan la probabilidad de AL MENOS un falso positivo, FWER).
    Devuelve un DataFrame (en el orden de entrada) con p_valor, rango (1 = el menor),
    umbral_bh (= rango/m * alpha, la recta de BH), p_ajustado y rechaza.

    Gauss: con 20 contrastes a alpha=0.05 y todos nulos, sin corregir sale ~1 "significativo" por azar.
    FDR es lo razonable para cribar muchas variables o segmentos (se tolera algún falso positivo);
    Bonferroni/Holm cuando un solo falso positivo es caro. BH asume independencia o dependencia
    positiva entre contrastes; si no, usar 'fdr_by'.
    """
    p = np.asarray(p_valores, dtype=float).ravel()
    if p.size == 0:
        raise ValueError("No hay p-valores.")
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("Los p-valores deben estar en [0, 1] y sin NaN.")
    if metodo not in _METODOS:
        raise ValueError(f"metodo debe ser uno de {sorted(_METODOS)}")
    rechaza, p_aj, _, _ = multipletests(p, alpha=alpha, method=_METODOS[metodo])
    m = p.size
    rango = pd.Series(p).rank(method="first").astype(int).to_numpy()
    out = pd.DataFrame({
        "p_valor": p, "rango": rango, "umbral_bh": rango / m * alpha,
        "p_ajustado": p_aj, "rechaza": rechaza.astype(bool),
    }, index=list(nombres) if nombres is not None else None)
    out.attrs.update(metodo=_METODOS[metodo], alpha=alpha,
                     n_rechazos=int(rechaza.sum()), n_sin_corregir=int((p < alpha).sum()))
    return out
