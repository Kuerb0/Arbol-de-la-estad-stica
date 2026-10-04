"""Rama CONTRASTES / comparaciones múltiples post-hoc.

Origen: bloque Tukey HSD de `notebook de consultoría` (comparar medias de las variables entre clusters).
Equivale a MEANS grupo / TUKEY de PROC ANOVA o LSMEANS / ADJUST=TUKEY de PROC GLM.
"""
from __future__ import annotations

import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def tukey_entre_grupos(df: pd.DataFrame, valor: str, grupo: str, alpha: float = 0.05) -> pd.DataFrame:
    """Tukey HSD por pares: diferencia de medias, IC ajustado y si se rechaza igualdad.

    Supuestos: normalidad aproximada y varianzas similares (si no, ver elegir_contraste).
    Columnas: grupo1, grupo2, diferencia_medias (grupo2 - grupo1), p_ajustado, ic_inf, ic_sup, rechaza.
    """
    d = df[[valor, grupo]].dropna()
    res = pairwise_tukeyhsd(endog=d[valor].to_numpy(dtype=float), groups=d[grupo].astype(str).to_numpy(), alpha=alpha)
    filas = res._results_table.data
    t = pd.DataFrame(filas[1:], columns=filas[0])
    return t.rename(columns={"meandiff": "diferencia_medias", "p-adj": "p_ajustado",
                             "lower": "ic_inf", "upper": "ic_sup", "reject": "rechaza"})
