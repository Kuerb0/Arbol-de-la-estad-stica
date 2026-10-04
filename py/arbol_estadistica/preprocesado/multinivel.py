"""Centrado de variables para modelos multinivel."""
from __future__ import annotations

import pandas as pd

from .._util import _columnas


def centrar_por_grupo(df: pd.DataFrame, variables, grupo: str, sufijos=("_dentro", "_media_grupo", "_gran_media")) -> pd.DataFrame:
    """Descompone cada variable de nivel 1 en la parte DENTRO del grupo (x − media del grupo, CWC) y la media del grupo
    (efecto contextual), además del centrado en la gran media (CGM). En un modelo mixto, meter x_dentro y
    x_media_grupo separa el efecto individual del efecto entre grupos, que con x sin centrar quedan mezclados
    (el coeficiente de x sería una media ponderada difícil de interpretar). Devuelve una copia con las columnas nuevas."""
    variables = [variables] if isinstance(variables, str) else list(variables)
    _columnas(df, variables + [grupo])
    d = df.copy()
    for v in variables:
        m = d.groupby(grupo)[v].transform("mean")
        d[v + sufijos[0]] = d[v] - m
        d[v + sufijos[1]] = m
        d[v + sufijos[2]] = d[v] - d[v].mean()
    return d
