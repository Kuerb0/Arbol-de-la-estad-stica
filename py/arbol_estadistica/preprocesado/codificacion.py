"""Rama PREPROCESADO / codificación y duraciones.

Origen: `encode_ordered_column` (notebook de consultoría) y `calc_duration` (notebook de consultoría).
"""
from __future__ import annotations

from typing import Hashable, Sequence

import pandas as pd


def codificar_ordinal(
    df: pd.DataFrame,
    columna: str,
    orden: Sequence[Hashable],
    valor_nulo: int = -1,
) -> tuple[pd.DataFrame, dict]:
    """Sustituye una categórica por enteros según `orden` (0 = primer valor de `orden`).

    Categorías observadas pero no previstas en `orden` se añaden al final (ordenadas
    alfabéticamente) para no perder datos. Los nulos reciben `valor_nulo`.
    Devuelve (df_nuevo, mapping) para poder invertir la codificación.
    """
    orden = list(orden)
    observados = df[columna].dropna().unique().tolist()
    extra = sorted(v for v in observados if v not in orden)
    mapping = {v: i for i, v in enumerate(orden + extra)}

    df_out = df.copy()
    df_out[columna] = df_out[columna].map(mapping).fillna(valor_nulo).astype(int)
    return df_out, mapping


def duracion_hasta_evento(
    df: pd.DataFrame,
    id_col: str,
    evento_col: str,
    orden_col: str | None = None,
) -> pd.Series:
    """Nº de periodos con evento == 0 antes del primer evento == 1, por cada id.

    Ejemplo: renewed = [0, 0, 1, 0] -> duración 2. Si nunca hay un 1, cuenta todos los 0
    (observación censurada: no distingue censura de evento; ver teoria/glm.md).
    `orden_col` (p. ej. el año) ordena cada grupo antes de contar.
    """
    datos = df.sort_values([id_col, orden_col]) if orden_col else df

    def _contar(valores: pd.Series) -> int:
        n = 0
        for v in valores.to_numpy():
            if v == 0:
                n += 1
            elif v == 1:
                break
        return n

    return datos.groupby(id_col)[evento_col].agg(_contar).rename("duracion")
