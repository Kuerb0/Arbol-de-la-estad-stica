"""Rama PREPROCESADO / binning: variables numéricas -> categóricas por cuantiles.

Origen: `crear_numericas_cat` (notebook de consultoría). Equivale a PROC RANK GROUPS=n de SAS.
Cambios respecto al original: no muta el DataFrame de entrada, no depende de nombres de
negocio (los patrones de "importe" son parámetro) y tolera cuantiles duplicados.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

PATRONES_IMPORTE = ("amount", "importe", "complement", "premium")


def categorizar_por_cuantiles(
    df: pd.DataFrame,
    columnas: Iterable[str],
    q: int = 5,
    q_importes: int = 10,
    patrones_importe: Iterable[str] = PATRONES_IMPORTE,
    sufijo: str = "_cat",
) -> tuple[pd.DataFrame, list[str]]:
    """Crea `<col><sufijo>` (categórica ordenada Q1..Qn o D1..Dn) para cada columna numérica.

    - Columnas cuyo nombre contiene algún patrón de `patrones_importe` usan `q_importes`
      tramos (deciles por defecto) si tienen >= `q_importes` valores distintos; el resto usa `q`.
    - Si hay cuantiles repetidos se descartan tramos (duplicates="drop"); se etiqueta con los
      tramos realmente obtenidos. Si queda < 2 tramos, la columna no se crea.

    Devuelve (df_nuevo, lista_de_columnas_creadas). El DataFrame original no se modifica.
    """
    df_out = df.copy()
    creadas: list[str] = []
    patrones = tuple(p.lower() for p in patrones_importe)

    for col in columnas:
        if col not in df_out.columns:
            continue
        serie = df_out[col]
        n_unicos = serie.dropna().nunique()
        if n_unicos < 2:
            continue

        es_importe = any(p in col.lower() for p in patrones)
        q_local = q_importes if (es_importe and n_unicos >= q_importes) else q

        try:
            codigos = pd.qcut(serie, q=q_local, labels=False, duplicates="drop")
        except ValueError:
            continue
        if codigos.notna().sum() == 0:
            continue

        n_tramos = int(codigos.max()) + 1
        if n_tramos < 2:
            continue

        prefijo = "D" if n_tramos >= 10 else "Q"
        etiquetas = [f"{prefijo}{i + 1}" for i in range(n_tramos)]
        nueva = col + sufijo
        df_out[nueva] = pd.Categorical.from_codes(
            codigos.fillna(-1).astype(int).to_numpy(), categories=etiquetas, ordered=True
        )
        creadas.append(nueva)

    return df_out, creadas
