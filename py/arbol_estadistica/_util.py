"""Utilidades internas compartidas (validación de entradas). No forman parte del árbol público."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _numerico(x, nombre: str = "x", minimo: int = 2) -> np.ndarray:
    """Vector float sin NaN; falla con mensaje claro si queda corto o no es numérico."""
    try:
        a = np.asarray(pd.to_numeric(pd.Series(np.ravel(np.asarray(x, dtype=object))), errors="coerce"), dtype=float)
    except Exception as e:  # pragma: no cover - entradas muy raras
        raise TypeError(f"{nombre}: no es numérico ({e})") from e
    a = a[np.isfinite(a)]
    if len(a) < minimo:
        raise ValueError(f"{nombre}: hacen falta al menos {minimo} valores numéricos válidos (hay {len(a)}).")
    return a


def _columnas(df: pd.DataFrame, columnas) -> None:
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise KeyError(f"Columnas que no están en el DataFrame: {faltan}")


def _rng(semilla):
    return np.random.default_rng(semilla)
