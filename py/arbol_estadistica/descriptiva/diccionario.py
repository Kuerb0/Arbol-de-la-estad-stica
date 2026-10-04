"""Diccionario de datos automático."""
from __future__ import annotations

import numpy as np
import pandas as pd


def diccionario_datos(df: pd.DataFrame, descripciones: dict | None = None, max_ejemplos: int = 3) -> pd.DataFrame:
    """Diccionario (catálogo) de un DataFrame: tipo, nº de valores distintos, % de faltantes, mínimo/máximo, ejemplos,
    memoria y un ROL sugerido (identificador, constante, binaria, categórica, categórica de alta cardinalidad, fecha,
    numérica, texto libre, posible fuga si el nombre empieza por y_/prob_/pred). Añade tus `descripciones`
    {columna: texto}. Punto de partida para documentar un dataset y detectar problemas antes de modelar."""
    filas = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        nd = s.nunique(dropna=True)
        num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        fecha = pd.api.types.is_datetime64_any_dtype(s)
        if nd <= 1:
            rol = "constante"
        elif nd == n and not num:
            rol = "identificador"
        elif num and nd == n and s.notna().all() and np.array_equal(np.sort(s.to_numpy()), np.arange(s.min(), s.min() + n)):
            rol = "identificador"
        elif nd == 2:
            rol = "binaria"
        elif fecha:
            rol = "fecha"
        elif num:
            rol = "numérica discreta" if nd <= 20 and np.allclose(s.dropna() % 1, 0) else "numérica"
        elif s.astype(str).str.len().mean() > 40:
            rol = "texto libre"
        else:
            rol = "categórica" if nd <= 50 else "categórica alta cardinalidad"
        nombre = str(c).lower()
        if nombre.startswith(("y_", "prob_", "pred_")) or nombre.endswith("_pred"):
            rol += " · ¡posible fuga!"
        ej = ", ".join(map(str, s.dropna().unique()[:max_ejemplos]))
        filas.append({"columna": c, "tipo": str(s.dtype), "rol_sugerido": rol, "distintos": nd, "pct_faltantes": 100 * s.isna().mean(),
                      "minimo": s.min() if (num or fecha) and s.notna().any() else None,
                      "maximo": s.max() if (num or fecha) and s.notna().any() else None,
                      "ejemplos": ej, "memoria_kb": s.memory_usage(deep=True) / 1024,
                      "descripcion": (descripciones or {}).get(c, "")})
    return pd.DataFrame(filas).set_index("columna")
