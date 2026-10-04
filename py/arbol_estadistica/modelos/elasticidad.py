"""Rama MODELOS / sensibilidad (elasticidad) al precio por segmento.

Origen: `elasticidad_por_grupo`, `calcular_elasticidad_segmento` y `plot_mpc_polynomial` de
notebook de consultoría.
OJO (Gauss): la pendiente se estima con regresión LINEAL sobre un objetivo binario, es decir un
modelo de probabilidad lineal. Es una medida descriptiva rápida y fácil de explicar a negocio,
pero no acotada a [0,1]; para inferencia usar la logística (logit_sas.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sensibilidad_por_grupo(df: pd.DataFrame, objetivo: str, precio: str, grupo: str,
                           min_n: int = 30, top: int | None = None) -> pd.DataFrame:
    """Pendiente de `objetivo` sobre el precio ESTANDARIZADO dentro de cada grupo.

    Es el cambio en la tasa de `objetivo` por cada desviación típica de precio. Ordenado de más
    negativo (más sensible a subidas de precio) a menos. Grupos con <= min_n filas o precio
    constante se omiten.
    """
    filas = []
    for g, t in df.groupby(grupo):
        if len(t) > min_n and t[precio].std() > 0:
            x = (t[precio] - t[precio].mean()) / t[precio].std()
            filas.append({grupo: g, "n": len(t),
                          "sensibilidad": float(np.polyfit(x, t[objetivo], 1)[0])})
    out = pd.DataFrame(filas, columns=[grupo, "n", "sensibilidad"])
    out = out.sort_values("sensibilidad").reset_index(drop=True)
    return out.head(top) if top else out


def ganancia_por_bajada_precio(df: pd.DataFrame, objetivo: str, precio: str, grupo: str,
                               bajada: float = 0.05, min_n: int = 30) -> pd.DataFrame:
    """Puntos porcentuales de tasa que se ganan al bajar el precio un `bajada` (5 % por defecto).

    ganancia_pp = pendiente_lineal(precio) * (-bajada * precio_medio) * 100, por grupo.
    Es una extrapolación lineal: válida para bajadas pequeñas dentro del rango observado.
    """
    filas = []
    for g, t in df.groupby(grupo):
        if len(t) > min_n and t[precio].std() > 0:
            b = float(np.polyfit(t[precio], t[objetivo], 1)[0])
            filas.append({grupo: g, "n": len(t), "ganancia_pp": b * (-bajada * t[precio].mean()) * 100})
    out = pd.DataFrame(filas, columns=[grupo, "n", "ganancia_pp"])
    return out.sort_values("ganancia_pp", ascending=False).reset_index(drop=True)


def tendencia_polinomica_ponderada(x, y, pesos=None, grado: int = 2):
    """Ajuste polinómico (grado 2 = 'sweet spot') ponderado por volumen.

    Devuelve (coeficientes de mayor a menor grado, función predictora). Los pesos se interpretan
    como sample_weight (varianza inversa); internamente se usa sqrt(peso) para np.polyfit.
    Si grado == 2 y el coeficiente cuadrático es < 0, el óptimo está en x* = -b / (2a).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = None if pesos is None else np.sqrt(np.asarray(pesos, float))
    coef = np.polyfit(x, y, grado, w=w)
    return coef, np.poly1d(coef)
