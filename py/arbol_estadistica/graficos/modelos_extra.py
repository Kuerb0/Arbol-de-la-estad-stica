"""Rama GRAFICOS / modelos 0.9: efecto de un spline, curva de validación de la regularización, relatividades de
tarificación e incidencia acumulada con riesgos competitivos."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._base import APAGADO, AZUL, NARANJA, TINTA2, _color, _ejes, _figura, _leyenda


def grafico_efecto_spline(resultado: dict, variable: str, ax=None):
    """Curva del efecto de `variable` (escala del predictor lineal) de modelos.ajustar_glm_splines, con el p de no linealidad."""
    if variable not in resultado.get("curvas", {}):
        raise KeyError(f"«{variable}» no está entre las variables con spline.")
    c = resultado["curvas"][variable]
    fig, ax = _figura(ax)
    ax.plot(c[variable], c["predictor_lineal"], color=AZUL, lw=2.2)
    _ejes(ax, f"Efecto de {variable} (spline)  ·  p no linealidad = {resultado['p_no_linealidad']:.2g}", variable, "predictor lineal")
    return fig


def grafico_regularizacion(resultado: dict, ax=None):
    """Error de validación cruzada frente a λ (escala log) con el λ elegido (salida de modelos.ajustar_regularizado)."""
    c = resultado["curva_cv"]
    col = [k for k in c.columns if k != "lambda"]
    if not col:
        raise ValueError("Esta combinación (ridge gaussiana) no guarda la curva de CV.")
    fig, ax = _figura(ax)
    ax.semilogx(c["lambda"], c[col[0]], "o-", color=AZUL, ms=3)
    ax.axvline(resultado["lambda"], color=NARANJA, lw=1.5, label=f"λ elegido = {resultado['lambda']:.3g} ({resultado['n_seleccionadas']} variables)")
    _ejes(ax, f"{resultado['tipo'].capitalize()}: validación cruzada", "λ (más penalización →)", col[0].replace("_", " "))
    _leyenda(ax)
    return fig


def grafico_relatividades(tabla: pd.DataFrame, titulo: str = "Relatividades", ax=None):
    """Barras de relatividades (exp(β) frente a la base = 1) con IC 95 %, como en Emblem/Radar
    (salida de modelos.tabla_relatividades)."""
    fig, ax = _figura(ax)
    x = np.arange(len(tabla))
    ax.bar(x, tabla["relatividad"], color=[APAGADO if r == 1 else AZUL for r in tabla["relatividad"]], alpha=0.85)
    err = np.vstack([tabla["relatividad"] - tabla["IC_inf"], tabla["IC_sup"] - tabla["relatividad"]])
    ax.errorbar(x, tabla["relatividad"], yerr=err, fmt="none", ecolor=TINTA2, capsize=4, lw=1)
    ax.axhline(1, color=TINTA2, lw=1, ls=":")
    ax.set_xticks(x, [str(i) for i in tabla.index])
    _ejes(ax, titulo, "", "relatividad (base = 1)")
    return fig


def grafico_incidencia_acumulada(tabla: pd.DataFrame, ax=None):
    """Curvas escalonadas de incidencia acumulada por causa (y grupo) de modelos.incidencia_acumulada."""
    fig, ax = _figura(ax)
    causas = [c for c in tabla.columns if c.startswith("cif_causa_")]
    i = 0
    for g, sub in tabla.groupby("grupo", sort=False):
        for c in causas:
            ax.step(sub["tiempo"], sub[c], where="post", color=_color(i % 8), lw=2,
                    label=f"{c.replace('cif_causa_', 'causa ')}" + ("" if g == "todos" else f" · {g}"))
            i += 1
    ax.set_ylim(0, 1)
    _ejes(ax, "Incidencia acumulada (riesgos competitivos)", "tiempo", "probabilidad acumulada")
    _leyenda(ax)
    return fig
