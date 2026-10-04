"""Rama MODELOS / regresión logística "estilo PROC LOGISTIC".

Orígenes (notebook de consultoría): `preparar_df_modelo`, `fit_logit`, `mostrar_odds_ratios`,
bloque E "Parameter Estimates" (Wald chi2) y bloque A "Response Profile".
Equivale a:  PROC LOGISTIC DESCENDING; CLASS var(REF='x') / PARAM=REF; MODEL y(EVENT='1') = ...;
             ODDSRATIO var / UNITS=(u);

Flujo típico:
    X, y, std_map = preparar_matriz_modelo(df, 'y', vars, categoricas, numericas, refs={'combustible': 'Diesel'})
    modelo = ajustar_logit(y, X)
    tabla_odds_ratios(modelo, unidades={'precio': 1000}, std_map=std_map)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2


def preparar_matriz_modelo(
    df: pd.DataFrame,
    objetivo: str,
    variables: list[str],
    categoricas: list[str] | tuple = (),
    numericas: list[str] | tuple = (),
    refs: dict | None = None,
    estandarizar: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Construye (X, y, std_map) como CLASS ... PARAM=REF + z-score de las numéricas.

    - Elimina filas con NaN/inf en objetivo o variables.
    - Cada categórica se convierte en dummies 0/1; la categoría de referencia es `refs[var]`
      si se indica, y si no la primera en orden alfabético (como pd.get_dummies).
    - Las numéricas se estandarizan (media 0, desv. 1) si `estandarizar`; `std_map` guarda su
      desviación ORIGINAL para poder reescalar los odds ratios a unidades interpretables.
    - X NO incluye constante: añadirla con sm.add_constant al ajustar.
    """
    refs = refs or {}
    d = df[[objetivo] + list(variables)].replace([np.inf, -np.inf], np.nan).dropna().copy()

    for cv in categoricas:
        dummies = pd.get_dummies(d[cv], prefix=cv).astype(int)
        ref_col = f"{cv}_{refs[cv]}" if cv in refs else dummies.columns[0]
        if ref_col not in dummies.columns:
            raise ValueError(f"Referencia {refs.get(cv)!r} no existe en '{cv}'.")
        d = pd.concat([d.drop(columns=[cv]), dummies.drop(columns=[ref_col])], axis=1)

    y = d[objetivo]
    X = d.drop(columns=[objetivo]).astype(float)

    std_map: dict = {}
    for col in numericas:
        if col in X.columns:
            std_map[col] = float(X[col].std())
            if estandarizar and std_map[col] > 0:
                X[col] = (X[col] - X[col].mean()) / std_map[col]
    return X, y, std_map


def ajustar_logit(y, X, pesos=None, metodo: str = "newton", constante: bool = True):
    """Ajusta una logística binaria. Con `pesos` (frecuencias) usa GLM Binomial.

    ATENCIÓN: sm.Logit NO admite freq_weights (los ignora con un aviso). Por eso, con pesos se
    usa sm.GLM(..., freq_weights=...). El notebook de consultoría original pasaba freq_weights a sm.Logit.
    """
    Xc = sm.add_constant(X, has_constant="add") if constante else X
    if pesos is not None:
        w = pd.Series(pesos).reindex(y.index).fillna(1.0)
        return sm.GLM(y, Xc, family=sm.families.Binomial(), freq_weights=w).fit()
    return sm.Logit(y, Xc).fit(method=metodo, disp=False, maxiter=200)


def _estrellas(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def tabla_odds_ratios(modelo, unidades: dict | None = None, std_map: dict | None = None) -> pd.DataFrame:
    """Odds ratios con IC 95 % y p-valor, equivalente a ODDSRATIO ... UNITS= de SAS.

    Para una variable numérica estandarizada con desviación s y unidad u:
        OR = exp(beta * u / s)   (efecto de aumentar la variable ORIGINAL en `u` unidades)
    `unidades` = {var: u}; `std_map` = desviaciones originales (de preparar_matriz_modelo).
    Las dummies llevan OR directo frente a su categoría de referencia.
    """
    unidades, std_map = unidades or {}, std_map or {}
    params = modelo.params.drop("const", errors="ignore")
    conf = modelo.conf_int().drop("const", errors="ignore")
    pv = modelo.pvalues.drop("const", errors="ignore")
    filas = []
    for v in params.index:
        f = unidades.get(v, 1) / std_map.get(v, 1)
        filas.append({
            "variable": v,
            "OR": float(np.exp(params[v] * f)),
            "IC_2.5%": float(np.exp(conf.loc[v].iloc[0] * f)),
            "IC_97.5%": float(np.exp(conf.loc[v].iloc[1] * f)),
            "p_valor": float(pv[v]),
            "sig": _estrellas(float(pv[v])),
        })
    return pd.DataFrame(filas).set_index("variable")


def tabla_parametros_wald(modelo) -> pd.DataFrame:
    """'Analysis of Maximum Likelihood Estimates' de SAS: estimación, SE, Wald chi2 (gl=1), Pr>chi2, exp(B)."""
    b, se = modelo.params, modelo.bse
    wald = (b / se) ** 2
    p = pd.Series(chi2.sf(wald, 1), index=b.index)
    return pd.DataFrame({
        "estimacion": b, "SE": se, "Wald_chi2": wald, "Pr>chi2": p,
        "sig": [_estrellas(x) for x in p], "exp(B)": np.exp(b),
    })


def perfil_respuesta(y, etiquetas: dict | None = None) -> pd.DataFrame:
    """'Response Profile' de SAS: frecuencia y proporción de cada nivel del objetivo."""
    t = y.value_counts().sort_index().rename("frecuencia").to_frame()
    t["proporcion"] = t["frecuencia"] / t["frecuencia"].sum()
    if etiquetas:
        t.index = [etiquetas.get(i, i) for i in t.index]
    return t
