"""Rama MODELOS / logit multinomial "estilo PROC LOGISTIC LINK=GLOGIT".

Origen: `multinomial_logit_sas_like` (notebook de consultoría), la función SAS-like que sacaste en consultoría.
Equivale a:  PROC LOGISTIC; CLASS x(REF='a') / PARAM=REF; MODEL y(REF='base') = ... / LINK=GLOGIT;
Notas SAS <-> Python:
  * Referencia del objetivo: aquí `base_class` (si no se indica, la primera categoría que aparece).
    En SAS la referencia por defecto con GLOGIT es la ÚLTIMA categoría ordenada: fíjala siempre.
  * `lsmeans_like` NO es el LSMEANS de SAS (que evalúa en las medias de las covariables): es
    estandarización marginal (se fija el nivel en todas las filas, se predice y se promedia).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from ..diagnostico.colinealidad import calcular_vif
from ..diagnostico.metricas import matrices_confusion, resumen_auc_multiclase


def logit_multinomial_sas(
    data: pd.DataFrame,
    y: str,
    x_num=None,
    x_cat=None,
    base_class=None,
    ref_levels: dict | None = None,
    maxiter: int = 200,
) -> dict:
    """Ajusta un logit multinomial y devuelve un dict con todo lo que da PROC LOGISTIC.

    Parámetros
    ----------
    data       : DataFrame con objetivo y predictores.
    y          : nombre del objetivo multiclase.
    x_num      : numéricas (se ajustan en bruto; los betas estandarizados se calculan aparte).
    x_cat      : categóricas (dummies por patsy, referencia = `ref_levels[var]` o la primera).
    base_class : categoría de referencia del objetivo.
    ref_levels : {'sexo': 'M'} referencias de predictores categóricos.

    Devuelve
    --------
    modelo, coeficientes (beta, se, z, pvalue, beta_std por comparación "clase vs base"),
    probabilidades_predichas, auc (dict), matriz_confusion, matriz_confusion_normalizada_filas,
    vif, marginal_effects_summary, ajuste (n, llf, aic, bic, pseudo_r2_mcfadden),
    lsmeans_like(factor) -> DataFrame con probabilidades medias ajustadas por nivel.
    """
    x_num = list(x_num) if x_num is not None else []
    x_cat = list(dict.fromkeys(x_cat)) if x_cat is not None else []
    ref_levels = ref_levels or {}
    if not x_num and not x_cat:
        raise ValueError("Debes indicar al menos una variable explicativa.")

    df = data[[y] + x_num + x_cat].dropna().copy()

    # 1) Orden del objetivo: la primera categoría es la base
    niveles = list(pd.unique(df[y]))
    if base_class is not None:
        if base_class not in niveles:
            raise ValueError(f"base_class {base_class!r} no está en '{y}'.")
        niveles = [base_class] + [c for c in niveles if c != base_class]
    df[y] = pd.Categorical(df[y], categories=niveles, ordered=False)
    df["_y_code"] = df[y].cat.codes
    categorias = list(df[y].cat.categories)

    # 2) Fórmula de predictores
    def _termino(v):
        return f"C({v}, Treatment(reference={ref_levels[v]!r}))" if v in ref_levels else f"C({v})"

    x_formula = " + ".join(x_num + [_termino(v) for v in x_cat])

    # 3) Ajuste
    X = patsy.dmatrix(x_formula, data=df, return_type="dataframe")
    diseno = X.design_info
    res = sm.MNLogit(df["_y_code"], X).fit(method="newton", maxiter=maxiter, disp=False)

    eq_map = {i: f"{categorias[i + 1]} vs {categorias[0]}" for i in range(len(categorias) - 1)}

    def _tabla(r, nombre_beta="beta"):
        b = r.params.rename(columns=eq_map)
        se = r.bse.rename(columns=eq_map)
        z = r.tvalues.rename(columns=eq_map)
        p = r.pvalues.rename(columns=eq_map)
        out = (b.stack().rename(nombre_beta).to_frame()
               .join(se.stack().rename("se")).join(z.stack().rename("z"))
               .join(p.stack().rename("pvalue")).reset_index()
               .rename(columns={"level_0": "variable", "level_1": "comparacion"}))
        return out[["comparacion", "variable", nombre_beta, "se", "z", "pvalue"]]

    coef_raw = _tabla(res)

    # 4) Betas estandarizados (solo numéricas)
    df_std = df.copy()
    if x_num:
        df_std[x_num] = StandardScaler().fit_transform(df_std[x_num])
    X_std = patsy.dmatrix(x_formula, data=df_std, return_type="dataframe")
    res_std = sm.MNLogit(df_std["_y_code"], X_std).fit(method="newton", maxiter=maxiter, disp=False)
    coef_std = _tabla(res_std, "beta_std")[["comparacion", "variable", "beta_std"]]
    coeficientes = (coef_raw.merge(coef_std, on=["comparacion", "variable"], how="left")
                    .sort_values(["comparacion", "pvalue", "variable"]).reset_index(drop=True))

    # 5) Probabilidades y clasificación
    probs_arr = np.asarray(res.predict(X))
    probs = pd.DataFrame(probs_arr, columns=categorias, index=df.index)
    y_pred = pd.Categorical.from_codes(probs_arr.argmax(axis=1), categories=categorias)
    cm, cm_fila = matrices_confusion(df[y], y_pred, categorias)

    # 6) AUC, 7) VIF, 8) efectos marginales
    auc = resumen_auc_multiclase(df["_y_code"], probs_arr, categorias)
    vif = calcular_vif(X)
    try:
        mfx_text = res.get_margeff(at="overall", method="dydx").summary().as_text()
    except Exception as e:  # noqa: BLE001
        mfx_text = f"No se pudieron calcular marginal effects: {e}"

    # 9) lsmeans_like
    def lsmeans_like(factor: str) -> pd.DataFrame:
        """Probabilidades medias ajustadas por nivel de `factor` (estandarización marginal)."""
        if factor not in x_cat:
            raise ValueError(f"{factor} no está en x_cat.")
        filas = []
        for lev in pd.unique(df[factor]):
            tmp = df.copy()
            tmp[factor] = lev
            Xn = patsy.build_design_matrices([diseno], tmp, return_type="dataframe")[0]
            p = pd.DataFrame(np.asarray(res.predict(Xn)), columns=categorias).mean(axis=0)
            filas.append({"factor": factor, "nivel": lev, **p.to_dict()})
        return pd.DataFrame(filas)

    ajuste = {"n": int(res.nobs), "llf": float(res.llf), "aic": float(res.aic),
              "bic": float(res.bic), "pseudo_r2_mcfadden": float(res.prsquared)}

    return {
        "modelo": res,
        "coeficientes": coeficientes,
        "probabilidades_predichas": probs,
        "auc": auc,
        "matriz_confusion": cm,
        "matriz_confusion_normalizada_filas": cm_fila,
        "vif": vif,
        "marginal_effects_summary": mfx_text,
        "ajuste": ajuste,
        "lsmeans_like": lsmeans_like,
    }
