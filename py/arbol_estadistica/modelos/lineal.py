"""Regresión lineal completa: MCO con errores robustos, diagnósticos, influencia, F parcial, Box-Cox, MCP,
regresión robusta / cuantílica, no lineal y falta de ajuste (PROC REG, ROBUSTREG, QUANTREG, NLIN, TRANSREG).

Origen: nuevo (temario de Modelos Lineales / ALSM, Kutner et al.). Regla Gauss: con MCO, errores robustos HC3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize, stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from .._util import _columnas, _numerico


def ajustar_ols(formula: str, df: pd.DataFrame, robusto: str | None = "HC3", alpha: float = 0.05) -> dict:
    """MCO con fórmula y errores estándar robustos a heterocedasticidad (HC3 por defecto; None = clásicos).

    Devuelve modelo, tabla (coef, SE, t, p, IC, coeficiente estandarizado β·sx/sy), R², R² ajustado, F global,
    sigma, AIC/BIC y diagnósticos: Breusch-Pagan (heterocedasticidad), Durbin-Watson (autocorrelación, ≈2 = nada),
    Jarque-Bera (normalidad de residuos) y número de condición (colinealidad > 30).
    Equivale a PROC REG (/ STB CLB HCC SPEC DW) o PROC GLM.
    """
    res_clasico = smf.ols(formula, df).fit()
    res = res_clasico.get_robustcov_results(robusto) if robusto else res_clasico
    nombres = res_clasico.params.index
    ci = np.asarray(res.conf_int(alpha))
    X = res_clasico.model.exog
    y = res_clasico.model.endog
    sx = X.std(axis=0, ddof=1); sy = y.std(ddof=1)
    tabla = pd.DataFrame({"coef": np.asarray(res.params), "SE": np.asarray(res.bse), "t": np.asarray(res.tvalues),
                          "p_valor": np.asarray(res.pvalues), "IC_inf": ci[:, 0], "IC_sup": ci[:, 1],
                          "beta_estandarizado": np.where(sx > 0, np.asarray(res.params) * sx / sy, np.nan)}, index=nombres)
    bp = het_breuschpagan(res_clasico.resid, X)
    jb = jarque_bera(res_clasico.resid)
    diag = {"breusch_pagan_p": float(bp[1]), "durbin_watson": float(durbin_watson(res_clasico.resid)),
            "jarque_bera_p": float(jb[1]), "numero_condicion": float(np.linalg.cond(X))}
    avisos = []
    if diag["breusch_pagan_p"] < 0.05:
        avisos.append("Heterocedasticidad (Breusch-Pagan): usa los SE robustos (ya aplicados si robusto='HC3') o MCP.")
    if not 1.5 < diag["durbin_watson"] < 2.5:
        avisos.append("Durbin-Watson lejos de 2: residuos autocorrelacionados (datos en el tiempo).")
    if diag["numero_condicion"] > 30 * (1e3 if np.abs(X).max() > 1e3 else 1):
        avisos.append("Número de condición alto: colinealidad o escalas muy distintas (mira calcular_vif).")
    return {"modelo": res_clasico, "modelo_robusto": res if robusto else None, "tabla": tabla,
            "r2": float(res_clasico.rsquared), "r2_ajustado": float(res_clasico.rsquared_adj),
            "f": float(res.fvalue) if np.ndim(res.fvalue) == 0 else float(np.ravel(res.fvalue)[0]),
            "p_f": float(res.f_pvalue), "sigma": float(np.sqrt(res_clasico.scale)), "aic": float(res_clasico.aic),
            "bic": float(res_clasico.bic), "diagnosticos": diag, "avisos": avisos, "robusto": robusto}


def medidas_influencia(modelo, umbrales: bool = True) -> pd.DataFrame:
    """Por observación: apalancamiento h_ii, residuo estandarizado y estudentizado (externo), distancia de Cook,
    DFFITS y COVRATIO, con marcas según los cortes clásicos (h > 2p/n, |r*| > 3, Cook > 4/n, |DFFITS| > 2√(p/n)).
    `modelo`: resultado de statsmodels OLS (o el dict de ajustar_ols). Equivale a PROC REG / INFLUENCE R."""
    res = modelo["modelo"] if isinstance(modelo, dict) else modelo
    inf = res.get_influence()
    n, p = res.model.exog.shape
    h, r = inf.hat_matrix_diag, inf.resid_studentized_internal
    # fórmulas cerradas (sin reajustar n veces: statsmodels lo hace en O(n²) para el estudentizado externo)
    t_ext = r * np.sqrt((n - p - 1) / np.maximum(n - p - r ** 2, 1e-12))
    t = pd.DataFrame({"apalancamiento": h, "residuo_estandarizado": r, "residuo_estudentizado": t_ext,
                      "cook": r ** 2 * h / (p * (1 - h)), "dffits": t_ext * np.sqrt(h / (1 - h)),
                      "covratio": 1 / ((1 - h) * ((n - p - 1 + t_ext ** 2) / (n - p)) ** p)})
    if umbrales:
        t["alto_apalancamiento"] = t.apalancamiento > 2 * p / n
        t["atipico_y"] = t.residuo_estudentizado.abs() > 3
        t["influyente"] = (t.cook > 4 / n) | (t.dffits.abs() > 2 * np.sqrt(p / n))
    return t


def contraste_f_parcial(modelo_reducido, modelo_completo) -> dict:
    """F parcial (sumas de cuadrados extra): ¿aporta el bloque de variables del modelo completo?
    F = [(SCE_r − SCE_c)/(gl_r − gl_c)] / (SCE_c/gl_c). Los modelos deben estar anidados y con las mismas filas.
    Equivale a TEST en PROC REG o a anova(m1, m2) en R."""
    r = modelo_reducido["modelo"] if isinstance(modelo_reducido, dict) else modelo_reducido
    c = modelo_completo["modelo"] if isinstance(modelo_completo, dict) else modelo_completo
    if r.nobs != c.nobs:
        raise ValueError("Los modelos no tienen las mismas observaciones.")
    q = r.df_resid - c.df_resid
    if q <= 0:
        raise ValueError("El modelo completo debe tener más parámetros que el reducido.")
    f = ((r.ssr - c.ssr) / q) / (c.ssr / c.df_resid)
    return {"F": float(f), "gl_num": int(q), "gl_den": int(c.df_resid), "p_valor": float(stats.f.sf(f, q, c.df_resid)),
            "r2_parcial": float((r.ssr - c.ssr) / r.ssr)}


def transformacion_box_cox(y, nivel: float = 0.95) -> dict:
    """λ de Box-Cox por máxima verosimilitud con IC (perfil), la variable transformada y una lectura del λ
    (≈1 nada, ≈0.5 raíz, ≈0 log, ≈−1 inversa). Si hay valores ≤ 0 usa Yeo-Johnson. Equivale a PROC TRANSREG BOXCOX."""
    v = _numerico(y, "y", 5)
    if (v > 0).all():
        yt, lam, ic = stats.boxcox(v, alpha=1 - nivel)
        metodo = "box-cox"
    else:
        yt, lam = stats.yeojohnson(v); ic = (np.nan, np.nan); metodo = "yeo-johnson"
    cerca = min({1: "sin transformar", 0.5: "raíz cuadrada", 0: "logaritmo", -1: "inversa"}.items(), key=lambda kv: abs(kv[0] - lam))
    return {"lambda": float(lam), "ic": tuple(map(float, ic)), "metodo": metodo, "transformada": yt,
            "sugerencia": cerca[1] if (np.isnan(ic[0]) or ic[0] <= cerca[0] <= ic[1]) else f"λ = {lam:.2f}"}


def ajustar_wls(formula: str, df: pd.DataFrame, pesos: str | None = None, estimar_pesos: bool = False) -> dict:
    """Mínimos cuadrados ponderados. Con `pesos` (columna) los usa tal cual (p. ej. exposición o 1/varianza);
    con `estimar_pesos=True` hace MCP factible: regresa log(residuo²) sobre las X y usa w = 1/σ̂²."""
    base = smf.ols(formula, df).fit()
    if pesos:
        _columnas(df, [pesos]); w = df.loc[base.model.data.row_labels, pesos].to_numpy(float)
    elif estimar_pesos:
        aux = sm.OLS(np.log(base.resid ** 2 + 1e-12), base.model.exog).fit()
        w = 1 / np.exp(aux.fittedvalues)
    else:
        raise ValueError("Indica `pesos` o estimar_pesos=True.")
    res = smf.wls(formula, df.loc[base.model.data.row_labels], weights=w).fit()
    return {"modelo": res, "tabla": res.summary2().tables[1], "r2": float(res.rsquared), "pesos": w}


def regresion_robusta(formula: str, df: pd.DataFrame, metodo: str = "huber", cuantil: float = 0.5) -> dict:
    """Regresión resistente a atípicos en Y.
    metodo: 'huber' (M-estimador de Huber, IRLS), 'bisquare' (Tukey, más resistente) o 'cuantil' (LAD si
    cuantil = 0.5; otros cuantiles modelan la cola: p. ej. 0.9 = el 10 % más caro).
    Devuelve modelo, tabla y los pesos finales (observaciones con peso bajo = atípicas). Equivale a PROC ROBUSTREG /
    PROC QUANTREG. Gauss: protege frente a atípicos en Y, no frente a puntos de alto apalancamiento en X."""
    if metodo in ("huber", "bisquare"):
        norma = sm.robust.norms.HuberT() if metodo == "huber" else sm.robust.norms.TukeyBiweight()
        res = smf.rlm(formula, df, M=norma).fit()
        pesos = res.weights
    elif metodo == "cuantil":
        res = smf.quantreg(formula, df).fit(q=cuantil); pesos = None
    else:
        raise ValueError("metodo: 'huber', 'bisquare' o 'cuantil'")
    ci = res.conf_int()
    tabla = pd.DataFrame({"coef": res.params, "SE": res.bse, "p_valor": res.pvalues, "IC_inf": ci[0], "IC_sup": ci[1]})
    return {"modelo": res, "tabla": tabla, "pesos": pesos, "metodo": metodo}


def regresion_no_lineal(funcion, x, y, p0, nombres=None, nivel: float = 0.95) -> dict:
    """Mínimos cuadrados no lineales y = f(x, θ) + ε (scipy curve_fit, Levenberg-Marquardt) con SE, IC de Wald y R².
    `funcion(x, *theta)`. Útil para curvas de desarrollo, leyes de mortalidad o saturación. Equivale a PROC NLIN.
    Gauss: depende del punto inicial p0; prueba varios si no converge. Los IC son asintóticos."""
    x = np.asarray(x, float); y = _numerico(y, "y", len(p0) + 2)
    theta, cov = optimize.curve_fit(funcion, x, y, p0=p0, maxfev=20000)
    se = np.sqrt(np.diag(cov)); gl = len(y) - len(theta)
    tc = stats.t.ppf(0.5 + nivel / 2, gl)
    pred = funcion(x, *theta); ssr = float(np.sum((y - pred) ** 2))
    nombres = list(nombres) if nombres else [f"theta{i}" for i in range(len(theta))]
    tabla = pd.DataFrame({"estimacion": theta, "SE": se, "IC_inf": theta - tc * se, "IC_sup": theta + tc * se}, index=nombres)
    return {"tabla": tabla, "parametros": theta, "covarianza": cov, "sigma": float(np.sqrt(ssr / gl)),
            "r2": float(1 - ssr / np.sum((y - y.mean()) ** 2)), "predichos": pred}


def contraste_falta_ajuste(df: pd.DataFrame, x: str, y: str, grado: int = 1) -> dict:
    """Test F de falta de ajuste de un polinomio de grado `grado` cuando hay RÉPLICAS (varios y para el mismo x):
    separa el error puro (dentro de cada x) de la falta de ajuste. p pequeño = la forma del modelo es incorrecta.
    Equivale a PROC REG / LACKFIT."""
    _columnas(df, [x, y])
    d = df[[x, y]].dropna()
    X = np.vander(d[x].to_numpy(float), grado + 1)
    res = sm.OLS(d[y].to_numpy(float), X).fit()
    g = d.groupby(x)[y]
    sc_puro = float(((d[y] - g.transform("mean")) ** 2).sum())
    c = g.ngroups; n = len(d); p = grado + 1
    if c <= p or n <= c:
        raise ValueError("Hacen falta réplicas y más niveles de x que parámetros.")
    sc_fa = float(res.ssr - sc_puro)
    f = (sc_fa / (c - p)) / (sc_puro / (n - c))
    return {"F": f, "gl_num": c - p, "gl_den": n - c, "p_valor": float(stats.f.sf(f, c - p, n - c)),
            "sc_falta_ajuste": sc_fa, "sc_error_puro": sc_puro}
