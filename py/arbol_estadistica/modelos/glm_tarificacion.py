"""GLM de tarificación: frecuencia (Poisson / binomial negativa con exposición), severidad (Gamma / inversa
gaussiana), Tweedie y prima pura = frecuencia × severidad (PROC GENMOD; Emblem/Radar en seguros).

Origen: ejercicios 5.2 y 5.3 de formación (frecuencia y severidad) y el temario de No Vida.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .._util import _columnas

_FAMILIAS_CONTEO = {"poisson": lambda a: sm.families.Poisson(),
                    "negbin": lambda a: sm.families.NegativeBinomial(alpha=a)}


def _tabla_exp(res, alpha=0.05, nombre="relatividad"):
    ci = res.conf_int(alpha)
    t = pd.DataFrame({"coef": res.params, "SE": res.bse, "p_valor": res.pvalues,
                      nombre: np.exp(res.params), "IC_inf": np.exp(ci[0]), "IC_sup": np.exp(ci[1])})
    t.index.name = "termino"
    return t


def ajustar_glm_conteo(formula: str, df: pd.DataFrame, familia: str = "poisson", exposicion: str | None = None,
                       alpha_nb: float | None = None) -> dict:
    """GLM de frecuencia (nº de siniestros) con enlace log y offset log(exposición).

    familia: 'poisson' o 'negbin' (binomial negativa NB2; si `alpha_nb` es None se estima por máxima verosimilitud
    con smf.negativebinomial y se reutiliza en el GLM). Devuelve modelo, tabla de relatividades exp(β) con IC
    (efecto multiplicativo sobre la FRECUENCIA), AIC, deviance, dispersión de Pearson (χ²/gl) y contraste de
    sobredispersión de Cameron-Trivedi (si la familia es Poisson).
    Equivale a PROC GENMOD DIST=POISSON|NEGBIN LINK=LOG OFFSET=log_expo.
    Gauss: dispersión ≫ 1 con Poisson = errores estándar demasiado pequeños (falsas significaciones): pasa a
    binomial negativa o usa errores cuasi-Poisson. Sin offset, una póliza de 1 mes cuenta igual que una de 1 año.
    """
    _columnas(df, [formula.split("~")[0].strip()] + ([exposicion] if exposicion else []))
    d = df.copy()
    offset = None
    if exposicion:
        if (d[exposicion] <= 0).any():
            raise ValueError("La exposición debe ser > 0.")
        offset = np.log(d[exposicion].to_numpy(float))
    if familia not in _FAMILIAS_CONTEO:
        raise ValueError("familia: 'poisson' o 'negbin'")
    if familia == "negbin" and alpha_nb is None:
        nb = smf.negativebinomial(formula, d, offset=offset).fit(disp=False, maxiter=200)
        alpha_nb = float(nb.params["alpha"])
    res = smf.glm(formula, d, family=_FAMILIAS_CONTEO[familia](alpha_nb or 1.0), offset=offset).fit()
    disp = float(res.pearson_chi2 / res.df_resid)
    out = {"modelo": res, "tabla": _tabla_exp(res), "aic": float(res.aic), "deviance": float(res.deviance),
           "gl_resid": float(res.df_resid), "dispersion": disp, "familia": familia, "alpha_nb": alpha_nb}
    if familia == "poisson":
        out["sobredispersion"] = contraste_sobredispersion(res)
    return out


def contraste_sobredispersion(modelo_poisson) -> dict:
    """Contraste de Cameron-Trivedi para un GLM Poisson: regresión auxiliar ((y−μ)² − y)/μ = α·μ + ε.
    H0: α = 0 (equidispersión). α > 0 significativo -> sobredispersión (usa binomial negativa)."""
    mu = np.asarray(modelo_poisson.fittedvalues, float)
    y = np.asarray(modelo_poisson.model.endog, float)
    aux = ((y - mu) ** 2 - y) / mu
    r = sm.OLS(aux, mu).fit()
    return {"alpha": float(r.params[0]), "t": float(r.tvalues[0]), "p_valor": float(stats.t.sf(r.tvalues[0], r.df_resid)),
            "dispersion_pearson": float(np.sum((y - mu) ** 2 / mu) / modelo_poisson.df_resid)}


def ajustar_glm_severidad(formula: str, df: pd.DataFrame, familia: str = "gamma", pesos: str | None = None) -> dict:
    """GLM de severidad (coste medio por siniestro) con enlace log: Gamma (varianza ∝ μ²) o inversa gaussiana
    (varianza ∝ μ³, colas más pesadas). `pesos` = nº de siniestros si la fila es un coste MEDIO de varios.
    Equivale a PROC GENMOD DIST=GAMMA|IGAUSSIAN LINK=LOG (WEIGHT=). Solo filas con importe > 0."""
    yname = formula.split("~")[0].strip()
    _columnas(df, [yname] + ([pesos] if pesos else []))
    d = df[df[yname] > 0].copy()
    if len(d) < 10:
        raise ValueError("Hacen falta al menos 10 importes positivos.")
    fam = {"gamma": sm.families.Gamma(sm.families.links.Log()),
           "inversa_gaussiana": sm.families.InverseGaussian(sm.families.links.Log())}.get(familia)
    if fam is None:
        raise ValueError("familia: 'gamma' o 'inversa_gaussiana'")
    res = smf.glm(formula, d, family=fam, var_weights=d[pesos] if pesos else None).fit(scale="X2")
    return {"modelo": res, "tabla": _tabla_exp(res), "aic": float(res.aic), "dispersion": float(res.scale), "familia": familia}


def ajustar_tweedie(formula: str, df: pd.DataFrame, potencia_var: float = 1.5, exposicion: str | None = None) -> dict:
    """GLM Tweedie (Poisson compuesta Gamma, 1 < p < 2) para la prima pura directamente (muchos ceros + importes).
    `potencia_var` = p de Var = φ·μ^p. Con offset log(exposición). Equivale a PROC GENMOD DIST=TWEEDIE."""
    d = df.copy()
    offset = np.log(d[exposicion].to_numpy(float)) if exposicion else None
    res = smf.glm(formula, d, family=sm.families.Tweedie(var_power=potencia_var, link=sm.families.links.Log()),
                  offset=offset).fit(scale="X2")
    return {"modelo": res, "tabla": _tabla_exp(res), "dispersion": float(res.scale), "potencia_var": potencia_var}


def prima_pura(modelo_frecuencia, modelo_severidad, df: pd.DataFrame, exposicion: str | None = None) -> pd.DataFrame:
    """Prima pura = frecuencia esperada × severidad esperada para cada fila de `df` (la exposición escala la
    frecuencia; sin ella, prima por unidad de exposición). Acepta los dict de ajustar_glm_* o los modelos.
    Devuelve frecuencia, severidad y prima_pura. Gauss: supone frecuencia y severidad independientes dadas las X."""
    mf = modelo_frecuencia["modelo"] if isinstance(modelo_frecuencia, dict) else modelo_frecuencia
    ms = modelo_severidad["modelo"] if isinstance(modelo_severidad, dict) else modelo_severidad
    expo = df[exposicion].to_numpy(float) if exposicion else np.ones(len(df))
    freq = np.asarray(mf.predict(df, offset=np.log(expo)))
    sev = np.asarray(ms.predict(df))
    return pd.DataFrame({"frecuencia": freq, "severidad": sev, "prima_pura": freq * sev}, index=df.index)


def tabla_relatividades(modelo, variable: str) -> pd.DataFrame:
    """Relatividades de una variable categórica (exp(β) frente al nivel base = 1), como las tablas de Emblem/Radar.
    `modelo`: dict de ajustar_glm_* o resultado de statsmodels con fórmula (C(variable) o variable categórica)."""
    res = modelo["modelo"] if isinstance(modelo, dict) else modelo
    t = _tabla_exp(res)
    filas = [i for i in t.index if i.startswith((f"C({variable})[", f"{variable}[")) or i.startswith(f"C({variable}, ")]
    if not filas:
        raise KeyError(f"No hay términos categóricos de «{variable}» en el modelo.")
    datos = res.model.data.frame
    niveles = sorted(datos[variable].dropna().unique(), key=str)
    nivel_de = {f: f.split("[T.")[-1].rstrip("]") for f in filas}
    base = [n for n in niveles if str(n) not in nivel_de.values()]
    out = [{"nivel": base[0] if base else "base", "relatividad": 1.0, "IC_inf": 1.0, "IC_sup": 1.0, "p_valor": np.nan}]
    out += [{"nivel": nivel_de[f], "relatividad": t.loc[f, "relatividad"], "IC_inf": t.loc[f, "IC_inf"],
             "IC_sup": t.loc[f, "IC_sup"], "p_valor": t.loc[f, "p_valor"]} for f in filas]
    return pd.DataFrame(out).set_index("nivel")
