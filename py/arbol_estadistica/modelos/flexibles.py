"""Modelos más flexibles: GLM con splines (tipo GAM), modelos mixtos con ICC, logit ordinal y regularización
(Ridge / Lasso / Elastic Net con validación cruzada). PROC GAMPL/GLIMMIX, MIXED, LOGISTIC (ordinal), GLMSELECT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .._util import _columnas

_FAM = {"binomial": sm.families.Binomial, "poisson": sm.families.Poisson, "gaussiana": sm.families.Gaussian,
        "gamma": lambda: sm.families.Gamma(sm.families.links.Log())}


def ajustar_glm_splines(df: pd.DataFrame, objetivo: str, splines: dict, otras: str = "", familia: str = "binomial",
                        tipo: str = "cr") -> dict:
    """GLM donde las variables de `splines` ({var: grados de libertad}) entran con splines cúbicos RESTRINGIDOS
    ('cr', naturales: lineales en los extremos) o B-splines ('bs'); `otras` = resto de la fórmula ('C(zona) + edad').
    Compara con el modelo lineal en esas variables (test de razón de verosimilitudes de no linealidad) y devuelve
    las curvas parciales (efecto de cada variable en la escala del predictor lineal, resto fijo).
    Es la versión «GLM + splines» de un GAM; equivale a EFFECT spl = SPLINE(x / NATURALCUBIC) en SAS.
    Gauss: más grados de libertad = más flexible y más varianza; 3-5 suele bastar (Harrell).
    """
    _columnas(df, [objetivo] + list(splines))
    fam = _FAM[familia]()
    if tipo not in ("cr", "bs"):
        raise ValueError("tipo: 'cr' (cúbicos restringidos) o 'bs' (B-splines)")
    extra_cr = ", constraints='center'" if tipo == "cr" else ""
    term = " + ".join(f"{tipo}({v}, df={g}{extra_cr})" for v, g in splines.items())
    lin = " + ".join(splines)
    extra = f" + {otras}" if otras else ""
    m_s = smf.glm(f"{objetivo} ~ {term}{extra}", df, family=fam).fit()
    m_l = smf.glm(f"{objetivo} ~ {lin}{extra}", df, family=_FAM[familia]()).fit()
    lr = max(m_l.deviance - m_s.deviance, 0.0) / (m_s.scale if familia in ("gaussiana", "gamma") else 1.0)
    gl = int(m_l.df_resid - m_s.df_resid)
    curvas = {}
    base = df.iloc[[0]].copy()
    for v in splines:
        grid = np.linspace(df[v].quantile(0.01), df[v].quantile(0.99), 100)
        nd = pd.concat([base] * len(grid), ignore_index=True); nd[v] = grid
        for c in df.columns:
            if c not in (v, objetivo) and c in nd and pd.api.types.is_numeric_dtype(df[c]):
                nd[c] = df[c].median()
        eta = m_s.get_prediction(nd).summary_frame(alpha=0.05)
        lin_pred = np.log(eta["mean"]) if familia in ("poisson", "gamma") else (
            np.log(eta["mean"] / (1 - eta["mean"])) if familia == "binomial" else eta["mean"])
        curvas[v] = pd.DataFrame({v: grid, "predictor_lineal": np.asarray(lin_pred), "media": eta["mean"].to_numpy()})
    return {"modelo": m_s, "modelo_lineal": m_l, "aic": float(m_s.aic), "aic_lineal": float(m_l.aic),
            "lr_no_linealidad": float(lr), "gl": gl, "p_no_linealidad": float(stats.chi2.sf(lr, gl)) if gl > 0 else np.nan,
            "curvas": curvas}


def ajustar_modelo_mixto(formula: str, df: pd.DataFrame, grupo: str, pendiente_aleatoria: str | None = None) -> dict:
    """Modelo lineal de efectos mixtos (intercepto aleatorio por `grupo` y, opcionalmente, pendiente aleatoria).
    Devuelve efectos fijos, varianzas de los efectos aleatorios, ICC (proporción de la varianza entre grupos),
    efectos aleatorios estimados (BLUP: «encogidos» hacia la media) y AIC (ML). Equivale a PROC MIXED.
    Gauss: con pocos grupos (< 10) la varianza entre grupos se estima mal; ignorar el agrupamiento (MCO) da SE
    demasiado pequeños para las variables de nivel grupo."""
    _columnas(df, [grupo])
    re = f"~{pendiente_aleatoria}" if pendiente_aleatoria else None
    m = smf.mixedlm(formula, df, groups=df[grupo], re_formula=re).fit(reml=True)
    var_g = float(m.cov_re.iloc[0, 0]); var_e = float(m.scale)
    ci = m.conf_int()
    fijos = pd.DataFrame({"coef": m.fe_params, "SE": m.bse_fe, "p_valor": m.pvalues[m.fe_params.index],
                          "IC_inf": ci.loc[m.fe_params.index, 0], "IC_sup": ci.loc[m.fe_params.index, 1]})
    blup = pd.DataFrame(m.random_effects).T
    ml = smf.mixedlm(formula, df, groups=df[grupo], re_formula=re).fit(reml=False)
    return {"modelo": m, "efectos_fijos": fijos, "varianza_grupo": var_g, "varianza_residual": var_e,
            "icc": var_g / (var_g + var_e), "efectos_aleatorios": blup, "aic_ml": float(ml.aic), "n_grupos": int(df[grupo].nunique())}


def coeficiente_icc(df: pd.DataFrame, valor: str, grupo: str, nivel: float = 0.95) -> dict:
    """ICC(1) por ANOVA de un factor: (CMentre − CMdentro) / (CMentre + (k0−1)·CMdentro), con IC (F) y el
    efecto diseño 1 + (k0−1)·ICC (cuánto infla la varianza ignorar los grupos). k0 = tamaño medio ajustado."""
    _columnas(df, [valor, grupo])
    d = df[[valor, grupo]].dropna()
    g = d.groupby(grupo)[valor]
    n, a = len(d), g.ngroups
    ni = g.size().to_numpy()
    k0 = (n - np.sum(ni ** 2) / n) / (a - 1)
    cme = float(((g.mean() - d[valor].mean()) ** 2 * g.size()).sum() / (a - 1))
    cmd = float(((d[valor] - g.transform("mean")) ** 2).sum() / (n - a))
    F = cme / cmd
    icc = (cme - cmd) / (cme + (k0 - 1) * cmd)
    fl, fu = F / stats.f.ppf(0.5 + nivel / 2, a - 1, n - a), F * stats.f.ppf(0.5 + nivel / 2, n - a, a - 1)
    return {"icc": float(icc), "ic_inf": float((fl - 1) / (fl + k0 - 1)), "ic_sup": float((fu - 1) / (fu + k0 - 1)),
            "F": float(F), "p_valor": float(stats.f.sf(F, a - 1, n - a)), "efecto_diseno": float(1 + (k0 - 1) * icc), "k0": float(k0)}


def ajustar_logit_ordinal(df: pd.DataFrame, objetivo: str, variables, orden=None, enlace: str = "logit") -> dict:
    """Modelo de odds proporcionales (logit acumulado) para una respuesta ORDENADA (satisfacción, gravedad, tramo).
    Devuelve umbrales, coeficientes con OR acumulados (efecto de subir de categoría, el mismo en todos los cortes),
    y un contraste aproximado de la hipótesis de odds proporcionales (compara los OR de logits binarios por corte).
    Equivale a PROC LOGISTIC con respuesta ordinal (LINK=CLOGIT). enlace: 'logit' o 'probit'."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    variables = list(variables)
    _columnas(df, [objetivo] + variables)
    d = df[[objetivo] + variables].dropna()
    niveles = list(orden) if orden is not None else sorted(d[objetivo].unique())
    y = pd.Series(pd.Categorical(d[objetivo], categories=niveles, ordered=True), index=d.index)
    X = pd.get_dummies(d[variables], drop_first=True, dtype=float)
    m = OrderedModel(y, X, distr=enlace).fit(method="bfgs", disp=False, maxiter=500)
    k = X.shape[1]
    coef = m.params.iloc[:k]
    ci = m.conf_int().iloc[:k]
    tabla = pd.DataFrame({"coef": coef, "SE": m.bse.iloc[:k], "p_valor": m.pvalues.iloc[:k],
                          "OR_acumulado": np.exp(coef), "IC_inf": np.exp(ci[0]), "IC_sup": np.exp(ci[1])})
    # odds proporcionales: logits binarios P(y > corte) y dispersión de sus coeficientes
    cortes = []
    for j, c in enumerate(niveles[:-1]):
        yb = pd.Series((y.cat.codes > j).astype(int), index=d.index)
        if 0 < yb.mean() < 1:
            r = sm.Logit(yb, sm.add_constant(X)).fit(disp=False)
            cortes.append(r.params.iloc[1:].rename(f"> {c}"))
    tb = pd.concat(cortes, axis=1) if cortes else pd.DataFrame()
    return {"modelo": m, "tabla": tabla, "umbrales": m.params.iloc[k:], "coef_por_corte": tb,
            "aic": float(m.aic), "niveles": niveles}


def _elegir(media, se, regla: str, creciente_en_penal: bool, penal=None) -> int:
    """Índice del hiperparámetro: mínimo de CV o regla de 1 error estándar (el más penalizado dentro de min + 1 SE).
    `creciente_en_penal`: si el vector va de MÁS a menos penalización en orden de índice (LassoCV: alphas_ decreciente)."""
    i = int(np.argmin(media))
    if regla == "min":
        return i
    if regla != "1se":
        raise ValueError("regla: '1se' o 'min'")
    validos = np.where(media <= media[i] + se[i])[0]
    # Cs crecientes (logística): más penalización = índice menor; alphas_ decrecientes (LassoCV): índice menor también
    return int(validos.min())


def _logistica(C: float, l1: float, semilla: int):
    """LogisticRegression penalizada compatible con scikit-learn antiguo (penalty=) y ≥ 1.8 (solo l1_ratio)."""
    import sklearn
    from sklearn.linear_model import LogisticRegression
    version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if version >= (1, 8):
        return LogisticRegression(C=C, l1_ratio=l1, solver="saga", max_iter=5000, random_state=semilla)
    pen = "l1" if l1 == 1 else "l2" if l1 == 0 else "elasticnet"
    return LogisticRegression(C=C, penalty=pen, l1_ratio=l1 if pen == "elasticnet" else None, solver="saga",
                              max_iter=5000, random_state=semilla)


def ajustar_regularizado(X, y, tipo: str = "lasso", familia: str = "binomial", cv: int = 5, l1_ratio: float = 0.5,
                         regla: str = "1se", semilla: int = 42) -> dict:
    """Regresión penalizada con el λ elegido por validación cruzada (estratificada si es binaria).

    tipo: 'lasso' (L1: pone coeficientes a 0 = selección de variables), 'ridge' (L2: encoge, útil con colinealidad)
    o 'elasticnet' (mezcla; `l1_ratio`). familia: 'binomial' (logística) o 'gaussiana' (lineal).
    Estandariza las X dentro (y devuelve los coeficientes en la escala ORIGINAL), indica qué variables quedan
    y la curva de error de CV por λ. regla: '1se' (el λ más grande a menos de 1 error estándar del mínimo:
    modelo más sencillo con el mismo error, recomendado) o 'min' (el mínimo de CV: más variables).
    Equivale a PROC GLMSELECT SELECTION=LASSO (CHOOSE=CV) / HPGENSELECT.
    Gauss: los coeficientes están sesgados hacia 0 a propósito; sus p-valores clásicos no valen.
    """
    from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    Xd = pd.DataFrame(X).astype(float)
    yv = np.asarray(y)
    ok = np.isfinite(Xd.to_numpy()).all(axis=1) & pd.notna(yv)
    Xd, yv = Xd[ok], yv[ok]
    esc = StandardScaler().fit(Xd)
    Z = esc.transform(Xd)
    if familia == "binomial":
        from sklearn.model_selection import cross_val_score
        folds = StratifiedKFold(cv, shuffle=True, random_state=semilla)
        l1 = {"lasso": 1.0, "ridge": 0.0, "elasticnet": l1_ratio}[tipo]
        Cs = np.logspace(-3, 2, 25)
        por_fold = np.array([-cross_val_score(_logistica(C, l1, semilla), Z, yv, cv=folds, scoring="neg_log_loss") for C in Cs])
        perdida = por_fold.mean(axis=1)
        mejor = _elegir(perdida, por_fold.std(axis=1, ddof=1) / np.sqrt(cv), regla, creciente_en_penal=False)
        m = _logistica(Cs[mejor], l1, semilla).fit(Z, yv)
        coef_z, inter, lam = m.coef_.ravel(), float(m.intercept_[0]), 1 / float(Cs[mejor])
        curva = pd.DataFrame({"lambda": 1 / Cs, "logloss_cv": perdida})
    elif familia == "gaussiana":
        folds = KFold(cv, shuffle=True, random_state=semilla)
        if tipo == "ridge":
            alphas = np.logspace(-3, 4, 40)
            m = RidgeCV(alphas=alphas, cv=folds).fit(Z, yv); lam = float(m.alpha_)
            curva = pd.DataFrame({"lambda": alphas})
        else:
            cls = LassoCV if tipo == "lasso" else ElasticNetCV
            kw = {} if tipo == "lasso" else {"l1_ratio": l1_ratio}
            try:                                   # scikit-learn ≥ 1.7: alphas=entero
                m = cls(cv=folds, random_state=semilla, alphas=60, **kw).fit(Z, yv)
            except (TypeError, ValueError):        # versiones anteriores: n_alphas
                m = cls(cv=folds, random_state=semilla, n_alphas=60, **kw).fit(Z, yv)
            ecm = m.mse_path_
            k = _elegir(ecm.mean(axis=1), ecm.std(axis=1, ddof=1) / np.sqrt(ecm.shape[1]), regla, creciente_en_penal=True,
                        penal=m.alphas_)
            from sklearn.linear_model import ElasticNet, Lasso
            lam = float(m.alphas_[k])
            curva = pd.DataFrame({"lambda": m.alphas_, "ecm_cv": ecm.mean(axis=1)})
            m = (Lasso(alpha=lam) if tipo == "lasso" else ElasticNet(alpha=lam, l1_ratio=l1_ratio)).fit(Z, yv)
        coef_z, inter = np.ravel(m.coef_), float(m.intercept_)
    else:
        raise ValueError("familia: 'binomial' o 'gaussiana'")
    coef = coef_z / esc.scale_
    inter_orig = inter - float(np.sum(coef * esc.mean_))
    tabla = pd.DataFrame({"coef_estandarizado": coef_z, "coef": coef, "seleccionada": np.abs(coef_z) > 1e-8}, index=Xd.columns)
    return {"modelo": m, "escalador": esc, "tabla": tabla, "intercepto": inter_orig, "lambda": lam,
            "n_seleccionadas": int(tabla.seleccionada.sum()), "curva_cv": curva, "tipo": tipo, "familia": familia}
