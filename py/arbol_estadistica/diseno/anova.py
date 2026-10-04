"""ANOVA y diseños clásicos: factorial con interacción (tipo II/III, η²p y ω²), bloques aleatorizados (con eficiencia
relativa), ANCOVA (pendientes homogéneas y medias ajustadas), medidas repetidas y diseño mixto (Mauchly,
Greenhouse-Geisser, Huynh-Feldt), diseños anidados (componentes de la varianza) y diagnóstico con remedios."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .._util import _columnas


def _q(c):
    return f'Q("{c}")'


def _tabla_anova(modelo, tipo):
    t = sm.stats.anova_lm(modelo, typ=tipo).rename(columns={"sum_sq": "suma_cuadrados", "df": "gl", "PR(>F)": "p_valor"})
    t.index = [i.replace('Q("', "").replace('")', "").replace("C(", "").replace(", Sum)", "").replace(")", "") for i in t.index]
    sse, gle = t.loc["Residual", "suma_cuadrados"], t.loc["Residual", "gl"]
    mse = sse / gle
    ef = t.index != "Residual"
    if "Intercept" in t.index:
        ef = ef & (t.index != "Intercept")
    t["media_cuadratica"] = t.suma_cuadrados / t.gl
    sst = t.loc[ef, "suma_cuadrados"].sum() + sse
    t.loc[ef, "eta2_parcial"] = t.loc[ef, "suma_cuadrados"] / (t.loc[ef, "suma_cuadrados"] + sse)
    t.loc[ef, "omega2"] = ((t.loc[ef, "suma_cuadrados"] - t.loc[ef, "gl"] * mse) / (sst + mse)).clip(lower=0)
    return t[[c for c in ["suma_cuadrados", "gl", "media_cuadratica", "F", "p_valor", "eta2_parcial", "omega2"] if c in t]]


def anova_factorial(df: pd.DataFrame, respuesta: str, factores, tipo: int = 2, interacciones: bool | int = True,
                    covariables=None) -> dict:
    """ANOVA de varios factores con interacciones (hasta el orden indicado; True = todas). Tipo II (por defecto: cada
    efecto ajustado por los demás del mismo orden; recomendado sin interacción relevante) o tipo III (contrastes de
    suma; necesario si la interacción importa y el diseño está desequilibrado — lo que da SAS con TYPE3).
    Devuelve la tabla con η² parcial y ω² (tamaños del efecto), medias por celda y avisos.
    Equivale a: PROC GLM / PROC MIXED con MODEL y ~ A|B. Gauss: si la interacción es significativa, interpreta los
    efectos simples (un factor dentro de cada nivel del otro), no los efectos principales."""
    factores = [factores] if isinstance(factores, str) else list(factores)
    cov = list(covariables or [])
    _columnas(df, [respuesta] + factores + cov)
    d = df[[respuesta] + factores + cov].dropna()
    ref = "C({}, Sum)" if tipo == 3 else "C({})"
    terminos = [ref.format(_q(f)) for f in factores]
    orden = len(factores) if interacciones is True else (1 if interacciones is False else int(interacciones))
    rhs = [":".join(c) for k in range(1, orden + 1) for c in itertools.combinations(terminos, k)] + [_q(c) for c in cov]
    m = smf.ols(f"{_q(respuesta)} ~ " + " + ".join(rhs), d).fit()
    t = _tabla_anova(m, tipo)
    celdas = d.groupby(factores, observed=True)[respuesta].agg(["count", "mean", "std"])
    avisos = []
    if celdas["count"].nunique() > 1 and tipo == 2 and orden > 1:
        inter = [i for i in t.index if ":" in i]
        if inter and (t.loc[inter, "p_valor"] < 0.05).any():
            avisos.append("Diseño desequilibrado con interacción significativa: considera tipo=3.")
    if (celdas["count"] < 2).any():
        avisos.append("Hay celdas con < 2 observaciones: la interacción completa no es estimable con error.")
    return {"tabla": t, "medias_celda": celdas, "modelo": m, "avisos": avisos}


def anova_bloques(df: pd.DataFrame, respuesta: str, tratamiento: str, bloque: str) -> dict:
    """Diseño de bloques aleatorizados (completos o incompletos): y = μ + tratamiento + bloque + ε, tratamiento ajustado
    por bloque (tipo II, válido también para bloques incompletos equilibrados). Devuelve la tabla, las medias ajustadas
    de los tratamientos y la EFICIENCIA RELATIVA frente a un diseño completamente aleatorizado (cuántas veces más
    observaciones habría necesitado sin bloques). Equivale a: PROC GLM con CLASS bloque tratamiento."""
    _columnas(df, [respuesta, tratamiento, bloque])
    d = df[[respuesta, tratamiento, bloque]].dropna()
    m = smf.ols(f"{_q(respuesta)} ~ C({_q(tratamiento)}) + C({_q(bloque)})", d).fit()
    t = _tabla_anova(m, 2)
    t.index = [tratamiento if i == tratamiento else (bloque if i == bloque else i) for i in t.index]
    b, r = d[bloque].nunique(), d[tratamiento].nunique()
    msb, mse = t.loc[bloque, "media_cuadratica"], t.loc["Residual", "media_cuadratica"]
    ef = ((b - 1) * msb + b * (r - 1) * mse) / ((b * r - 1) * mse)
    malla = d[[bloque]].drop_duplicates().merge(pd.DataFrame({tratamiento: d[tratamiento].unique()}), how="cross")
    adj = malla.assign(pred=m.predict(malla)).groupby(tratamiento)["pred"].mean()
    completo = d.groupby([bloque, tratamiento]).size().unstack().notna().all().all()
    return {"tabla": t, "medias_ajustadas": adj, "eficiencia_relativa": float(ef), "completo": bool(completo), "modelo": m}


def ancova(df: pd.DataFrame, respuesta: str, grupo: str, covariables) -> dict:
    """Análisis de la covarianza: compara grupos ajustando por covariables continuas (p. ej. la medida basal).
    1) Contrasta la homogeneidad de pendientes (interacción grupo × covariable; si es significativa, la ANCOVA
    clásica no vale: las diferencias dependen del valor de la covariable). 2) Ajusta el modelo de pendientes comunes y
    devuelve la tabla (tipo II), las medias AJUSTADAS por grupo (en la media de las covariables, LSMEANS) y la
    reducción del error frente al ANOVA sin covariable. Equivale a: PROC GLM con LSMEANS grupo / covariables."""
    cov = [covariables] if isinstance(covariables, str) else list(covariables)
    _columnas(df, [respuesta, grupo] + cov)
    d = df[[respuesta, grupo] + cov].dropna()
    g = f"C({_q(grupo)})"
    base = f"{_q(respuesta)} ~ {g} + " + " + ".join(_q(c) for c in cov)
    m = smf.ols(base, d).fit()
    mi = smf.ols(base + " + " + " + ".join(f"{g}:{_q(c)}" for c in cov), d).fit()
    f_hom = sm.stats.anova_lm(m, mi)
    p_hom = float(f_hom["Pr(>F)"].iloc[1])
    sin = smf.ols(f"{_q(respuesta)} ~ {g}", d).fit()
    malla = pd.DataFrame({grupo: d[grupo].unique()}).assign(**{c: d[c].mean() for c in cov})
    pr = m.get_prediction(malla).summary_frame()
    medias = pd.DataFrame({"media_bruta": d.groupby(grupo)[respuesta].mean().reindex(malla[grupo]).to_numpy(),
                           "media_ajustada": pr["mean"].to_numpy(), "ee": pr["mean_se"].to_numpy()}, index=malla[grupo])
    avisos = [] if p_hom >= 0.05 else ["Pendientes NO homogéneas (p < 0.05): el efecto del grupo depende de la covariable; informa por tramos o usa el modelo con interacción."]
    return {"tabla": _tabla_anova(m, 2), "medias_ajustadas": medias, "p_homogeneidad_pendientes": p_hom,
            "reduccion_error": float(1 - m.mse_resid / sin.mse_resid), "modelo": m, "avisos": avisos}


def _esfericidad(Y):
    """Mauchly, ε de Greenhouse-Geisser y de Huynh-Feldt para una matriz sujetos × niveles (centrada por grupos)."""
    n, k = Y.shape
    C = np.linalg.qr(np.c_[np.ones(k), np.eye(k)[:, :-1]])[0][:, 1:]           # contrastes ortonormales
    S = np.cov(Y @ C, rowvar=False).reshape(k - 1, k - 1)
    p = k - 1
    W = np.linalg.det(S) / (np.trace(S) / p) ** p if p > 1 else 1.0
    f = 1 - (2 * p ** 2 + p + 2) / (6 * p * (n - 1))
    chi = -(n - 1) * f * np.log(W) if p > 1 else 0.0
    gl = p * (p + 1) / 2 - 1
    p_m = float(stats.chi2.sf(chi, gl)) if p > 1 else np.nan
    gg = np.trace(S) ** 2 / (p * np.trace(S @ S))
    hf = min(1.0, (n * p * gg - 2) / (p * (n - 1 - p * gg)))
    return W, p_m, float(gg), float(hf)


def anova_medidas_repetidas(df: pd.DataFrame, sujeto: str, dentro: str, respuesta: str, entre: str | None = None) -> dict:
    """ANOVA de medidas repetidas (un factor intra-sujeto) y, con `entre`, diseño MIXTO (split-plot: un factor
    entre-sujetos y uno intra). Datos en formato largo, un valor por sujeto y nivel (si hay varios se promedian).
    Devuelve la tabla con F, p sin corregir y con las correcciones de Greenhouse-Geisser y Huynh-Feldt, η² parcial y el
    contraste de esfericidad de Mauchly. Equivale a: PROC GLM con REPEATED. Gauss: si Mauchly rechaza o ε < 0.75 usa
    GG; con datos faltantes o tiempos irregulares es mejor un modelo mixto (`ajustar_modelo_mixto`)."""
    cols = [sujeto, dentro, respuesta] + ([entre] if entre else [])
    _columnas(df, cols)
    d = df[cols].dropna().groupby([sujeto, dentro] + ([entre] if entre else []), observed=True)[respuesta].mean().reset_index()
    ancho = d.pivot(index=sujeto, columns=dentro, values=respuesta)
    completos = ancho.dropna()
    if len(completos) < len(ancho):
        d = d[d[sujeto].isin(completos.index)]
    Y = completos.to_numpy(float); n, k = Y.shape
    if k < 2 or n < 3:
        raise ValueError("Hacen falta ≥ 2 niveles intra-sujeto y ≥ 3 sujetos completos.")
    gm = Y.mean()
    filas = []
    if entre is None:
        ss_d = n * np.sum((Y.mean(axis=0) - gm) ** 2)
        ss_s = k * np.sum((Y.mean(axis=1) - gm) ** 2)
        ss_e = np.sum((Y - gm) ** 2) - ss_d - ss_s
        gl_d, gl_e = k - 1, (k - 1) * (n - 1)
        W, p_m, gg, hf = _esfericidad(Y)
        intra = [(dentro, ss_d, gl_d)]
        err_intra = (f"Error({dentro})", ss_e, gl_e)
        filas.append({"efecto": "sujetos", "suma_cuadrados": ss_s, "gl": n - 1})
    else:
        grupo = d.drop_duplicates(sujeto).set_index(sujeto)[entre].reindex(completos.index)
        niveles = grupo.unique(); a = len(niveles)
        ss_a = sum(k * (grupo == g).sum() * (Y[(grupo == g).to_numpy()].mean() - gm) ** 2 for g in niveles)
        ss_suj = k * np.sum((Y.mean(axis=1) - gm) ** 2)
        ss_err_a = ss_suj - ss_a
        celdas = np.array([Y[(grupo == g).to_numpy()].mean(axis=0) for g in niveles])
        nj = np.array([(grupo == g).sum() for g in niveles])
        ss_b = n * np.sum((Y.mean(axis=0) - gm) ** 2)
        ss_ab = np.sum(nj[:, None] * (celdas - gm) ** 2) - ss_a - ss_b
        ss_err_b = np.sum((Y - gm) ** 2) - ss_suj - ss_b - ss_ab
        gl_a, gl_ea, gl_b, gl_ab, gl_eb = a - 1, n - a, k - 1, (a - 1) * (k - 1), (n - a) * (k - 1)
        filas.append({"efecto": entre, "suma_cuadrados": ss_a, "gl": gl_a, "F": (ss_a / gl_a) / (ss_err_a / gl_ea),
                      "p_valor": stats.f.sf((ss_a / gl_a) / (ss_err_a / gl_ea), gl_a, gl_ea), "eta2_parcial": ss_a / (ss_a + ss_err_a)})
        filas.append({"efecto": f"Error({entre})", "suma_cuadrados": ss_err_a, "gl": gl_ea})
        Yc = Y - celdas[[list(niveles).index(g) for g in grupo]]           # residuos dentro de cada grupo
        W, p_m, gg, hf = _esfericidad(Yc + gm)
        intra = [(dentro, ss_b, gl_b), (f"{entre}:{dentro}", ss_ab, gl_ab)]
        err_intra = (f"Error({dentro})", ss_err_b, gl_eb)
        ss_e, gl_e = ss_err_b, gl_eb
    mse = ss_e / gl_e
    for nom, ss, gl in intra:
        F = (ss / gl) / mse
        filas.append({"efecto": nom, "suma_cuadrados": ss, "gl": gl, "F": F, "p_valor": stats.f.sf(F, gl, gl_e),
                      "p_greenhouse_geisser": stats.f.sf(F, gl * gg, gl_e * gg), "p_huynh_feldt": stats.f.sf(F, gl * hf, gl_e * hf),
                      "eta2_parcial": ss / (ss + ss_e)})
    filas.append({"efecto": err_intra[0], "suma_cuadrados": err_intra[1], "gl": err_intra[2]})
    t = pd.DataFrame(filas).set_index("efecto")
    esf = {"W_mauchly": float(W), "p_mauchly": p_m, "epsilon_gg": gg, "epsilon_hf": hf}
    avisos = []
    if (np.isfinite(p_m) and p_m < 0.05) or gg < 0.75:
        avisos.append("Se viola la esfericidad: usa p_greenhouse_geisser.")
    if len(completos) < len(ancho):
        avisos.append(f"Se han descartado {len(ancho) - len(completos)} sujetos incompletos (un modelo mixto los aprovecharía).")
    return {"tabla": t, "esfericidad": esf, "medias": d.groupby([dentro] + ([entre] if entre else []), observed=True)[respuesta].mean(),
            "avisos": avisos}


def anova_anidado(df: pd.DataFrame, respuesta: str, factor: str, anidado: str) -> dict:
    """Diseño anidado (jerárquico) de dos etapas con efectos aleatorios: unidades `anidado` (p. ej. lotes, oficinas,
    réplicas de submuestreo) dentro de cada nivel de `factor`. El factor se contrasta frente a la variabilidad ENTRE
    unidades anidadas (no frente al error residual: hacerlo sería pseudorreplicación). Devuelve la tabla con las F
    correctas y los componentes de la varianza (por EMS, equilibrado) con su porcentaje."""
    _columnas(df, [respuesta, factor, anidado])
    d = df[[respuesta, factor, anidado]].dropna().copy()
    d["_u"] = d[factor].astype(str) + "/" + d[anidado].astype(str)
    gm = d[respuesta].mean()
    a = d[factor].nunique()
    ua = d.groupby(factor)["_u"].nunique(); nu = d.groupby("_u").size()
    b_med, n_med = ua.mean(), nu.mean()
    ss_a = sum(len(g) * (g[respuesta].mean() - gm) ** 2 for _, g in d.groupby(factor))
    ss_b = sum(len(g) * (g[respuesta].mean() - d.loc[d[factor] == g[factor].iloc[0], respuesta].mean()) ** 2 for _, g in d.groupby("_u"))
    ss_e = sum(((g[respuesta] - g[respuesta].mean()) ** 2).sum() for _, g in d.groupby("_u"))
    gl_a, gl_b, gl_e = a - 1, int(ua.sum() - a), int(len(d) - ua.sum())
    if gl_b < 1 or gl_e < 1:
        raise ValueError("Hacen falta varias unidades anidadas por nivel y varias observaciones por unidad.")
    ms_a, ms_b, ms_e = ss_a / gl_a, ss_b / gl_b, ss_e / gl_e
    Fa, Fb = ms_a / ms_b, ms_b / ms_e
    t = pd.DataFrame({"suma_cuadrados": [ss_a, ss_b, ss_e], "gl": [gl_a, gl_b, gl_e], "media_cuadratica": [ms_a, ms_b, ms_e],
                      "F": [Fa, Fb, np.nan], "p_valor": [stats.f.sf(Fa, gl_a, gl_b), stats.f.sf(Fb, gl_b, gl_e), np.nan]},
                     index=[factor, f"{anidado}({factor})", "Residual"])
    s2_e = ms_e; s2_b = max(0.0, (ms_b - ms_e) / n_med); s2_a = max(0.0, (ms_a - ms_b) / (n_med * b_med))
    comp = pd.DataFrame({"varianza": [s2_a, s2_b, s2_e]}, index=[factor, anidado, "residual"])
    comp["porcentaje"] = 100 * comp.varianza / comp.varianza.sum()
    avisos = [] if nu.nunique() == 1 and ua.nunique() == 1 else ["Diseño desequilibrado: componentes aproximados (para exactitud usa REML con ajustar_modelo_mixto)."]
    return {"tabla": t, "componentes_varianza": comp, "avisos": avisos}


def diagnostico_anova(df: pd.DataFrame, respuesta: str, factores) -> dict:
    """Diagnóstico y remedios del ANOVA: normalidad de residuos (Shapiro sobre una muestra si n grande), homogeneidad de
    varianzas (Levene-mediana), relación media-varianza (pendiente de log s frente a log media → potencia de Box-Cox
    sugerida: 0 = log, 0.5 = raíz, 1 = ninguna) y atípicos (residuo estudentizado > 3). Devuelve la recomendación:
    ANOVA clásico, Welch, transformación o Kruskal-Wallis."""
    factores = [factores] if isinstance(factores, str) else list(factores)
    _columnas(df, [respuesta] + factores)
    d = df[[respuesta] + factores].dropna()
    m = smf.ols(f"{_q(respuesta)} ~ " + ":".join(f"C({_q(f)})" for f in factores), d).fit()
    r = m.resid.to_numpy()
    sw = stats.shapiro(r if len(r) <= 5000 else np.random.default_rng(0).choice(r, 5000, replace=False))
    grupos = [g[respuesta].to_numpy() for _, g in d.groupby(factores, observed=True) if len(g) > 1]
    lev = stats.levene(*grupos, center="median")
    ms = d.groupby(factores, observed=True)[respuesta].agg(["mean", "std"]).query("mean > 0 and std > 0")
    pend = float(np.polyfit(np.log(ms["mean"]), np.log(ms["std"]), 1)[0]) if len(ms) >= 3 else np.nan
    lam = 1 - pend if np.isfinite(pend) else np.nan
    est = m.get_influence().resid_studentized_external
    recom = "ANOVA clásico"
    if lev.pvalue < 0.05:
        recom = "Welch (varianzas distintas)" if not (np.isfinite(lam) and abs(lam - 1) > 0.4) else f"transformar (Box-Cox λ ≈ {lam:.1f}) y repetir"
    if sw.pvalue < 0.01 and lev.pvalue >= 0.05 and min(len(g) for g in grupos) < 30:
        recom = "Kruskal-Wallis (residuos no normales con grupos pequeños)"
    return {"shapiro_residuos": {"W": float(sw.statistic), "p_valor": float(sw.pvalue)},
            "levene": {"estadistico": float(lev.statistic), "p_valor": float(lev.pvalue)},
            "pendiente_log_sd_log_media": pend, "lambda_box_cox_sugerida": lam,
            "atipicos": d.index[np.abs(est) > 3].tolist(), "recomendacion": recom}
