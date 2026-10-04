"""Supervivencia avanzada: riesgos proporcionales (Schoenfeld), modelos paramétricos AFT (Weibull, lognormal,
log-logística, exponencial), Nelson-Aalen, riesgos competitivos (incidencia acumulada) y RMST.

Equivale a PROC PHREG (ZPH / ASSESS PH), PROC LIFEREG, PROC LIFETEST (NELSON, RMST) y %CIF / PROC LIFETEST EVENTCODE=.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .supervivencia import _validar


def contraste_schoenfeld(resultado_cox, transformacion: str = "rango") -> pd.DataFrame:
    """¿Se cumplen los riesgos proporcionales? Correlación entre los residuos de Schoenfeld de cada covariable
    y el tiempo del evento (transformado: 'rango', 'log' o 'identidad'). p pequeño = el efecto (HR) cambia con
    el tiempo. Incluye una fila GLOBAL (suma de χ²). Aproximación de Grambsch-Therneau sin escalar.
    `resultado_cox`: dict de ajustar_cox o PHRegResults. Equivale a cox.zph() de R o ASSESS PH en PROC PHREG.
    """
    res = resultado_cox["modelo"] if isinstance(resultado_cox, dict) else resultado_cox
    sch = np.asarray(res.schoenfeld_residuals)
    t = np.asarray(res.model.endog, float)
    ev = np.asarray(res.model.status, bool)
    nombres = list(res.model.exog_names) if res.model.exog_names else [f"x{i}" for i in range(sch.shape[1])]
    ok = ev & np.isfinite(sch).all(axis=1)
    tt = {"rango": stats.rankdata(t[ok]), "log": np.log(t[ok]), "identidad": t[ok]}[transformacion]
    filas = []
    for j, nom in enumerate(nombres):
        r, p = stats.pearsonr(tt, sch[ok, j])
        filas.append({"variable": nom, "rho": float(r), "chi2": float(r ** 2 * ok.sum()), "p_valor": float(p)})
    out = pd.DataFrame(filas)
    glob = out["chi2"].sum()
    out = pd.concat([out, pd.DataFrame([{"variable": "GLOBAL", "rho": np.nan, "chi2": glob,
                                         "p_valor": float(stats.chi2.sf(glob, len(nombres)))}])], ignore_index=True)
    out["incumple_ph"] = out["p_valor"] < 0.05
    return out.set_index("variable")


_DISTS = {
    # (log f de W, log S de W) para log T = Xβ + σ W
    "weibull": (lambda w: w - np.exp(w), lambda w: -np.exp(w)),
    "exponencial": (lambda w: w - np.exp(w), lambda w: -np.exp(w)),
    "lognormal": (stats.norm.logpdf, stats.norm.logsf),
    "loglogistica": (stats.logistic.logpdf, stats.logistic.logsf),
}


def ajustar_supervivencia_parametrica(df: pd.DataFrame, duracion: str, evento: str, covariables: Sequence[str] = (),
                                      distribucion: str = "weibull", categoricas: Sequence[str] = ()) -> dict:
    """Modelo de tiempo de fallo acelerado (AFT): log T = β0 + Xβ + σ·W, con W valor extremo (Weibull/exponencial),
    normal (lognormal) o logística (log-logística). Máxima verosimilitud con censura por la derecha.

    exp(β) = «factor de aceleración del tiempo» (> 1 alarga la supervivencia). En Weibull, el HR equivalente es
    exp(−β/σ) (también es un modelo de riesgos proporcionales). Devuelve tabla, σ, forma, AIC y la mediana predicha.
    Equivale a PROC LIFEREG DIST=WEIBULL|EXPONENTIAL|LNORMAL|LLOGISTIC.
    Gauss: compara distribuciones por AIC; frente a Cox, gana eficiencia si la forma es correcta y permite extrapolar.
    """
    if distribucion not in _DISTS:
        raise ValueError(f"distribucion: {sorted(_DISTS)}")
    cols = list(covariables) + list(categoricas)
    d = _validar(df, duracion, evento, cols)
    if (d[duracion] <= 0).any():
        raise ValueError("Las duraciones deben ser > 0 en un modelo AFT.")
    X = pd.get_dummies(d[cols], columns=list(categoricas), drop_first=True, dtype=float) if cols else pd.DataFrame(index=d.index)
    X.insert(0, "Intercept", 1.0)
    Xm, y, e = X.to_numpy(float), np.log(d[duracion].to_numpy(float)), d[evento].to_numpy(int)
    logf, logS = _DISTS[distribucion]
    fija_sigma = distribucion == "exponencial"

    def nll(par):
        beta = par[:Xm.shape[1]]
        sigma = 1.0 if fija_sigma else np.exp(par[-1])
        w = (y - Xm @ beta) / sigma
        return -(np.sum(e * (logf(w) - np.log(sigma))) + np.sum((1 - e) * logS(w)))
    p0 = np.r_[np.linalg.lstsq(Xm, y, rcond=None)[0], [] if fija_sigma else [0.0]]
    opt = optimize.minimize(nll, p0, method="BFGS")
    opt = optimize.minimize(nll, opt.x, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10}) if not opt.success else opt
    H = _hessiana(nll, opt.x)
    cov = np.linalg.pinv(H)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    k = Xm.shape[1]
    b, sb = opt.x[:k], se[:k]
    sigma = 1.0 if fija_sigma else float(np.exp(opt.x[-1]))
    z = stats.norm.ppf(0.975)
    tabla = pd.DataFrame({"coef": b, "SE": sb, "p_valor": 2 * stats.norm.sf(np.abs(b / np.where(sb > 0, sb, np.nan))),
                          "factor_tiempo": np.exp(b), "IC_inf": np.exp(b - z * sb), "IC_sup": np.exp(b + z * sb)}, index=X.columns)
    if distribucion in ("weibull", "exponencial"):
        tabla["HR_equivalente"] = np.exp(-b / sigma)
    mediana_w = {"weibull": np.log(np.log(2)), "exponencial": np.log(np.log(2)), "lognormal": 0.0, "loglogistica": 0.0}[distribucion]
    npar = k + (0 if fija_sigma else 1)
    return {"tabla": tabla, "sigma": sigma, "forma_weibull": 1 / sigma if distribucion in ("weibull", "exponencial") else np.nan,
            "loglik": float(-opt.fun), "aic": float(2 * npar + 2 * opt.fun), "convergio": bool(opt.success),
            "mediana_predicha": pd.Series(np.exp(Xm @ b + sigma * mediana_w), index=d.index), "distribucion": distribucion}


def _hessiana(f, x, h=1e-4):
    n = len(x); H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


def nelson_aalen(df: pd.DataFrame, duracion: str, evento: str, grupo: str | None = None) -> pd.DataFrame:
    """Riesgo acumulado H(t) de Nelson-Aalen = Σ d_i/n_i con su error estándar, y S(t) = exp(−H(t)) (Fleming-Harrington).
    Mejor que −log(KM) con pocos individuos en riesgo. Equivale a PROC LIFETEST NELSON."""
    d = _validar(df, duracion, evento, [grupo] if grupo else [])
    grupos = [("todos", d)] if grupo is None else [(g, d[d[grupo] == g]) for g in sorted(d[grupo].unique())]
    salida = []
    for g, sub in grupos:
        t = sub.groupby(duracion)[evento].agg(eventos="sum", total="size").sort_index()
        riesgo = t["total"][::-1].cumsum()[::-1]
        t = t[t.eventos > 0]; n = riesgo.loc[t.index]
        H = (t.eventos / n).cumsum(); var = (t.eventos / n ** 2).cumsum()
        salida.append(pd.DataFrame({"grupo": g, "tiempo": t.index, "en_riesgo": n.to_numpy(), "eventos": t.eventos.to_numpy(),
                                    "riesgo_acumulado": H.to_numpy(), "se": np.sqrt(var.to_numpy()),
                                    "supervivencia_fh": np.exp(-H.to_numpy())}))
    return pd.concat(salida, ignore_index=True)


def incidencia_acumulada(df: pd.DataFrame, duracion: str, causa: str, grupo: str | None = None) -> pd.DataFrame:
    """Riesgos competitivos: función de incidencia acumulada (Aalen-Johansen) de cada causa.
    `causa`: 0 = censurado, 1, 2, … = tipo de evento (p. ej. 1 = rescate, 2 = fallecimiento). La CIF de una causa
    es la probabilidad de que ocurra ESA causa antes de t, teniendo en cuenta que las otras lo impiden.
    OJO: 1 − KM tratando las otras causas como censura SOBREESTIMA el riesgo (supone que podrían ocurrir después)."""
    cols = [duracion, causa] + ([grupo] if grupo else [])
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise KeyError(f"Columnas inexistentes: {faltan}")
    d = df[cols].dropna()
    causas = sorted(c for c in d[causa].unique() if c != 0)
    grupos = [("todos", d)] if grupo is None else [(g, d[d[grupo] == g]) for g in sorted(d[grupo].unique())]
    salida = []
    for g, sub in grupos:
        cuenta = pd.crosstab(sub[duracion], sub[causa]).sort_index()          # filas = tiempos distintos
        tiempos = cuenta.index.to_numpy(float)
        n_riesgo = cuenta.sum(axis=1).to_numpy()[::-1].cumsum()[::-1]         # en riesgo = con t ≥ tiempo
        d_c = {c: (cuenta[c].to_numpy() if c in cuenta.columns else np.zeros(len(cuenta))) for c in causas}
        d_tot = sum(d_c.values())
        S_prev = np.r_[1.0, np.cumprod(1 - d_tot / n_riesgo)[:-1]]
        fila = {"grupo": g, "tiempo": tiempos, "en_riesgo": n_riesgo, "supervivencia_global": np.cumprod(1 - d_tot / n_riesgo)}
        for c in causas:
            fila[f"cif_causa_{c}"] = np.cumsum(S_prev * d_c[c] / n_riesgo)
        salida.append(pd.DataFrame(fila))
    return pd.concat(salida, ignore_index=True)


def rmst(df: pd.DataFrame, duracion: str, evento: str, tau: float, grupo: str | None = None) -> pd.DataFrame:
    """Tiempo medio de supervivencia restringido a τ: área bajo la curva de Kaplan-Meier en [0, τ] con su SE.
    Con 2 grupos añade la diferencia, su IC 95 % y p. Útil cuando los riesgos NO son proporcionales (las curvas se
    cruzan) y para comunicar («de media, 3.2 meses más de permanencia en los primeros 24»). PROC LIFETEST RMST."""
    d = _validar(df, duracion, evento, [grupo] if grupo else [])
    grupos = [("todos", d)] if grupo is None else [(g, d[d[grupo] == g]) for g in sorted(d[grupo].unique())]
    filas = []
    for g, sub in grupos:
        t = sub.groupby(duracion)[evento].agg(eventos="sum", total="size").sort_index()
        n = t["total"][::-1].cumsum()[::-1]
        t = t[(t.eventos > 0) & (t.index <= tau)]; n = n.loc[t.index]
        S = np.cumprod(1 - t.eventos / n)
        tiempos = np.r_[0.0, t.index.to_numpy(float), tau]
        Ss = np.r_[1.0, S.to_numpy()]
        areas = np.diff(tiempos) * Ss
        area = float(areas.sum())
        # varianza (Klein-Moeschberger): Σ A_i² d_i / (n_i (n_i − d_i)), A_i = área desde t_i hasta τ
        cola = np.cumsum(areas[::-1])[::-1][1:]
        dn = (t.eventos / (n * (n - t.eventos))).replace([np.inf], np.nan).fillna(0).to_numpy()
        var = float(np.sum(cola ** 2 * dn))
        filas.append({"grupo": g, "rmst": area, "se": np.sqrt(var), "tau": tau, "n": len(sub)})
    out = pd.DataFrame(filas).set_index("grupo")
    if len(out) == 2:
        a, b = out.index
        dif = out.loc[b, "rmst"] - out.loc[a, "rmst"]; se = np.hypot(out.loc[a, "se"], out.loc[b, "se"])
        out.attrs.update({"diferencia": float(dif), "ic": (float(dif - 1.96 * se), float(dif + 1.96 * se)),
                          "p_valor": float(2 * stats.norm.sf(abs(dif / se))), "comparacion": f"{b} − {a}"})
    return out
