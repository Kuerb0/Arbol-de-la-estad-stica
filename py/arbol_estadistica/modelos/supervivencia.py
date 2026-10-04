"""Rama MODELOS / análisis de supervivencia: Kaplan-Meier, log-rank y modelo de Cox.

Origen: hueco del catálogo (Very Normal: «The Statistics of Life and Death | Survival Analysis»,
«2-Minute Kaplan-Meier Curves in R», «2-Minute Proportional Hazards Model in R») y temario de
Demografía/Vida del máster. No venía de consultoría.
Equivale a PROC LIFETEST (curvas KM, STRATA -> log-rank) y PROC PHREG (TIES=EFRON) de SAS.

Datos: una fila por individuo con `duracion` (tiempo observado) y `evento` (1 = ocurrió el evento,
0 = censurado: el seguimiento acabó sin evento). Censurar NO es lo mismo que «no le pasó»:
quitar a los censurados o tratarlos como «sin evento» sesga la supervivencia hacia arriba.
OJO: `preprocesado.duracion_hasta_evento` no distingue censura; construye la columna `evento` aparte.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight, survdiff


def _validar(df: pd.DataFrame, duracion: str, evento: str, extra: Sequence[str] = ()) -> pd.DataFrame:
    faltan = [c for c in [duracion, evento, *extra] if c not in df.columns]
    if faltan:
        raise KeyError(f"Columnas inexistentes: {faltan}")
    d = df[[duracion, evento, *extra]].dropna()
    if (d[duracion] < 0).any():
        raise ValueError("Hay duraciones negativas.")
    if not set(pd.unique(d[evento])) <= {0, 1, True, False}:
        raise ValueError("`evento` debe ser 0/1 (1 = evento, 0 = censurado).")
    return d


def _km_un_grupo(t: np.ndarray, e: np.ndarray, z: float) -> pd.DataFrame:
    sf = SurvfuncRight(t, e)
    s, se = np.asarray(sf.surv_prob), np.asarray(sf.surv_prob_se)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.log(-np.log(s))                      # IC log(-log), el de PROC LIFETEST
        se_t = se / (s * np.abs(np.log(s)))
        inf = np.exp(-np.exp(theta + z * se_t))
        sup = np.exp(-np.exp(theta - z * se_t))
    inf = np.where(np.isfinite(inf), inf, s)
    sup = np.where(np.isfinite(sup), sup, s)
    cens = [int(((t == x) & (e == 0)).sum()) for x in sf.surv_times]
    tabla = pd.DataFrame({"tiempo": sf.surv_times, "en_riesgo": sf.n_risk.astype(int),
                          "eventos": sf.n_events.astype(int), "censurados_en_t": cens,
                          "supervivencia": s, "ic_inf": inf, "ic_sup": sup})
    inicio = pd.DataFrame({"tiempo": [0.0], "en_riesgo": [len(t)], "eventos": [0], "censurados_en_t": [0],
                           "supervivencia": [1.0], "ic_inf": [1.0], "ic_sup": [1.0]})
    return pd.concat([inicio, tabla], ignore_index=True)


def kaplan_meier(df: pd.DataFrame, duracion: str, evento: str, grupo: str | None = None,
                 nivel: float = 0.95) -> dict:
    """Curva de supervivencia de Kaplan-Meier (por grupo si se indica) con IC puntual log(-log).

    Devuelve dict con:
      tabla    : grupo, tiempo, en_riesgo, eventos, censurados_en_t, supervivencia, ic_inf, ic_sup
                 (empieza en tiempo 0 con S = 1; S(t) es escalonada: se mantiene hasta el siguiente evento)
      medianas : grupo, n, eventos, censurados, mediana = primer tiempo con S(t) <= 0.5
                 (NaN si la curva no baja de 0.5). SAS/R, si S vale justo 0.5 en un tramo, dan el
                 punto medio de ese tramo: puede diferir ligeramente.
    Gauss: la cola derecha tiene pocos individuos en riesgo -> IC muy anchos; no interpretar su forma.
    """
    d = _validar(df, duracion, evento, [grupo] if grupo else [])
    z = float(norm.ppf(0.5 + nivel / 2))
    grupos = [("todos", d)] if grupo is None else [(g, d[d[grupo] == g]) for g in sorted(d[grupo].unique())]
    tablas, meds = [], []
    for g, sub in grupos:
        t, e = sub[duracion].to_numpy(float), sub[evento].to_numpy(int)
        tg = _km_un_grupo(t, e, z)
        tablas.append(tg.assign(grupo=g))
        bajo = tg.loc[tg["supervivencia"] <= 0.5 + 1e-12, "tiempo"]
        meds.append({"grupo": g, "n": len(t), "eventos": int(e.sum()), "censurados": int((e == 0).sum()),
                     "mediana": float(bajo.iloc[0]) if len(bajo) else np.nan})
    tabla = pd.concat(tablas, ignore_index=True)
    return {"tabla": tabla[["grupo"] + [c for c in tabla.columns if c != "grupo"]], "medianas": pd.DataFrame(meds)}


def contraste_log_rank(df: pd.DataFrame, duracion: str, evento: str, grupo: str, alpha: float = 0.05) -> dict:
    """Log-rank: H0 = las curvas de supervivencia de todos los grupos son iguales.

    Equivale a STRATA grupo / TEST=LOGRANK de PROC LIFETEST. gl = nº de grupos - 1.
    OJO: tiene buena potencia si los riesgos son proporcionales; si las curvas se cruzan puede no
    detectar diferencias reales (mirar la gráfica de Kaplan-Meier antes de concluir).
    """
    d = _validar(df, duracion, evento, [grupo])
    k = d[grupo].nunique()
    if k < 2:
        raise ValueError("`grupo` necesita al menos 2 niveles.")
    chi2, p = survdiff(d[duracion].to_numpy(float), d[evento].to_numpy(int), d[grupo].to_numpy())
    sig = bool(p < alpha)
    return {"estadistico": float(chi2), "gl": k - 1, "p_valor": float(p), "significativo": sig,
            "interpretacion": f"Log-rank chi2={chi2:.2f} (gl={k - 1}), p={p:.4g} -> "
                              f"{'las curvas difieren' if sig else 'no hay evidencia de que las curvas difieran'} a alpha={alpha}."}


def ajustar_cox(df: pd.DataFrame, duracion: str, evento: str, covariables: Sequence[str],
                categoricas: Sequence[str] = (), refs: dict | None = None, ties: str = "efron",
                nivel: float = 0.95) -> dict:
    """Modelo de riesgos proporcionales de Cox: h(t|x) = h0(t) * exp(x'beta).

    `categoricas` se convierten en dummies frente a `refs[var]` (o la primera alfabéticamente).
    Devuelve dict con `modelo` (PHRegResults) y `tabla` (variable, coef, HR = exp(coef), IC del HR,
    p_valor). HR = 1.5 -> en cada instante el riesgo es un 50 % mayor por cada unidad de la variable.
    OJO: supone riesgos PROPORCIONALES (HR constante en el tiempo). Comprobarlo: curvas KM por grupo
    que no se cruzan / log(-log S) paralelas, o residuos de Schoenfeld (no implementado aquí).
    """
    refs = refs or {}
    cols = list(dict.fromkeys(list(covariables) + list(categoricas)))
    d = _validar(df, duracion, evento, cols)
    X = d[[c for c in cols if c not in categoricas]].astype(float)
    for cv in categoricas:
        dum = pd.get_dummies(d[cv], prefix=cv).astype(float)
        ref = f"{cv}_{refs[cv]}" if cv in refs else dum.columns[0]
        if ref not in dum.columns:
            raise ValueError(f"Referencia {refs.get(cv)!r} no existe en '{cv}'.")
        X = pd.concat([X, dum.drop(columns=[ref])], axis=1)
    if X.shape[1] == 0:
        raise ValueError("Indica al menos una covariable.")
    res = PHReg(d[duracion].to_numpy(float), X, status=d[evento].to_numpy(int), ties=ties).fit(disp=False)
    z = float(norm.ppf(0.5 + nivel / 2))
    b, se = np.asarray(res.params), np.asarray(res.bse)
    tabla = pd.DataFrame({"coef": b, "SE": se, "HR": np.exp(b), "IC_inf": np.exp(b - z * se),
                          "IC_sup": np.exp(b + z * se), "p_valor": np.asarray(res.pvalues)},
                         index=pd.Index(list(X.columns), name="variable"))
    return {"modelo": res, "tabla": tabla, "n": len(d), "eventos": int(d[evento].sum())}
