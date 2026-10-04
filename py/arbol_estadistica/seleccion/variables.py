"""Más selección de variables: backward, stepwise por p-valor (SLENTRY/SLSTAY de SAS), mejor subconjunto,
filtros de varianza casi nula y correlación alta, y eliminación recursiva con validación cruzada (RFE).
"""
from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .._util import _columnas
from .stepwise import _familia_por_defecto


def _preparar(df, objetivo, candidatas, categoricas, familia):
    candidatas = list(candidatas)
    _columnas(df, [objetivo] + candidatas)
    categoricas = set(categoricas)
    familia = familia or _familia_por_defecto(df[objetivo])
    datos = df[[objetivo] + candidatas].dropna()
    term = lambda v: f"C({v})" if v in categoricas else v                               # noqa: E731
    formula = lambda vs: f"{objetivo} ~ " + (" + ".join(term(v) for v in vs) if vs else "1")   # noqa: E731
    return candidatas, familia, datos, formula


def seleccion_backward(df: pd.DataFrame, objetivo: str, candidatas: Iterable[str], categoricas: Iterable[str] = (),
                       familia=None, criterio: str = "aic", protegidas: Iterable[str] = ()) -> dict:
    """Parte del modelo con TODAS las candidatas y quita en cada paso la que más mejora el criterio (AIC/BIC);
    para cuando quitar cualquiera empeora. `protegidas` nunca salen. Equivale a SELECTION=BACKWARD.
    Gauss: con muchas variables y pocos eventos el modelo completo es inestable: mejor forward, Lasso o cribado previo."""
    if criterio not in ("aic", "bic"):
        raise ValueError("criterio: 'aic' o 'bic'")
    candidatas, familia, datos, formula = _preparar(df, objetivo, candidatas, categoricas, familia)
    medir = (lambda m: m.aic) if criterio == "aic" else (lambda m: m.bic_llf)
    actuales, protegidas = list(candidatas), set(protegidas)
    mejor = float(medir(smf.glm(formula(actuales), datos, family=familia).fit()))
    historial = [{"paso": 0, "quitada": None, "criterio": mejor}]
    while True:
        intentos = []
        for v in [x for x in actuales if x not in protegidas]:
            try:
                intentos.append((float(medir(smf.glm(formula([x for x in actuales if x != v]), datos, family=familia).fit())), v))
            except Exception:  # noqa: BLE001
                continue
        if not intentos:
            break
        nuevo, v = min(intentos)
        if nuevo >= mejor:
            break
        mejor = nuevo; actuales.remove(v)
        historial.append({"paso": len(historial), "quitada": v, "criterio": nuevo})
    return {"formula": formula(actuales), "seleccionadas": actuales, "criterio_final": mejor,
            "modelo": smf.glm(formula(actuales), datos, family=familia).fit(), "historial": pd.DataFrame(historial)}


def seleccion_por_pvalor(df: pd.DataFrame, objetivo: str, candidatas: Iterable[str], categoricas: Iterable[str] = (),
                         familia=None, sle: float = 0.05, sls: float = 0.05, max_pasos: int = 100) -> dict:
    """Stepwise «a la SAS»: entra la variable con menor p (test de razón de verosimilitudes) si p < SLENTRY y, tras
    cada entrada, sale la de mayor p si p > SLSTAY. Equivale a SELECTION=STEPWISE SLENTRY= SLSTAY=.
    Gauss: los p-valores del modelo final están sesgados (se eligieron mirando los datos); úsalo para replicar SAS,
    no para inferencia. Necesita sle ≤ sls para no entrar en bucle."""
    from scipy import stats
    if sle > sls:
        raise ValueError("Debe cumplirse sle <= sls (si no, una variable puede entrar y salir sin fin).")
    candidatas, familia, datos, formula = _preparar(df, objetivo, candidatas, categoricas, familia)

    def p_lr(con, sin):
        m1, m0 = smf.glm(formula(con), datos, family=familia).fit(), smf.glm(formula(sin), datos, family=familia).fit()
        gl = max(m1.df_model - m0.df_model, 1)
        return float(stats.chi2.sf(max(m0.deviance - m1.deviance, 0) / (m1.scale if familia.__class__.__name__ == "Gaussian" else 1), gl))
    dentro, historial = [], []
    for paso in range(1, max_pasos + 1):
        fuera = [v for v in candidatas if v not in dentro]
        ps = []
        for v in fuera:
            try:
                ps.append((p_lr(dentro + [v], dentro), v))
            except Exception:  # noqa: BLE001
                continue
        if not ps or min(ps)[0] >= sle:
            break
        p, v = min(ps); dentro.append(v); historial.append({"paso": paso, "accion": "entra", "variable": v, "p_valor": p})
        salen = [(p_lr(dentro, [x for x in dentro if x != w]), w) for w in dentro if w != v]
        if salen and max(salen)[0] > sls:
            p, w = max(salen); dentro.remove(w); historial.append({"paso": paso, "accion": "sale", "variable": w, "p_valor": p})
    return {"formula": formula(dentro), "seleccionadas": dentro, "modelo": smf.glm(formula(dentro), datos, family=familia).fit(),
            "historial": pd.DataFrame(historial)}


def mejor_subconjunto(df: pd.DataFrame, objetivo: str, candidatas: Iterable[str], categoricas: Iterable[str] = (),
                      familia=None, max_variables: int | None = None) -> pd.DataFrame:
    """Ajusta TODOS los subconjuntos (2^p modelos; limita p ≤ 15) y devuelve, por tamaño, el mejor modelo con su AIC,
    BIC, deviance y (si es gaussiano) R² ajustado y Cp de Mallows. Equivale a SELECTION=SCORE / CP (PROC REG).
    Gauss: elige el tamaño por BIC o validación, no por R² (siempre sube)."""
    candidatas, familia, datos, formula = _preparar(df, objetivo, candidatas, categoricas, familia)
    p = len(candidatas)
    if p > 15:
        raise ValueError(f"{p} candidatas = {2 ** p} modelos: demasiados. Criba antes o usa Lasso.")
    gauss = familia.__class__.__name__ == "Gaussian"
    completo = smf.glm(formula(candidatas), datos, family=familia).fit()
    sigma2 = completo.scale
    filas = []
    for k in range(0, (max_variables or p) + 1):
        mejores = []
        for sub in itertools.combinations(candidatas, k):
            m = smf.glm(formula(list(sub)), datos, family=familia).fit()
            fila = {"n_variables": k, "variables": ", ".join(sub) or "(solo constante)", "aic": m.aic, "bic": m.bic_llf, "deviance": m.deviance}
            if gauss:
                n, q = m.nobs, m.df_model + 1
                fila["r2_ajustado"] = 1 - (m.deviance / (n - q)) / (m.null_deviance / (n - 1))
                fila["cp_mallows"] = m.deviance / sigma2 - n + 2 * q
            mejores.append(fila)
        filas.append(min(mejores, key=lambda f: f["deviance"]))
    t = pd.DataFrame(filas)
    t["mejor_bic"] = t["bic"] == t["bic"].min()
    return t


def filtrar_varianza_casi_nula(df: pd.DataFrame, ratio_frecuencias: float = 95 / 5, pct_unicos: float = 10.0,
                               excluir: Iterable[str] = ()) -> pd.DataFrame:
    """Detecta predictores casi constantes (criterio de Kuhn, `nearZeroVar`): ratio entre el valor más frecuente y el
    segundo > 95/5 Y menos de un 10 % de valores distintos. Devuelve una fila por columna con la marca `quitar`.
    Gauss: un predictor casi constante aporta poco y puede romper el ajuste en validación cruzada (folds sin variación)."""
    filas = []
    for c in [x for x in df.columns if x not in set(excluir)]:
        vc = df[c].value_counts(dropna=True)
        if len(vc) == 0:
            filas.append({"variable": c, "ratio_frecuencias": np.inf, "pct_unicos": 0.0, "constante": True, "quitar": True}); continue
        ratio = vc.iloc[0] / vc.iloc[1] if len(vc) > 1 else np.inf
        pu = 100 * len(vc) / max(df[c].notna().sum(), 1)
        filas.append({"variable": c, "ratio_frecuencias": float(ratio), "pct_unicos": float(pu), "constante": len(vc) == 1,
                      "quitar": bool(len(vc) == 1 or (ratio > ratio_frecuencias and pu < pct_unicos))})
    return pd.DataFrame(filas).set_index("variable")


def filtrar_correlacion_alta(df: pd.DataFrame, umbral: float = 0.9, metodo: str = "pearson", protegidas: Iterable[str] = ()) -> dict:
    """Quita variables numéricas hasta que ninguna pareja supere |r| > umbral (algoritmo de `findCorrelation`: de cada
    pareja problemática quita la que tiene más correlación media con el resto). Devuelve quitar, quedan y la matriz."""
    num = df.select_dtypes("number")
    r = num.corr(method=metodo).abs()
    protegidas = set(protegidas)
    quitar = []
    while True:
        rr = r.drop(index=quitar, columns=quitar)
        arr = rr.to_numpy(copy=True)
        np.fill_diagonal(arr, 0)
        if arr.size == 0 or np.nanmax(arr) <= umbral:
            break
        i, j = np.unravel_index(np.nanargmax(arr), arr.shape)
        a, b = rr.index[i], rr.columns[j]
        cand = [x for x in (a, b) if x not in protegidas]
        if not cand:
            break
        medias = dict(zip(rr.columns, np.nanmean(arr, axis=0)))
        quitar.append(max(cand, key=lambda x: medias[x]))
    return {"quitar": quitar, "quedan": [c for c in num.columns if c not in quitar], "matriz": r}


def eliminacion_recursiva(X, y, familia: str = "binomial", cv: int = 5, semilla: int = 42) -> dict:
    """Eliminación recursiva de variables con validación cruzada (RFE-CV, scikit-learn) sobre una logística o una
    regresión lineal con las X estandarizadas. Devuelve ranking, seleccionadas y la puntuación de CV por nº de variables.
    Gauss: el ranking depende de la escala (por eso se estandariza) y de la colinealidad."""
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xd = pd.DataFrame(X).astype(float)
    Z = StandardScaler().fit_transform(Xd)
    if familia == "binomial":
        est, folds, sc = LogisticRegression(max_iter=2000), StratifiedKFold(cv, shuffle=True, random_state=semilla), "roc_auc"
    elif familia == "gaussiana":
        est, folds, sc = LinearRegression(), KFold(cv, shuffle=True, random_state=semilla), "neg_root_mean_squared_error"
    else:
        raise ValueError("familia: 'binomial' o 'gaussiana'")
    r = RFECV(est, step=1, cv=folds, scoring=sc, min_features_to_select=1).fit(Z, np.asarray(y))
    curva = pd.DataFrame({"n_variables": np.arange(1, len(r.cv_results_["mean_test_score"]) + 1),
                          "puntuacion_cv": r.cv_results_["mean_test_score"], "sd": r.cv_results_["std_test_score"]})
    return {"seleccionadas": list(Xd.columns[r.support_]), "ranking": pd.Series(r.ranking_, index=Xd.columns).sort_values(),
            "curva_cv": curva, "metrica": sc}
