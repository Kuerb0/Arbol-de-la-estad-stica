"""Datos faltantes e ingeniería de variables: diagnóstico de faltantes (patrones y test MCAR de Little), imputación
simple / iterativa / k-NN, imputación múltiple con reglas de Rubin, categorías raras, codificación por objetivo y SMOTE.
Equivale a PROC MI / MIANALYZE (imputación múltiple) y a recetas típicas de preparación de datos.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .._util import _columnas


def resumen_faltantes(df: pd.DataFrame) -> dict:
    """Por columna: nº y % de faltantes; patrones de faltantes (combinaciones y su frecuencia) y el test MCAR de Little
    sobre las columnas numéricas (H0: faltan completamente al azar; p pequeño = MAR o MNAR: imputar con cuidado).
    Gauss: no rechazar MCAR no prueba MCAR; y si faltan por el propio valor (MNAR) ningún método lo arregla del todo."""
    porcol = pd.DataFrame({"faltantes": df.isna().sum(), "pct": df.isna().mean() * 100}).sort_values("pct", ascending=False)
    patrones = df.isna().value_counts().rename("n").reset_index()
    patrones["pct"] = patrones["n"] / len(df) * 100
    return {"por_columna": porcol, "patrones": patrones, "little": contraste_mcar_little(df.select_dtypes("number")),
            "filas_completas_pct": float(df.notna().all(axis=1).mean() * 100)}


def contraste_mcar_little(df: pd.DataFrame) -> dict:
    """Test MCAR de Little (1988) con medias y covarianzas por máxima verosimilitud (EM) de las columnas numéricas."""
    X = df.to_numpy(float)
    n, p = X.shape
    if p == 0 or not np.isnan(X).any():
        return {"chi2": 0.0, "gl": 0, "p_valor": 1.0, "nota": "sin faltantes"}
    mu, S = _em_normal(X)
    patron = np.isnan(X)
    claves, inversos = np.unique(patron, axis=0, return_inverse=True)
    d2, gl = 0.0, 0
    for k, pat in enumerate(claves):
        obs = ~pat
        if not obs.any():
            continue
        filas = X[inversos.ravel() == k][:, obs]
        m = filas.mean(axis=0) - mu[obs]
        d2 += len(filas) * m @ np.linalg.pinv(S[np.ix_(obs, obs)]) @ m
        gl += obs.sum()
    gl -= p
    return {"chi2": float(d2), "gl": int(gl), "p_valor": float(stats.chi2.sf(d2, gl)) if gl > 0 else np.nan}


def _em_normal(X, iters: int = 200, tol: float = 1e-6):
    """Medias y covarianza normales por EM con faltantes (vectorizado por patrón de faltantes)."""
    mu = np.nanmean(X, axis=0)
    S = np.diag(np.nanvar(X, axis=0)) + 1e-9 * np.eye(X.shape[1])
    patrones, inv = np.unique(np.isnan(X), axis=0, return_inverse=True)
    inv = inv.ravel()
    grupos = [(pat, np.where(inv == k)[0]) for k, pat in enumerate(patrones)]
    for _ in range(iters):
        Xh, C = X.copy(), np.zeros_like(S)
        for m, filas in grupos:
            if not m.any():
                continue
            o = ~m
            if not o.any():
                Xh[np.ix_(filas, m)] = mu[m]; C[np.ix_(m, m)] += len(filas) * S[np.ix_(m, m)]; continue
            B = S[np.ix_(m, o)] @ np.linalg.pinv(S[np.ix_(o, o)])
            Xh[np.ix_(filas, m)] = mu[m] + (X[np.ix_(filas, o)] - mu[o]) @ B.T
            C[np.ix_(m, m)] += len(filas) * (S[np.ix_(m, m)] - B @ S[np.ix_(o, m)])
        mu_n = Xh.mean(axis=0)
        S_n = (Xh - mu_n).T @ (Xh - mu_n) / len(X) + C / len(X)
        fin = np.max(np.abs(mu_n - mu)) < tol and np.max(np.abs(S_n - S)) < tol
        mu, S = mu_n, S_n
        if fin:
            break
    return mu, S


def imputar(df: pd.DataFrame, metodo: str = "mediana", columnas=None, vecinos: int = 5, semilla: int = 42,
            indicadores: bool = False) -> pd.DataFrame:
    """Devuelve una COPIA con los faltantes imputados.
    metodo: 'mediana' / 'media' (numéricas) y moda en categóricas; 'iterativa' (MICE de una pasada con regresión
    bayesiana, numéricas); 'knn' (media de los k vecinos, numéricas escaladas).
    `indicadores=True` añade <col>_falta (0/1): útil si faltar es informativo.
    Gauss: la imputación simple subestima la varianza (trata lo imputado como dato real); para inferencia usa
    imputacion_multiple."""
    cols = list(columnas) if columnas is not None else list(df.columns)
    _columnas(df, cols)
    out = df.copy()
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    if indicadores:
        for c in cols:
            if df[c].isna().any():
                out[f"{c}_falta"] = df[c].isna().astype(int)
    for c in cat:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].mode(dropna=True).iloc[0])
    if not num:
        return out
    if metodo in ("mediana", "media"):
        for c in num:
            out[c] = out[c].fillna(out[c].median() if metodo == "mediana" else out[c].mean())
    elif metodo == "iterativa":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        out[num] = IterativeImputer(random_state=semilla, max_iter=20, sample_posterior=False).fit_transform(out[num])
    elif metodo == "knn":
        from sklearn.impute import KNNImputer
        m, s = out[num].mean(), out[num].std().replace(0, 1)
        out[num] = KNNImputer(n_neighbors=vecinos).fit_transform((out[num] - m) / s) * s.to_numpy() + m.to_numpy()
    else:
        raise ValueError("metodo: 'mediana', 'media', 'iterativa' o 'knn'")
    return out


def imputacion_multiple(df: pd.DataFrame, formula: str, m: int = 10, familia: str = "gaussiana", semilla: int = 42) -> dict:
    """Imputación múltiple: crea `m` datos completos (IterativeImputer con muestreo de la posterior) para las numéricas,
    ajusta el modelo de `formula` (GLM) en cada uno y combina con las REGLAS DE RUBIN: estimación = media; varianza =
    dentro + (1 + 1/m)·entre; gl de Barnard-Rubin aproximados. Devuelve la tabla combinada y la fracción de información
    faltante (FMI) por coeficiente. Equivale a PROC MI + PROC MIANALYZE."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    fam = {"gaussiana": sm.families.Gaussian, "binomial": sm.families.Binomial, "poisson": sm.families.Poisson}[familia]
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    ests, vars_ = [], []
    for i in range(m):
        d = df.copy()
        d[num] = IterativeImputer(random_state=semilla + i, sample_posterior=True, max_iter=10).fit_transform(df[num])
        r = smf.glm(formula, d, family=fam()).fit()
        ests.append(r.params); vars_.append(r.bse ** 2)
    Q, U = pd.concat(ests, axis=1), pd.concat(vars_, axis=1)
    qbar, ubar, b = Q.mean(axis=1), U.mean(axis=1), Q.var(axis=1, ddof=1)
    T = ubar + (1 + 1 / m) * b
    lam = ((1 + 1 / m) * b / T).clip(1e-12, 1)
    gl = (m - 1) / lam ** 2
    se = np.sqrt(T)
    tc = stats.t.ppf(0.975, gl)
    tabla = pd.DataFrame({"coef": qbar, "SE": se, "IC_inf": qbar - tc * se, "IC_sup": qbar + tc * se,
                          "p_valor": 2 * stats.t.sf(np.abs(qbar / se), gl), "gl": gl, "fmi": lam})
    return {"tabla": tabla, "m": m, "estimaciones": Q}


def agrupar_categorias_raras(df: pd.DataFrame, columna: str, min_frecuencia: float = 0.01, etiqueta: str = "Otros") -> tuple[pd.DataFrame, list]:
    """Junta en `etiqueta` los niveles con frecuencia relativa < `min_frecuencia` (o con menos casos que el número si es ≥ 1).
    Devuelve (copia, niveles agrupados). Evita dummies con 3 observaciones que dan coeficientes absurdos."""
    _columnas(df, [columna])
    vc = df[columna].value_counts(normalize=min_frecuencia < 1)
    raras = list(vc[vc < min_frecuencia].index)
    out = df.copy()
    out[columna] = out[columna].where(~out[columna].isin(raras), etiqueta)
    return out, raras


def codificar_por_objetivo(df: pd.DataFrame, columna: str, objetivo: str, suavizado: float = 20.0, folds: int = 5,
                           semilla: int = 42) -> tuple[pd.Series, pd.DataFrame]:
    """Codificación por objetivo (target encoding) con suavizado bayesiano y FUERA DE FOLD para no filtrar el objetivo:
    valor = (n·media_nivel + m·media_global)/(n + m). Devuelve (columna codificada para estos datos, tabla para datos nuevos).
    Para categóricas de alta cardinalidad (código postal, modelo de coche). Gauss: sin el fuera-de-fold hay fuga."""
    _columnas(df, [columna, objetivo])
    rng = np.random.default_rng(semilla)
    fold = rng.integers(0, folds, len(df))
    glob = df[objetivo].mean()
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for k in range(folds):
        tr = fold != k
        g = df[tr].groupby(columna, observed=True)[objetivo].agg(["sum", "count"])
        enc = (g["sum"] + suavizado * glob) / (g["count"] + suavizado)
        out[~tr] = df.loc[~tr, columna].map(enc).fillna(glob).to_numpy()
    g = df.groupby(columna, observed=True)[objetivo].agg(["sum", "count"])
    tabla = pd.DataFrame({"n": g["count"], "media_bruta": g["sum"] / g["count"], "codificado": (g["sum"] + suavizado * glob) / (g["count"] + suavizado)})
    tabla.attrs["media_global"] = float(glob)
    return out, tabla


def smote(df: pd.DataFrame, objetivo: str, clase_minoritaria=1, ratio: float = 1.0, vecinos: int = 5, semilla: int = 42) -> pd.DataFrame:
    """SMOTE: crea casos SINTÉTICOS de la clase minoritaria interpolando entre cada caso y uno de sus k vecinos de la
    misma clase (solo variables numéricas; las demás se copian del caso base) hasta que minoritaria = ratio·mayoritaria.
    Gauss: solo en TRAIN; descalibra las probabilidades (corrígelas: corregir_probabilidades_por_balanceo); con
    variables categóricas o ruido puede crear casos imposibles. A menudo basta con pesos_por_clase."""
    from sklearn.neighbors import NearestNeighbors
    _columnas(df, [objetivo])
    rng = np.random.default_rng(semilla)
    num = [c for c in df.columns if c != objetivo and pd.api.types.is_numeric_dtype(df[c])]
    mino = df[df[objetivo] == clase_minoritaria]
    n_may = (df[objetivo] != clase_minoritaria).sum()
    faltan = int(ratio * n_may) - len(mino)
    if faltan <= 0 or len(mino) < 2:
        return df.copy()
    Xm = mino[num].to_numpy(float)
    k = min(vecinos, len(mino) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm).kneighbors(Xm, return_distance=False)[:, 1:]
    base = rng.integers(0, len(mino), faltan)
    vec = nn[base, rng.integers(0, k, faltan)]
    lam = rng.random((faltan, 1))
    nuevos = mino.iloc[base].copy()
    nuevos[num] = Xm[base] + lam * (Xm[vec] - Xm[base])
    nuevos.index = range(df.index.max() + 1 if len(df) else 0, (df.index.max() + 1 if len(df) else 0) + faltan) \
        if pd.api.types.is_integer_dtype(df.index) else pd.RangeIndex(faltan)
    out = pd.concat([df, nuevos])
    out["sintetico"] = np.r_[np.zeros(len(df), int), np.ones(faltan, int)]
    return out
