"""Inferencia multivariante: distancia de Mahalanobis (robusta), T² de Hotelling, M de Box, MANOVA y correlación
canónica (PROC CANCORR, GLM MANOVA, DISCRIM POOL=TEST)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .._util import _columnas


def _matriz(X, minimo=3):
    M = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").dropna()
    if len(M) < minimo or M.shape[1] < 1:
        raise ValueError(f"Hacen falta al menos {minimo} filas completas.")
    return M


def distancia_mahalanobis(X, robusta: bool = True, alpha: float = 0.01, semilla: int = 42) -> pd.DataFrame:
    """Distancia de Mahalanobis² de cada fila al centro, con p-valor χ²(p) y marca de atípico multivariante.
    robusta=True usa el determinante de covarianza mínima (MCD): el centro y la covarianza no se dejan arrastrar por
    los propios atípicos (con la clásica, varios atípicos juntos se «enmascaran»). Equivale a PROC ROBUSTCOV / MCD."""
    M = _matriz(X, 5)
    p = M.shape[1]
    if robusta:
        from sklearn.covariance import MinCovDet
        mcd = MinCovDet(random_state=semilla).fit(M.to_numpy())
        d2 = mcd.mahalanobis(M.to_numpy())
    else:
        c = M - M.mean()
        d2 = np.einsum("ij,jk,ik->i", c.to_numpy(), np.linalg.pinv(np.cov(M.T)), c.to_numpy())
    pv = stats.chi2.sf(d2, p)
    return pd.DataFrame({"d2": d2, "p_valor": pv, "atipico": pv < alpha}, index=M.index)


def contraste_hotelling(X1, X2) -> dict:
    """T² de Hotelling para dos muestras: ¿difieren los VECTORES de medias? (covarianza común). Se transforma a
    F(p, n1+n2−p−1). Hacerlo variable a variable con t infla el error tipo I y no ve diferencias conjuntas."""
    A, B = _matriz(X1), _matriz(X2)
    if A.shape[1] != B.shape[1]:
        raise ValueError("Las dos muestras deben tener las mismas variables.")
    n1, n2, p = len(A), len(B), A.shape[1]
    d = (A.mean() - B.mean()).to_numpy()
    Sp = ((n1 - 1) * np.cov(A.T) + (n2 - 1) * np.cov(B.T)) / (n1 + n2 - 2)
    t2 = float(n1 * n2 / (n1 + n2) * d @ np.linalg.pinv(np.atleast_2d(Sp)) @ d)
    f = t2 * (n1 + n2 - p - 1) / (p * (n1 + n2 - 2))
    return {"T2": t2, "F": float(f), "gl": (p, n1 + n2 - p - 1), "p_valor": float(stats.f.sf(f, p, n1 + n2 - p - 1)),
            "diferencia_medias": pd.Series(d, index=A.columns)}


def contraste_box_m(df: pd.DataFrame, variables, grupo: str) -> dict:
    """M de Box: ¿son iguales las matrices de covarianzas de los grupos? (supuesto de LDA y MANOVA). Aproximación χ².
    Gauss: muy sensible a la no normalidad y con n grande rechaza casi siempre; si rechaza, considera QDA o Pillai."""
    variables = list(variables)
    _columnas(df, variables + [grupo])
    d = df[variables + [grupo]].dropna()
    gs = [g[variables].to_numpy(float) for _, g in d.groupby(grupo)]
    k, p = len(gs), len(variables)
    ns = np.array([len(g) for g in gs]); N = ns.sum()
    Si = [np.cov(g.T) for g in gs]
    Sp = sum((n - 1) * S for n, S in zip(ns, Si)) / (N - k)
    M = (N - k) * np.log(np.linalg.det(Sp)) - sum((n - 1) * np.log(np.linalg.det(S)) for n, S in zip(ns, Si))
    c = (np.sum(1 / (ns - 1)) - 1 / (N - k)) * (2 * p ** 2 + 3 * p - 1) / (6 * (p + 1) * (k - 1))
    chi = M * (1 - c); gl = p * (p + 1) * (k - 1) / 2
    return {"M": float(M), "chi2": float(chi), "gl": float(gl), "p_valor": float(stats.chi2.sf(chi, gl))}


def manova(df: pd.DataFrame, respuestas, grupo: str) -> pd.DataFrame:
    """MANOVA de un factor: Wilks, Pillai, Hotelling-Lawley y Roy con su F aproximada (statsmodels).
    Pillai es el más robusto si fallan los supuestos. Equivale a PROC GLM / MANOVA H=grupo."""
    from statsmodels.multivariate.manova import MANOVA
    respuestas = list(respuestas)
    _columnas(df, respuestas + [grupo])
    d = df[respuestas + [grupo]].dropna().rename(columns={grupo: "_g"})
    r = MANOVA.from_formula(" + ".join(f"Q('{v}')" for v in respuestas) + " ~ C(_g)", d).mv_test()
    t = r.results["C(_g)"]["stat"].copy()
    t.columns = ["valor", "gl_num", "gl_den", "F", "p_valor"]
    return t.astype(float)


def correlacion_canonica(X, Y) -> dict:
    """Correlaciones canónicas entre dos bloques de variables, con contrastes de Bartlett (Λ de Wilks) de que las
    correlaciones a partir de la j-ésima son 0, y cargas (correlación de cada variable con su variable canónica).
    Equivale a PROC CANCORR."""
    A, B = _matriz(X), _matriz(Y)
    idx = A.index.intersection(B.index); A, B = A.loc[idx], B.loc[idx]
    n, p, q = len(A), A.shape[1], B.shape[1]
    Za, Zb = (A - A.mean()) / A.std(), (B - B.mean()) / B.std()
    Raa, Rbb = np.corrcoef(Za.T).reshape(p, p), np.corrcoef(Zb.T).reshape(q, q)
    Rab = (Za.T @ Zb).to_numpy() / (n - 1)
    Ia = np.linalg.inv(np.linalg.cholesky(Raa))
    Ib = np.linalg.inv(np.linalg.cholesky(Rbb))
    U, s, Vt = np.linalg.svd(Ia @ Rab @ Ib.T)
    k = min(p, q); s = np.clip(s[:k], 0, 0.999999)
    filas = []
    for j in range(k):
        lam = np.prod(1 - s[j:] ** 2)
        chi = -(n - 1 - (p + q + 1) / 2) * np.log(lam); gl = (p - j) * (q - j)
        filas.append({"canonica": j + 1, "correlacion": float(s[j]), "wilks": float(lam), "chi2": float(chi), "gl": gl,
                      "p_valor": float(stats.chi2.sf(chi, gl))})
    wa, wb = Ia.T @ U[:, :k], Ib.T @ Vt.T[:, :k]
    cargas_x = pd.DataFrame(Raa @ wa, index=A.columns, columns=[f"U{j + 1}" for j in range(k)])
    cargas_y = pd.DataFrame(Rbb @ wb, index=B.columns, columns=[f"V{j + 1}" for j in range(k)])
    return {"tabla": pd.DataFrame(filas).set_index("canonica"), "cargas_x": cargas_x, "cargas_y": cargas_y}
