"""Reducción de dimensión y escalas: PCA completo, análisis factorial (con KMO y Bartlett), alfa de Cronbach,
escalamiento multidimensional y análisis de correspondencias (PROC PRINCOMP, FACTOR, CORR ALPHA, MDS, CORRESP)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .inferencia import _matriz


def pca_completo(X, escalar: bool = True, n_componentes: int | None = None) -> dict:
    """PCA con varianza explicada (y acumulada), cargas (correlaciones variable-componente si se escala), puntuaciones,
    criterio de Kaiser (autovalor > 1) y contribución de cada variable. Equivale a PROC PRINCOMP.
    Gauss: sin escalar, la variable con más varianza (p. ej. importes en euros) domina las componentes."""
    M = _matriz(X)
    Z = (M - M.mean()) / (M.std(ddof=1) if escalar else 1)
    U, s, Vt = np.linalg.svd(Z.to_numpy(), full_matrices=False)
    autoval = s ** 2 / (len(M) - 1)
    k = n_componentes or len(autoval)
    nombres = [f"PC{i + 1}" for i in range(k)]
    var = pd.DataFrame({"autovalor": autoval[:k], "pct_varianza": autoval[:k] / autoval.sum() * 100}, index=nombres)
    var["pct_acumulado"] = var.pct_varianza.cumsum()
    cargas = pd.DataFrame(Vt[:k].T * np.sqrt(autoval[:k]), index=M.columns, columns=nombres)
    return {"varianza": var, "cargas": cargas, "puntuaciones": pd.DataFrame(U[:, :k] * s[:k], index=M.index, columns=nombres),
            "componentes_kaiser": int((autoval > 1).sum()) if escalar else None,
            "contribucion_pct": (cargas ** 2 / (cargas ** 2).sum()) * 100}


def adecuacion_factorial(X) -> dict:
    """¿Tiene sentido un análisis factorial? KMO global y por variable (> 0.6 aceptable, > 0.8 bueno) y test de
    esfericidad de Bartlett (H0: la matriz de correlaciones es la identidad: no hay nada que factorizar)."""
    M = _matriz(X)
    R = np.corrcoef(M.T); n, p = M.shape
    Ri = np.linalg.pinv(R)
    P = -Ri / np.sqrt(np.outer(np.diag(Ri), np.diag(Ri)))
    np.fill_diagonal(P, 0); R0 = R.copy(); np.fill_diagonal(R0, 0)
    kmo = (R0 ** 2).sum() / ((R0 ** 2).sum() + (P ** 2).sum())
    kmo_var = (R0 ** 2).sum(0) / ((R0 ** 2).sum(0) + (P ** 2).sum(0))
    chi = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R)); gl = p * (p - 1) / 2
    return {"kmo": float(kmo), "kmo_por_variable": pd.Series(kmo_var, index=M.columns),
            "bartlett_chi2": float(chi), "bartlett_gl": gl, "bartlett_p": float(stats.chi2.sf(chi, gl))}


def analisis_factorial(X, n_factores: int, rotacion: str | None = "varimax", semilla: int = 42) -> dict:
    """Análisis factorial (máxima verosimilitud, scikit-learn) sobre las variables estandarizadas, con rotación
    'varimax' (ortogonal) o 'quartimax'. Devuelve cargas, comunalidades, unicidades, % de varianza por factor,
    puntuaciones y la adecuación (KMO, Bartlett). Equivale a PROC FACTOR METHOD=ML ROTATE=VARIMAX."""
    from sklearn.decomposition import FactorAnalysis
    M = _matriz(X)
    Z = (M - M.mean()) / M.std(ddof=1)
    fa = FactorAnalysis(n_components=n_factores, rotation=rotacion, random_state=semilla).fit(Z)
    nombres = [f"F{i + 1}" for i in range(n_factores)]
    cargas = pd.DataFrame(fa.components_.T, index=M.columns, columns=nombres)
    comun = (cargas ** 2).sum(axis=1)
    var = pd.DataFrame({"varianza": (cargas ** 2).sum(), "pct": (cargas ** 2).sum() / M.shape[1] * 100})
    return {"cargas": cargas, "comunalidades": comun, "unicidades": pd.Series(fa.noise_variance_, index=M.columns),
            "varianza": var, "puntuaciones": pd.DataFrame(fa.transform(Z), index=M.index, columns=nombres),
            "adecuacion": adecuacion_factorial(M)}


def alfa_cronbach(items, nivel: float = 0.95) -> dict:
    """Fiabilidad de una escala: α de Cronbach con IC (Feldt), α si se elimina cada ítem y correlación ítem-total
    corregida (< 0.3 = el ítem no encaja). α > 0.7 aceptable, > 0.9 quizá ítems redundantes."""
    M = _matriz(items)
    n, k = M.shape
    if k < 2:
        raise ValueError("Hacen falta al menos 2 ítems.")

    def alfa(D):
        kk = D.shape[1]
        return kk / (kk - 1) * (1 - D.var(ddof=1).sum() / D.sum(axis=1).var(ddof=1))
    a = alfa(M)
    fl, fu = stats.f.ppf([0.5 + nivel / 2, 0.5 - nivel / 2], n - 1, (n - 1) * (k - 1))
    por_item = pd.DataFrame({"alfa_si_se_elimina": [alfa(M.drop(columns=c)) if k > 2 else np.nan for c in M.columns],
                             "correlacion_item_total": [np.corrcoef(M[c], M.drop(columns=c).sum(axis=1))[0, 1] for c in M.columns]},
                            index=M.columns)
    return {"alfa": float(a), "ic_inf": float(1 - (1 - a) * fl), "ic_sup": float(1 - (1 - a) * fu), "n": n, "items": k, "por_item": por_item}


def escalamiento_multidimensional(D=None, X=None, dimensiones: int = 2) -> dict:
    """MDS clásico (Torgerson): coordenadas que reproducen una matriz de distancias `D` (o las euclídeas de `X`
    estandarizadas), con la bondad de ajuste (proporción de autovalores positivos explicada)."""
    if D is None:
        if X is None:
            raise ValueError("Pasa una matriz de distancias D o los datos X.")
        M = _matriz(X)
        if len(M) > 5000:
            raise ValueError(f"{len(M)} filas: MDS clásico es O(n²) en memoria y O(n³) en tiempo; muestrea (≤ 5000).")
        Z = ((M - M.mean()) / M.std()).to_numpy()
        from scipy.spatial.distance import pdist, squareform
        D = pd.DataFrame(squareform(pdist(Z)), index=M.index, columns=M.index)
    D = pd.DataFrame(D)
    n = len(D)
    if n > 5000:
        raise ValueError(f"{n} objetos: MDS clásico es O(n²) en memoria y O(n³) en tiempo; muestrea (≤ 5000).")
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D.to_numpy() ** 2) @ J
    w, V = np.linalg.eigh(B)
    orden = np.argsort(w)[::-1]; w, V = w[orden], V[:, orden]
    coords = V[:, :dimensiones] * np.sqrt(np.clip(w[:dimensiones], 0, None))
    return {"coordenadas": pd.DataFrame(coords, index=D.index, columns=[f"D{i + 1}" for i in range(dimensiones)]),
            "bondad_ajuste": float(w[:dimensiones].sum() / w[w > 0].sum()), "autovalores": w}


def analisis_correspondencias(tabla: pd.DataFrame, dimensiones: int = 2) -> dict:
    """Análisis de correspondencias simple de una tabla de contingencia: inercia total (= χ²/n), % por dimensión y
    coordenadas principales de filas y columnas (niveles cercanos = perfiles parecidos). Equivale a PROC CORRESP."""
    N = pd.DataFrame(tabla).astype(float)
    P = N / N.to_numpy().sum()
    r, c = P.sum(axis=1).to_numpy(), P.sum(axis=0).to_numpy()
    S = (P.to_numpy() - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    U, s, Vt = np.linalg.svd(S, full_matrices=False)
    k = min(dimensiones, len(s))
    filas = pd.DataFrame(U[:, :k] * s[:k] / np.sqrt(r)[:, None], index=N.index, columns=[f"Dim{i + 1}" for i in range(k)])
    cols = pd.DataFrame(Vt.T[:, :k] * s[:k] / np.sqrt(c)[:, None], index=N.columns, columns=[f"Dim{i + 1}" for i in range(k)])
    inercia = s ** 2
    return {"inercia_total": float(inercia.sum()), "pct_inercia": pd.Series(inercia / inercia.sum() * 100, index=[f"Dim{i + 1}" for i in range(len(s))]),
            "filas": filas, "columnas": cols, "chi2": float(inercia.sum() * N.to_numpy().sum())}
