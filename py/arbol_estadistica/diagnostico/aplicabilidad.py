"""Dominio de aplicabilidad: ¿las observaciones nuevas se parecen a las de entrenamiento o el modelo extrapola?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def dominio_aplicabilidad(X_entrenamiento, X_nuevo, k: int = 5, cuantil: float = 0.99) -> pd.DataFrame:
    """Para cada fila nueva: apalancamiento h = xᵀ(XᵀX)⁻¹x frente al umbral 3p/n (zona de extrapolación lineal),
    distancia de Mahalanobis frente al cuantil χ²_p(`cuantil`), distancia media a los k vecinos de entrenamiento
    (variables estandarizadas) frente al cuantil empírico de esa misma distancia dentro del entrenamiento, y fuera de
    rango univariante. `dentro` = no falla ningún criterio. Las predicciones fuera del dominio no son fiables aunque
    el modelo valide bien. Variables numéricas (codifica antes las categóricas)."""
    A = pd.DataFrame(X_entrenamiento).astype(float); B = pd.DataFrame(X_nuevo)[A.columns].astype(float)
    n, p = A.shape
    mu, sd = A.mean(), A.std(ddof=1).replace(0, 1)
    Za, Zb = ((A - mu) / sd).to_numpy(), ((B - mu) / sd).to_numpy()
    Xc = np.column_stack([np.ones(n), Za]); XtXi = np.linalg.pinv(Xc.T @ Xc)
    Xn = np.column_stack([np.ones(len(Zb)), Zb])
    h = np.einsum("ij,jk,ik->i", Xn, XtXi, Xn)
    S = np.cov(Za, rowvar=False).reshape(p, p); Si = np.linalg.pinv(S)
    md = np.einsum("ij,jk,ik->i", Zb, Si, Zb)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Za)
    dtr = nn.kneighbors(Za)[0][:, 1:].mean(axis=1)
    dnew = nn.kneighbors(Zb, n_neighbors=k)[0].mean(axis=1)
    fuera_rango = ((B < A.min()) | (B > A.max())).sum(axis=1).to_numpy()
    out = pd.DataFrame({"apalancamiento": h, "umbral_h": 3 * (p + 1) / n, "mahalanobis2": md,
                        "umbral_mahalanobis": stats.chi2.ppf(cuantil, p), "dist_knn": dnew, "umbral_knn": np.quantile(dtr, cuantil),
                        "variables_fuera_rango": fuera_rango}, index=B.index)
    out["dentro"] = (out.apalancamiento <= out.umbral_h) & (out.mahalanobis2 <= out.umbral_mahalanobis) & (out.dist_knn <= out.umbral_knn) & (out.variables_fuera_rango == 0)
    return out
