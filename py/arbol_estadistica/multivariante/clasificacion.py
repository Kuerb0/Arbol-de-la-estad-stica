"""Agrupar y clasificar: clustering jerárquico, mezclas gaussianas y análisis discriminante (LDA / QDA).
Equivale a PROC CLUSTER / TREE, PROC FASTCLUS (no), PROC DISCRIM y a mclust (R)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .inferencia import _matriz


def clustering_jerarquico(X, k: int | None = None, metodo: str = "ward", escalar: bool = True, umbral: float | None = None) -> dict:
    """Clustering aglomerativo ('ward', 'average', 'complete', 'single') con la correlación cofenética (cuánto respeta
    el dendrograma las distancias originales; > 0.75 bien), etiquetas para `k` grupos (o cortando a una altura `umbral`)
    y la matriz de enlace para dibujar el dendrograma. Gauss: O(n²) en memoria: para n > 6 000 usa K-Means o muestrea (≈ 450 MB con n = 4 000)."""
    from scipy.cluster.hierarchy import cophenet, fcluster, linkage
    from scipy.spatial.distance import pdist
    M = _matriz(X)
    if len(M) > 6000:
        raise ValueError(f"{len(M)} filas: demasiadas para jerárquico (memoria O(n²)); muestrea o usa ajustar_kmeans.")
    Z = ((M - M.mean()) / M.std(ddof=1)).to_numpy() if escalar else M.to_numpy()
    dist = pdist(Z)
    L = linkage(Z, method=metodo) if metodo == "ward" else linkage(dist, method=metodo)
    coph = float(cophenet(L, dist)[0])
    if k is not None:
        etiquetas = fcluster(L, k, criterion="maxclust")
    elif umbral is not None:
        etiquetas = fcluster(L, umbral, criterion="distance")
    else:
        etiquetas = None
    return {"enlace": L, "cofenetica": coph, "etiquetas": None if etiquetas is None else pd.Series(etiquetas, index=M.index),
            "alturas_ultimas_fusiones": L[-10:, 2][::-1]}


def mezclas_gaussianas(X, k_max: int = 8, covarianza: str = "full", escalar: bool = True, semilla: int = 42) -> dict:
    """Clustering basado en modelos: mezcla de normales ajustada por EM para k = 1..k_max, eligiendo k por BIC.
    Da probabilidades de pertenencia (no solo etiquetas) y admite clusters elípticos de distinto tamaño (a diferencia
    de K-Means). covarianza: 'full', 'tied', 'diag' o 'spherical'."""
    from sklearn.mixture import GaussianMixture
    M = _matriz(X)
    Z = ((M - M.mean()) / M.std(ddof=1)).to_numpy() if escalar else M.to_numpy()
    filas, modelos = [], {}
    for k in range(1, k_max + 1):
        g = GaussianMixture(k, covariance_type=covarianza, random_state=semilla, n_init=3).fit(Z)
        modelos[k] = g; filas.append({"k": k, "bic": g.bic(Z), "aic": g.aic(Z)})
    t = pd.DataFrame(filas).set_index("k")
    kb = int(t.bic.idxmin()); g = modelos[kb]
    prob = g.predict_proba(Z)
    return {"tabla": t, "k": kb, "etiquetas": pd.Series(prob.argmax(1) + 1, index=M.index),
            "probabilidades": pd.DataFrame(prob, index=M.index, columns=[f"c{i + 1}" for i in range(kb)]),
            "pesos": g.weights_, "incertidumbre_media": float(1 - prob.max(1).mean())}


def analisis_discriminante(X, y, tipo: str = "lda", cv: int = 5, semilla: int = 42) -> dict:
    """LDA (covarianza común: fronteras lineales, coeficientes interpretables) o QDA (una covarianza por clase).
    Devuelve exactitud en validación cruzada, matriz de confusión (CV), coeficientes / funciones discriminantes (LDA),
    varianza explicada por cada función y el modelo. Equivale a PROC DISCRIM (y CANDISC para las funciones)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    M = _matriz(X)
    yy = pd.Series(np.asarray(y), index=pd.DataFrame(X).index).loc[M.index]
    ok = yy.notna(); M, yy = M[ok], yy[ok]
    mod = LinearDiscriminantAnalysis() if tipo == "lda" else QuadraticDiscriminantAnalysis(reg_param=1e-6) if tipo == "qda" else None
    if mod is None:
        raise ValueError("tipo: 'lda' o 'qda'")
    pred = cross_val_predict(mod, M, yy, cv=StratifiedKFold(cv, shuffle=True, random_state=semilla))
    mod.fit(M, yy)
    clases = list(mod.classes_)
    out = {"modelo": mod, "exactitud_cv": float((pred == yy.to_numpy()).mean()),
           "confusion_cv": pd.crosstab(pd.Series(yy.to_numpy(), name="real"), pd.Series(pred, name="predicho")),
           "clases": clases, "priors": dict(zip(clases, mod.priors_))}
    if tipo == "lda":
        out["funciones"] = pd.DataFrame(mod.scalings_, index=M.columns, columns=[f"LD{i + 1}" for i in range(mod.scalings_.shape[1])])
        out["varianza_explicada"] = mod.explained_variance_ratio_
    return out
