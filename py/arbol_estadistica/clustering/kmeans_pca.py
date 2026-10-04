"""Rama CLUSTERING / K-Means parametrizable, selección de K y PCA.

Orígenes: `notebook de consultoría` (MinMax + OneHot + Ordinal, k por silhouette, PCA 2D) y
notebook de consultoría (StandardScaler + pesos por variable, silhouette con muestra,
Davies-Bouldin, Calinski-Harabasz, remapeo de clusters por precio, pseudo-R2).
Equivale a PROC STDIZE + PROC FASTCLUS + (PROC PRINCOMP para la proyección).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler


def preparar_matriz_clustering(
    df: pd.DataFrame,
    num_vars: Iterable[str],
    cat_vars: Iterable[str] = (),
    ord_vars: Iterable[str] = (),
    escalado: str = "minmax",
    pesos: dict | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Matriz numérica lista para K-Means: numéricas escaladas + categóricas one-hot + ordinales.

    escalado : 'minmax' (por defecto, notebook de consultoría) o 'standard' (notebook de consultoría).
    pesos    : {prefijo_o_nombre: factor}; multiplica las columnas resultantes cuyo nombre
               empieza por esa clave (p. ej. {'precio': 1.25, 'marca_': 1.5}) para dar más
               importancia a unas variables en la distancia. Sin filas NaN: imputar antes.
    Devuelve (X, nombres_de_columnas).
    """
    num_vars, cat_vars, ord_vars = list(num_vars), list(cat_vars), list(ord_vars)
    escalador = {"minmax": MinMaxScaler, "standard": StandardScaler}[escalado]()
    pasos = []
    if num_vars:
        pasos.append(("num", escalador, num_vars))
    if cat_vars:
        pasos.append(("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_vars))
    if ord_vars:
        pasos.append(("ord", OrdinalEncoder(), ord_vars))
    if not pasos:
        raise ValueError("Indica al menos una variable.")
    ct = ColumnTransformer(pasos, verbose_feature_names_out=False)
    X = np.asarray(ct.fit_transform(df), dtype=float)
    nombres = list(ct.get_feature_names_out())
    for clave, f in (pesos or {}).items():
        for i, nom in enumerate(nombres):
            if nom.startswith(clave):
                X[:, i] *= f
    return X, nombres


def buscar_k_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 10, semilla: int = 42,
                        n_init: int = 2, muestra: int = 5000) -> pd.DataFrame:
    """Silhouette (con muestra, O(n^2) si no), Davies-Bouldin y Calinski-Harabasz para cada K.

    Elegir K = argmax silhouette (mejor: >0.5 fuerte, 0.25-0.5 débil) y contrastarlo con
    Davies-Bouldin (menor es mejor) y Calinski-Harabasz (mayor es mejor). Devuelve un DataFrame
    con columna `elegido` marcando el K de mayor silhouette.
    """
    filas = []
    for k in range(k_min, min(k_max, len(X) - 1) + 1):
        km = KMeans(n_clusters=k, random_state=semilla, n_init=n_init).fit(X)
        filas.append({
            "k": k,
            "silhouette": float(silhouette_score(X, km.labels_, sample_size=min(muestra, len(X)),
                                                 random_state=semilla)),
            "davies_bouldin": float(davies_bouldin_score(X, km.labels_)),
            "calinski_harabasz": float(calinski_harabasz_score(X, km.labels_)),
            "inercia": float(km.inertia_),
        })
    out = pd.DataFrame(filas)
    out["elegido"] = out["silhouette"] == out["silhouette"].max()
    return out


def ajustar_kmeans(X: np.ndarray, k: int, semilla: int = 42, n_init: int = 10,
                   ordenar_por: pd.Series | None = None) -> dict:
    """Ajusta K-Means y devuelve etiquetas 1..k, pseudo-R2 (1 - inercia/SS_total) y el modelo.

    Si pasas `ordenar_por` (serie alineada con las filas de X, p. ej. el precio), los clusters se
    renumeran de menor a mayor media de esa serie, para que C1 sea siempre "el más barato".
    """
    km = KMeans(n_clusters=k, random_state=semilla, n_init=n_init).fit(X)
    etiquetas = pd.Series(km.labels_)
    if ordenar_por is not None:
        orden = pd.Series(np.asarray(ordenar_por)).groupby(km.labels_).mean().sort_values().index.tolist()
        etiquetas = etiquetas.map({old: new for new, old in enumerate(orden, start=1)})
    else:
        etiquetas = etiquetas + 1
    ss_tot = float(np.sum((X - X.mean(axis=0)) ** 2))
    return {"etiquetas": etiquetas.to_numpy(), "pseudo_r2": 1 - km.inertia_ / ss_tot, "modelo": km}


def proyeccion_pca(X: np.ndarray, n_componentes: int = 2, semilla: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Proyecta X en sus primeras componentes principales (para dibujar clusters).

    Devuelve (DataFrame PC1..PCn, varianza_explicada_por_componente).
    OJO: la PCA es solo visualización; el clustering se hace en el espacio completo.
    """
    pca = PCA(n_components=n_componentes, random_state=semilla)
    Z = pca.fit_transform(X)
    return pd.DataFrame(Z, columns=[f"PC{i + 1}" for i in range(n_componentes)]), pca.explained_variance_ratio_
