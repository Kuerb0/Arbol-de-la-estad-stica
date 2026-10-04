"""Piezas comunes de la rama ML: codificar X, partir train/test y medir igual en todos los modelos."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _tarea(y, tarea: str = "auto") -> str:
    if tarea != "auto":
        if tarea not in ("clasificacion", "regresion"):
            raise ValueError("tarea: 'auto', 'clasificacion' o 'regresion'")
        return tarea
    v = pd.Series(y).dropna()
    return "clasificacion" if (v.dtype == object or str(v.dtype) == "category" or v.nunique() <= 10) else "regresion"


def _codificar(X, columnas=None) -> pd.DataFrame:
    """One-hot de las columnas no numéricas (drop_first=False: los árboles no sufren colinealidad) y float."""
    D = pd.get_dummies(pd.DataFrame(X), dtype=float)
    if columnas is not None:
        D = D.reindex(columns=columnas, fill_value=0.0)
    return D.astype(float)


def _metricas(tarea, y, pred, proba=None):
    from sklearn import metrics as m
    if tarea == "clasificacion":
        out = {"exactitud": float(m.accuracy_score(y, pred))}
        if proba is not None:
            clases = np.unique(y)
            if proba.shape[1] == 2:
                out["auc"] = float(m.roc_auc_score(y, proba[:, 1]))
                out["brier"] = float(m.brier_score_loss(y, proba[:, 1], pos_label=clases.max()))
            else:
                out["auc"] = float(m.roc_auc_score(y, proba, multi_class="ovr"))
            out["logloss"] = float(m.log_loss(y, np.clip(proba, 1e-12, 1)))
        return out
    return {"rmse": float(np.sqrt(m.mean_squared_error(y, pred))), "mae": float(m.mean_absolute_error(y, pred)),
            "r2": float(m.r2_score(y, pred))}


def _ajustar_y_evaluar(estimador, X, y, tarea="auto", test: float = 0.3, semilla: int = 42, escalar: bool = False) -> dict:
    """Parte en train/test (estratificado si es clasificación), ajusta, mide en ambos y devuelve todo lo común."""
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xd = _codificar(X)
    yv = pd.Series(np.asarray(y), index=Xd.index)
    ok = yv.notna() & np.isfinite(Xd.to_numpy()).all(axis=1)
    Xd, yv = Xd[ok], yv[ok]
    t = _tarea(yv, tarea)
    Xtr, Xte, ytr, yte = train_test_split(Xd, yv, test_size=test, random_state=semilla, stratify=yv if t == "clasificacion" else None)
    mod = make_pipeline(StandardScaler(), estimador) if escalar else estimador
    mod.fit(Xtr, ytr)
    res = {"modelo": mod, "tarea": t, "columnas": list(Xd.columns), "n_train": len(Xtr), "n_test": len(Xte)}
    for nombre, A, b in (("train", Xtr, ytr), ("test", Xte, yte)):
        proba = mod.predict_proba(A) if t == "clasificacion" and hasattr(mod, "predict_proba") else None
        res[f"metricas_{nombre}"] = _metricas(t, b, mod.predict(A), proba)
    clave = "auc" if t == "clasificacion" and "auc" in res["metricas_test"] else ("exactitud" if t == "clasificacion" else "r2")
    res["sobreajuste"] = float(res["metricas_train"][clave] - res["metricas_test"][clave])
    res["predecir"] = lambda nuevo, _m=mod, _c=res["columnas"]: (_m.predict_proba(_codificar(nuevo, _c))[:, 1]
                                                                 if t == "clasificacion" and hasattr(_m, "predict_proba") and len(np.unique(yv)) == 2
                                                                 else _m.predict(_codificar(nuevo, _c)))
    res["X_test"], res["y_test"] = Xte, yte
    return res
