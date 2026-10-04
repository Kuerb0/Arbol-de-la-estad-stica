"""Explicar modelos de caja negra: importancia por permutación y dependencia parcial / ICE."""
from __future__ import annotations

import numpy as np
import pandas as pd


def importancia_permutacion(resultado: dict, n_repeticiones: int = 10, semilla: int = 42) -> pd.DataFrame:
    """Cuánto empeora la métrica en TEST al barajar cada variable (media ± sd sobre `n_repeticiones`). Válida para
    cualquier modelo y sin el sesgo de la importancia por impureza. `resultado`: salida de cualquier ajustar_* de ml.
    Gauss: con variables correlacionadas, barajar una deja su información en la otra y ambas parecen poco importantes."""
    from sklearn.inspection import permutation_importance
    clas = resultado["tarea"] == "clasificacion"
    sc = "roc_auc" if clas and pd.Series(resultado["y_test"]).nunique() == 2 else ("accuracy" if clas else "r2")
    r = permutation_importance(resultado["modelo"], resultado["X_test"], resultado["y_test"], scoring=sc,
                               n_repeats=n_repeticiones, random_state=semilla, n_jobs=-1)
    t = pd.DataFrame({"importancia": r.importances_mean, "sd": r.importances_std}, index=resultado["columnas"])
    t.attrs["metrica"] = sc
    return t.sort_values("importancia", ascending=False)


def dependencia_parcial(resultado: dict, variable: str, puntos: int = 30, ice: int = 0, semilla: int = 42) -> dict:
    """Dependencia parcial: predicción media cuando `variable` toma cada valor de una rejilla (resto como en los datos).
    Con `ice` > 0 devuelve también esas curvas individuales (ICE) de una muestra: si se cruzan, hay interacciones.
    Gauss: con variables muy correlacionadas la DP evalúa combinaciones que no existen en la realidad."""
    X = resultado["X_test"]
    if variable not in X.columns:
        raise KeyError(f"«{variable}» no está entre las columnas del modelo (las categóricas van en dummies).")
    grid = np.unique(np.quantile(X[variable], np.linspace(0.02, 0.98, puntos)))
    mod = resultado["modelo"]
    clas = resultado["tarea"] == "clasificacion" and hasattr(mod, "predict_proba")
    pred = (lambda A: mod.predict_proba(A)[:, 1]) if clas else mod.predict
    medias, curvas = [], []
    rng = np.random.default_rng(semilla)
    muestra = X.iloc[rng.choice(len(X), min(ice, len(X)), replace=False)] if ice else None
    for v in grid:
        A = X.copy(); A[variable] = v
        medias.append(float(np.mean(pred(A))))
        if ice:
            B = muestra.copy(); B[variable] = v; curvas.append(pred(B))
    out = {"tabla": pd.DataFrame({variable: grid, "prediccion_media": medias}), "variable": variable}
    if ice:
        out["ice"] = pd.DataFrame(np.array(curvas).T, columns=grid)
    return out
