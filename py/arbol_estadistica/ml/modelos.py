"""Modelos de aprendizaje automático con la MISMA salida (métricas train/test, sobreajuste, función predecir):
árbol de decisión (con poda por CV), random forest, gradient boosting, AdaBoost, k-NN, SVM, naive Bayes, red neuronal y
stacking. Todos parten en train/test estratificado con semilla 42.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ._comun import _ajustar_y_evaluar, _codificar, _tarea


def ajustar_arbol_decision(X, y, tarea: str = "auto", profundidad_max: int | None = None, poda: str | float = "cv",
                           min_hoja: int = 20, semilla: int = 42) -> dict:
    """Árbol CART. poda='cv' elige el parámetro de complejidad (ccp_alpha) por validación cruzada (como PROC HPSPLIT
    PRUNE=COSTCOMPLEXITY); un número fija ccp_alpha; None = sin poda. Devuelve además las reglas en texto e importancias.
    Gauss: un árbol solo es muy inestable (cambia mucho con otra muestra): para predecir usa random forest o boosting."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
    t = _tarea(y, tarea)
    Cls = DecisionTreeClassifier if t == "clasificacion" else DecisionTreeRegressor
    base = Cls(max_depth=profundidad_max, min_samples_leaf=min_hoja, random_state=semilla)
    if poda == "cv":
        Xd = _codificar(X)
        alphas = base.cost_complexity_pruning_path(Xd, np.asarray(y)).ccp_alphas
        alphas = np.unique(np.quantile(alphas[:-1], np.linspace(0, 1, 15))) if len(alphas) > 2 else [0.0]
        est = GridSearchCV(base, {"ccp_alpha": alphas}, cv=5).fit(Xd, np.asarray(y)).best_estimator_
        est = Cls(**{**est.get_params()})
    elif poda is None:
        est = base
    else:
        est = Cls(max_depth=profundidad_max, min_samples_leaf=min_hoja, random_state=semilla, ccp_alpha=float(poda))
    r = _ajustar_y_evaluar(est, X, y, t, semilla=semilla)
    m = r["modelo"]
    r["reglas"] = export_text(m, feature_names=r["columnas"], max_depth=6)
    r["importancias"] = pd.Series(m.feature_importances_, index=r["columnas"]).sort_values(ascending=False)
    r["hojas"], r["profundidad"], r["ccp_alpha"] = int(m.get_n_leaves()), int(m.get_depth()), float(m.ccp_alpha)
    return r


def ajustar_random_forest(X, y, tarea: str = "auto", n_arboles: int = 300, max_variables="sqrt", min_hoja: int = 5,
                          semilla: int = 42) -> dict:
    """Random forest (bagging de árboles + variables al azar en cada corte). Error OOB (fuera de la bolsa: validación
    gratis), importancias por impureza. max_variables=None convierte el bosque en bagging puro.
    Gauss: las importancias por impureza favorecen variables continuas o con muchos niveles: confirma con importancia_permutacion."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    t = _tarea(y, tarea)
    Cls = RandomForestClassifier if t == "clasificacion" else RandomForestRegressor
    r = _ajustar_y_evaluar(Cls(n_estimators=n_arboles, max_features=max_variables if t == "clasificacion" or max_variables != "sqrt" else 1.0,
                               min_samples_leaf=min_hoja, oob_score=True, n_jobs=-1, random_state=semilla), X, y, t, semilla=semilla)
    m = r["modelo"]
    r["oob"] = float(m.oob_score_)
    r["importancias"] = pd.Series(m.feature_importances_, index=r["columnas"]).sort_values(ascending=False)
    return r


def ajustar_gradient_boosting(X, y, tarea: str = "auto", tasa: float = 0.05, max_iter: int = 500, profundidad_max: int | None = 3,
                              semilla: int = 42) -> dict:
    """Gradient boosting por histogramas (estilo LightGBM/XGBoost) con PARADA TEMPRANA sobre una validación interna:
    árboles pequeños que corrigen los errores de los anteriores. Suele ser el mejor en datos tabulares.
    Devuelve también el nº de iteraciones usadas. Gauss: muy flexible: vigila el sobreajuste (train vs test)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    t = _tarea(y, tarea)
    Cls = HistGradientBoostingClassifier if t == "clasificacion" else HistGradientBoostingRegressor
    r = _ajustar_y_evaluar(Cls(learning_rate=tasa, max_iter=max_iter, max_depth=profundidad_max, early_stopping=True,
                               validation_fraction=0.15, n_iter_no_change=30, random_state=semilla), X, y, t, semilla=semilla)
    r["iteraciones"] = int(r["modelo"].n_iter_)
    return r


def ajustar_adaboost(X, y, n_estimadores: int = 200, tasa: float = 0.5, semilla: int = 42) -> dict:
    """AdaBoost con tocones (árboles de profundidad 1): reponderar los casos mal clasificados (pérdida exponencial).
    Solo clasificación. Sensible a etiquetas erróneas y atípicos (les da cada vez más peso)."""
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    r = _ajustar_y_evaluar(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n_estimadores, learning_rate=tasa,
                                              random_state=semilla), X, y, "clasificacion", semilla=semilla)
    r["importancias"] = pd.Series(r["modelo"].feature_importances_, index=r["columnas"]).sort_values(ascending=False)
    return r


def ajustar_knn(X, y, tarea: str = "auto", k: int | str = "cv", semilla: int = 42) -> dict:
    """k vecinos más cercanos con las X ESTANDARIZADAS (si no, manda la variable de mayor escala). k='cv' elige k
    por validación cruzada entre 3 y 51. Sin entrenamiento pero lento al predecir con n grande; sufre con muchas variables."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t = _tarea(y, tarea)
    Cls = KNeighborsClassifier if t == "clasificacion" else KNeighborsRegressor
    if k == "cv":
        Xd = _codificar(X)
        g = GridSearchCV(make_pipeline(StandardScaler(), Cls()), {f"{Cls.__name__.lower()}__n_neighbors": [3, 5, 9, 15, 25, 35, 51]}, cv=5)
        k = int(list(g.fit(Xd, np.asarray(y)).best_params_.values())[0])
    r = _ajustar_y_evaluar(Cls(n_neighbors=int(k)), X, y, t, semilla=semilla, escalar=True)
    r["k"] = int(k)
    return r


def ajustar_svm(X, y, tarea: str = "auto", kernel: str = "rbf", C: float = 1.0, semilla: int = 42) -> dict:
    """Máquina de vectores soporte (X estandarizadas). kernel 'rbf' (no lineal), 'linear' o 'poly'. Probabilidades por
    Platt (internamente con CV). Coste O(n²)-O(n³): para n > 20 000 usa boosting o un SVM lineal."""
    from sklearn.svm import SVC, SVR
    t = _tarea(y, tarea)
    est = SVC(kernel=kernel, C=C, probability=True, random_state=semilla) if t == "clasificacion" else SVR(kernel=kernel, C=C)
    n = len(pd.DataFrame(X))
    if n > 30000:
        raise ValueError(f"{n} filas: SVM con kernel escala O(n²) o peor; muestrea o usa ajustar_gradient_boosting.")
    return _ajustar_y_evaluar(est, X, y, t, semilla=semilla, escalar=True)


def ajustar_naive_bayes(X, y, semilla: int = 42) -> dict:
    """Naive Bayes gaussiano: supone las variables independientes dada la clase. Muy rápido y estable con pocos datos;
    las probabilidades suelen estar mal calibradas (extremas)."""
    from sklearn.naive_bayes import GaussianNB
    return _ajustar_y_evaluar(GaussianNB(), X, y, "clasificacion", semilla=semilla)


def ajustar_red_neuronal(X, y, tarea: str = "auto", capas=(32, 16), regularizacion: float = 1e-3, semilla: int = 42) -> dict:
    """Perceptrón multicapa (X estandarizadas, ReLU, Adam, parada temprana). En datos tabulares rara vez supera al
    boosting y es menos interpretable; útil como componente de stacking o con muchas interacciones suaves."""
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    t = _tarea(y, tarea)
    Cls = MLPClassifier if t == "clasificacion" else MLPRegressor
    return _ajustar_y_evaluar(Cls(hidden_layer_sizes=tuple(capas), alpha=regularizacion, early_stopping=True, max_iter=500,
                                  random_state=semilla), X, y, t, semilla=semilla, escalar=True)


def ajustar_stacking(X, y, semilla: int = 42) -> dict:
    """Stacking: logística + random forest + gradient boosting como modelos base y una logística que aprende a combinar
    sus predicciones fuera de fold. Solo clasificación. Gana poco cuando un modelo ya domina; más coste y menos interpretable."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    base = [("logit", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
            ("rf", RandomForestClassifier(200, min_samples_leaf=5, n_jobs=-1, random_state=semilla)),
            ("gb", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, early_stopping=True, random_state=semilla))]
    r = _ajustar_y_evaluar(StackingClassifier(base, LogisticRegression(max_iter=2000), cv=5), X, y, "clasificacion", semilla=semilla)
    r["pesos_meta"] = pd.Series(r["modelo"].final_estimator_.coef_.ravel()[:3], index=[b[0] for b in base]) \
        if r["modelo"].final_estimator_.coef_.shape[1] == 3 else None
    return r


def comparar_clasificadores(X, y, modelos=("logit", "arbol", "rf", "gb", "knn", "nb"), cv: int = 5, semilla: int = 42) -> pd.DataFrame:
    """Compara varios clasificadores con la MISMA validación cruzada estratificada: AUC, log-loss y Brier medios (± sd)
    y tiempo de ajuste. modelos ⊂ {logit, arbol, rf, gb, ada, knn, svm, nb, mlp}."""
    import time

    from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    cat = {"logit": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
           "arbol": DecisionTreeClassifier(min_samples_leaf=20, max_depth=6, random_state=semilla),
           "rf": RandomForestClassifier(300, min_samples_leaf=5, n_jobs=-1, random_state=semilla),
           "gb": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, early_stopping=True, random_state=semilla),
           "ada": AdaBoostClassifier(n_estimators=200, random_state=semilla),
           "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(25)),
           "svm": make_pipeline(StandardScaler(), SVC(probability=True, random_state=semilla)),
           "nb": GaussianNB(),
           "mlp": make_pipeline(StandardScaler(), MLPClassifier((32, 16), early_stopping=True, max_iter=500, random_state=semilla))}
    Xd = _codificar(X); yv = np.asarray(y)
    folds = StratifiedKFold(cv, shuffle=True, random_state=semilla)
    filas = []
    for nombre in modelos:
        if nombre not in cat:
            raise ValueError(f"modelo desconocido: {nombre}")
        t0 = time.perf_counter()
        r = cross_validate(cat[nombre], Xd, yv, cv=folds, scoring=("roc_auc", "neg_log_loss", "neg_brier_score"))
        filas.append({"modelo": nombre, "auc": r["test_roc_auc"].mean(), "auc_sd": r["test_roc_auc"].std(ddof=1),
                      "logloss": -r["test_neg_log_loss"].mean(), "brier": -r["test_neg_brier_score"].mean(),
                      "segundos": time.perf_counter() - t0})
    return pd.DataFrame(filas).set_index("modelo").sort_values("auc", ascending=False)
