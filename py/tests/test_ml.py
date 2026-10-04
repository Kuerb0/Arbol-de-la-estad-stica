"""Rama ML (0.9): salida común, los modelos flexibles captan lo no lineal y la explicabilidad funciona."""
import numpy as np
import pandas as pd
import pytest

from arbol_estadistica.ml import (ajustar_adaboost, ajustar_arbol_decision, ajustar_gradient_boosting, ajustar_knn,
                                  ajustar_naive_bayes, ajustar_random_forest, ajustar_red_neuronal, ajustar_stacking,
                                  ajustar_svm, comparar_clasificadores, dependencia_parcial, importancia_permutacion)


def _datos(seed=0, n=2500):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.uniform(0, 10, n), "ruido": rng.normal(size=n),
                      "z": rng.choice(["p", "q", "r"], n)})
    eta = 1.5 * X.a * X.b + 1.2 * np.sin(X.c) + (X.z == "q") - 0.3
    return X, (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int), rng


@pytest.mark.parametrize("f", [ajustar_arbol_decision, ajustar_random_forest, ajustar_gradient_boosting, ajustar_adaboost,
                               ajustar_knn, ajustar_svm, ajustar_naive_bayes, ajustar_red_neuronal, ajustar_stacking])
def test_salida_comun(f):
    X, y, _ = _datos(n=1200)
    r = f(X, y)
    assert {"modelo", "metricas_train", "metricas_test", "sobreajuste", "predecir", "columnas"} <= set(r)
    assert 0.5 <= r["metricas_test"]["auc"] <= 1
    p = r["predecir"](X.head(5))
    assert len(p) == 5 and ((0 <= p) & (p <= 1)).all()


def test_boosting_capta_interacciones_que_la_logistica_no():
    X, y, _ = _datos(1)
    t = comparar_clasificadores(X, y, ("logit", "gb", "rf"), cv=3)
    assert t.loc["gb", "auc"] > t.loc["logit", "auc"] + 0.08 and set(t.columns) >= {"auc", "brier", "segundos"}
    with pytest.raises(ValueError):
        comparar_clasificadores(X, y, ("nada",))


def test_regresion_y_arbol_podado():
    X, _, rng = _datos(2)
    yr = 2 * X.a + np.sin(X.c) + rng.normal(size=len(X))
    rf = ajustar_random_forest(X, yr)
    assert rf["tarea"] == "regresion" and rf["metricas_test"]["r2"] > 0.7 and 0 < rf["oob"] < 1
    X2, y2, _ = _datos(3)
    podado, sin = ajustar_arbol_decision(X2, y2), ajustar_arbol_decision(X2, y2, poda=None, min_hoja=1)
    assert podado["hojas"] < sin["hojas"] and podado["sobreajuste"] < sin["sobreajuste"]
    assert "|---" in podado["reglas"]


def test_importancia_y_dependencia_parcial():
    X, y, _ = _datos(4)
    r = ajustar_gradient_boosting(X, y)
    imp = importancia_permutacion(r, 5)
    assert imp.index[-1] in ("ruido", "z_p", "z_r") or imp.loc["ruido", "importancia"] < imp.loc["a", "importancia"]
    assert imp.loc["a", "importancia"] > 0.03
    dp = dependencia_parcial(r, "c", 20, ice=10)
    t = dp["tabla"].set_index("c")["prediccion_media"]
    assert t.iloc[t.index.get_indexer([np.pi / 2], method="nearest")[0]] > t.iloc[t.index.get_indexer([3 * np.pi / 2], method="nearest")[0]]
    assert dp["ice"].shape == (10, 20)
    with pytest.raises(KeyError):
        dependencia_parcial(r, "no_existe")
