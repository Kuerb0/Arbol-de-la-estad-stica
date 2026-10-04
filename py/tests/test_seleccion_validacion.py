"""Selección (backward, stepwise SAS, mejor subconjunto, filtros, RFE), validación y métricas de negocio (0.9)."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from arbol_estadistica.diagnostico import (calibrar_probabilidades, curva_precision_recall, descomposicion_brier,
                                           estadistico_ks_gini, tabla_ganancia_lift)
from arbol_estadistica.seleccion import (comparar_modelos_cv, eliminacion_recursiva, filtrar_correlacion_alta,
                                         filtrar_varianza_casi_nula, mejor_subconjunto, metricas_regresion,
                                         optimismo_bootstrap, seleccion_backward, seleccion_por_pvalor, tabla_criterios,
                                         validacion_cruzada)


def _datos(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n), "z": rng.choice(["x", "y", "w"], n),
                      "r1": rng.normal(size=n), "r2": rng.normal(size=n)})
    d["y"] = (rng.random(n) < 1 / (1 + np.exp(-(d.a - 0.7 * d.b + 0.6 * (d.z == "y"))))).astype(int)
    d["yg"] = d.a + 0.5 * d.c + rng.normal(size=n)
    return d, rng


def test_backward_y_stepwise_sas_encuentran_las_verdaderas():
    d, _ = _datos()
    assert set(seleccion_backward(d, "y", ["a", "b", "c", "z", "r1", "r2"], ["z"], criterio="bic")["seleccionadas"]) == {"a", "b", "z"}
    s = seleccion_por_pvalor(d, "y", ["a", "b", "c", "z", "r1", "r2"], ["z"], sle=0.01, sls=0.01)
    assert set(s["seleccionadas"]) == {"a", "b", "z"} and (s["historial"].accion == "entra").all()
    with pytest.raises(ValueError):
        seleccion_por_pvalor(d, "y", ["a"], sle=0.1, sls=0.05)


def test_mejor_subconjunto_cp_y_bic():
    d, _ = _datos(1)
    t = mejor_subconjunto(d, "yg", ["a", "b", "c", "r1", "r2"])
    mejor = t.loc[t.mejor_bic].iloc[0]
    assert {"a", "c"} <= set(mejor.variables.split(", ")) and mejor.n_variables <= 3
    assert mejor.cp_mallows == pytest.approx(3, abs=3)             # Cp ≈ nº de parámetros si el modelo es correcto
    assert t.loc[t.n_variables == 5, "cp_mallows"].item() == pytest.approx(6)
    with pytest.raises(ValueError):
        mejor_subconjunto(d.assign(**{f"v{i}": 0.0 for i in range(16)}), "yg", [f"v{i}" for i in range(16)])


def test_filtros_y_rfe():
    d, rng = _datos(2)
    nz = filtrar_varianza_casi_nula(d.assign(rara=np.r_[np.ones(1495), np.zeros(5)], cte=1.0))
    assert nz.loc["rara", "quitar"] and nz.loc["cte", "constante"] and not nz.loc["a", "quitar"]
    fc = filtrar_correlacion_alta(d.assign(a2=2 * d.a + rng.normal(0, .05, len(d))), 0.9, protegidas=["a"])
    assert fc["quitar"] == ["a2"]
    r = eliminacion_recursiva(d[["a", "b", "c", "r1", "r2"]], d.y)
    assert {"a", "b"} <= set(r["seleccionadas"]) and r["ranking"]["a"] == 1


def test_validacion_cruzada_y_optimismo():
    d, _ = _datos(3, 800)
    v = validacion_cruzada(d, "y ~ a + b + C(z)", "binomial", k=5, repeticiones=2)
    assert len(v["por_fold"]) == 10 and 0.7 < v["resumen"].loc["auc", "media"] < 0.9
    g = validacion_cruzada(d, "yg ~ a + c", "gaussiana")
    assert g["resumen"].loc["rmse", "media"] == pytest.approx(1, abs=0.15)
    o = optimismo_bootstrap(d.assign(**{f"ruido{i}": np.random.default_rng(100 + i).normal(size=len(d)) for i in range(15)}),
                            "y ~ a + b + " + " + ".join(f"ruido{i}" for i in range(15)), n_boot=60)
    assert o.loc["auc", "optimismo"] > 0 and o.loc["auc", "corregida"] < o.loc["auc", "aparente"]


def test_comparar_modelos_cv_corrige_el_t():
    d, _ = _datos(4, 1000)
    r = comparar_modelos_cv(d, "y ~ a", "y ~ a + b + C(z)", k=5, repeticiones=3)
    assert r["diferencia"] > 0 and r["p_valor"] < 0.05 and r["p_valor"] > r["p_valor_sin_corregir"]


def test_tabla_criterios_y_metricas_regresion():
    d, _ = _datos(5)
    ms = {"a": smf.ols("yg ~ a", d).fit(), "ac": smf.ols("yg ~ a + c", d).fit(), "todo": smf.ols("yg ~ a + b + c + r1 + r2", d).fit()}
    t = tabla_criterios(ms, ms["todo"].scale)
    assert t.index[0] == "ac" and t.peso_akaike.sum() == pytest.approx(1)
    m = metricas_regresion([1, 2, 3, 4], [1.1, 1.9, 3.2, 3.8])
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean([.01, .01, .04, .04]))) and m["r2"] > 0.95


def test_lift_ks_pr_brier_y_calibracion():
    rng = np.random.default_rng(6)
    x = rng.normal(size=6000); p = 1 / (1 + np.exp(-(-2.5 + 1.2 * x))); y = (rng.random(6000) < p).astype(int)
    t = tabla_ganancia_lift(y, p)
    assert t.lift.iloc[0] > 2.5 and t.ganancia_acum.iloc[-1] == pytest.approx(100) and t.lift_acum.iloc[:6].is_monotonic_decreasing
    k = estadistico_ks_gini(y, p)
    assert 0.3 < k["ks"] < 0.6 and k["gini"] == pytest.approx(2 * k["auc"] - 1)
    pr = curva_precision_recall(y, p)
    assert pr["precision_media"] > pr["prevalencia"]
    bien, mal = descomposicion_brier(y, p), descomposicion_brier(y, np.clip(2 * p, 0, 1))
    assert bien["fiabilidad"] < 0.002 < mal["fiabilidad"]
    assert bien["brier"] == pytest.approx(bien["fiabilidad"] - bien["resolucion"] + bien["incertidumbre"], abs=0.003)
    c = calibrar_probabilidades(y[:3000], np.clip(2 * p[:3000], 0, 1), np.clip(2 * p[3000:], 0, 1))
    assert abs(c["probabilidades"].mean() - y[3000:].mean()) < 0.015
    with pytest.raises(ValueError):
        tabla_ganancia_lift([0, 0, 0], [0.1, 0.2, 0.3])
