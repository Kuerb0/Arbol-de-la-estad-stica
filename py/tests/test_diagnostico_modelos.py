import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from arbol_estadistica.diagnostico import (
    auc_train_test, calcular_vif, estadisticos_asociacion, filtrar_vif_iterativo,
    hosmer_lemeshow, matrices_confusion, tabla_umbrales, umbral_optimo_youden,
)
from arbol_estadistica.modelos import (
    ajustar_glm_binomial, ajustar_logit, comparar_enlaces, comparar_tecnicas_estimacion,
    dividir_train_test, ganancia_por_bajada_precio, logit_multinomial_sas, perfil_respuesta,
    preparar_matriz_modelo, regresion_logistica_firth, sensibilidad_por_grupo,
    tabla_odds_ratios, tabla_parametros_wald, tendencia_polinomica_ponderada,
)
from arbol_estadistica.preprocesado import pesos_por_clase


# ---------- VIF ----------
def test_vif_detecta_colinealidad_y_filtro_la_elimina(datos_bin):
    X = datos_bin[["x1", "x2"]].copy()
    X["x1_copia"] = X.x1 * 2 + np.random.default_rng(3).normal(scale=0.01, size=len(X))
    X = sm.add_constant(X)
    v = calcular_vif(X).set_index("variable")
    assert v.loc["x1", "VIF"] > 10 and v.loc["x1_copia", "diagnostico"].startswith("ALTO")
    assert "const" not in v.index
    cols = filtrar_vif_iterativo(X, umbral=8)
    assert "const" in cols and not ({"x1", "x1_copia"} <= set(cols))
    assert "x1" in filtrar_vif_iterativo(X, umbral=8, protegidas=["x1"])


# ---------- asociación: exacta y == fuerza bruta ----------
def test_asociacion_exacta_vs_fuerza_bruta():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 300)
    p = np.round(rng.random(300) * 0.6 + 0.2 * y, 2)  # redondeo -> muchos empates
    a = estadisticos_asociacion(y, p)
    d = p[y == 1][:, None] - p[y == 0][None, :]
    conc, disc, emp = (d > 0).sum(), (d < 0).sum(), (d == 0).sum()
    assert (a["concordantes"], a["discordantes"], a["empates"]) == (conc, disc, emp)
    assert np.isclose(a["c"], roc_auc_score(y, p))
    assert np.isclose(a["Somers_D"], 2 * a["c"] - 1)


def test_hosmer_lemeshow_modelo_bien_calibrado(datos_bin):
    X, y, _ = preparar_matriz_modelo(datos_bin, "y", ["x1", "x2", "region"], ["region"], ["x1", "x2"])
    m = ajustar_logit(y, X)
    hl = hosmer_lemeshow(y, m.predict(sm.add_constant(X)))
    assert hl["gl"] == 8 and hl["p_valor"] > 0.01 and hl["tabla"]["N"].sum() == len(y)
    # un modelo claramente mal calibrado debe rechazar
    mal = hosmer_lemeshow(y, np.clip(m.predict(sm.add_constant(X)) * 0.3, 0.001, 0.999))
    assert mal["p_valor"] < 0.001 and not mal["ajuste_aceptable"]


def test_umbrales_y_youden(datos_bin):
    p = 1 / (1 + np.exp(-(datos_bin.x1 - 0.2)))
    t = tabla_umbrales(datos_bin.y, p)
    assert {"sens", "espec", "Youden_J"} <= set(t.columns) and len(t) == 17
    assert 0 < umbral_optimo_youden(datos_bin.y, p) < 1


def test_matrices_confusion():
    cm, cmf = matrices_confusion(["a", "b", "a", "b"], ["a", "a", "a", "b"], ["a", "b"])
    assert cm.to_numpy().tolist() == [[2, 0], [1, 1]] and np.allclose(cmf.sum(axis=1), 1)


# ---------- logit estilo PROC LOGISTIC ----------
def test_preparar_matriz_referencias_y_estandarizacion(datos_bin):
    X, y, sd = preparar_matriz_modelo(datos_bin, "y", ["x1", "region"], ["region"], ["x1"], refs={"region": "N"})
    assert "region_N" not in X.columns and {"region_S", "region_E"} <= set(X.columns)
    assert abs(X.x1.mean()) < 1e-9 and np.isclose(X.x1.std(), 1) and np.isclose(sd["x1"], datos_bin.x1.std())
    with pytest.raises(ValueError):
        preparar_matriz_modelo(datos_bin, "y", ["region"], ["region"], refs={"region": "XX"})


def test_odds_ratios_recuperan_parametros_y_unidades(datos_bin):
    X, y, sd = preparar_matriz_modelo(datos_bin, "y", ["x1", "x2", "region"], ["region"], ["x1", "x2"],
                                      refs={"region": "N"})
    m = ajustar_logit(y, X)
    t = tabla_odds_ratios(m, std_map=sd, unidades={"x1": 1})
    assert abs(np.log(t.loc["x1", "OR"]) - 1.0) < 0.15      # beta real = 1.0
    assert abs(np.log(t.loc["x2", "OR"]) + 0.5) < 0.15      # beta real = -0.5
    assert abs(np.log(t.loc["region_S", "OR"]) - 0.6) < 0.2
    assert (t["IC_2.5%"] < t["OR"]).all() and (t["OR"] < t["IC_97.5%"]).all()
    t2 = tabla_odds_ratios(m, std_map=sd, unidades={"x1": 2})
    assert np.isclose(np.log(t2.loc["x1", "OR"]), 2 * np.log(t.loc["x1", "OR"]))  # UNITS=2 duplica el log-OR


def test_wald_y_perfil(datos_bin):
    X, y, _ = preparar_matriz_modelo(datos_bin, "y", ["x1", "x2"], numericas=["x1", "x2"])
    m = ajustar_logit(y, X)
    w = tabla_parametros_wald(m)
    assert np.allclose(w["Pr>chi2"], m.pvalues, atol=1e-6)
    assert np.isclose(perfil_respuesta(y)["proporcion"].sum(), 1)


def test_logit_ignora_freq_weights_pero_ajustar_logit_con_pesos_si_los_usa(datos_bin):
    """Documenta el fallo del notebook de consultoría: sm.Logit(freq_weights=...) NO pondera."""
    X, y, _ = preparar_matriz_modelo(datos_bin, "y", ["x1", "x2"], numericas=["x1", "x2"])
    Xc = sm.add_constant(X)
    w = pesos_por_clase(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sin_pesos = sm.Logit(y, Xc).fit(disp=False).params
        mal = sm.Logit(y, Xc, freq_weights=w).fit(disp=False).params
    bien = ajustar_logit(y, X, pesos=w).params
    assert np.allclose(sin_pesos, mal)                  # Logit ignora los pesos
    assert not np.allclose(sin_pesos["const"], bien["const"], atol=1e-3)  # GLM sí pondera (cambia el intercepto)


def test_firth_coincide_con_newton_en_datos_grandes_y_converge_con_separacion():
    rng = np.random.default_rng(2)
    n = 3000
    X = sm.add_constant(pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}))
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.2 + 0.8 * X.a - 0.4 * X.b)))).astype(int)
    f = regresion_logistica_firth(X, y)
    nw = sm.Logit(y, X).fit(disp=False)
    assert f["convergio"] and np.allclose(f["beta"], nw.params, atol=0.02)
    # separación perfecta: Newton explota, Firth da coeficientes finitos
    Xs = sm.add_constant(pd.DataFrame({"a": [-3, -2, -1, 1, 2, 3.0]}))
    ys = np.array([0, 0, 0, 1, 1, 1])
    fs = regresion_logistica_firth(Xs, ys)
    assert np.isfinite(fs["beta"]).all() and fs["beta"]["a"] > 0


def test_comparativas(datos_bin):
    tr, te = dividir_train_test(datos_bin, "y")
    cols = ["x1", "x2"]
    Xtr, Xte = sm.add_constant(tr[cols]), sm.add_constant(te[cols])
    tabla, delta = comparar_tecnicas_estimacion(tr.y, Xtr, te.y, Xte)
    assert len(tabla) == 3 and delta < 0.05 and tabla["AUC_test"].min() > 0.7
    assert list(comparar_enlaces(tr.y, Xtr, te.y, Xte)["metodo"]) == ["Logit", "Probit", "CLogLog"]


def test_glm_binomial_y_split_estratificado(datos_bin):
    tr, te = dividir_train_test(datos_bin, "y")
    assert abs(tr.y.mean() - te.y.mean()) < 0.005 and len(tr) == 2800
    r = ajustar_glm_binomial("y ~ x1 + x2 + C(region)", tr, te, "y")
    assert r["auc_test"] > 0.7 and "OR" in r["tabla"] and not r["alerta_sobreajuste"]
    assert r["aic"] > 0 and np.isfinite(r["bic"])
    assert auc_train_test(r["modelo"], tr, te, "y")["gap"] == pytest.approx(r["gap"])


# ---------- multinomial ----------
def test_multinomial_sas(datos_multi):
    r = logit_multinomial_sas(datos_multi, "clase", x_num=["x1", "x2"], x_cat=["g"],
                              base_class="base", ref_levels={"g": "a"})
    c = r["coeficientes"]
    assert set(c["comparacion"]) == {"media vs base", "alta vs base"}
    b_x1 = c[(c.comparacion == "media vs base") & (c.variable == "x1")].beta.iloc[0]
    assert abs(b_x1 - 1.2) < 0.2
    p = r["probabilidades_predichas"]
    assert np.allclose(p.sum(axis=1), 1) and list(p.columns)[0] == "base"
    assert r["auc"]["macro_ovr"] > 0.65 and set(r["auc"]["por_clase_ovr"]["clase"]) == {"base", "media", "alta"}
    assert r["ajuste"]["n"] == len(datos_multi) and 0 < r["ajuste"]["pseudo_r2_mcfadden"] < 1
    ls = r["lsmeans_like"]("g")
    assert set(ls["nivel"]) == {"a", "b"} and np.allclose(ls[["base", "media", "alta"]].sum(axis=1), 1)
    assert "x1" in set(r["vif"]["variable"]) and "Intercept" not in set(r["vif"]["variable"])
    with pytest.raises(ValueError):
        r["lsmeans_like"]("x1")
    with pytest.raises(ValueError):
        logit_multinomial_sas(datos_multi, "clase", x_num=["x1"], base_class="inexistente")


# ---------- elasticidad ----------
def test_elasticidad_sensibilidad_y_ganancia():
    rng = np.random.default_rng(4)
    filas = []
    for g, pend in {"sensible": -0.25, "rigido": -0.02}.items():
        precio = rng.normal(30000, 5000, 600)
        p = np.clip(0.4 + pend * (precio - 30000) / 5000 + rng.normal(0, 0.05, 600), 0, 1)
        filas.append(pd.DataFrame({"seg": g, "precio": precio, "vende": (rng.random(600) < p).astype(int)}))
    df = pd.concat(filas)
    s = sensibilidad_por_grupo(df, "vende", "precio", "seg")
    assert s.iloc[0]["seg"] == "sensible" and s.iloc[0]["sensibilidad"] < s.iloc[1]["sensibilidad"] < 0.05
    g = ganancia_por_bajada_precio(df, "vende", "precio", "seg")
    assert g.iloc[0]["seg"] == "sensible" and g.iloc[0]["ganancia_pp"] > 0
    assert sensibilidad_por_grupo(df, "vende", "precio", "seg", min_n=10_000).empty


def test_tendencia_polinomica_ponderada_recupera_optimo():
    x = np.linspace(10, 50, 40)
    y = -0.001 * (x - 30) ** 2 + 0.5
    coef, f = tendencia_polinomica_ponderada(x, y, pesos=np.linspace(1, 5, 40), grado=2)
    assert coef[0] < 0 and abs(-coef[1] / (2 * coef[0]) - 30) < 0.5 and np.isclose(f(30), 0.5, atol=1e-6)
