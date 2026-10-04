"""Modelos añadidos en 0.9: tarificación GLM, lineal completo, flexibles y supervivencia avanzada."""
import numpy as np
import pandas as pd
import pytest

from arbol_estadistica.modelos import (ajustar_cox, ajustar_glm_conteo, ajustar_glm_severidad, ajustar_glm_splines,
                                       ajustar_logit_ordinal, ajustar_modelo_mixto, ajustar_ols, ajustar_regularizado,
                                       ajustar_supervivencia_parametrica, ajustar_tweedie, ajustar_wls, coeficiente_icc,
                                       contraste_f_parcial, contraste_falta_ajuste, contraste_schoenfeld,
                                       contraste_sobredispersion, incidencia_acumulada, medidas_influencia, nelson_aalen,
                                       prima_pura, regresion_no_lineal, regresion_robusta, rmst, tabla_relatividades,
                                       transformacion_box_cox)


def _cartera(seed=0, n=6000, heterogeneidad=True):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"zona": rng.choice(["A", "B", "C"], n), "edad": rng.uniform(18, 80, n), "expo": rng.uniform(0.2, 1, n)})
    lam = 0.15 * np.exp(0.5 * (d.zona == "B") - 0.4 * (d.zona == "C") - 0.01 * (d.edad - 40)) * d.expo
    d["nsin"] = rng.poisson(lam * (rng.gamma(1.0, 1.0, n) if heterogeneidad else 1))
    return d, rng


def test_frecuencia_recupera_relatividades_y_detecta_sobredispersion():
    d, _ = _cartera()
    p = ajustar_glm_conteo("nsin ~ C(zona) + edad", d, "poisson", "expo")
    assert p["tabla"].loc["C(zona)[T.B]", "relatividad"] == pytest.approx(np.exp(0.5), rel=0.2)
    assert p["sobredispersion"]["p_valor"] < 0.01
    nb = ajustar_glm_conteo("nsin ~ C(zona) + edad", d, "negbin", "expo")
    assert nb["aic"] < p["aic"] and nb["alpha_nb"] > 0.3
    sin, _ = _cartera(1, heterogeneidad=False)
    assert contraste_sobredispersion(ajustar_glm_conteo("nsin ~ C(zona) + edad", sin, "poisson", "expo")["modelo"])["p_valor"] > 0.01
    rel = tabla_relatividades(nb, "zona")
    assert list(rel.index) == ["A", "B", "C"] and rel.loc["A", "relatividad"] == 1
    with pytest.raises(ValueError):
        ajustar_glm_conteo("nsin ~ edad", d.assign(expo=0), "poisson", "expo")


def test_severidad_prima_pura_y_tweedie():
    d, rng = _cartera(2)
    s = d[d.nsin > 0].copy()
    s["coste"] = rng.gamma(2.0, 1500 * np.exp(0.3 * (s.zona == "B")) / 2.0)
    sev = ajustar_glm_severidad("coste ~ C(zona)", s)
    assert sev["tabla"].loc["C(zona)[T.B]", "relatividad"] == pytest.approx(np.exp(0.3), rel=0.15)
    assert sev["dispersion"] == pytest.approx(0.5, rel=0.3)                  # 1/forma de la gamma
    fr = ajustar_glm_conteo("nsin ~ C(zona) + edad", d, "negbin", "expo")
    pp = prima_pura(fr, sev, d.head(50), "expo")
    assert np.allclose(pp.prima_pura, pp.frecuencia * pp.severidad) and (pp.prima_pura > 0).all()
    d["coste"] = 0.0; d.loc[s.index, "coste"] = s.coste * s.nsin
    tw = ajustar_tweedie("coste ~ C(zona) + edad", d, 1.5, "expo")
    assert tw["tabla"].loc["C(zona)[T.B]", "relatividad"] > 1.3


def test_ols_robusto_influencia_y_f_parcial():
    rng = np.random.default_rng(3)
    d = pd.DataFrame({"x": rng.uniform(0, 10, 400), "z": rng.normal(size=400)})
    d["y"] = 1 + 0.5 * d.x + rng.normal(0, 0.2 + 0.3 * d.x)
    r = ajustar_ols("y ~ x + z", d)
    assert r["diagnosticos"]["breusch_pagan_p"] < 0.01 and r["avisos"]
    clasico = ajustar_ols("y ~ x + z", d, robusto=None)
    assert r["tabla"].loc["x", "SE"] > clasico["tabla"].loc["x", "SE"]      # HC3 corrige el SE optimista
    d.loc[0, ["x", "y"]] = [30, -20]
    inf = medidas_influencia(ajustar_ols("y ~ x + z", d))
    assert inf.loc[0, "influyente"] and inf.loc[0, "alto_apalancamiento"] and inf.cook.idxmax() == 0
    f = contraste_f_parcial(ajustar_ols("y ~ x", d), ajustar_ols("y ~ x + z", d))
    assert f["gl_num"] == 1 and f["p_valor"] > 0.01


def test_box_cox_wls_robusta_no_lineal_y_falta_de_ajuste():
    rng = np.random.default_rng(4)
    assert transformacion_box_cox(rng.lognormal(0, 1, 500))["sugerencia"] == "logaritmo"
    assert transformacion_box_cox(rng.normal(size=100))["metodo"] == "yeo-johnson"
    d = pd.DataFrame({"x": rng.uniform(0, 10, 300)}); d["y"] = 2 + d.x + rng.standard_t(1.5, 300)
    hub, mco = regresion_robusta("y ~ x", d), ajustar_ols("y ~ x", d, robusto=None)
    assert abs(hub["tabla"].loc["x", "coef"] - 1) < abs(mco["tabla"].loc["x", "coef"] - 1) + 0.05
    assert regresion_robusta("y ~ x", d, "cuantil", 0.5)["tabla"].loc["x", "coef"] == pytest.approx(1, abs=0.15)
    assert ajustar_wls("y ~ x", d.assign(w=1.0), pesos="w")["r2"] > 0
    t = np.linspace(0, 10, 80); y = 5 * (1 - np.exp(-0.4 * t)) + rng.normal(0, 0.1, 80)
    nl = regresion_no_lineal(lambda t, a, b: a * (1 - np.exp(-b * t)), t, y, [1, 1], ["a", "b"])
    assert nl["tabla"].loc["a", "IC_inf"] < 5 < nl["tabla"].loc["a", "IC_sup"] and nl["r2"] > 0.95
    rep = pd.DataFrame({"x": np.repeat(np.arange(1, 8), 6)}); rep["y"] = (rep.x - 4) ** 2 + rng.normal(0, 1, len(rep))
    assert contraste_falta_ajuste(rep, "x", "y", 1)["p_valor"] < 1e-6 and contraste_falta_ajuste(rep, "x", "y", 2)["p_valor"] > 0.01


def test_splines_detectan_curvatura_y_mixto_recupera_icc():
    rng = np.random.default_rng(5)
    d = pd.DataFrame({"edad": rng.uniform(18, 80, 3000), "z": rng.normal(size=3000)})
    d["y"] = (rng.random(3000) < 1 / (1 + np.exp(-(-1 + 0.002 * (d.edad - 45) ** 2 + 0.3 * d.z)))).astype(int)
    r = ajustar_glm_splines(d, "y", {"edad": 4}, "z")
    assert r["p_no_linealidad"] < 1e-6 and r["aic"] < r["aic_lineal"]
    curva = r["curvas"]["edad"]
    assert curva.predictor_lineal.iloc[0] > curva.predictor_lineal.iloc[50] < curva.predictor_lineal.iloc[-1]   # forma de U
    g = pd.DataFrame({"g": np.repeat(np.arange(40), 25)}); g["x"] = rng.normal(size=1000)
    g["y"] = 1 + 0.5 * g.x + np.repeat(rng.normal(0, 1, 40), 25) + rng.normal(0, 1, 1000)     # ICC verdadero 0.5
    m = ajustar_modelo_mixto("y ~ x", g, "g")
    assert m["icc"] == pytest.approx(0.5, abs=0.15) and m["efectos_fijos"].loc["x", "coef"] == pytest.approx(0.5, abs=0.1)
    ic = coeficiente_icc(g, "y", "g")
    assert ic["ic_inf"] < 0.5 < ic["ic_sup"] and ic["efecto_diseno"] > 5


def test_ordinal_y_regularizacion():
    rng = np.random.default_rng(6)
    o = pd.DataFrame({"x": rng.normal(size=1500)})
    o["sat"] = pd.cut(1.0 * o.x + rng.logistic(size=1500), [-np.inf, -1, 0, 1.5, np.inf], labels=["mal", "reg", "bien", "muy"])
    r = ajustar_logit_ordinal(o, "sat", ["x"], orden=["mal", "reg", "bien", "muy"])
    assert r["tabla"].loc["x", "coef"] == pytest.approx(1.0, abs=0.15) and len(r["umbrales"]) == 3
    assert r["coef_por_corte"].loc["x"].std() < 0.2                       # odds proporcionales: mismo efecto en cada corte
    X = pd.DataFrame(rng.normal(size=(1000, 15)), columns=[f"v{i}" for i in range(15)])
    y = (rng.random(1000) < 1 / (1 + np.exp(-(1.2 * X.v0 - X.v1)))).astype(int)
    las = ajustar_regularizado(X, y, "lasso")
    sel = set(las["tabla"].query("seleccionada").index)
    assert {"v0", "v1"} <= sel and len(sel) <= 8
    assert ajustar_regularizado(X, y, "ridge")["n_seleccionadas"] == 15     # ridge no pone ceros
    yg = 2 * X.v0 + rng.normal(size=1000)
    assert set(ajustar_regularizado(X, yg, "lasso", "gaussiana")["tabla"].query("seleccionada").index) == {"v0"}


def _weibull(seed, n=800, ph=True):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"g": rng.choice(["A", "B"], n), "x": rng.normal(size=n)})
    lam = 0.05 * np.exp(0.5 * (d.g == "B") + 0.3 * d.x)
    forma = 1.5 if ph else np.where(d.g == "B", 0.6, 2.5)
    T = (-np.log(rng.random(n)) / lam) ** (1 / forma); C = rng.uniform(0, 15, n)
    d["t"], d["e"] = np.minimum(T, C), (T <= C).astype(int)
    return d


def test_supervivencia_parametrica_schoenfeld_y_rmst():
    d = _weibull(7)
    w = ajustar_supervivencia_parametrica(d, "t", "e", ["x"], "weibull", ["g"])
    assert w["forma_weibull"] == pytest.approx(1.5, rel=0.12)
    assert np.log(w["tabla"].loc["g_B", "HR_equivalente"]) == pytest.approx(0.5, abs=0.25)
    assert w["aic"] < ajustar_supervivencia_parametrica(d, "t", "e", ["x"], "exponencial", ["g"])["aic"]
    assert contraste_schoenfeld(ajustar_cox(d, "t", "e", ["x"], ["g"], {"g": "A"})).loc["GLOBAL", "p_valor"] > 0.01
    nph = _weibull(8, ph=False)
    assert contraste_schoenfeld(ajustar_cox(nph, "t", "e", ["x"], ["g"], {"g": "A"})).loc["g_B", "incumple_ph"]
    r = rmst(d, "t", "e", 10, "g")
    assert r.attrs["diferencia"] < 0 and r.attrs["p_valor"] < 0.001
    na = nelson_aalen(d, "t", "e")
    assert na.riesgo_acumulado.is_monotonic_increasing


def test_incidencia_acumulada_suma_con_la_supervivencia():
    d = _weibull(9)
    rng = np.random.default_rng(9)
    d["causa"] = np.where(d.e == 1, rng.choice([1, 2], len(d), p=[0.7, 0.3]), 0)
    c = incidencia_acumulada(d, "t", "causa")
    ult = c.iloc[-1]
    assert ult.cif_causa_1 + ult.cif_causa_2 + ult.supervivencia_global == pytest.approx(1, abs=1e-9)
    assert ult.cif_causa_1 > ult.cif_causa_2
