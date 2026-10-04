"""Rama finanzas (0.9): resultados de libro (Black-Scholes, paridad, bonos, CRR) y propiedades de riesgo."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from arbol_estadistica.finanzas import (ajustar_gpd, arbol_binomial, beta_capm, black_litterman, black_scholes,
                                        bootstrapping_etti, dominancia_estocastica, equivalente_cierto, estimador_hill,
                                        fraccion_anio, frontera_eficiente, funcion_exceso_medio, inmunizacion,
                                        matriz_covarianzas, modelo_factores, precio_bono, prueba_estres, simular_copula,
                                        tir_bono, valorar_swap, var_tvar)


def test_var_tvar_y_evt():
    x = stats.t.rvs(3, size=40000, random_state=np.random.default_rng(0))
    h = var_tvar(x)
    assert (h.tvar >= h["var"]).all() and h.loc[0.99, "var"] == pytest.approx(stats.t.ppf(0.99, 3), rel=0.08)
    assert var_tvar(x, metodo="normal").loc[0.995, "var"] < h.loc[0.995, "var"]       # la normal subestima la cola
    g = ajustar_gpd(x, cuantil_umbral=0.95)
    assert g["xi"] == pytest.approx(1 / 3, abs=0.12) and g["tabla"].loc[0.999, "var"] == pytest.approx(stats.t.ppf(0.999, 3), rel=0.2)
    hill = estimador_hill(np.abs(x))
    assert hill.loc[200, "xi"] == pytest.approx(1 / 3, abs=0.1)
    em = funcion_exceso_medio(np.random.default_rng(1).exponential(2, 20000))
    assert em.exceso_medio.std() / em.exceso_medio.mean() < 0.1                     # exponencial: exceso medio plano


@pytest.mark.parametrize("tipo,param,tau", [("gaussiana", 0.5, 2 / np.pi * np.arcsin(0.5)), ("t", 0.5, 2 / np.pi * np.arcsin(0.5)),
                                            ("clayton", 2.0, 0.5), ("gumbel", 2.0, 0.5)])
def test_copulas_reproducen_su_tau(tipo, param, tau):
    c = simular_copula(tipo, param, 20000)
    assert stats.kendalltau(c.u, c.v)[0] == pytest.approx(tau, abs=0.03) and c.u.between(0, 1).all()


def test_colas_de_clayton_y_gumbel():
    cl, gu = simular_copula("clayton", 2, 40000), simular_copula("gumbel", 2, 40000)
    abajo = lambda c: ((c.u < .05) & (c.v < .05)).mean() / .05                          # noqa: E731
    arriba = lambda c: ((c.u > .95) & (c.v > .95)).mean() / .05                         # noqa: E731
    assert abajo(cl) > arriba(cl) and arriba(gu) > abajo(gu)


def test_carteras_capm_y_bl():
    rng = np.random.default_rng(2)
    R = pd.DataFrame(rng.multivariate_normal([.08, .05, .03], [[.04, .006, .001], [.006, .01, .0005], [.001, .0005, .0025]], 600), columns=list("abc"))
    mc = matriz_covarianzas(R)
    assert 0 <= mc["contraccion"] <= 1
    fe = frontera_eficiente(mc["medias"], mc["covarianza"], 12, tasa_libre=0.01)
    assert fe["frontera"].volatilidad.min() == pytest.approx(fe["minima_varianza_rent_vol"][1], rel=1e-3)
    assert fe["tangente"].sum() == pytest.approx(1) and (fe["tangente"] >= -1e-9).all()
    m = rng.normal(.006, .04, 300); a = 1.3 * m + rng.normal(0, .02, 300)
    c = beta_capm(a, m)
    assert c["ic_beta"][0] < 1.3 < c["ic_beta"][1] and c["p_alfa"] > 0.01
    assert modelo_factores(pd.DataFrame({"a": a}), pd.DataFrame({"mkt": m}))["beta_mkt"].iloc[0] == pytest.approx(c["beta"], rel=1e-6)
    w = np.array([.5, .3, .2])
    sin_op = black_litterman(mc["covarianza"], w, [[1, 0, 0]], [0.0], confianza=[1e6])
    assert np.allclose(sin_op["pesos"], w, atol=1e-3)                                # sin opinión creíble = pesos de mercado
    con_op = black_litterman(mc["covarianza"], w, [[1, -1, 0]], [0.10])
    assert con_op["pesos"]["a"] > w[0]


def test_dominancia_y_utilidad():
    rng = np.random.default_rng(3)
    assert dominancia_estocastica(rng.normal(1, 1, 3000), rng.normal(0, 1, 3000))["A_domina_FSD"]
    a, b = rng.normal(0, 1, 3000), rng.normal(0, 3, 3000)
    d = dominancia_estocastica(a - a.mean(), b - b.mean())
    assert not d["A_domina_FSD"] and d["A_domina_SSD"]
    e = equivalente_cierto([0, 100], [.5, .5], "exponencial", 0.02)
    assert 0 < e["equivalente_cierto"] < 50 and e["prima_riesgo"] > 0
    assert equivalente_cierto([50, 50], utilidad="logaritmica")["prima_riesgo"] == pytest.approx(0, abs=1e-12)


def test_bonos_etti_inmunizacion_swap():
    assert precio_bono(100, .05, 10, .05)["precio"] == pytest.approx(100)
    pb = precio_bono(100, .05, 10, .04)
    assert tir_bono(pb["precio"], 100, .05, 10) == pytest.approx(.04)
    dP = precio_bono(100, .05, 10, .041)["precio"] - pb["precio"]
    aprox = -pb["duracion_modificada"] * .001 * pb["precio"] + .5 * pb["convexidad"] * .001 ** 2 * pb["precio"]
    assert dP == pytest.approx(aprox, rel=0.01)
    assert precio_bono(100, 0, 5, .03)["duracion_macaulay"] == pytest.approx(5)
    e = bootstrapping_etti([1, 2, 3], [.03, .04, .05], [100.5, 101, 102])
    assert (e.factor_descuento.diff().dropna() < 0).all()
    assert 100 * .04 * e.factor_descuento.iloc[0] + 104 * e.factor_descuento.iloc[1] == pytest.approx(101)
    inm = inmunizacion(5, 1e6, pd.DataFrame({"precio": [98, 105], "duracion": [2, 9]}, index=["c", "l"]))
    assert inm["pesos"] @ np.array([2, 9]) == pytest.approx(5) and inm["importe"].sum() == pytest.approx(1e6)
    s = valorar_swap(0.0, e.factor_descuento)
    assert valorar_swap(s["tipo_swap_par"], e.factor_descuento)["valor_pagador_fijo"] == pytest.approx(0, abs=1e-12)
    assert fraccion_anio("2024-01-01", "2025-01-01", "act/act") == pytest.approx(1)
    assert fraccion_anio("2024-01-31", "2024-03-31", "30/360") == pytest.approx(60 / 360)


def test_opciones_black_scholes_paridad_y_crr():
    c, p = black_scholes(100, 100, 1, .03, .2), black_scholes(100, 100, 1, .03, .2, "put")
    assert c["precio"] == pytest.approx(9.4134, abs=1e-3)
    assert c["precio"] - p["precio"] == pytest.approx(100 - 100 * np.exp(-.03), abs=1e-9)           # paridad put-call
    assert arbol_binomial(100, 100, 1, .03, .2, 800)["precio"] == pytest.approx(c["precio"], abs=0.01)
    am = arbol_binomial(100, 100, 1, .03, .2, 400, "put", americana=True)["precio"]
    assert am > p["precio"]
    assert arbol_binomial(100, 100, 1, .03, .2, 400, "call", americana=True)["precio"] == pytest.approx(c["precio"], abs=0.02)


def test_prueba_estres():
    t = prueba_estres({"rv": 1e6, "bonos": 5e6}, {"crisis": {"rv": -.39, "bonos": -.05}, "tipos": {"bonos": -.10}})
    assert t.index[0] == "crisis" and t.loc["crisis", "perdida_total"] == pytest.approx(640000)
