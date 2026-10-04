"""Descriptiva: casos donde la teoría da la respuesta."""
import numpy as np
import pandas as pd
import pytest

from arbol_estadistica.descriptiva import (ajustar_distribucion_discreta, ajustar_distribuciones, contraste_normalidad,
                                          correlacion_con_ic, correlacion_parcial, detectar_atipicos, estimar_densidad,
                                          funcion_distribucion_empirica, homogeneidad_varianzas, indice_dispersion,
                                          matriz_correlaciones, medidas_riesgo_2x2, resumen_descriptivo, tabla_contingencia)

rng = np.random.default_rng(3)


def test_resumen_descriptivo_cuadra_con_numpy_y_no_muta():
    rng = np.random.default_rng(101)
    df = pd.DataFrame({"x": [1.0, 2, 3, 4, 100, np.nan], "t": list("abcdef")})
    copia = df.copy()
    r = resumen_descriptivo(df, recorte=0.2)
    assert list(r.index) == ["x"] and r.loc["x", "n"] == 5 and r.loc["x", "faltantes"] == 1
    assert r.loc["x", "mediana"] == 3 and r.loc["x", "media"] == pytest.approx(22)
    assert r.loc["x", "media_recortada"] < r.loc["x", "media"]          # el 100 pesa menos
    pd.testing.assert_frame_equal(df, copia)
    with pytest.raises(ValueError):
        resumen_descriptivo(df, recorte=0.6)


def test_detectar_atipicos_robusto_frente_a_z():
    rng = np.random.default_rng(102)
    x = np.r_[rng.normal(0, 1, 50), [8, 9, 10, 11, 12, 13]]
    assert detectar_atipicos(x, "mad").atipico.sum() >= 6
    assert detectar_atipicos(x, "iqr").atipico.sum() >= 6
    # con muchos atípicos la desviación se infla y z los esconde (por eso z no es robusto)
    assert detectar_atipicos(x, "z").atipico.sum() < detectar_atipicos(x, "mad").atipico.sum()
    with pytest.raises(ValueError):
        detectar_atipicos(x, "otro")


def test_correlacion_ic_contiene_rho_y_parcial_elimina_confusion():
    rng = np.random.default_rng(103)
    z = rng.normal(size=2000)
    x, y = z + rng.normal(size=2000), z + rng.normal(size=2000)          # correlación 0.5 solo por z
    r = correlacion_con_ic(x, y)
    assert r["ic_inf"] < 0.5 < r["ic_sup"]
    p = correlacion_parcial(pd.DataFrame({"x": x, "y": y, "z": z}), "x", "y", ["z"])
    assert abs(p["parcial"]) < 0.06 and p["bruta"] > 0.4
    for m in ("spearman", "kendall"):
        assert correlacion_con_ic(x, y, m)["r"] > 0.3
    mc = matriz_correlaciones(pd.DataFrame({"x": x, "y": y, "z": z}))
    assert mc["r"].shape == (3, 3) and mc["r"].loc["x", "x"] == 1 and mc["n"].loc["x", "y"] == 2000


def test_normalidad_rechaza_lognormal_y_no_normal():
    rng = np.random.default_rng(104)
    assert contraste_normalidad(rng.lognormal(0, 1, 300))["tabla"]["rechaza_normalidad"].all()
    assert contraste_normalidad(rng.normal(size=300))["tabla"]["rechaza_normalidad"].mean() <= 0.4
    assert contraste_normalidad(rng.normal(size=20))["avisos"]


def test_homogeneidad_detecta_varianzas_distintas():
    rng = np.random.default_rng(105)
    d = pd.DataFrame({"g": np.repeat(["a", "b"], 200), "v": np.r_[rng.normal(0, 1, 200), rng.normal(0, 3, 200)]})
    assert homogeneidad_varianzas(d, "v", "g")["rechaza_igualdad"].all()


def test_ajustar_distribuciones_elige_la_verdadera():
    rng = np.random.default_rng(106)
    t = ajustar_distribuciones(rng.gamma(2.0, 3.0, 3000), candidatas=("gamma", "lognorm", "expon", "norm"))
    assert t.loc[0, "distribucion"] == "gamma"
    forma, _, escala = t.loc[0, "parametros"]
    assert forma == pytest.approx(2.0, rel=0.1) and escala == pytest.approx(3.0, rel=0.1)


def test_discreta_y_dispersion_detectan_binomial_negativa():
    rng = np.random.default_rng(107)
    nb = rng.negative_binomial(2, 0.4, 2000)              # media 3, var 7.5
    t = ajustar_distribucion_discreta(nb)
    assert t.loc[0, "distribucion"] == "binomial negativa" and t.attrs["p_valor_lr"] < 0.001
    assert t.loc[0, "parametros"]["r"] == pytest.approx(2, rel=0.2)
    assert indice_dispersion(nb)["p_sobredispersion"] < 0.001
    assert indice_dispersion(rng.poisson(3, 2000))["p_sobredispersion"] > 0.01
    with pytest.raises(ValueError):
        ajustar_distribucion_discreta([1.5, 2, 3, 4, 5])


def test_ecdf_banda_y_densidad_integran():
    rng = np.random.default_rng(108)
    x = rng.normal(size=500)
    e = funcion_distribucion_empirica(x)
    assert e["F"].iloc[-1] == 1 and (e["banda_inf"] <= e["F"]).all()
    d = estimar_densidad(x)
    assert np.trapezoid(d.densidad, d.x) == pytest.approx(1, abs=0.02)


def test_tabla_contingencia_y_riesgo_relativo():
    rng = np.random.default_rng(109)
    df = pd.DataFrame({"f": np.repeat(["a", "b"], 300), "c": np.r_[rng.random(300) < 0.6, rng.random(300) < 0.3]})
    t = tabla_contingencia(df, "f", "c")
    assert t["p_valor"] < 1e-6 and 0.2 < t["cramer_v"] < 0.45 and "p_fisher" in t
    assert t["residuos_ajustados"].abs().to_numpy().max() > 2
    m = medidas_riesgo_2x2(30, 100, 15, 100)
    assert m.loc["riesgo_relativo", "valor"] == pytest.approx(2) and m.loc["odds_ratio", "valor"] > 2
    assert m.loc["riesgo_relativo", "ic_inf"] > 1
