import numpy as np
import pandas as pd

from arbol_estadistica.seleccion import contraste_chi2_variable, cribar_variables, es_posible_fuga, seleccion_forward


def test_chi2_variable_detecta_asociacion(datos_bin):
    r = contraste_chi2_variable(datos_bin, "region", "y")
    assert r["p_valor"] < 0.001 and 0 < r["cramer_v"] < 1 and r["n_categorias"] == 3
    assert contraste_chi2_variable(datos_bin.assign(c=1), "c", "y") is None


def test_cribar_variables_motivos(datos_bin):
    df = datos_bin.assign(y_pred=datos_bin.y, constante=7)
    antes = df.copy()
    t = cribar_variables(df, "y").set_index("variable")
    pd.testing.assert_frame_equal(df, antes)  # no muta
    assert t.loc["y", "motivo"] == "target"
    assert t.loc["y_pred", "motivo"] == "posible_fuga_o_variable_modelo"
    assert t.loc["constante", "motivo"] == "sin_variabilidad"
    assert t.loc["x1", "incluir"] and t.loc["region", "incluir"]
    assert not t.loc["ruido", "incluir"]  # ruido puro no debe pasar el cribado


def test_es_posible_fuga():
    assert es_posible_fuga("prob_renovar") and es_posible_fuga("y_max") and es_posible_fuga("tasa_pred")
    assert not es_posible_fuga("edad")


def test_forward_elige_señal_y_bic_descarta_ruido(datos_bin):
    # AIC es permisivo: una variable de ruido entra ~16 % de las veces (p~0.12 aquí); BIC la rechaza.
    r = seleccion_forward(datos_bin, "y", ["x1", "x2", "region", "ruido"], categoricas=["region"])
    assert {"x1", "x2", "region"} <= set(r["seleccionadas"])
    rb = seleccion_forward(datos_bin, "y", ["x1", "x2", "region", "ruido"], categoricas=["region"], criterio="bic")
    assert set(rb["seleccionadas"]) == {"x1", "x2", "region"}
    assert "C(region)" in r["formula"]
    assert r["historial"]["criterio"].is_monotonic_decreasing
    # familia binomial por defecto (el original usaba gaussiana sin querer)
    assert r["modelo"].model.family.__class__.__name__ == "Binomial"


def test_forward_bic_y_base(datos_bin):
    r = seleccion_forward(datos_bin, "y", ["x2", "ruido"], base=["x1"], criterio="bic")
    assert r["seleccionadas"][0] == "x1" and "x2" in r["seleccionadas"]
