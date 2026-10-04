import numpy as np
import pandas as pd

from arbol_estadistica.preprocesado import (
    balancear_clases, categorizar_por_cuantiles, codificar_ordinal,
    duracion_hasta_evento, pesos_por_clase, submuestreo_por_ratio,
    corregir_probabilidades_por_balanceo,
)


def test_categorizar_no_muta_y_crea_tramos(datos_bin):
    original = datos_bin.copy()
    out, creadas = categorizar_por_cuantiles(datos_bin, ["x1", "x2", "no_existe"], q=5)
    pd.testing.assert_frame_equal(datos_bin, original)  # no muta
    assert creadas == ["x1_cat", "x2_cat"]
    assert out["x1_cat"].nunique() == 5
    assert list(out["x1_cat"].cat.categories) == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_categorizar_importes_usan_deciles_y_toleran_duplicados():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"annual_premium": rng.normal(size=500),
                       "binaria": [0] * 450 + [1] * 50})  # cuantiles repetidos
    out, creadas = categorizar_por_cuantiles(df, ["annual_premium", "binaria"])
    assert "annual_premium_cat" in creadas and out["annual_premium_cat"].nunique() == 10
    assert out["annual_premium_cat"].cat.categories[0] == "D1"


def test_balancear_oversample_y_undersample():
    df = pd.DataFrame({"c": ["a"] * 90 + ["b"] * 10, "v": range(100)})
    assert balancear_clases(df, "c", "oversample")["c"].value_counts().tolist() == [90, 90]
    assert balancear_clases(df, "c", "undersample")["c"].value_counts().tolist() == [10, 10]
    assert len(balancear_clases(df, "c", None)) == 100


def test_pesos_y_submuestreo():
    y = pd.Series([0] * 90 + [1] * 10)
    w = pesos_por_clase(y)
    assert np.isclose(w[y == 1].iloc[0], 100 / (2 * 10)) and np.isclose(w[y == 0].iloc[0], 100 / (2 * 90))
    assert np.isclose(w.sum(), 100)  # los pesos conservan el total
    df = pd.DataFrame({"y": y, "v": range(100)})
    s = submuestreo_por_ratio(df, "y", 3)
    assert (s.y == 1).sum() == 10 and (s.y == 0).sum() == 30


def test_codificar_ordinal_con_categoria_extra_y_nulos():
    df = pd.DataFrame({"t": ["High", "Normal", None, "Raro"]})
    out, mapa = codificar_ordinal(df, "t", ["Normal", "High"])
    assert mapa == {"Normal": 0, "High": 1, "Raro": 2}
    assert out["t"].tolist() == [1, 0, -1, 2]


def test_duracion_hasta_evento():
    df = pd.DataFrame({"id": [1, 1, 1, 2, 2, 3, 3, 3],
                       "anio": [1, 2, 3, 1, 2, 1, 2, 3],
                       "renewed": [0, 0, 1, 1, 0, 0, 0, 0]})
    d = duracion_hasta_evento(df, "id", "renewed", "anio")
    assert d.to_dict() == {1: 2, 2: 0, 3: 3}


def test_correccion_prior_tras_submuestreo_recupera_calibracion():
    import statsmodels.api as sm
    from arbol_estadistica.diagnostico import hosmer_lemeshow
    rng = np.random.default_rng(10)
    n = 60000
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-3.0 + 1.0 * x)))).astype(int)   # prevalencia ~ 7 %
    df = pd.DataFrame({"x": x, "y": y})
    bal = submuestreo_por_ratio(df, "y", 1)
    m = sm.Logit(bal.y, sm.add_constant(bal[["x"]])).fit(disp=False)
    p_raw = m.predict(sm.add_constant(df[["x"]]))
    p_cor = corregir_probabilidades_por_balanceo(p_raw, df.y.mean(), bal.y.mean())
    assert p_raw.mean() > 0.3 and abs(p_cor.mean() - df.y.mean()) < 0.01        # sin corregir: muy inflada
    assert hosmer_lemeshow(df.y, p_cor)["p_valor"] > 0.01 > hosmer_lemeshow(df.y, p_raw)["p_valor"]
    from sklearn.metrics import roc_auc_score
    assert np.isclose(roc_auc_score(df.y, p_raw), roc_auc_score(df.y, p_cor))   # el AUC no cambia
