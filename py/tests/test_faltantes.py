"""Faltantes, imputación (simple y múltiple), categorías raras, codificación por objetivo y SMOTE (0.9)."""
import numpy as np
import pandas as pd
import pytest

from arbol_estadistica.preprocesado import (agrupar_categorias_raras, codificar_por_objetivo, contraste_mcar_little,
                                            imputacion_multiple, imputar, resumen_faltantes, smote)


def _datos(seed=0, n=600):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "z": rng.choice(["x", "y"], n)})
    d["c"] = d.a + d.b + rng.normal(0, .5, n)
    return d, rng


def test_little_distingue_mcar_de_mar_y_resumen():
    d, rng = _datos()
    mcar = d.copy(); mcar.loc[rng.random(len(d)) < .2, "c"] = np.nan
    mar = d.copy(); mar.loc[(d.a > .5) & (rng.random(len(d)) < .6), "c"] = np.nan
    assert contraste_mcar_little(mcar[["a", "b", "c"]])["p_valor"] > 0.01
    assert contraste_mcar_little(mar[["a", "b", "c"]])["p_valor"] < 1e-6
    r = resumen_faltantes(mar)
    assert r["por_columna"].index[0] == "c" and r["filas_completas_pct"] < 100 and len(r["patrones"]) == 2


def test_imputar_iterativa_supera_a_la_mediana_y_no_muta():
    d, rng = _datos(1)
    mar = d.copy(); mar.loc[(d.a > .5) & (rng.random(len(d)) < .6), "c"] = np.nan; mar.loc[:5, "z"] = None
    copia = mar.copy()
    med, it = imputar(mar, "mediana"), imputar(mar, "iterativa")
    assert not med.isna().any().any() and not it.isna().any().any()
    assert np.corrcoef(it.c, d.c)[0, 1] > np.corrcoef(med.c, d.c)[0, 1] + 0.05
    assert "c_falta" in imputar(mar, "knn", indicadores=True)
    pd.testing.assert_frame_equal(mar, copia)
    with pytest.raises(ValueError):
        imputar(mar, "otro")


def test_imputacion_multiple_reglas_de_rubin():
    d, rng = _datos(2)
    mar = d[["a", "b", "c"]].copy(); mar.loc[(d.a > .5) & (rng.random(len(d)) < .6), "c"] = np.nan
    r = imputacion_multiple(mar, "c ~ a + b", m=8)
    t = r["tabla"]
    assert t.loc["a", "IC_inf"] < 1 < t.loc["a", "IC_sup"] and (t.fmi > 0).all() and r["estimaciones"].shape[1] == 8
    completo = imputar(mar, "iterativa")
    import statsmodels.formula.api as smf
    assert t.loc["a", "SE"] > smf.ols("c ~ a + b", completo).fit().bse["a"]      # la simple subestima la incertidumbre


def test_raras_codificacion_y_smote():
    rng = np.random.default_rng(3)
    z = pd.DataFrame({"k": rng.choice(list("ABCDEFG"), 2000, p=[.4, .3, .2, .05, .03, .01, .01])})
    out, raras = agrupar_categorias_raras(z, "k", 0.02)
    assert set(raras) == {"F", "G"} and "Otros" in set(out.k)
    z["y"] = (rng.random(2000) < z.k.map(dict(zip("ABCDEFG", [.1, .2, .3, .4, .5, .6, .7])))).astype(int)
    enc, tabla = codificar_por_objetivo(z, "k", "y", suavizado=10)
    assert enc.notna().all() and tabla.loc["A", "codificado"] < tabla.loc["E", "codificado"]
    assert abs(tabla.loc["G", "codificado"] - tabla.attrs["media_global"]) < abs(tabla.loc["G", "media_bruta"] - tabla.attrs["media_global"])
    dd = pd.DataFrame({"x1": rng.normal(size=500), "x2": rng.normal(size=500), "y": (rng.random(500) < .08).astype(int)})
    s = smote(dd, "y")
    assert (s.y == 1).sum() == (s.y == 0).sum() and s.sintetico.sum() == s.y.eq(1).sum() - dd.y.sum()
    sint = s[s.sintetico == 1]
    assert sint.x1.between(dd.x1.min(), dd.x1.max()).all()
