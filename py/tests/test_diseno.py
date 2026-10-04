"""Rama diseño: ANOVA y diseños, experimentos y causal (0.10)."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from arbol_estadistica.diseno import (aleatorizar_ensayo, analisis_intencion_tratar, ancova, anova_anidado, anova_bloques,
                                      anova_cuadrado_latino, anova_factorial, anova_medidas_repetidas, cuadrado_latino,
                                      diagnostico_anova, diferencias_en_diferencias, diseno_central_compuesto,
                                      diseno_factorial_2k, efectos_factorial_2k, mediacion, metaanalisis, moderacion,
                                      puntuacion_propension, superficie_respuesta)


def test_anova_factorial_detecta_interaccion_y_tipos():
    rng = np.random.default_rng(0)
    d = pd.DataFrame([(a, b) for a in "xyz" for b in "pq" for _ in range(20)], columns=["A", "B"])
    d["y"] = (d.A == "z") * 2 + (d.B == "q") * 1 + ((d.A == "z") & (d.B == "q")) * 1.5 + rng.normal(0, 1, len(d))
    r = anova_factorial(d, "y", ["A", "B"])
    t = r["tabla"]
    assert t.loc["A:B", "p_valor"] < 0.01 and t.loc["A", "eta2_parcial"] > t.loc["B", "eta2_parcial"] > 0
    assert (t.loc[["A", "B", "A:B"], "omega2"] <= t.loc[["A", "B", "A:B"], "eta2_parcial"]).all()
    # equilibrado: tipo II = tipo III en efectos principales
    r3 = anova_factorial(d, "y", ["A", "B"], tipo=3)
    assert r3["tabla"].loc["A:B", "F"] == pytest.approx(t.loc["A:B", "F"])
    sin = anova_factorial(d, "y", ["A", "B"], interacciones=False)
    assert "A:B" not in sin["tabla"].index


def test_bloques_ganan_eficiencia():
    rng = np.random.default_rng(1)
    d = pd.DataFrame([(b, t) for b in range(10) for t in "abcd"], columns=["bloque", "trat"])
    d["y"] = d.bloque * 2.0 + d.trat.map({"a": 0, "b": 0.5, "c": 1, "d": 1.5}) + rng.normal(0, 0.5, len(d))
    r = anova_bloques(d, "y", "trat", "bloque")
    assert r["tabla"].loc["trat", "p_valor"] < 0.001 and r["eficiencia_relativa"] > 10 and r["completo"]
    assert r["medias_ajustadas"]["d"] - r["medias_ajustadas"]["a"] == pytest.approx(1.5, abs=0.4)
    sin = smf.ols("y ~ C(trat)", d).fit()
    assert sin.f_pvalue > r["tabla"].loc["trat", "p_valor"]


def test_ancova_ajusta_y_detecta_pendientes():
    rng = np.random.default_rng(2)
    n = 300
    g = rng.choice(["c", "t"], n)
    base = rng.normal(50, 10, n) + 5 * (g == "t")                   # grupos desequilibrados en la basal
    y = 10 + 0.8 * base + 3 * (g == "t") + rng.normal(0, 3, n)
    d = pd.DataFrame({"g": g, "base": base, "y": y})
    r = ancova(d, "y", "g", "base")
    ma = r["medias_ajustadas"]
    assert ma.loc["t", "media_ajustada"] - ma.loc["c", "media_ajustada"] == pytest.approx(3, abs=1)
    assert ma.loc["t", "media_bruta"] - ma.loc["c", "media_bruta"] > 6 and r["p_homogeneidad_pendientes"] > 0.05
    assert r["reduccion_error"] > 0.7 and not r["avisos"]
    d2 = d.assign(y=10 + np.where(g == "t", 1.5, 0.5) * base + rng.normal(0, 3, n))
    assert ancova(d2, "y", "g", "base")["avisos"]


def test_medidas_repetidas_coincide_con_anovarm_y_corrige():
    from statsmodels.stats.anova import AnovaRM
    rng = np.random.default_rng(3)
    s = 25
    filas = []
    for i in range(s):
        u = rng.normal(0, 2)
        for k, t in enumerate(["t1", "t2", "t3", "t4"]):
            filas.append({"suj": i, "tiempo": t, "y": u + 0.5 * k + rng.normal(0, 1 + k * 0.6)})
    d = pd.DataFrame(filas)
    r = anova_medidas_repetidas(d, "suj", "tiempo", "y")
    ref = AnovaRM(d, "y", "suj", within=["tiempo"]).fit().anova_table
    assert r["tabla"].loc["tiempo", "F"] == pytest.approx(ref["F Value"].iloc[0])
    assert r["tabla"].loc["tiempo", "p_greenhouse_geisser"] >= r["tabla"].loc["tiempo", "p_valor"]
    assert 1 / 3 <= r["esfericidad"]["epsilon_gg"] <= 1
    # diseño mixto
    d["grupo"] = np.where(d.suj < 12, "A", "B")
    d.loc[(d.grupo == "B"), "y"] += 1.0
    m = anova_medidas_repetidas(d, "suj", "tiempo", "y", entre="grupo")
    assert {"grupo", "tiempo", "grupo:tiempo"} <= set(m["tabla"].index)
    assert m["tabla"].loc["tiempo", "F"] > 1


def test_anidado_no_pseudorreplica():
    rng = np.random.default_rng(4)
    filas = []
    for f in "AB":
        for lote in range(4):
            u = rng.normal(0, 2)                                     # mucha variación entre lotes, sin efecto del factor
            for _ in range(10):
                filas.append({"f": f, "lote": lote, "y": u + rng.normal(0, 0.5)})
    d = pd.DataFrame(filas)
    r = anova_anidado(d, "y", "f", "lote")
    ing = smf.ols("y ~ C(f)", d).fit().f_pvalue
    assert r["tabla"].loc["f", "gl"] == 1 and r["tabla"].loc["lote(f)", "gl"] == 6
    assert r["componentes_varianza"].loc["lote", "porcentaje"] > 70
    assert r["tabla"].loc["f", "p_valor"] > ing                   # el contraste correcto es más conservador


def test_diagnostico_anova_recomienda():
    rng = np.random.default_rng(5)
    d = pd.DataFrame({"g": np.repeat(list("abcd"), 40)})
    d["y"] = rng.lognormal(d.g.map({"a": 1, "b": 1.5, "c": 2, "d": 2.5}), 0.6)
    r = diagnostico_anova(d, "y", "g")
    assert r["levene"]["p_valor"] < 0.05 and r["lambda_box_cox_sugerida"] == pytest.approx(0, abs=0.35) and "transformar" in r["recomendacion"]
    d2 = d.assign(y=rng.normal(5, 1, len(d)))
    assert diagnostico_anova(d2, "y", "g")["recomendacion"] == "ANOVA clásico"


def test_factorial_2k_y_fraccion():
    c = diseno_factorial_2k(3, aleatorizar=False)
    D = c["diseno"]
    assert len(D) == 8 and (D[["A", "B", "C"]].sum() == 0).all() and list(D.A[:2]) == [-1, 1]
    f = diseno_factorial_2k(4, {"D": "ABC"})
    assert f["n_ensayos"] == 8 and f["resolucion"] == 4 and "ABCD" in f["relacion_definicion"]
    assert "CD" in f["alias"]["AB"]
    f3 = diseno_factorial_2k(5, {"D": "AB", "E": "AC"})
    assert f3["resolucion"] == 3 and f3["n_ensayos"] == 8


def test_efectos_lenth_y_curvatura():
    rng = np.random.default_rng(6)
    D = diseno_factorial_2k(4, aleatorizar=False, centros=4)["diseno"]
    y = 50 + 4 * D.A + 3 * D.C + 2.5 * D.A * D.C + rng.normal(0, 0.5, len(D))
    y[D.A.eq(0)] += 3                                               # curvatura en el centro
    r = efectos_factorial_2k(D.assign(y=y), "y", ["A", "B", "C", "D"])
    t = r["tabla"]
    assert set(t.index[t.activo_me]) == {"A", "C", "A:C"}
    assert t.loc["A", "estimacion"] == pytest.approx(8, abs=0.6)
    assert r["curvatura"]["p_valor"] < 0.01
    with pytest.raises(ValueError):
        efectos_factorial_2k(D.assign(y=y, A=D.A * 2), "y", ["A", "B"])


def test_cuadrado_latino():
    L = cuadrado_latino(5)
    assert (L.groupby("fila").tratamiento.nunique() == 5).all() and (L.groupby("columna").tratamiento.nunique() == 5).all()
    rng = np.random.default_rng(7)
    L["y"] = L.fila * 1.0 + L.columna * 0.5 + L.tratamiento.map(dict(zip("ABCDE", [0, 0, 0, 2, 4]))) + rng.normal(0, 0.5, 25)
    r = anova_cuadrado_latino(L, "y")
    assert r["tabla"].loc["tratamiento", "p_valor"] < 0.001 and r["tabla"].loc["Residual", "gl"] == 12


def test_superficie_respuesta_encuentra_maximo():
    rng = np.random.default_rng(8)
    D = diseno_central_compuesto(2, centros=5)
    assert len(D) == 13 and D.iloc[4:8, 1:].abs().max().max() == pytest.approx(np.sqrt(2))
    y = 80 - 2 * (D.x1 - 0.4) ** 2 - 3 * (D.x2 + 0.3) ** 2 + 1 * (D.x1 - .4) * (D.x2 + .3) + rng.normal(0, 0.1, len(D))
    r = superficie_respuesta(D.assign(y=y), "y", ["x1", "x2"])
    assert r["tipo"] == "máximo" and r["punto_estacionario"].to_numpy() == pytest.approx([0.4, -0.3], abs=0.08)
    assert r["respuesta_estacionaria"] == pytest.approx(80, abs=0.3) and r["dentro_region"] and r["p_curvatura"] < 0.01
    silla = superficie_respuesta(D.assign(y=D.x1 ** 2 - D.x2 ** 2 + rng.normal(0, .05, len(D))), "y", ["x1", "x2"])
    assert silla["tipo"] == "punto de silla"


def _observacional(n=4000, seed=9):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    zona = rng.choice(["n", "s"], n)
    e = 1 / (1 + np.exp(-(-0.3 + 1.0 * x1 - 0.8 * x2 + 0.5 * (zona == "s"))))
    t = (rng.random(n) < e).astype(int)
    y = 2 * t + 3 * x1 - 2 * x2 + 1 * (zona == "s") + rng.normal(0, 1, n)
    return pd.DataFrame({"t": t, "y": y, "x1": x1, "x2": x2, "zona": zona})


@pytest.mark.parametrize("metodo", ["ipw", "emparejamiento", "estratificacion"])
def test_propension_corrige_confusion(metodo):
    d = _observacional()
    r = puntuacion_propension(d, "t", "y", ["x1", "x2", "zona"], metodo=metodo, n_boot=60)
    assert abs(r["efecto_ingenuo"] - 2) > 1.5
    tol = 0.6 if metodo == "estratificacion" else 0.3
    assert r["efecto"] == pytest.approx(2, abs=tol) and r["ic"][0] < r["efecto"] < r["ic"][1]
    if metodo != "estratificacion":
        assert (r["balance"].smd_despues.abs() < r["balance"].smd_antes.abs()).all()


def test_dif_en_dif_y_tendencias():
    rng = np.random.default_rng(10)
    filas = []
    for u in range(200):
        tr = int(u < 100); a = rng.normal(0, 2)
        for per in range(6):
            post = int(per >= 4)
            filas.append({"u": u, "tr": tr, "per": per, "post": post, "y": a + 3 * tr + 0.5 * per + 2.0 * tr * post + rng.normal()})
    d = pd.DataFrame(filas)
    r = diferencias_en_diferencias(d, "y", "tr", "post", cluster="u", periodo="per")
    assert r["efecto"] == pytest.approx(2, abs=0.3) and r["tendencias_previas"]["p_valor"] > 0.05
    d2 = d.assign(y=d.y + 0.6 * d.tr * d.per)                       # tendencias no paralelas
    assert "aviso" in diferencias_en_diferencias(d2, "y", "tr", "post", cluster="u", periodo="per")


def test_aleatorizacion_e_itt():
    a = aleatorizar_ensayo(100, tamano_bloque=4)
    assert (a.brazo.iloc[:4].value_counts() == 2).all() and (a.brazo.value_counts() == 50).all()
    e = aleatorizar_ensayo(60, ("A", "B", "C"), (2, 1, 1), estratos=pd.Series(np.repeat(["x", "y"], 30)))
    assert e.groupby("estrato").brazo.apply(lambda s: (s == "A").mean()).between(0.4, 0.6).all()
    rng = np.random.default_rng(11)
    n = 20000
    z = rng.integers(0, 2, n); cumplidor = rng.random(n) < 0.6
    sano = rng.normal(0, 1, n)
    t = np.where(cumplidor, z, (sano > 0.5).astype(int))            # los no cumplidores eligen según su salud
    y = 1.0 * t + 2 * sano + rng.normal(0, 1, n)
    r = analisis_intencion_tratar(pd.DataFrame({"z": z, "t": t, "y": y}), "z", "t", "y")
    assert r.loc["ITT", "efecto"] == pytest.approx(0.6, abs=0.1) and r.loc["CACE (VI)", "efecto"] == pytest.approx(1, abs=0.15)
    assert abs(r.loc["según recibido", "efecto"] - 1) > 0.3


def test_mediacion_y_moderacion():
    rng = np.random.default_rng(12)
    n = 2000
    x = rng.normal(size=n); m = 0.5 * x + rng.normal(size=n); y = 0.4 * m + 0.3 * x + rng.normal(size=n)
    r = mediacion(pd.DataFrame({"x": x, "m": m, "y": y}), "x", "m", "y", n_boot=300)
    assert r["indirecto"] == pytest.approx(0.2, abs=0.04) and r["ic_indirecto"][0] > 0
    assert r["total"] == pytest.approx(r["directo"] + r["indirecto"])
    w = rng.normal(size=n)
    yy = 1 + 0.5 * x + 0.2 * w + 0.4 * x * w + rng.normal(size=n)
    mo = moderacion(pd.DataFrame({"x": x, "w": w, "y": yy}), "y", "x", "w")
    ps = mo["pendientes_simples"].pendiente_x
    assert mo["p_interaccion"] < 1e-6 and ps.iloc[2] - ps.iloc[0] == pytest.approx(0.8, abs=0.15)
    assert len(mo["johnson_neyman"]) == 2


def test_metaanalisis_dl_reml_y_fijo():
    # Ejemplo BCG simplificado: efectos y EE con heterogeneidad clara
    y = np.array([-0.89, -1.59, -1.35, -1.44, -0.22, -0.79, -1.62, 0.01, -0.47, -1.37, -0.34, 0.45, -0.02])
    se = np.sqrt([0.33, 0.19, 0.42, 0.02, 0.05, 0.01, 0.22, 0.004, 0.06, 0.07, 0.01, 0.53, 0.07])
    f = metaanalisis(y, se, "fijo"); dl = metaanalisis(y, se, "dl"); re = metaanalisis(y, se, "reml")
    assert f["tau2"] == 0 and dl["tau2"] > 0.2 and dl["I2"] > 0.8
    assert dl["ee"] > f["ee"] and dl["intervalo_prediccion"][0] < dl["ic"][0]
    assert re["tau2"] == pytest.approx(dl["tau2"], rel=0.6) and dl["tabla"].peso.sum() == pytest.approx(100)
    hom = metaanalisis(np.random.default_rng(13).normal(0.3, 0.1, 10), np.full(10, 0.1))
    assert hom["p_heterogeneidad"] > 0.01 and hom["efecto"] == pytest.approx(0.3, abs=0.1)
    hk = metaanalisis(y, se, "dl", hartung_knapp=True)
    assert hk["efecto"] == pytest.approx(dl["efecto"]) and hk["ee"] >= dl["ee"] and hk["metodo"] == "dl+HK"
    with pytest.raises(ValueError):
        metaanalisis([1], [0.1])
