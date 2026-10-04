"""Funciones añadidas en 0.10 a ramas existentes: regresión, interpretación, aplicabilidad, multinivel, diccionario,
multivariante, no supervisado, IA generativa y BDT."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from arbol_estadistica.descriptiva import diccionario_datos
from arbol_estadistica.diagnostico import dominio_aplicabilidad
from arbol_estadistica.finanzas import arbol_black_derman_toy
from arbol_estadistica.ml import (ajustar_bradley_terry, autoconsistencia_votacion, mapa_autoorganizado, modelo_ngramas,
                                  muestrear_siguiente, muestrear_texto, proceso_difusion, recuperar_tfidf, reglas_asociacion)
from arbol_estadistica.modelos import (ajustar_mars, bandas_confianza_regresion, correccion_error_medida, funciones_escalonadas,
                                       modelo_loglineal, regresion_inversa, regresion_inversa_cortes, regresion_local,
                                       regresion_pls_pcr, regresion_polinomica, regresion_por_origen, tamano_muestral_modelo)
from arbol_estadistica.multivariante import (analisis_conjunto, analisis_perfiles, centroides_contraidos,
                                             control_t2_multivariante, modelo_grafico_gaussiano, pca_funcional, pls_da,
                                             regresion_matriz_indicadora, regresion_multivariante)
from arbol_estadistica.preprocesado import centrar_por_grupo
from arbol_estadistica.seleccion import aproximar_modelo, tabla_nomograma


def _lineal(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    return x, 2 + 0.5 * x + rng.normal(0, 1, n)


def test_bandas_working_hotelling_mas_anchas():
    x, y = _lineal()
    p = bandas_confianza_regresion(x, y, metodo="puntual"); w = bandas_confianza_regresion(x, y)
    b = bandas_confianza_regresion(x, y, x_nuevos=[2, 5, 8], metodo="bonferroni")
    assert (w.banda_sup - w.banda_inf > p.banda_sup - p.banda_inf).all()
    assert w.multiplicador.iloc[0] == pytest.approx(np.sqrt(2 * stats.f.ppf(0.95, 2, 198)))
    assert b.multiplicador.iloc[0] == pytest.approx(stats.t.ppf(1 - 0.05 / 6, 198))
    assert ((p.prediccion_sup - p.prediccion_inf) > (p.banda_sup - p.banda_inf)).all()


def test_regresion_inversa_y_origen():
    x, y = _lineal(seed=1)
    r = regresion_inversa(x, y, y0=4.5)
    assert r["x0"] == pytest.approx(5, abs=0.4) and r["ic"][0] < 5 < r["ic"][1] and r["g"] < 0.05
    ruido = regresion_inversa(x, np.random.default_rng(2).normal(size=len(x)), 0.0)
    assert ruido["ic"] == (-np.inf, np.inf)
    rng = np.random.default_rng(3); xo = rng.uniform(1, 10, 100); yo = 3 * xo + rng.normal(0, 1, 100)
    o = regresion_por_origen(xo, yo)
    assert o["pendiente"] == pytest.approx(3, abs=0.05) and o["r2_no_centrado"] >= o["r2_con_ordenada"] and o["recomendacion"] == "por el origen"


def test_error_medida_corrige_atenuacion():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, 5000); Xo = X + rng.normal(0, 0.75, 5000); y = 2 * X + rng.normal(0, 0.5, 5000)
    fi = 1 / (1 + 0.75 ** 2)
    c = correccion_error_medida(Xo, y, fiabilidad=fi, metodo="calibracion")
    s = correccion_error_medida(Xo, y, var_error=0.75 ** 2)
    assert c["pendiente_ingenua"] == pytest.approx(2 * fi, abs=0.06) and c["pendiente_corregida"] == pytest.approx(2, abs=0.1)
    assert abs(s["pendiente_corregida"] - 2) < abs(s["pendiente_ingenua"] - 2) and s["curva_simex"].is_monotonic_decreasing


def test_polinomica_elige_grado_y_tamano_riley():
    rng = np.random.default_rng(5)
    x = rng.uniform(-2, 2, 300); y = 1 + x - 0.8 * x ** 2 + 0.5 * x ** 3 + rng.normal(0, 0.5, 300)
    r = regresion_polinomica(x, y)
    assert r["grado"] == 3 and regresion_polinomica(x, y, criterio="cv")["grado"] in (3, 4)
    assert np.polyval(r["coeficientes_crudos"], 1.0) == pytest.approx(1.7, abs=0.2)
    t = tamano_muestral_modelo(20, "binario", prevalencia=0.174, r2=0.255)
    assert t.attrs["n_minimo"] == int(np.ceil(20 / ((0.9 - 1) * np.log(1 - 0.255 / 0.9))))   # criterio 1 de Riley (601)
    c = tamano_muestral_modelo(10, "continuo", r2=0.5)
    assert c.attrs["n_minimo"] >= 100


def test_regresion_local_escalonada_y_mars():
    rng = np.random.default_rng(6)
    x = rng.uniform(0, 10, 400); f = np.sin(x) + 0.1 * x; y = f + rng.normal(0, 0.3, 400)
    for m in ("loess", "nadaraya_watson"):
        r = regresion_local(x, y, m, x_nuevos=np.linspace(1, 9, 30))
        assert np.sqrt(np.mean((r["ajuste"] - (np.sin(r["x"]) + 0.1 * r["x"])) ** 2)) < 0.2
    e = funciones_escalonadas(pd.DataFrame({"x": x, "y": y}), "y", "x", 8)
    assert len(e["tabla"]) == 8 and e["aic_escalonado"] < e["aic_lineal"]
    X = pd.DataFrame({"a": rng.uniform(0, 10, 500), "b": rng.uniform(0, 10, 500), "c": rng.uniform(0, 10, 500)})
    yy = 2 * np.maximum(0, X.a - 6) - 1 * np.maximum(0, 3 - X.b) + rng.normal(0, 0.2, 500)
    m = ajustar_mars(X, yy, max_terminos=11)
    assert m["r2"] > 0.95 and not m["terminos"].termino.str.contains(r"\bc\b").any()
    assert np.corrcoef(m["predecir"](X), yy)[0, 1] > 0.97


def test_sir_y_pls():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(1000, 6)), columns=list("abcdef"))
    beta = np.array([1, -1, 0, 0, 0.5, 0]) / 1.5
    y = np.exp(X.to_numpy() @ beta) + rng.normal(0, 0.1, 1000)
    s = regresion_inversa_cortes(X, y, n_direcciones=1)
    d = s["direcciones"].dir1.to_numpy()
    assert abs(d @ beta / np.linalg.norm(beta)) > 0.97 and s["autovalores"].iloc[0] > 5 * s["autovalores"].iloc[1]
    lat = rng.normal(size=(300, 2))
    Xc = pd.DataFrame(lat @ rng.normal(size=(2, 30)) + rng.normal(0, 0.1, (300, 30)))
    yc = lat[:, 0] - lat[:, 1] + rng.normal(0, 0.1, 300)
    p = regresion_pls_pcr(Xc, yc, "pls"); q = regresion_pls_pcr(Xc, yc, "pcr")
    assert p["n_componentes"] <= 6 and p["curva_cv"].min() < 0.2 and q["curva_cv"].min() < 0.2


def test_loglineal_independencia_y_asociacion():
    rng = np.random.default_rng(8)
    n = 3000
    a = rng.choice(["x", "y"], n); b = np.where(rng.random(n) < np.where(a == "x", 0.7, 0.3), "p", "q"); c = rng.choice(["u", "v", "w"], n)
    d = pd.DataFrame({"A": a, "B": b, "C": c})
    ind = modelo_loglineal(d, ["A", "B", "C"])
    assert ind["p_valor"] < 1e-10
    ab = modelo_loglineal(d, ["A", "B", "C"], modelo='A*B + C')
    assert ab["p_valor"] > 0.01 and ab["frente_a_independencia"]["p_valor"] < 1e-10
    sat = modelo_loglineal(d, ["A", "B", "C"], modelo="saturado")
    assert sat["G2"] == pytest.approx(0, abs=1e-6)


def test_nomograma_y_aproximar():
    rng = np.random.default_rng(9)
    n = 2000
    d = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "z": rng.choice(["a", "b"], n), "r": rng.normal(size=n)})
    d["y"] = (rng.random(n) < 1 / (1 + np.exp(-(2 * d.x1 + 0.5 * d.x2 + 0.3 * (d.z == "b"))))).astype(int)
    m = smf.glm("y ~ x1 + x2 + z", d, family=__import__("statsmodels.api").api.families.Binomial()).fit()
    t = tabla_nomograma(m, d, ["x1", "x2", "z"])
    assert t["importancia_puntos"].index[0] == "x1" and t["importancia_puntos"].iloc[0] == pytest.approx(100)
    assert t["conversion"].prediccion.is_monotonic_increasing
    eta = m.predict(d, which="linear") + 0.02 * d.r
    a = aproximar_modelo(d[["x1", "x2", "z", "r"]], eta, 0.95)
    assert "x1" in a["variables"] and "r" not in a["variables"] and a["r2"] >= 0.95


def test_dominio_centrado_diccionario():
    rng = np.random.default_rng(10)
    A = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
    B = pd.DataFrame([[0, 0, 0], [6, 0, 0], [2.5, 2.5, -2.5]], columns=list("abc"))
    r = dominio_aplicabilidad(A, B)
    assert r.dentro.tolist() == [True, False, False]
    g = pd.DataFrame({"g": np.repeat([1, 2, 3], 4), "x": np.arange(12.0)})
    c = centrar_por_grupo(g, "x", "g")
    assert c.groupby("g").x_dentro.mean().abs().max() < 1e-12 and (c.x_dentro + c.x_media_grupo == c.x).all() and "x_dentro" not in g
    dd = pd.DataFrame({"id": range(100), "cte": 1, "sexo": ["h", "m"] * 50, "importe": rng.gamma(2, 100, 100),
                       "prob_fuga": rng.random(100), "zona": rng.choice(list("abcde"), 100)})
    dic = diccionario_datos(dd, {"importe": "prima anual"})
    assert dic.loc["id", "rol_sugerido"] == "identificador" and dic.loc["cte", "rol_sugerido"] == "constante"
    assert dic.loc["sexo", "rol_sugerido"] == "binaria" and "fuga" in dic.loc["prob_fuga", "rol_sugerido"]
    assert dic.loc["importe", "descripcion"] == "prima anual"


def test_regresion_multivariante_y_perfiles():
    rng = np.random.default_rng(11)
    n = 300
    x = rng.normal(size=n); g = rng.choice(["a", "b"], n)
    E = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], n)
    d = pd.DataFrame({"x": x, "g": g, "y1": 1 + 0.5 * x + E[:, 0], "y2": 2 + 0.0 * x + E[:, 1], "r": rng.normal(size=n)})
    r = regresion_multivariante(d, ["y1", "y2"], ["x", "r"])
    assert r["contrastes"].loc["x", "p_wilks"] < 1e-6 and r["contrastes"].loc["r", "p_wilks"] > 0.01
    assert r["correlacion_residuos"].iloc[0, 1] == pytest.approx(0.7, abs=0.1)
    pf = pd.DataFrame({"g": np.repeat(["a", "b"], 80)})
    base = rng.normal(0, 1, 160)
    for k in range(4):
        pf[f"t{k}"] = base + k * 0.5 + (pf.g == "b") * 1.0 + rng.normal(0, 0.5, 160)
    t = analisis_perfiles(pf, ["t0", "t1", "t2", "t3"], "g")
    assert t.loc["paralelismo (forma)", "p_valor"] > 0.01 and t.loc["niveles (altura)", "p_valor"] < 1e-4
    assert t.loc["planitud (cambio entre medidas)", "p_valor"] < 1e-6


def test_control_t2_conjoint_y_grafico_gaussiano():
    rng = np.random.default_rng(12)
    S = [[1, 0.9], [0.9, 1]]
    f1 = pd.DataFrame(rng.multivariate_normal([0, 0], S, 100), columns=["a", "b"])
    f2 = pd.DataFrame([[1.5, -1.5], [0.2, 0.1]], columns=["a", "b"])     # 1º: rompe la correlación sin salirse de ±3σ
    c = control_t2_multivariante(f1, f2)
    assert c["fase2"].fuera.tolist() == [True, False] and c["fase1"].fuera.mean() < 0.05
    perfiles = pd.DataFrame([(p, m, k) for p in ["bajo", "medio", "alto"] for m in ["A", "B"] for k in ["sí", "no"]], columns=["precio", "marca", "garantia"])
    filas = []
    for e in range(30):
        for _, r in perfiles.iterrows():
            v = {"bajo": 2, "medio": 0, "alto": -2}[r.precio] + (0.5 if r.marca == "A" else -0.5) + (0.2 if r.garantia == "sí" else -0.2)
            filas.append({**r.to_dict(), "id": e, "nota": 5 + v + rng.normal(0, 0.5)})
    cj = analisis_conjunto(pd.DataFrame(filas), "nota", ["precio", "marca", "garantia"], encuestado="id")
    assert cj["importancia"].index[0] == "precio" and cj["importancia"]["precio"] == pytest.approx(100 * 4 / 5.4, abs=4)
    assert len(cj["individuales"]) == 30
    z = rng.normal(size=(2000, 4)); X = pd.DataFrame(np.c_[z[:, 0], z[:, 0] + 0.5 * z[:, 1], z[:, 0] + 0.5 * z[:, 1] + 0.5 * z[:, 2], z[:, 3]], columns=list("abcd"))
    gg = modelo_grafico_gaussiano(X)
    pares = {frozenset((r.de, r.a)) for r in gg["aristas"].itertuples() if abs(r.correlacion_parcial) > 0.1}
    assert frozenset("ab") in pares and frozenset("bc") in pares and not any("d" in p for p in pares)


def test_pca_funcional_y_clasificadores_lineales():
    rng = np.random.default_rng(13)
    t = np.linspace(0, 1, 50)
    a, b = rng.normal(0, 2, 200), rng.normal(0, 0.5, 200)
    Y = 5 + np.outer(a, np.sin(2 * np.pi * t)) + np.outer(b, np.cos(2 * np.pi * t)) + rng.normal(0, 0.1, (200, 50))
    f = pca_funcional(Y, t, n_componentes=3)
    assert f["varianza_explicada"].iloc[:2].sum() > 0.95 and abs(np.corrcoef(f["puntuaciones"].fpc1, a)[0, 1]) > 0.98
    X = pd.DataFrame({"x": np.r_[rng.normal(-3, .5, 100), rng.normal(0, .5, 100), rng.normal(3, .5, 100)]})
    yy = np.repeat(["a", "b", "c"], 100)
    ri = regresion_matriz_indicadora(X, yy)
    assert ri["clases_enmascaradas"] == ["b"]
    Xp = pd.DataFrame(rng.normal(size=(120, 200))); yp = np.repeat([0, 1, 2], 40)
    Xp.iloc[:40, :5] += 2.0; Xp.iloc[40:80, 5:10] += 2.0
    cc = centroides_contraidos(Xp, yp)
    assert cc["exactitud_cv"] > 0.85 and cc["umbral"] > 0 and len(cc["variables_activas"]) < 100
    pd_ = pls_da(Xp, yp, max_componentes=5)
    assert pd_["exactitud_cv"] > 0.8 and set(pd_["vip"].index[:10]) <= set(range(10)) | set(range(200))
    assert pd_["vip"].iloc[:10].index.isin(range(10)).mean() > 0.7


def test_reglas_y_som():
    rng = np.random.default_rng(14)
    cestas = []
    for _ in range(2000):
        c = set(rng.choice(list("abcdefg"), rng.integers(1, 4), replace=False))
        if "a" in c and rng.random() < 0.8:
            c.add("pan")
        cestas.append(sorted(c))
    r = reglas_asociacion(cestas, 0.02, 0.5)
    top = r["reglas"].query("antecedente == 'a' and consecuente == 'pan'").iloc[0]
    assert r["reglas"].lift.iloc[0] >= top.lift and top.consecuente == "pan" and top.lift > 2 and top.confianza == pytest.approx(0.8, abs=0.05)
    X = pd.DataFrame(np.r_[rng.normal(0, .3, (150, 3)), rng.normal(4, .3, (150, 3))])
    s = mapa_autoorganizado(X, 5, 5, 3000)
    c1 = s["celda"].iloc[:150].mean(); c2 = s["celda"].iloc[150:].mean()
    assert np.hypot(*(c1 - c2)) > 2 and s["conteos"].sum() == 300


def test_generativa():
    m = modelo_ngramas(["el gato come pescado", "el perro come carne", "el gato duerme"], n=2)
    d = m["distribucion"]("el")
    assert d["gato"] > d["perro"] > d["carne"] and m["perplejidad"] > 1
    p = pd.Series({"a": 0.6, "b": 0.3, "c": 0.1})
    assert muestrear_siguiente(p, temperatura=0)["distribucion"]["a"] == 1
    assert muestrear_siguiente(p, 2.0)["entropia_bits"] > muestrear_siguiente(p, 0.5)["entropia_bits"]
    assert muestrear_siguiente(p, top_k=2)["distribucion"]["c"] == 0 and muestrear_siguiente(p, top_p=0.5)["distribucion"]["b"] == 0
    assert muestrear_texto(m, "el", 5, temperatura=0).startswith("el gato")
    rng = np.random.default_rng(15)
    fuerza = {"A": 1.5, "B": 0.5, "C": 0.0, "D": -2.0}
    filas = []
    for _ in range(3000):
        i, j = rng.choice(list(fuerza), 2, replace=False)
        g = i if rng.random() < 1 / (1 + np.exp(-(fuerza[i] - fuerza[j]))) else j
        filas.append({"ganador": g, "perdedor": j if g == i else i})
    bt = ajustar_bradley_terry(pd.DataFrame(filas))
    assert list(bt.index) == ["A", "B", "C", "D"] and (bt.puntuacion["A"] - bt.puntuacion["D"]) == pytest.approx(3.5, abs=0.3)
    rec = recuperar_tfidf(["la prima se paga anualmente", "el siniestro se declara en 7 días", "la franquicia reduce la prima"], "cuándo declarar un siniestro", 1)
    assert rec.index[0] == 1
    a = autoconsistencia_votacion(0.6, (1, 11, 41))
    assert a.precision.is_monotonic_increasing and a.loc[41, "precision"] > 0.85
    assert autoconsistencia_votacion(0.4, (1, 41)).precision.is_monotonic_decreasing
    df_ = proceso_difusion(rng.choice([-3, 3], 5000).astype(float))
    assert abs(df_["media_final"]) < 0.05 and df_["sd_final"] == pytest.approx(1, abs=0.05)


def test_black_derman_toy_reproduce_curva():
    z = [0.03, 0.035, 0.04, 0.042, 0.045]
    r = arbol_black_derman_toy(z, 0.15)
    T = r["arbol_tipos"]
    # reprecia el bono cupón cero a 5 años por inducción hacia atrás
    V = np.ones(6)
    for i in range(4, -1, -1):
        rr = T[i].dropna().to_numpy()
        V = 0.5 * (V[:-1] + V[1:]) / (1 + rr)
    assert V[0] == pytest.approx(1.045 ** -5, rel=1e-8)
    assert np.log(T[3].iloc[1] / T[3].iloc[0]) / 2 == pytest.approx(0.15)
