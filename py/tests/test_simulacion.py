"""Rama simulación: bayes, procesos estocásticos, Monte Carlo, inferencia por simulación y muestreo (0.10)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from arbol_estadistica.simulacion import (ab_bayesiano, accion_bayes, ajuste_no_respuesta, analizar_distribucion_conjunta,
                                          asignacion_estratos, bayes_empirico_beta, bonus_malus, cadena_markov,
                                          cadena_markov_continua, chequeo_predictivo, comparar_estimadores,
                                          contraste_proceso_poisson, contrastes_aleatoriedad, convergencia_media_muestral,
                                          coste_de_dicotomizar, diagnostico_mcmc, distribucion_estadistico_orden,
                                          distribucion_muestral_simulada, estimador_razon, estimar_estratificado, estimar_mas,
                                          estimar_montecarlo, extraer_muestra, generador_congruencial,
                                          generar_normal_multivariante, generar_por_aceptacion_rechazo, generar_por_inversion,
                                          informacion_fisher, metodo_delta, metodo_momentos, metropolis, momentos_distribucion,
                                          nacimiento_muerte, paseo_aleatorio, posterior_conjugado, ruina_jugador,
                                          simular_bandido, simular_browniano, simular_cadena_markov,
                                          simular_proceso_poisson, simular_regresion_a_la_media, tamano_muestra_encuesta,
                                          teorema_bayes)


# ------------------------------------------------------------------ bayes
def test_conjugadas_formulas_exactas():
    b = posterior_conjugado("beta_binomial", (7, 20), (1, 1))
    assert b["posterior"] == {"a": 8, "b": 14} and b["media"] == pytest.approx(8 / 22)
    assert b["media"] == pytest.approx(b["credibilidad_Z"] * 7 / 20 + (1 - b["credibilidad_Z"]) * 0.5)
    assert posterior_conjugado("beta_binomial", np.r_[np.ones(7), np.zeros(13)], (1, 1))["posterior"] == b["posterior"]
    g = posterior_conjugado("gamma_poisson", [2, 0, 1, 3], (2, 1), exposicion=[1, 1, 1, 1])
    assert g["posterior"] == {"a": 8, "b": 5} and g["media"] == pytest.approx(1.6)
    n = posterior_conjugado("normal_normal", [10, 12, 11, 13], (0, 1), sigma=2)
    assert n["media"] == pytest.approx((46 / 4) / (1 + 1)) and n["posterior"]["desviacion"] == pytest.approx(np.sqrt(0.5))
    with pytest.raises(ValueError):
        posterior_conjugado("normal_normal", [1, 2], (0, 1))


def test_bayes_empirico_contrae_mas_a_los_pequenos():
    rng = np.random.default_rng(0)
    p = rng.beta(20, 80, 40); n = rng.integers(5, 400, 40); x = rng.binomial(n, p)
    r = bayes_empirico_beta(x, n)
    assert r["previa"]["media"] == pytest.approx(0.2, abs=0.03)
    t = r["tabla"]
    assert (t.tasa_contraida - 0.2).abs().mean() < (t.tasa_bruta - 0.2).abs().mean()
    assert stats.spearmanr(t.ensayos, t.peso_datos)[0] == pytest.approx(1)
    # el error frente a la verdad baja
    assert np.mean((t.tasa_contraida - p) ** 2) < np.mean((t.tasa_bruta - p) ** 2)


def test_metropolis_recupera_normal_y_diagnostica():
    lp = lambda th: -0.5 * ((th[0] - 3) ** 2 / 4 + (th[1] + 1) ** 2)      # N(3, 2²) × N(−1, 1)  # noqa: E731
    r = metropolis(lp, [0, 0], n_iter=4000, cadenas=4, nombres=["a", "b"])
    s = r["resumen"]
    assert s.loc["a", "media"] == pytest.approx(3, abs=0.15) and s.loc["a", "desviacion"] == pytest.approx(2, rel=0.1)
    assert s.loc["b", "media"] == pytest.approx(-1, abs=0.1) and (s.r_hat < 1.02).all() and 0.15 < r["aceptacion"] < 0.6
    malas = np.stack([np.random.default_rng(i).normal(i * 3, 1, 500) for i in range(4)])   # cadenas en sitios distintos
    assert diagnostico_mcmc(malas).r_hat.iloc[0] > 1.5
    with pytest.raises(ValueError):
        metropolis(lambda th: -np.inf, [0.0])


def test_chequeo_predictivo_detecta_sobredispersion():
    rng = np.random.default_rng(1)
    y = rng.negative_binomial(1, 0.2, 300)                            # media 4, muy dispersa
    lam = rng.gamma(1 + y.sum(), 1 / (0.001 + len(y)), 400)          # posterior Poisson
    t = chequeo_predictivo(y, lam, lambda l, n, r: r.poisson(l, n))
    assert t.loc["desviacion", "alarma"] and t.loc["prop_ceros", "alarma"] and not t.loc["media", "alarma"]


def test_accion_bayes_segun_perdida():
    x = np.random.default_rng(2).lognormal(0, 1, 50_000)
    assert accion_bayes(x)["accion"] == pytest.approx(np.exp(0.5), rel=0.03)
    assert accion_bayes(x, "absoluta")["accion"] == pytest.approx(1, rel=0.03)
    assert accion_bayes(x, ("asimetrica", 3))["accion"] == pytest.approx(stats.lognorm(1).ppf(0.75), rel=0.03)
    L = pd.DataFrame([[0, 10], [2, 2]], index=["aceptar", "rechazar"], columns=["bueno", "malo"])
    assert accion_bayes([0.9, 0.1], L)["accion"] == "aceptar" and accion_bayes([0.6, 0.4], L)["accion"] == "rechazar"


def test_ab_y_bandido():
    r = ab_bayesiano(100, 1000, 140, 1000)
    assert r["prob_b_mejor"] > 0.99 and r["mejor"] == "B" and r["uplift_medio"] == pytest.approx(0.4, abs=0.08)
    assert 0.4 < ab_bayesiano(100, 1000, 100, 1000)["prob_b_mejor"] < 0.6
    P = [0.04, 0.05, 0.08]
    th = simular_bandido(P, 3000, "thompson", repeticiones=5); un = simular_bandido(P, 3000, "uniforme")
    assert th["arrepentimiento"].iloc[-1] < 0.5 * un["arrepentimiento"].iloc[-1]
    assert un["arrepentimiento"].iloc[-1] == pytest.approx(3000 * (0.04 + 0.03) / 3, rel=0.01)
    assert th["prop_mejor_brazo"] > 0.6


# ------------------------------------------------------------------ procesos
def test_cadena_markov_estacionaria_y_absorcion():
    P = [[0.9, 0.1], [0.5, 0.5]]
    r = cadena_markov(P, inicial=0, pasos=50)
    assert r["estacionaria"].to_numpy() == pytest.approx([5 / 6, 1 / 6]) and r["regular"]
    assert r["trayectoria"].iloc[-1].to_numpy() == pytest.approx([5 / 6, 1 / 6], abs=1e-6)
    assert r["tiempo_medio_retorno"].iloc[0] == pytest.approx(1.2)
    # ruina del jugador como cadena absorbente (0..4, p = 0.5)
    G = np.zeros((5, 5)); G[0, 0] = G[4, 4] = 1
    for i in range(1, 4):
        G[i, i - 1] = G[i, i + 1] = 0.5
    a = cadena_markov(G)
    assert a["prob_absorcion"].loc[1, 0] == pytest.approx(0.75) and a["tiempo_hasta_absorcion"].loc[2] == pytest.approx(4)
    assert set(a["clases"].tipo) == {"recurrente", "transitoria"}
    per = cadena_markov([[0, 1], [1, 0]])
    assert per["clases"].periodo.iloc[0] == 2 and not per["regular"]
    with pytest.raises(ValueError):
        cadena_markov([[0.5, 0.4], [0.5, 0.5]])


def test_simular_cadena_converge_a_la_estacionaria():
    s = simular_cadena_markov(pd.DataFrame([[0.9, 0.1], [0.5, 0.5]], index=["a", "b"], columns=["a", "b"]), 20000, "a")
    assert (s[0] == "a").mean() == pytest.approx(5 / 6, abs=0.02)


def test_bonus_malus_eficiencia():
    coef = [0.5, 0.7, 1.0, 1.3, 1.6]
    reglas = [[0, 2, 4], [0, 3, 4], [1, 4, 4], [2, 4, 4], [3, 4, 4]]     # baja 1 nivel sin siniestros, sube 2 por siniestro
    r = bonus_malus(coef, reglas, 0.1)
    assert r["estacionaria"].sum() == pytest.approx(1) and r["estacionaria"].iloc[0] > 0.5
    assert 0 < r["eficiencia_loimaranta"] < 1
    assert bonus_malus(coef, reglas, 0.3)["coeficiente_medio"] > r["coeficiente_medio"]


def test_markov_continua_y_colas():
    Q = [[-0.2, 0.2], [1.0, -1.0]]
    r = cadena_markov_continua(Q, t=[0.5, 100])
    assert r["estacionaria"].to_numpy() == pytest.approx([5 / 6, 1 / 6]) and r["P_t"][100.0].iloc[0].to_numpy() == pytest.approx([5 / 6, 1 / 6])
    assert r["permanencia_media"].iloc[0] == pytest.approx(5)
    mm1 = nacimiento_muerte(0.8, 1.0)
    assert mm1["L"] == pytest.approx(4, rel=1e-3) and mm1["W"] == pytest.approx(5, rel=1e-3) and mm1["prob_esperar"] == pytest.approx(0.8, rel=1e-3)
    mm2 = nacimiento_muerte(1.5, 1.0, servidores=2)
    assert mm2["prob_esperar"] == pytest.approx(0.6428571, rel=1e-3)                     # Erlang C(2, 1.5)
    k = nacimiento_muerte(1.0, 1.0, capacidad=4)
    assert k["prob_rechazo"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        nacimiento_muerte(2.0, 1.0)


def test_proceso_poisson_y_contraste():
    t = simular_proceso_poisson(3.0, 100, n_trayectorias=200)
    assert np.mean([len(x) for x in t]) == pytest.approx(300, rel=0.02)
    assert (contraste_proceso_poisson(t[0], 100).p_valor > 0.01).all()
    nh = simular_proceso_poisson(lambda s: 2 + 2 * np.sin(s), 100, n_trayectorias=100)
    assert np.mean([len(x) for x in nh]) == pytest.approx(200 + 2 * (1 - np.cos(100)), rel=0.03)
    agrup = np.sort(np.r_[np.random.default_rng(0).uniform(0, 10, 150), np.random.default_rng(1).uniform(50, 60, 150)])
    assert contraste_proceso_poisson(agrup, 100).p_valor.min() < 1e-4


def test_ruina_paseo_y_browniano():
    r = ruina_jugador(10, 20, 0.5, simular=4000)
    assert r["prob_ruina"] == pytest.approx(0.5) and r["duracion_esperada"] == pytest.approx(100)
    assert r["prob_ruina_simulada"] == pytest.approx(0.5, abs=0.03)
    r2 = ruina_jugador(3, 10, 0.45)
    q = 0.55 / 0.45
    assert r2["prob_ruina"] == pytest.approx((q ** 3 - q ** 10) / (1 - q ** 10))
    p = paseo_aleatorio(200, 0.55, 5000)
    m = p["martingalas"]
    assert m.iloc[-1, 0] == pytest.approx(0, abs=0.3) and m.iloc[-1, 1] == pytest.approx(1, abs=0.1)
    b = simular_browniano(2, 500, 4000, mu=0.05, sigma=0.3, inicio=100, geometrico=True)
    c = b["comprobacion"]
    assert c.loc["media final", "simulado"] == pytest.approx(c.loc["media final", "teorico"], abs=2.5)          # 3 EE
    assert c.iloc[2]["simulado"] == pytest.approx(0.18, rel=0.02)


# ------------------------------------------------------------------ Monte Carlo y probabilidad
def test_generadores_y_contrastes_aleatoriedad():
    bueno = contrastes_aleatoriedad(generador_congruencial(30000))
    assert (bueno.p_valor > 0.001).all()
    randu = contrastes_aleatoriedad(generador_congruencial(30000, 1, 65539, 0, 2 ** 31))
    assert randu.loc[randu.index.str.startswith("tripletas"), "p_valor"].iloc[0] < 1e-6
    assert contrastes_aleatoriedad(np.sort(np.random.default_rng(0).random(5000))).p_valor.min() < 1e-6


def test_inversion_aceptacion_y_normal_multivariante():
    x = generar_por_inversion(20000, cuantil=lambda u: -np.log(1 - u) / 2)
    assert x.mean() == pytest.approx(0.5, rel=0.03)
    y = generar_por_inversion(20000, cdf=lambda v: 1 - np.exp(-2 * v), soporte=(0, 20))
    assert stats.kstest(y, "expon", args=(0, 0.5)).pvalue > 0.01
    ar = generar_por_aceptacion_rechazo(20000, lambda v: stats.beta(2, 5).pdf(v), stats.uniform(0, 1))
    assert stats.kstest(ar["muestra"], stats.beta(2, 5).cdf).pvalue > 0.01 and ar["tasa_aceptacion"] == pytest.approx(1 / ar["M"], rel=0.05)
    S = np.array([[4, 1.2], [1.2, 1]])
    for m in ("cholesky", "eigen"):
        d = generar_normal_multivariante([1, 2], S, 40000, m)
        assert np.cov(d.T) == pytest.approx(S, abs=0.08)
    sing = generar_normal_multivariante([0, 0], [[1, 1], [1, 1]], 100, "eigen")
    assert np.allclose(sing.x1, sing.x2)
    with pytest.raises(ValueError):
        generar_normal_multivariante([0, 0], [[1, 2], [2, 1]], 10)


def test_montecarlo_reduccion_varianza():
    f = np.exp                                                     # ∫ e^u du = e − 1
    s = estimar_montecarlo(f, 20000)
    a = estimar_montecarlo(f, 20000, metodo="antiteticas")
    c = estimar_montecarlo(f, 20000, metodo="control", control=lambda u: u, media_control=0.5)
    for r in (s, a, c):
        assert r["estimacion"] == pytest.approx(np.e - 1, abs=4 * r["error_estandar"])
    assert a["reduccion_varianza"] > 0.9 and c["reduccion_varianza"] > 0.9
    d2 = estimar_montecarlo(lambda U: (U ** 2).sum(axis=1) <= 1, 40000, dim=2)
    assert 4 * d2["estimacion"] == pytest.approx(np.pi, abs=0.03)


def test_bayes_y_conjunta():
    t = teorema_bayes(0.01, (0.99, 0.95))
    assert t.loc["VPP = P(enfermo | +)", "valor"] == pytest.approx(0.0099 / (0.0099 + 0.0495))
    h = teorema_bayes([0.5, 0.3, 0.2], [0.01, 0.02, 0.05])
    assert h.posterior.sum() == pytest.approx(1) and h.posterior.idxmax() == 2
    T = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=[0, 1], columns=[0, 1])
    r = analizar_distribucion_conjunta(T)
    assert r["E_E_y_dado_x"] == pytest.approx(r["E_y"]) and r["E_y"] == pytest.approx(0.6)
    assert r["covarianza"] == pytest.approx(0.4 - 0.7 * 0.6)
    ind = analizar_distribucion_conjunta(np.outer([0.3, 0.7], [0.2, 0.8]))
    assert ind["max_desviacion_independencia"] < 1e-12


def test_orden_momentos_tcl():
    o = distribucion_estadistico_orden(stats.uniform(), 10, 10, n_sim=20000)
    assert o["media"] == pytest.approx(10 / 11) and o["media_simulada"] == pytest.approx(10 / 11, abs=0.005)
    assert o["cdf"](0.9) == pytest.approx(0.9 ** 10)
    m = momentos_distribucion(stats.expon(scale=2))
    assert m["crudos"][2] == pytest.approx(8) and m["fgm"][0.1] == pytest.approx(1 / (1 - 0.2), rel=1e-4)
    assert m["fgm_derivada1_en_0"] == pytest.approx(2, rel=1e-3) and m["curtosis_exceso"] == pytest.approx(6)
    assert np.isinf(momentos_distribucion(stats.lognorm(1))["fgm"][0.5])
    c = convergencia_media_muestral(stats.expon(), (1, 30, 200), 4000)["tabla"]
    assert c.ks_normal.is_monotonic_decreasing and c.loc[200, "desviacion"] == pytest.approx(1 / np.sqrt(200), rel=0.05)
    cau = convergencia_media_muestral(stats.cauchy(), (1, 100), 4000)["tabla"]
    assert cau.loc[100, "ks_normal"] > 0.1                                  # el TCL no se cumple


# ------------------------------------------------------------------ inferencia por simulación
def test_metodo_momentos():
    x = stats.gamma(3, scale=1 / 2).rvs(5000, random_state=np.random.default_rng(0))
    t = metodo_momentos(x, "gamma")
    assert t.loc["forma", "momentos"] == pytest.approx(3, rel=0.08) and t.loc["tasa", "max_verosimilitud"] == pytest.approx(2, rel=0.08)
    nb = metodo_momentos(np.random.default_rng(1).negative_binomial(2, 0.3, 5000), "binomial_negativa")
    assert nb.loc["r", "momentos"] == pytest.approx(2, rel=0.15) and nb.loc["p", "max_verosimilitud"] == pytest.approx(0.3, rel=0.1)
    with pytest.raises(ValueError):
        metodo_momentos(np.random.default_rng(2).poisson(3, 500) * 0 + 3, "binomial_negativa")


def test_fisher_cramer_rao_y_eficiencia():
    ld = lambda x, l: stats.poisson.logpmf(x, l)                              # noqa: E731  I(λ) = 1/λ
    e = informacion_fisher(ld, 4.0, distribucion=stats.poisson(4), n=50)
    assert e["informacion"] == pytest.approx(0.25, rel=0.03) and e["cota_cramer_rao"] == pytest.approx(4 / 50, rel=0.03)
    x = np.random.default_rng(0).normal(5, 2, 400)
    o = informacion_fisher(lambda v, th: stats.norm.logpdf(v, th[0], th[1]), [x.mean(), x.std()], datos=x)
    assert o["error_estandar_minimo"][0] == pytest.approx(x.std() / 20, rel=0.01)
    t = comparar_estimadores({"media": np.mean, "mediana": np.median}, lambda n, r: r.normal(0, 1, n), 0.0, n=100, n_sim=3000,
                             cota_cr=1 / 100)
    assert t.loc["media", "eficiencia_cramer_rao"] == pytest.approx(1, abs=0.08)
    assert t.loc["mediana", "eficiencia_relativa"] == pytest.approx(2 / np.pi, abs=0.08)


def test_metodo_delta_y_distribuciones_muestrales():
    d = metodo_delta(lambda b: np.exp(b), 0.5, [[0.01]])
    assert d["error_estandar"] == pytest.approx(np.exp(0.5) * 0.1, rel=1e-4)
    r = metodo_delta(lambda t: t[0] / t[1], [2, 4], [[0.04, 0], [0, 0.16]])
    assert r["error_estandar"] == pytest.approx(np.sqrt(0.04 / 16 + 4 * 0.16 / 256), rel=1e-4)
    for tipo, gl in [("chi2", 5), ("t", 4), ("f", (3, 8)), ("hotelling", (3, 12))]:
        assert distribucion_muestral_simulada(tipo, gl, 8000)["p_valor_ks"] > 0.001
    v = distribucion_muestral_simulada("varianza", 5, 40000)["tabla"]
    assert v.loc["divide entre n", "media_estimador"] == pytest.approx(0.8, abs=0.02)


def test_regresion_a_la_media_y_dicotomizar():
    r = simular_regresion_a_la_media(0.5, 50000)
    assert r["media_2a_seleccionados"] == pytest.approx(r["prediccion_teorica_2a"], abs=0.5) and r["cambio_aparente"] < -5
    c = coste_de_dicotomizar(0.25, n=150, n_sim=600)
    assert c["factor_atenuacion"] == pytest.approx(0.7979, abs=1e-3) and c["potencia_dicotomizada"] < c["potencia_continua"]


# ------------------------------------------------------------------ muestreo
def _poblacion(N=20000, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.choice(["a", "b", "c"], N, p=[0.6, 0.3, 0.1])
    x = rng.gamma(2, 50, N) * pd.Series(e).map({"a": 1, "b": 3, "c": 10}).to_numpy()
    return pd.DataFrame({"estrato": e, "x": x, "y": 1.2 * x + rng.normal(0, 10, N), "conglo": rng.integers(0, 400, N)})


def test_extraer_y_estimar_mas():
    pob = _poblacion()
    for m, kw in [("mas", {}), ("sistematica", {}), ("estratificada", {"estrato": "estrato"}), ("conglomerados", {"conglomerado": "conglo"})]:
        n = 40 if m == "conglomerados" else 2000
        s = extraer_muestra(pob, n, m, **kw)
        assert (s.peso * 1).sum() == pytest.approx(len(pob), rel=0.12)
    s = extraer_muestra(pob, 2000)
    e = estimar_mas(s.y, N=len(pob))
    assert e.loc["media", "ic_inf"] < pob.y.mean() < e.loc["media", "ic_sup"]
    assert e.loc["media", "error_estandar"] == pytest.approx(s.y.std() / np.sqrt(2000) * np.sqrt(0.9), rel=1e-6)
    p = estimar_mas(np.r_[np.ones(30), np.zeros(70)])
    assert p.loc["proporcion", "estimacion"] == pytest.approx(0.3)


def test_tamano_y_estratificado_y_razon():
    t = tamano_muestra_encuesta(0.03)
    assert t["n_necesario"] == 1068
    assert tamano_muestra_encuesta(0.03, N=2000)["n_necesario"] < 1068
    pob = _poblacion(seed=1)
    Nh = pob.estrato.value_counts(); Sh = pob.groupby("estrato").y.std()
    ney = asignacion_estratos(Nh, Sh, 1000, "neyman")
    assert ney.attrs["var_media"] < ney.attrs["var_proporcional"] and ney.n_h.sum() == pytest.approx(1000)
    assert ney.loc["c", "n_h"] / ney.loc["c", "N_h"] > ney.loc["a", "n_h"] / ney.loc["a", "N_h"]
    rng = np.random.default_rng(2)
    partes = [g.sample(int(round(ney.loc[h, "n_h"])), random_state=rng) for h, g in pob.groupby("estrato")]
    e = estimar_estratificado(pd.concat(partes), "y", "estrato", Nh)
    assert e["ic"][0] < pob.y.mean() < e["ic"][1] and e["efecto_diseno"] < 1
    s = pob.sample(300, random_state=3)
    r = estimador_razon(s.y, s.x, pob.x.mean(), N=len(pob))
    assert r["ganancia_varianza"] > 0.9 and abs(r["media_estimada"] - pob.y.mean()) < 3 * r["error_estandar"] and r["conviene"]


def test_ajuste_no_respuesta_corrige_sesgo():
    rng = np.random.default_rng(4)
    n = 20000
    d = pd.DataFrame({"edad": rng.choice(["joven", "mayor"], n)})
    d["y"] = np.where(d.edad == "joven", 10, 20) + rng.normal(0, 1, n)
    d["resp"] = rng.random(n) < np.where(d.edad == "joven", 0.2, 0.8)
    r = ajuste_no_respuesta(d, "resp", "edad")
    R = r["respondentes"]
    sin = R.y.mean(); con = np.average(R.y, weights=R.peso_ajustado)
    assert abs(con - d.y.mean()) < 0.1 < abs(sin - d.y.mean()) and r["tasa_respuesta"] == pytest.approx(0.5, abs=0.02) and r["avisos"]
