"""Rama actuarial (0.9): identidades actuariales exactas y resultados publicados (Taylor-Ashe / Mack)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from arbol_estadistica.actuarial import (ajustar_ley_mortalidad, bootstrap_chain_ladder, chain_ladder, conmutados,
                                         credibilidad_buhlmann, decrementos_multiples, estandarizar_tasas,
                                         exposicion_por_edad, indicadores_fecundidad, prima_neta, prima_por_principios,
                                         prima_tarifa, probabilidad_ruina, probabilidad_supervivencia, provision_matematica,
                                         proyeccion_leslie, qx_ley, recursion_panjer, renta_actuarial, seguro_vida,
                                         sensibilidad_longevidad, simular_siniestralidad_agregada, tabla_conjunta,
                                         tabla_mortalidad, tasas_especificas, vida_futura)

T = tabla_mortalidad(qx_ley(range(0, 121)))
I = 0.03
TAYLOR_ASHE = [[357848, 766940, 610542, 482940, 527326, 574398, 146342, 139950, 227229, 67948],
               [352118, 884021, 933894, 1183289, 445745, 320996, 527804, 266172, 425046, None],
               [290507, 1001799, 926219, 1016654, 750816, 146923, 495992, 280405, None, None],
               [310608, 1108250, 776189, 1562400, 272482, 352053, 206286, None, None, None],
               [443160, 693190, 991983, 769488, 504851, 470639, None, None, None, None],
               [396132, 937085, 847498, 805037, 705960, None, None, None, None, None],
               [440832, 847631, 1131398, 1063269, None, None, None, None, None, None],
               [359480, 1061648, 1443370, None, None, None, None, None, None, None],
               [376686, 986608, None, None, None, None, None, None, None, None],
               [344014, None, None, None, None, None, None, None, None, None]]


def test_tabla_y_esperanza_de_vida():
    assert T.lx.iloc[0] == 100000 and T.qx.iloc[-1] == 1
    assert T.loc[0, "e_completa"] == pytest.approx(T.loc[0, "e_abreviada"] + 0.5, abs=1e-6)
    vf = vida_futura(T, 65)
    assert vf["esperanza"] == pytest.approx(T.loc[65, "e_abreviada"]) and vf["funcion_probabilidad"].sum() == pytest.approx(1)
    assert probabilidad_supervivencia(T, 40, 10) == pytest.approx(T.loc[50, "lx"] / T.loc[40, "lx"])
    udd, cte = probabilidad_supervivencia(T, 80, 0.5, "udd"), probabilidad_supervivencia(T, 80, 0.5, "constante")
    assert cte < udd < 1                                         # (1−q)^s ≤ 1 − s·q (Bernoulli): fuerza constante, algo menos de supervivencia


def test_identidades_de_seguros_y_rentas():
    d = I / (1 + I)
    A, a = seguro_vida(T, 40, I)["valor_unitario"], renta_actuarial(T, 40, I)
    assert A + d * a == pytest.approx(1)                         # A_x = 1 − d·ä_x
    m = seguro_vida(T, 40, I, "mixto", 20)["valor_unitario"]
    assert m + d * renta_actuarial(T, 40, I, 20) == pytest.approx(1)
    c = conmutados(T, I)
    assert c.loc[40, "Nx"] / c.loc[40, "Dx"] == pytest.approx(a) and c.loc[40, "Mx"] / c.loc[40, "Dx"] == pytest.approx(A)
    assert renta_actuarial(T, 65, I) - renta_actuarial(T, 65, I, anticipada=False) == pytest.approx(1)
    assert renta_actuarial(T, 65, I, fraccionamiento=12) == pytest.approx(renta_actuarial(T, 65, I) - 11 / 24, abs=1e-9)
    assert seguro_vida(T, 40, I, momento="momento_muerte")["valor"] > A
    with pytest.raises(ValueError):
        seguro_vida(T, 40, I, "temporal")


def test_primas_y_provisiones():
    p = prima_neta(T, 40, I, "mixto", 20, capital=10000)
    pm = provision_matematica(T, 40, I, "mixto", 20, capital=10000)
    assert pm.loc[0, "provision"] == pytest.approx(0, abs=1e-6) and pm.loc[20, "provision"] == 10000
    assert pm.provision.is_monotonic_increasing and pm.attrs["prima_anual"] == pytest.approx(p["prima_anual"])
    g = prima_tarifa(T, 40, I, "mixto", 20, capital=10000)
    assert g["prima_tarifa"] > g["prima_inventario"] > g["prima_neta"]
    s = sensibilidad_longevidad(T, 65, I)
    assert s["impacto_pct"] > 5 and s["e_choque"] > s["e_base"]


def test_varias_cabezas_y_decrementos():
    tc = tabla_conjunta(T, 65, T, 62)
    assert (tc.tpxy <= tc[["tpx", "tpy"]].min(axis=1) + 1e-12).all() and (tc.tp_ultimo >= tc[["tpx", "tpy"]].max(axis=1) - 1e-12).all()
    qi = pd.DataFrame({"muerte": [0.01, 0.02], "rescate": [0.1, 0.08]})
    dm = decrementos_multiples(q_independientes=qi)
    vuelta = decrementos_multiples(q_dependientes=dm["dependientes"])["independientes"]
    assert np.allclose(vuelta, qi, atol=1e-4)


def test_ley_de_makeham_recuperada():
    x = np.arange(40, 95); E = np.full(len(x), 20000.0)
    mu = 0.0005 + 0.00007 * 1.1 ** (x + 0.5)
    D = np.random.default_rng(0).poisson(E * mu)
    p = ajustar_ley_mortalidad(x, D, E)["parametros"]
    assert p["c"] == pytest.approx(1.10, abs=0.01) and p["B"] == pytest.approx(7e-5, rel=0.4)


def test_chain_ladder_y_mack_reproducen_taylor_ashe():
    tri = pd.DataFrame(TAYLOR_ASHE, dtype=float).cumsum(axis=1).where(pd.DataFrame(TAYLOR_ASHE).notna())
    cl = chain_ladder(tri)
    assert cl["reserva_total"] == pytest.approx(18_680_856, rel=1e-6)
    assert cl["se_total"] == pytest.approx(2_447_095, rel=1e-4)
    assert cl["tabla"].se_mack.iloc[-1] == pytest.approx(1_363_155, rel=1e-4)
    assert chain_ladder(pd.DataFrame(TAYLOR_ASHE, dtype=float), acumulado=False)["reserva_total"] == pytest.approx(cl["reserva_total"])
    b = bootstrap_chain_ladder(tri, 500)
    assert b["media"] == pytest.approx(cl["reserva_total"], rel=0.05) and b["percentiles"][99.5] > b["percentiles"][75]


def test_panjer_coincide_con_la_simulacion_y_la_media():
    sev = np.r_[0, 0.25, 0.25, 0.25, 0.25]
    pj = recursion_panjer("poisson", {"lambda": 3}, sev, 80)
    assert pj.attrs["media"] == pytest.approx(7.5, rel=1e-6) and pj.probabilidad.iloc[0] == pytest.approx(np.exp(-3))
    s = simular_siniestralidad_agregada(stats.poisson(3), stats.randint(1, 5), 100000)
    assert s["media"] == pytest.approx(7.5, rel=0.02) and s["tvar"][0.99] >= s["var"][0.99]
    nb = recursion_panjer("binomial_negativa", {"r": 2, "p": 0.4}, sev, 200)
    assert nb.attrs["media"] == pytest.approx(2 * 0.6 / 0.4 * 2.5, rel=1e-4)


def test_ruina_primas_y_credibilidad():
    r = probabilidad_ruina(10, 0.2, 1, horizonte=300, n_sim=1500)
    assert r["psi_exponencial"] == pytest.approx(np.exp(-0.2 * 10 / 1.2) / 1.2) and r["cota_lundberg"] >= r["psi_exponencial"]
    assert abs(r["psi_horizonte"] - r["psi_exponencial"]) < 0.04
    assert np.isnan(probabilidad_ruina(10, 0.2, 1, severidad=stats.lognorm(1, scale=0.6))["coef_ajuste"])
    t = prima_por_principios(distribucion=stats.gamma(2, scale=500))
    assert t.loc["valor esperado (θ)", "recargo_pct"] == pytest.approx(10, abs=0.01) and (t.prima >= t.prima.iloc[0] - 1e-9).all()
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"r": np.repeat(np.arange(40), 6)}); d["x"] = np.repeat(rng.gamma(5, 20, 40), 6) + rng.normal(0, 60, 240)
    cb = credibilidad_buhlmann(d, "r", "x")
    assert 0 < cb["tabla"].credibilidad_Z.iloc[0] < 1
    assert (abs(cb["tabla"].prima_credibilidad - cb["media_colectiva"]) <= abs(cb["tabla"].media_propia - cb["media_colectiva"]) + 1e-9).all()
    homog = pd.DataFrame({"r": np.repeat(np.arange(40), 6), "x": rng.normal(100, 30, 240)})
    assert credibilidad_buhlmann(homog, "r", "x")["tabla"].credibilidad_Z.max() < 0.5


def test_demografia():
    t = tasas_especificas([5, 20, 80], [10000, 8000, 4000])
    assert t.attrs["tasa_bruta"] == pytest.approx(105 / 22000 * 1000) and (t.ic_inf < t.tasa).all()
    dirc = estandarizar_tasas([5, 20, 80], [10000, 8000, 4000], [20000, 10000, 2000])
    assert dirc["tasa_estandarizada"] < dirc["tasa_bruta"]                   # población joven estándar
    ind = estandarizar_tasas([5, 20, 80], [10000, 8000, 4000], [0.0004, 0.002, 0.015], "indirecta")
    assert ind["smr"] == pytest.approx(105 / 80) and ind["ic_inf"] > 1
    nac = pd.to_datetime(["1950-06-30"] * 2); ini = pd.to_datetime(["2020-01-01"] * 2); fin = pd.to_datetime(["2021-12-31", "2020-06-30"])
    e = exposicion_por_edad(pd.DataFrame({"n": nac, "i": ini, "f": fin, "e": [0, 1]}), "n", "i", "f", "e")
    assert e.exposicion.sum() == pytest.approx((730 + 181) / 365.25, rel=1e-3) and e.eventos.sum() == 1
    f = indicadores_fecundidad([10, 60, 90, 50, 10], [5000] * 5, [15, 20, 25, 30, 35], amplitud=5)
    assert f["isf"] == pytest.approx(0.22)
    l = proyeccion_leslie([100, 100, 100], [0.9, 0.8], [0, 1.2, 0.8], 30)
    ult = l["poblaciones"].iloc[-1].to_numpy()
    assert np.allclose(ult / ult.sum(), l["estructura_estable"], atol=1e-3)
