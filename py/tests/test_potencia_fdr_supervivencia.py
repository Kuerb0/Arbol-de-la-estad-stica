import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests

from arbol_estadistica.contrastes import (
    ajustar_p_valores, curva_potencia, potencia_contraste_medias, potencia_por_simulacion,
    tamano_muestral_medias, tamano_muestral_proporciones,
)
from arbol_estadistica.modelos import ajustar_cox, contraste_log_rank, kaplan_meier


# ---------- FDR ----------
def test_bh_caso_de_libro_y_coincide_con_statsmodels():
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    t = ajustar_p_valores(p)
    assert np.allclose(t["p_ajustado"], 0.05) and t["rechaza"].all()          # BH: todos a 0.05
    assert not ajustar_p_valores(p, "bonferroni")["rechaza"][1:].any()          # Bonferroni: solo el 1º
    p2 = np.random.default_rng(0).uniform(size=40) ** 3
    assert np.allclose(ajustar_p_valores(p2)["p_ajustado"], multipletests(p2, method="fdr_bh")[1])


def test_bh_controla_fdr_y_valida_entradas():
    rng = np.random.default_rng(1)
    falsos, descub = 0, 0
    for _ in range(30):                                  # 900 nulos + 100 efectos reales
        p = np.r_[rng.uniform(size=900), stats.norm.sf(rng.normal(3.5, 1, 100))]
        r = ajustar_p_valores(p, alpha=0.05)["rechaza"].to_numpy()
        falsos += r[:900].sum(); descub += r.sum()
    assert falsos / descub < 0.08                        # FDR teórico <= 0.05 * 900/1000
    t = ajustar_p_valores([0.2, 0.001], nombres=["a", "b"])
    assert list(t.index) == ["a", "b"] and t.loc["b", "rango"] == 1 and t.attrs["n_rechazos"] == 1
    with pytest.raises(ValueError):
        ajustar_p_valores([0.1, np.nan])
    with pytest.raises(ValueError):
        ajustar_p_valores([0.1], metodo="magia")


# ---------- potencia ----------
def test_tamano_muestral_casos_de_libro():
    r = tamano_muestral_medias(efecto=0.5)
    assert r["n_grupo1"] == 64 and r["potencia_real"] >= 0.80
    assert tamano_muestral_medias(diferencia=5, desviacion=10)["n_grupo1"] == 64      # mismo d
    assert tamano_muestral_medias(efecto=0.2)["n_grupo1"] in (393, 394)
    pr = tamano_muestral_proporciones(0.5, 0.6)
    assert 385 <= pr["n_grupo1"] <= 390
    desig = tamano_muestral_medias(efecto=0.5, ratio=2)
    assert desig["n_grupo2"] == 2 * desig["n_grupo1"] and desig["n_grupo1"] < 64
    with pytest.raises(ValueError):
        tamano_muestral_medias()


def test_potencia_monotona_y_simulacion_cuadra_con_formula():
    c = curva_potencia(0.5, [10, 30, 64, 150])
    assert c["potencia"].is_monotonic_increasing and abs(c.loc[2, "potencia"] - 0.80) < 0.01
    assert abs(potencia_contraste_medias(64, 0.0) - 0.05) < 1e-6          # sin efecto: potencia = alpha

    def gen(rng, n):
        return rng.normal(0, 1, n), rng.normal(0.5, 1, n)
    sim = potencia_por_simulacion(gen, lambda a, b: stats.ttest_ind(a, b).pvalue, n=64, n_sim=1500)
    assert abs(sim["potencia"] - 0.80) < 3 * sim["error_mc"] + 0.01
    nulo = potencia_por_simulacion(lambda rng, n: (rng.normal(size=n), rng.normal(size=n)),
                                   lambda a, b: stats.ttest_ind(a, b).pvalue, n=30, n_sim=1500)
    assert abs(nulo["potencia"] - 0.05) < 0.02                              # error tipo I real


# ---------- supervivencia ----------
@pytest.fixture(scope="module")
def datos_surv():
    rng = np.random.default_rng(7)
    n = 3000
    x = rng.normal(size=n)
    g = rng.choice(["A", "B"], n)
    lam = 0.1 * np.exp(0.7 * x + 0.5 * (g == "B"))
    t_ev = rng.exponential(1 / lam)
    t_cen = rng.uniform(0, 25, n)
    return pd.DataFrame({"t": np.minimum(t_ev, t_cen), "e": (t_ev <= t_cen).astype(int), "x": x, "g": g})


def test_km_sin_censura_es_la_empirica_y_con_censura_bien():
    t = np.array([1, 2, 2, 3, 5, 8, 8, 9, 12, 15], float)
    km = kaplan_meier(pd.DataFrame({"t": t, "e": 1}), "t", "e")
    tab = km["tabla"]
    for _, f in tab.iloc[1:].iterrows():
        assert np.isclose(f["supervivencia"], (t > f["tiempo"]).mean())
    assert tab.iloc[0]["supervivencia"] == 1 and km["medianas"].loc[0, "mediana"] == 5
    # ejemplo con censura: S(3) = (1 - 1/5) * (1 - 1/3) tras censurar en t=2
    d = pd.DataFrame({"t": [1, 2, 2, 3, 4], "e": [1, 0, 0, 1, 1]})
    s = kaplan_meier(d, "t", "e")["tabla"].set_index("tiempo")["supervivencia"]
    assert np.isclose(s[3.0], 0.8 * (1 - 1 / 2)) and np.isclose(s[4.0], 0.0)
    assert (kaplan_meier(d, "t", "e")["tabla"]["ic_inf"] <= kaplan_meier(d, "t", "e")["tabla"]["supervivencia"] + 1e-12).all()


def test_log_rank_y_cox_recuperan_el_efecto(datos_surv):
    lr = contraste_log_rank(datos_surv, "t", "e", "g")
    assert lr["significativo"] and lr["gl"] == 1
    iguales = datos_surv.assign(g2=np.random.default_rng(3).choice(["u", "v"], len(datos_surv)))
    assert contraste_log_rank(iguales, "t", "e", "g2")["p_valor"] > 0.01
    cox = ajustar_cox(datos_surv, "t", "e", ["x"], categoricas=["g"], refs={"g": "A"})
    tab = cox["tabla"]
    assert abs(tab.loc["x", "coef"] - 0.7) < 0.08 and abs(tab.loc["g_B", "coef"] - 0.5) < 0.12
    assert tab.loc["x", "IC_inf"] < np.exp(0.7) < tab.loc["x", "IC_sup"]
    with pytest.raises(ValueError):
        kaplan_meier(datos_surv.assign(e=2), "t", "e")
