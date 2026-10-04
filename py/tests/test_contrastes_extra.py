"""Contrastes añadidos en 0.9: apareados, no paramétricos, permutación, intervalos y bootstrap."""
import numpy as np
import pandas as pd
import pytest

from arbol_estadistica.contrastes import (bondad_ajuste_multinomial, bootstrap_ic, contraste_apareado,
                                          contraste_friedman, contraste_jonckheere, contraste_mcnemar,
                                          contraste_permutacion, games_howell, intervalo_proporcion, posthoc_dunn)

rng = np.random.default_rng(5)


def test_apareado_gana_potencia_frente_a_independientes():
    rng = np.random.default_rng(101)
    base = rng.normal(100, 15, 25)
    despues = base + 3 + rng.normal(0, 2, 25)          # efecto pequeño frente a la variación entre sujetos
    r = contraste_apareado(base, despues)
    assert r["contraste"] == "t apareado" and r["p_valor"] < 0.001
    assert abs(r["diferencia_media"] - 3) < 1.5 and r["ic_diferencia"][0] < r["diferencia_media"] < r["ic_diferencia"][1]
    from scipy import stats
    assert stats.ttest_ind(base, despues).pvalue > 0.05      # el t de grupos independientes no lo ve
    w = contraste_apareado(base, base + rng.exponential(1, 25) ** 3, tipo="wilcoxon")
    assert w["contraste"].startswith("Wilcoxon") and w["p_valor"] < 0.01
    with pytest.raises(ValueError):
        contraste_apareado([1, 2, 3], [1, 2])


def test_friedman_y_mcnemar():
    rng = np.random.default_rng(102)
    n = 30
    d = pd.DataFrame({"s": np.repeat(np.arange(n), 3), "c": np.tile(["a", "b", "c"], n)})
    d["v"] = rng.normal(size=3 * n) + d.c.map({"a": 0, "b": 0.5, "c": 1.5}) + np.repeat(rng.normal(0, 3, n), 3)
    f = contraste_friedman(d, "s", "c", "v")
    assert f["p_valor"] < 0.001 and 0 < f["w_kendall"] <= 1
    antes = np.r_[np.ones(40), np.zeros(60)].astype(int)
    despues = antes.copy(); despues[:15] = 0; despues[40:42] = 1           # 15 dejan, 2 empiezan
    m = contraste_mcnemar(antes, despues)
    assert (m["b"], m["c"]) == (15, 2) and m["metodo"].startswith("exacto") and m["p_valor"] < 0.01


def test_permutacion_controla_alfa_y_detecta_efecto():
    rng = np.random.default_rng(103)
    r = contraste_permutacion(rng.normal(0, 1, 40), rng.normal(1.2, 1, 40), n_perm=2000)
    assert r["p_valor"] < 0.01
    pv = [contraste_permutacion(rng.normal(size=15), rng.normal(size=15), n_perm=300, semilla=i)["p_valor"] for i in range(200)]
    assert 0.01 < np.mean(np.array(pv) < 0.05) < 0.11
    assert 0 < contraste_permutacion([1, 2, 3], [4, 5, 6], n_perm=100)["p_valor"] <= 1


def test_jonckheere_ve_la_tendencia():
    rng = np.random.default_rng(104)
    d = pd.DataFrame({"g": np.repeat(["bajo", "medio", "alto"], 40)})
    d["v"] = rng.normal(d.g.map({"bajo": 0, "medio": 0.5, "alto": 1.0}), 1)
    r = contraste_jonckheere(d, "v", "g", orden=["bajo", "medio", "alto"])
    assert r["p_valor"] < 0.01 and r["z"] > 0
    assert contraste_jonckheere(d, "v", "g", orden=["alto", "medio", "bajo"])["p_valor"] > 0.9


def test_dunn_y_games_howell_encuentran_el_par_distinto():
    rng = np.random.default_rng(105)
    d = pd.DataFrame({"g": np.repeat(["a", "b", "c"], 50)})
    d["v"] = np.r_[rng.normal(0, 1, 50), rng.normal(0, 1, 50), rng.normal(1.2, 4, 50)]
    gh = games_howell(d, "v", "g")
    assert not gh.loc[(gh.grupo_1 == "a") & (gh.grupo_2 == "b"), "rechaza"].item()
    du = posthoc_dunn(d.assign(v=np.r_[rng.normal(0, 1, 50), rng.normal(0, 1, 50), rng.normal(2, 1, 50)]), "v", "g")
    assert du.loc[(du.grupo_1 == "a") & (du.grupo_2 == "c"), "rechaza"].item()
    assert (du.p_ajustado >= du.p_valor).all()


def test_intervalo_proporcion_metodos():
    rng = np.random.default_rng(106)
    w = intervalo_proporcion(0, 20, metodo="wilson")
    assert w["ic_inf"] == 0 and 0.1 < w["ic_sup"] < 0.2
    assert intervalo_proporcion(0, 20, metodo="wald")["ic_sup"] == 0          # Wald degenera con 0 éxitos
    cp, wi = intervalo_proporcion(7, 40, metodo="clopper_pearson"), intervalo_proporcion(7, 40)
    assert cp["ic_sup"] - cp["ic_inf"] > wi["ic_sup"] - wi["ic_inf"]          # exacto = conservador
    with pytest.raises(ValueError):
        intervalo_proporcion(5, 3)


def test_bootstrap_ic_contiene_la_media_y_bca_se_mueve_con_la_asimetria():
    rng = np.random.default_rng(107)
    x = rng.lognormal(0, 1, 200)
    p, b = bootstrap_ic(x, metodo="percentil"), bootstrap_ic(x, metodo="bca")
    real = np.exp(0.5)
    assert b["ic_inf"] < real < b["ic_sup"] and p["ic_inf"] < real < p["ic_sup"]
    assert b["ic_sup"] > p["ic_sup"]                         # cola derecha: BCa estira el extremo superior
    med = bootstrap_ic(x, np.median, metodo="percentil", n_boot=1000)
    assert med["ic_inf"] < 1 < med["ic_sup"]


def test_bondad_ajuste_multinomial():
    rng = np.random.default_rng(108)
    dado = rng.integers(1, 7, 600)
    r = bondad_ajuste_multinomial(np.bincount(dado)[1:])
    assert r["gl"] == 5 and r["p_valor"] > 0.01
    trucado = bondad_ajuste_multinomial([150, 90, 90, 90, 90, 90])
    assert trucado["p_valor"] < 0.001
