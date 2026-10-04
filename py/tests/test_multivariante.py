"""Multivariante (0.9): casos con respuesta conocida."""
import numpy as np
import pandas as pd
import pytest

from arbol_estadistica.multivariante import (adecuacion_factorial, alfa_cronbach, analisis_correspondencias,
                                             analisis_discriminante, analisis_factorial, clustering_jerarquico,
                                             contraste_box_m, contraste_hotelling, correlacion_canonica,
                                             distancia_mahalanobis, escalamiento_multidimensional, manova,
                                             mezclas_gaussianas, pca_completo)


def test_mahalanobis_robusta_no_se_deja_enmascarar():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.multivariate_normal([0, 0], [[1, .9], [.9, 1]], 300), columns=["a", "b"])
    X.iloc[:15] = [[2.5, -2.5]] * 15                      # atípicos «contra la correlación», en grupo
    rob, cla = distancia_mahalanobis(X), distancia_mahalanobis(X, robusta=False)
    assert rob.atipico.iloc[:15].all() and rob.atipico.iloc[:15].sum() >= cla.atipico.iloc[:15].sum()


def test_hotelling_box_y_manova():
    rng = np.random.default_rng(1)
    A = pd.DataFrame(rng.normal(size=(100, 3))); B = pd.DataFrame(rng.normal(size=(100, 3))) + [0.4, 0.4, 0]
    assert contraste_hotelling(A, B)["p_valor"] < 0.01
    assert contraste_hotelling(A, pd.DataFrame(rng.normal(size=(100, 3))))["p_valor"] > 0.01
    g = pd.DataFrame({"g": np.repeat(["x", "y", "z"], 100)})
    for v in "pq":
        g[v] = rng.normal(size=300) + (g.g == "z") * 0.6
    assert contraste_box_m(g, ["p", "q"], "g")["p_valor"] > 0.01
    assert contraste_box_m(g.assign(p=np.where(g.g == "x", g.p * 3, g.p)), ["p", "q"], "g")["p_valor"] < 0.001
    m = manova(g, ["p", "q"], "g")
    assert m.loc["Pillai's trace", "p_valor"] < 0.001


def test_canonica_pca_y_factorial():
    rng = np.random.default_rng(2)
    z = rng.normal(size=500)
    X = pd.DataFrame({"x1": z + rng.normal(size=500), "x2": rng.normal(size=500)})
    Y = pd.DataFrame({"y1": z + rng.normal(size=500), "y2": rng.normal(size=500)})
    cc = correlacion_canonica(X, Y)
    assert cc["tabla"].loc[1, "correlacion"] == pytest.approx(0.5, abs=0.08) and cc["tabla"].loc[2, "p_valor"] > 0.01
    p = pca_completo(pd.DataFrame(rng.multivariate_normal([0, 0, 0], [[1, .8, .8], [.8, 1, .8], [.8, .8, 1]], 600)))
    assert p["varianza"].pct_varianza.iloc[0] == pytest.approx(86.7, abs=3) and p["componentes_kaiser"] == 1
    L = rng.normal(size=(400, 2))
    items = pd.DataFrame(np.c_[L[:, [0]] + rng.normal(0, .5, (400, 3)), L[:, [1]] + rng.normal(0, .5, (400, 3))])
    fa = analisis_factorial(items, 2)
    c = fa["cargas"].abs()
    assert (c.iloc[:3].max(axis=1) > 0.75).all() and set(c.iloc[:3].idxmax(axis=1)) != set(c.iloc[3:].idxmax(axis=1))
    assert adecuacion_factorial(items)["bartlett_p"] < 1e-10
    assert adecuacion_factorial(pd.DataFrame(rng.normal(size=(300, 5))))["bartlett_p"] > 0.01


def test_cronbach_mds_y_correspondencias():
    rng = np.random.default_rng(3)
    t = rng.normal(size=(300, 1))
    a = alfa_cronbach(pd.DataFrame(t + rng.normal(0, .7, (300, 5))))
    assert 0.85 < a["alfa"] < 0.95 and a["ic_inf"] < a["alfa"] < a["ic_sup"]
    malo = pd.DataFrame(np.c_[t + rng.normal(0, .7, (300, 4)), rng.normal(size=(300, 1))])
    assert alfa_cronbach(malo)["por_item"]["correlacion_item_total"].idxmin() == 4
    pts = pd.DataFrame(rng.normal(size=(30, 2)))
    from scipy.spatial.distance import pdist, squareform
    m = escalamiento_multidimensional(D=squareform(pdist(pts)))
    assert m["bondad_ajuste"] == pytest.approx(1, abs=1e-6)
    tab = pd.DataFrame([[50, 10, 5], [10, 50, 5], [5, 5, 40]], index=list("ABC"), columns=list("xyz"))
    ca = analisis_correspondencias(tab)
    from scipy.stats import chi2_contingency
    assert ca["chi2"] == pytest.approx(chi2_contingency(tab, correction=False)[0])


def test_jerarquico_mezclas_y_discriminante():
    rng = np.random.default_rng(4)
    Z = pd.DataFrame(np.vstack([rng.normal(0, .5, (100, 2)), rng.normal(5, .5, (100, 2)), rng.normal([0, 5], .5, (100, 2))]))
    h = clustering_jerarquico(Z, k=3)
    assert h["cofenetica"] > 0.7 and sorted(h["etiquetas"].value_counts()) == [100, 100, 100]
    m = mezclas_gaussianas(Z, 5)
    assert m["k"] == 3 and m["incertidumbre_media"] < 0.01
    y = np.repeat(["a", "b", "c"], 100)
    d = analisis_discriminante(Z, y)
    assert d["exactitud_cv"] > 0.95 and d["funciones"].shape == (2, 2)
    assert analisis_discriminante(Z, y, "qda")["exactitud_cv"] > 0.95
