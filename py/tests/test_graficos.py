import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from matplotlib.figure import Figure
from scipy import stats

from arbol_estadistica.clustering import ajustar_kmeans, buscar_k_silhouette, preparar_matriz_clustering
from arbol_estadistica.contrastes import ajustar_p_valores
from arbol_estadistica.diagnostico import calcular_vif
from arbol_estadistica.graficos import (
    grafico_calibracion, grafico_clusters_pca, grafico_comparar_grupos, grafico_curva_potencia, grafico_curva_roc,
    grafico_diagnostico_residuos, grafico_fdr, grafico_kaplan_meier, grafico_odds_ratios, grafico_potencia,
    grafico_qq, grafico_regresion_simple, grafico_region_rechazo, grafico_residuos_agrupados, grafico_seleccion_k,
    grafico_umbrales, grafico_vif,
)
from arbol_estadistica.modelos import ajustar_logit, preparar_matriz_modelo, tabla_odds_ratios


@pytest.fixture(autouse=True)
def _cerrar():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def logit(datos_bin):
    X, y, std = preparar_matriz_modelo(datos_bin, "y", ["x1", "x2", "region"], ["region"], ["x1", "x2"])
    m = ajustar_logit(y, X)
    return X, y, m, m.predict(sm.add_constant(X))


def _titulo(fig):
    return " ".join(ax.get_title(loc="left") for ax in fig.axes)


def test_regresion_simple_no_muta_y_recupera_pendiente():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.uniform(0, 10, 300)})
    df["y"] = 2 + 3 * df.x + rng.normal(0, 1, 300)
    copia = df.copy()
    fig = grafico_regresion_simple(df, "x", "y")
    assert isinstance(fig, Figure) and "R²" in _titulo(fig)
    pd.testing.assert_frame_equal(df, copia)
    linea = [l for l in fig.axes[0].lines if l.get_label() == "recta MCO"][0]
    xs, ys = linea.get_data()
    assert abs((ys[-1] - ys[0]) / (xs[-1] - xs[0]) - 3) < 0.1


def test_diagnostico_residuos_ols_y_glm(logit, datos_bin):
    ols = sm.OLS(datos_bin["x1"], sm.add_constant(datos_bin[["x2"]])).fit()
    assert len(grafico_diagnostico_residuos(ols).axes) == 4
    X, y, m, _ = logit
    glm = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
    assert len(grafico_diagnostico_residuos(glm).axes) == 4
    with pytest.raises(TypeError):
        grafico_diagnostico_residuos(object())          # sin get_influence -> mensaje claro


def test_roc_calibracion_umbrales_residuos(logit):
    X, y, m, p = logit
    fig = grafico_curva_roc(y, {"completo": p, "azar": np.random.default_rng(1).random(len(y))})
    etiquetas = [l.get_label() for l in fig.axes[0].lines]
    assert any("completo (AUC" in e for e in etiquetas) and any("azar (AUC = 0.5" in e for e in etiquetas)
    roc = [l for l in fig.axes[0].lines if l.get_label().startswith("completo")][0].get_data()
    assert roc[0][0] == 0 and roc[1][-1] == 1
    assert "Hosmer-Lemeshow" in _titulo(grafico_calibracion(y, p))
    assert len(grafico_umbrales(y, p).axes[0].lines) >= 3
    assert "dentro de ±2 EE" in _titulo(grafico_residuos_agrupados(y, p))


def test_forest_odds_ratios(logit):
    X, y, m, _ = logit
    t = tabla_odds_ratios(m)
    fig = grafico_odds_ratios(t)
    assert [tk.get_text() for tk in fig.axes[0].get_yticklabels()] == list(t.sort_values("OR").index)
    assert fig.axes[0].get_xscale() == "log"
    with pytest.raises(KeyError):
        grafico_odds_ratios(pd.DataFrame({"x": [1]}))


def test_region_rechazo_pvalor_correcto():
    fig = grafico_region_rechazo(2.5, "t", gl=30)
    p = 2 * stats.t(30).sf(2.5)
    assert f"{p:.3g}" in _titulo(fig) and "se rechaza" in _titulo(fig)
    assert "no se rechaza" in _titulo(grafico_region_rechazo(1.0, "normal"))
    assert "0.0" in _titulo(grafico_region_rechazo(12.0, "chi2", gl=3))
    with pytest.raises(ValueError):
        grafico_region_rechazo(1.0, "t")


def test_potencia_y_curva():
    fig = grafico_potencia(0.5, 64)
    assert "Potencia 80%" in _titulo(fig)
    fig2 = grafico_curva_potencia([0.5, 0.8], n_max=200)
    textos = [t.get_text() for t in fig2.axes[0].texts]
    assert "n = 64" in textos and "n = 26" in textos


def test_comparar_grupos_numerico_y_categorico(datos_bin):
    assert "ANOVA" in _titulo(grafico_comparar_grupos(datos_bin, "x1", "region")) or \
        "Kruskal" in _titulo(grafico_comparar_grupos(datos_bin, "x1", "region"))
    assert "Chi-cuadrado" in _titulo(grafico_comparar_grupos(datos_bin, "y", "region"))


def test_qq_y_fdr():
    assert "Shapiro" in _titulo(grafico_qq(np.random.default_rng(0).normal(size=100)))
    p = np.r_[np.full(5, 1e-6), np.random.default_rng(0).uniform(size=95)]
    k = int(ajustar_p_valores(p)["rechaza"].sum())
    assert k >= 5 and _titulo(grafico_fdr(p)).startswith(f"BH: {k} descubrimientos")


def test_kaplan_meier_grafico():
    rng = np.random.default_rng(0)
    t = rng.exponential(10, 200)
    c = rng.uniform(0, 20, 200)
    df = pd.DataFrame({"t": np.minimum(t, c), "e": (t <= c).astype(int), "g": rng.choice(["a", "b"], 200)})
    fig = grafico_kaplan_meier(df, "t", "e", "g")
    assert "log-rank" in _titulo(fig)
    pasos = [l for l in fig.axes[0].lines if l.get_label().startswith("a ·")][0].get_data()
    assert pasos[1][0] == 1.0 and np.all(np.diff(pasos[1]) <= 1e-12)          # empieza en 1 y nunca sube
    assert np.isclose(pasos[0][-1], df.loc[df.g == "a", "t"].max())            # llega al último seguimiento


def test_clustering_y_vif():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(np.vstack([rng.normal(c, 0.5, (80, 2)) for c in ([0, 0], [5, 5], [0, 6])]), columns=["a", "b"])
    X, _ = preparar_matriz_clustering(df, ["a", "b"])
    fig = grafico_seleccion_k(buscar_k_silhouette(X, 2, 5))
    assert len(fig.axes) == 2 and "K = 3" in [t.get_text() for t in fig.axes[0].texts]
    assert len(grafico_clusters_pca(X, ajustar_kmeans(X, 3)["etiquetas"]).axes[0].get_legend().get_texts()) == 3
    v = calcular_vif(sm.add_constant(df.assign(c=df.a * 2 + rng.normal(0, 0.01, len(df)))))
    textos = [t.get_text() for t in grafico_vif(v).axes[0].texts]
    assert any("ALTO" in t for t in textos)
