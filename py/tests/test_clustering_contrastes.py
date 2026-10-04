import numpy as np
import pandas as pd
import pytest

from arbol_estadistica.clustering import ajustar_kmeans, buscar_k_silhouette, preparar_matriz_clustering, proyeccion_pca
from arbol_estadistica.contrastes import elegir_contraste, tukey_entre_grupos


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    centros = [(0, 0), (6, 6), (0, 8)]
    df = pd.concat([pd.DataFrame(rng.normal(c, 0.6, (150, 2)), columns=["a", "b"]).assign(real=i, precio=(i + 1) * 1000.0)
                    for i, c in enumerate(centros)], ignore_index=True)
    df["marca"] = rng.choice(["x", "y"], len(df))
    return df


def test_matriz_clustering_pesos_y_nombres(blobs):
    X, nombres = preparar_matriz_clustering(blobs, ["a", "b"], ["marca"], escalado="standard", pesos={"a": 2.0})
    assert X.shape == (450, 4) and nombres[:2] == ["a", "b"] and "marca_x" in nombres
    assert np.isclose(X[:, 0].std(), 2.0, atol=0.05) and np.isclose(X[:, 1].std(), 1.0, atol=0.05)
    with pytest.raises(ValueError):
        preparar_matriz_clustering(blobs, [])


def test_buscar_k_encuentra_3_y_ajustar_ordena_por_precio(blobs):
    X, _ = preparar_matriz_clustering(blobs, ["a", "b"])
    t = buscar_k_silhouette(X, 2, 6)
    assert int(t.loc[t.elegido, "k"].iloc[0]) == 3 and t["silhouette"].max() > 0.5
    r = ajustar_kmeans(X, 3, ordenar_por=blobs["precio"])
    assert sorted(set(r["etiquetas"])) == [1, 2, 3] and r["pseudo_r2"] > 0.7
    # el cluster 1 es el de menor precio medio
    assert blobs.groupby(r["etiquetas"])["precio"].mean().is_monotonic_increasing


def test_pca(blobs):
    X, _ = preparar_matriz_clustering(blobs, ["a", "b"])
    z, var = proyeccion_pca(X, 2)
    assert list(z.columns) == ["PC1", "PC2"] and np.isclose(var.sum(), 1.0)


# ---------- contrastes ----------
def _df(rng, mu_a, mu_b, n=200, sd_a=1, sd_b=1):
    return pd.concat([pd.DataFrame({"v": rng.normal(mu_a, sd_a, n), "g": "A"}),
                      pd.DataFrame({"v": rng.normal(mu_b, sd_b, n), "g": "B"})])


def test_dos_grupos_normales_varianza_igual_student():
    r = elegir_contraste(_df(np.random.default_rng(1), 0, 0.5), "v", "g")
    assert r["contraste"] == "t de Student" and r["significativo"] and r["efecto_nombre"] == "d_Cohen" and r["efecto"] < 0
    assert "p=" in r["interpretacion"]


def test_dos_grupos_varianza_distinta_welch():
    r = elegir_contraste(_df(np.random.default_rng(2), 0, 0.3, sd_a=1, sd_b=4), "v", "g")
    assert r["contraste"] == "t de Welch" and r["supuestos"]["levene_p"] < 0.05


def test_muestra_pequeña_no_normal_mann_whitney():
    rng = np.random.default_rng(3)
    d = pd.concat([pd.DataFrame({"v": rng.exponential(1, 12), "g": "A"}),
                   pd.DataFrame({"v": rng.exponential(5, 12), "g": "B"})])
    r = elegir_contraste(d, "v", "g")
    assert r["contraste"] == "U de Mann-Whitney" and r["efecto_nombre"] == "r_biserial_rangos"


def test_tres_grupos_anova_y_posthoc():
    rng = np.random.default_rng(4)
    d = pd.concat([pd.DataFrame({"v": rng.normal(m, 1, 150), "g": g}) for g, m in [("A", 0), ("B", 0), ("C", 1.5)]])
    r = elegir_contraste(d, "v", "g")
    assert r["contraste"] == "ANOVA de un factor" and r["significativo"] and any("post-hoc" in a for a in r["avisos"])
    t = tukey_entre_grupos(d, "v", "g").set_index(["group1", "group2"])
    assert not t.loc[("A", "B"), "rechaza"] and t.loc[("A", "C"), "rechaza"] and t.loc[("B", "C"), "rechaza"]


def test_tres_grupos_no_normales_kruskal():
    rng = np.random.default_rng(5)
    d = pd.concat([pd.DataFrame({"v": rng.exponential(s, 15), "g": g}) for g, s in [("A", 1), ("B", 1), ("C", 6)]])
    assert elegir_contraste(d, "v", "g")["contraste"] == "Kruskal-Wallis"


def test_categorico_chi2_y_fisher():
    rng = np.random.default_rng(6)
    g = rng.choice(["A", "B"], 500)
    y = (rng.random(500) < np.where(g == "A", 0.2, 0.4)).astype(int)
    r = elegir_contraste(pd.DataFrame({"y": y, "g": g}), "y", "g")
    assert r["contraste"].startswith("Chi-cuadrado") and r["significativo"] and r["efecto_nombre"] == "V_Cramer"
    peq = pd.DataFrame({"y": [1, 1, 0, 0, 0, 1, 0, 0], "g": list("AAAABBBB")})
    assert elegir_contraste(peq, "y", "g")["contraste"] == "Fisher exacto"


def test_aviso_n_grande_y_errores():
    rng = np.random.default_rng(7)
    r = elegir_contraste(_df(rng, 0, 0.05, n=4000), "v", "g")
    assert any("tamaño del efecto" in a for a in r["avisos"])
    with pytest.raises(ValueError):
        elegir_contraste(pd.DataFrame({"v": [1, 2, 3], "g": "A"}), "v", "g")
