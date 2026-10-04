"""Galería: genera todos los gráficos del árbol con datos sintéticos y los guarda como PNG.

Uso:  python ejemplos/galeria_graficos.py [carpeta_salida]     (por defecto ./galeria_graficos)
Sirve de chuleta: cada bloque muestra la llamada mínima de un gráfico.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py"))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from arbol_estadistica.clustering import ajustar_kmeans, buscar_k_silhouette, preparar_matriz_clustering
from arbol_estadistica.diagnostico import calcular_vif
from arbol_estadistica.graficos import (
    grafico_calibracion, grafico_clusters_pca, grafico_comparar_grupos, grafico_curva_potencia, grafico_curva_roc,
    grafico_diagnostico_residuos, grafico_fdr, grafico_kaplan_meier, grafico_odds_ratios, grafico_potencia,
    grafico_qq, grafico_regresion_simple, grafico_region_rechazo, grafico_residuos_agrupados, grafico_seleccion_k,
    grafico_umbrales, grafico_vif,
)
from arbol_estadistica.modelos import ajustar_logit, preparar_matriz_modelo, tabla_odds_ratios


def main(salida: Path) -> list[Path]:
    salida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    figuras = {}

    # ---- regresión lineal ----
    n = 200
    x = rng.uniform(0, 10, n)
    reg = pd.DataFrame({"x": x, "y": 3 + 1.5 * x + rng.normal(0, 1 + 0.3 * x)})   # heterocedástico a propósito
    figuras["01_regresion_simple"] = grafico_regresion_simple(reg, "x", "y")
    ols = sm.OLS(reg["y"], sm.add_constant(reg[["x"]])).fit()
    figuras["02_diagnostico_residuos"] = grafico_diagnostico_residuos(ols)

    # ---- logística ----
    m = 3000
    df = pd.DataFrame({"edad": rng.normal(45, 12, m), "precio": rng.normal(20000, 5000, m),
                       "zona": rng.choice(["norte", "sur", "este"], m)})
    eta = -1 + 0.04 * (df.edad - 45) - 0.00012 * (df.precio - 20000) + df.zona.map({"norte": 0, "sur": 0.6, "este": -0.3})
    df["compra"] = (rng.random(m) < 1 / (1 + np.exp(-eta))).astype(int)
    X, y, std = preparar_matriz_modelo(df, "compra", ["edad", "precio", "zona"], ["zona"], ["edad", "precio"],
                                       refs={"zona": "norte"})
    modelo = ajustar_logit(y, X)
    p = modelo.predict(sm.add_constant(X))
    figuras["03_curva_roc"] = grafico_curva_roc(y, {"logística": p, "solo edad": 1 / (1 + np.exp(-0.04 * (df.edad - 45)))})
    figuras["04_calibracion"] = grafico_calibracion(y, p)
    figuras["05_odds_ratios"] = grafico_odds_ratios(tabla_odds_ratios(modelo, {"edad": 10, "precio": 1000}, std))
    figuras["06_umbrales"] = grafico_umbrales(y, p)
    figuras["07_residuos_agrupados"] = grafico_residuos_agrupados(y, p)
    figuras["08_vif"] = grafico_vif(calcular_vif(sm.add_constant(X)))

    # ---- contrastes ----
    t = stats.ttest_ind(rng.normal(0, 1, 40), rng.normal(0.55, 1, 40))
    figuras["09_region_rechazo"] = grafico_region_rechazo(t.statistic, "t", gl=78)
    figuras["10_potencia"] = grafico_potencia(0.5, 40)
    figuras["11_curva_potencia"] = grafico_curva_potencia([0.2, 0.5, 0.8])
    figuras["12_comparar_grupos"] = grafico_comparar_grupos(df, "precio", "zona")
    figuras["13_qq"] = grafico_qq(rng.exponential(1, 150))
    pv = np.r_[rng.uniform(size=180), stats.norm.sf(rng.normal(3, 1, 20))]
    figuras["14_fdr"] = grafico_fdr(pv)

    # ---- supervivencia ----
    k = 400
    grupo = rng.choice(["tarifa A", "tarifa B"], k)
    t_ev = rng.exponential(np.where(grupo == "tarifa B", 18, 10))
    t_cen = rng.uniform(0, 30, k)
    surv = pd.DataFrame({"meses": np.minimum(t_ev, t_cen), "baja": (t_ev <= t_cen).astype(int), "tarifa": grupo})
    figuras["15_kaplan_meier"] = grafico_kaplan_meier(surv, "meses", "baja", "tarifa")

    # ---- clustering ----
    blobs = pd.DataFrame(np.vstack([rng.normal(c, 0.7, (150, 3)) for c in ([0, 0, 0], [5, 5, 0], [0, 6, 4])]),
                         columns=["a", "b", "c"])
    Xc, _ = preparar_matriz_clustering(blobs, ["a", "b", "c"], escalado="standard")
    figuras["16_seleccion_k"] = grafico_seleccion_k(buscar_k_silhouette(Xc, 2, 7))
    figuras["17_clusters_pca"] = grafico_clusters_pca(Xc, ajustar_kmeans(Xc, 3)["etiquetas"])

    rutas = []
    for nombre, fig in figuras.items():
        ruta = salida / f"{nombre}.png"
        fig.savefig(ruta, dpi=110)
        matplotlib.pyplot.close(fig)
        rutas.append(ruta)
    return rutas


if __name__ == "__main__":
    destino = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("galeria_graficos")
    for r in main(destino):
        print(r)
