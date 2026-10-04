"""Mide las propiedades medibles de cada función y las guarda en py/propiedades/medidas.json.

    python py/medir_propiedades.py             # todo (unos minutos)
    python py/medir_propiedades.py --rapido    # menos tamaños y simulaciones (para probar)
    python py/medir_propiedades.py ajustar_logit kaplan_meier   # solo esas funciones

Qué mide:
  * Código (benchmark): velocidad (tiempo en n = 16 000), escalabilidad (exponente b de tiempo ∝ n^b
    con n = 1 000 … 64 000) y memoria (pico con tracemalloc en el n más grande).
  * Estadística (Monte Carlo, datos simulados donde se conoce la verdad): tamaño α real, cobertura
    de los IC, sesgo relativo y potencia relativa frente a la mejor alternativa.
Las notas 0-10 salen de reglas fijas (ver NOTAS); el detalle (tiempos, α real…) se guarda junto a la nota.
Lo medido sustituye en el visor a la nota estimada de py/propiedades/fichas.json y se marca como «medido».
Los tiempos dependen del ordenador: si lo ejecutas en el tuyo, las notas reflejan tu equipo.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import platform
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CODIGO = Path(__file__).resolve().parent
sys.path.insert(0, str(CODIGO))
SALIDA = CODIGO / "propiedades" / "medidas.json"
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

from arbol_estadistica import clustering as cl  # noqa: E402
from arbol_estadistica import contrastes as co  # noqa: E402
from arbol_estadistica import descriptiva as de  # noqa: E402
from arbol_estadistica import diagnostico as di  # noqa: E402
from arbol_estadistica import graficos as gr  # noqa: E402
from arbol_estadistica import modelos as mo  # noqa: E402
from arbol_estadistica import multivariante as mv  # noqa: E402
from arbol_estadistica import ml  # noqa: E402
from arbol_estadistica import actuarial as ac  # noqa: E402
from arbol_estadistica import finanzas as fi  # noqa: E402
from arbol_estadistica import preprocesado as pr  # noqa: E402
from arbol_estadistica import seleccion as se  # noqa: E402

N_REF = 16_000


# ============================================================== reglas de nota (0-10)
def _clip(x):
    return float(round(min(10.0, max(0.0, x)), 1))


def nota_velocidad(t):          # 0.01 s → 10 · 0.1 s → 8 · 1 s → 6 · 10 s → 4
    return _clip(10 - 2 * math.log10(max(t, 1e-4) / 0.01))


def nota_escalabilidad(b):      # n^1 → 10 · n^1.5 → 6.5 · n^2 → 3
    return _clip(10 - 7 * max(b - 1, 0))


def nota_memoria(mb):           # 1 MB → 10 · 10 MB → 8 · 100 MB → 6 · 1 GB → 4
    return _clip(10 - 2 * math.log10(max(mb, 0.1)))


def nota_desviacion(d, n_sim, p=0.05):
    """|real − nominal| en tamaño α o cobertura: 10 dentro del error de Monte Carlo; −2 por cada punto (pp) de más."""
    mc = 1.96 * math.sqrt(p * (1 - p) / n_sim)
    return _clip(10 - 200 * max(d - mc, 0))


def nota_sesgo(rel):            # sesgo relativo: 0 → 10 · 5 % → 8 · 12.5 % → 5 · 25 % → 0
    return _clip(10 - 40 * abs(rel))


NOTAS = {"velocidad": nota_velocidad.__doc__, "escalabilidad": nota_escalabilidad.__doc__,
         "memoria": nota_memoria.__doc__, "tamano_alfa/cobertura_ic": nota_desviacion.__doc__, "insesgadez": nota_sesgo.__doc__}


# ============================================================== datos sintéticos de referencia
def datos(n: int, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    d = pd.DataFrame({"edad": rng.normal(45, 12, n).round(), "precio": rng.normal(20000, 5000, n).round(-1),
                      "zona": rng.choice(["norte", "sur", "este"], n), "ruido": rng.normal(size=n),
                      "grupo": rng.choice(["A", "B", "C"], n)})
    eta = -1 + 0.04 * (d.edad - 45) - 0.00012 * (d.precio - 20000) + d.zona.map({"norte": 0, "sur": 0.6, "este": -0.3})
    d["compra"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    d["valor"] = rng.normal(d.grupo.map({"A": 10, "B": 10.5, "C": 11}), 2.0)
    riesgo = 0.08 * np.exp(0.03 * (d.edad - 45) - 0.4 * (d.grupo == "B"))
    tb, tf = rng.exponential(1 / riesgo), rng.uniform(0, 36, n)
    d["meses"], d["baja"] = np.minimum(tb, tf).round(3) + 1e-3, (tb <= tf).astype(int)
    return d


def _logit_listo(d):
    X, y, smap = mo.preparar_matriz_modelo(d, "compra", ["edad", "precio", "zona"], ["zona"], ["edad", "precio"], refs={"zona": "norte"})
    m = mo.ajustar_logit(y, X)
    return X, y, smap, m, np.asarray(m.predict(sm.add_constant(X)))


def _cerrar(f):
    def g():
        r = f()
        plt.close("all")
        return r
    return g


# ============================================================== benchmarks: nombre -> preparar(n) -> llamada()
# «fijo»: la función no depende del tamaño de los datos (se mide una vez y la escalabilidad es 10).
BENCH: dict[str, tuple[callable, bool]] = {}


def bench(nombre, fijo=False):
    def deco(fn):
        BENCH[nombre] = (fn, fijo)
        return fn
    return deco


@bench("balancear_clases")
def _(n): d = datos(n); return lambda: pr.balancear_clases(d, "compra", "oversample")
@bench("submuestreo_por_ratio")
def _(n): d = datos(n); return lambda: pr.submuestreo_por_ratio(d, "compra", 1.0)
@bench("pesos_por_clase")
def _(n): d = datos(n); return lambda: pr.pesos_por_clase(d.compra)
@bench("corregir_probabilidades_por_balanceo")
def _(n): p = np.random.default_rng(0).random(n); return lambda: pr.corregir_probabilidades_por_balanceo(p, 0.1, 0.5)
@bench("categorizar_por_cuantiles")
def _(n): d = datos(n); return lambda: pr.categorizar_por_cuantiles(d, ["edad", "precio"], q=5)
@bench("codificar_ordinal")
def _(n): d = datos(n); return lambda: pr.codificar_ordinal(d, "zona", ["norte", "sur", "este"])
@bench("duracion_hasta_evento")
def _(n):
    rng = np.random.default_rng(0); k = max(n // 5, 1)
    h = pd.DataFrame({"id": np.repeat(np.arange(k), 5), "t": np.tile(np.arange(5), k), "ev": (rng.random(5 * k) < 0.2).astype(int)})
    return lambda: pr.duracion_hasta_evento(h, "id", "ev", "t")
@bench("es_posible_fuga", fijo=True)
def _(n): cols = [f"var_{i}" for i in range(500)] + ["y_pred", "prob_x"]; return lambda: [se.es_posible_fuga(c) for c in cols]
@bench("contraste_chi2_variable")
def _(n): d = datos(n); return lambda: se.contraste_chi2_variable(d, "zona", "compra")
@bench("cribar_variables")
def _(n): d = datos(n).drop(columns=["meses", "baja"]); return lambda: se.cribar_variables(d, "compra")
@bench("seleccion_forward")
def _(n): d = datos(n); return lambda: se.seleccion_forward(d, "compra", ["edad", "precio", "zona", "ruido", "grupo"], categoricas=["zona", "grupo"], criterio="bic")
@bench("preparar_matriz_modelo")
def _(n): d = datos(n); return lambda: mo.preparar_matriz_modelo(d, "compra", ["edad", "precio", "zona"], ["zona"], ["edad", "precio"], refs={"zona": "norte"})
@bench("ajustar_logit")
def _(n): X, y, *_ = _logit_listo(datos(n)); return lambda: mo.ajustar_logit(y, X)
@bench("regresion_logistica_firth")
def _(n): X, y, *_ = _logit_listo(datos(n)); Xc = sm.add_constant(X); return lambda: mo.regresion_logistica_firth(Xc, y)
@bench("tabla_odds_ratios")
def _(n): X, y, smap, m, _p = _logit_listo(datos(n)); return lambda: mo.tabla_odds_ratios(m, {"edad": 10}, smap)
@bench("tabla_parametros_wald")
def _(n): m = _logit_listo(datos(n))[3]; return lambda: mo.tabla_parametros_wald(m)
@bench("perfil_respuesta")
def _(n): d = datos(n); return lambda: mo.perfil_respuesta(d.compra)
@bench("dividir_train_test")
def _(n): d = datos(n); return lambda: mo.dividir_train_test(d, "compra")
@bench("ajustar_glm_binomial")
def _(n): tr, te = mo.dividir_train_test(datos(n), "compra"); return lambda: mo.ajustar_glm_binomial("compra ~ edad + precio + C(zona)", tr, te, "compra")
@bench("tabla_coeficientes")
def _(n):
    m = sm.GLM.from_formula("compra ~ edad + precio + C(zona)", datos(n), family=sm.families.Binomial()).fit()
    return lambda: mo.tabla_coeficientes(m)
@bench("comparar_tecnicas_estimacion")
def _(n):
    tr, te = mo.dividir_train_test(datos(n), "compra")
    a, b = sm.add_constant(tr[["edad", "precio"]]), sm.add_constant(te[["edad", "precio"]])
    return lambda: mo.comparar_tecnicas_estimacion(tr.compra, a, te.compra, b)
@bench("comparar_enlaces")
def _(n):
    tr, te = mo.dividir_train_test(datos(n), "compra")
    a, b = sm.add_constant(tr[["edad", "precio"]]), sm.add_constant(te[["edad", "precio"]])
    return lambda: mo.comparar_enlaces(tr.compra, a, te.compra, b)
@bench("logit_multinomial_sas")
def _(n):
    d = datos(n).assign(clase=lambda x: np.where(x.valor > 11, "alta", np.where(x.valor > 9.5, "media", "base")))
    return lambda: mo.logit_multinomial_sas(d, "clase", x_num=["edad", "ruido"], x_cat=["zona"], base_class="base")
@bench("sensibilidad_por_grupo")
def _(n): d = datos(n); return lambda: mo.sensibilidad_por_grupo(d, "compra", "precio", "zona")
@bench("ganancia_por_bajada_precio")
def _(n): d = datos(n); return lambda: mo.ganancia_por_bajada_precio(d, "compra", "precio", "zona")
@bench("tendencia_polinomica_ponderada")
def _(n): rng = np.random.default_rng(0); x = rng.uniform(0, 10, n); y = (x - 5) ** 2 + rng.normal(size=n); return lambda: mo.tendencia_polinomica_ponderada(x, y)
@bench("kaplan_meier")
def _(n): d = datos(n); return lambda: mo.kaplan_meier(d, "meses", "baja", "grupo")
@bench("contraste_log_rank")
def _(n): d = datos(n); return lambda: mo.contraste_log_rank(d, "meses", "baja", "grupo")
@bench("ajustar_cox")
def _(n): d = datos(n); return lambda: mo.ajustar_cox(d, "meses", "baja", ["edad"], categoricas=["grupo"], refs={"grupo": "A"})
@bench("hosmer_lemeshow")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return lambda: di.hosmer_lemeshow(y, p)
@bench("estadisticos_asociacion")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return lambda: di.estadisticos_asociacion(y, p)
@bench("calcular_vif")
def _(n): X = sm.add_constant(_logit_listo(datos(n))[0]); return lambda: di.calcular_vif(X)
@bench("filtrar_vif_iterativo")
def _(n):
    X = sm.add_constant(_logit_listo(datos(n))[0]); X = X.assign(copia=X.edad * 2 + np.random.default_rng(0).normal(0, .01, len(X)))
    return lambda: di.filtrar_vif_iterativo(X, umbral=8)
@bench("auc_train_test")
def _(n):
    tr, te = mo.dividir_train_test(datos(n), "compra")
    m = sm.GLM.from_formula("compra ~ edad + precio + C(zona)", tr, family=sm.families.Binomial()).fit()
    return lambda: di.auc_train_test(m, tr, te, "compra")
@bench("metricas_binarias")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return lambda: di.metricas_binarias(y, p)
@bench("tabla_umbrales")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return lambda: di.tabla_umbrales(y, p)
@bench("umbral_optimo_youden")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return lambda: di.umbral_optimo_youden(y, p)
@bench("resumen_auc_multiclase")
def _(n):
    rng = np.random.default_rng(0); P = rng.dirichlet([1, 1, 1], n); yv = np.array([rng.choice(3, p=q) for q in P[:2000]])
    yv = np.resize(yv, n)
    return lambda: di.resumen_auc_multiclase(yv, P, [0, 1, 2])
@bench("matrices_confusion")
def _(n): rng = np.random.default_rng(0); a, b = rng.choice(["a", "b", "c"], n), rng.choice(["a", "b", "c"], n); return lambda: di.matrices_confusion(a, b, ["a", "b", "c"])
@bench("preparar_matriz_clustering")
def _(n): d = datos(n); return lambda: cl.preparar_matriz_clustering(d, ["edad", "precio", "valor"], ["zona"], escalado="standard")
@bench("buscar_k_silhouette")
def _(n): X, _ = cl.preparar_matriz_clustering(datos(n), ["edad", "precio", "valor"], escalado="standard"); return lambda: cl.buscar_k_silhouette(X, 2, 5)
@bench("ajustar_kmeans")
def _(n): X, _ = cl.preparar_matriz_clustering(datos(n), ["edad", "precio", "valor"], escalado="standard"); return lambda: cl.ajustar_kmeans(X, 3)
@bench("proyeccion_pca")
def _(n): X, _ = cl.preparar_matriz_clustering(datos(n), ["edad", "precio", "valor"], escalado="standard"); return lambda: cl.proyeccion_pca(X)
@bench("elegir_contraste")
def _(n): d = datos(n); return lambda: co.elegir_contraste(d, "valor", "grupo")
@bench("tukey_entre_grupos")
def _(n): d = datos(n); return lambda: co.tukey_entre_grupos(d, "valor", "grupo")
@bench("ajustar_p_valores")
def _(n): p = np.random.default_rng(0).random(n); return lambda: co.ajustar_p_valores(p)
@bench("tamano_muestral_medias", fijo=True)
def _(n): return lambda: co.tamano_muestral_medias(efecto=0.3)
@bench("tamano_muestral_proporciones", fijo=True)
def _(n): return lambda: co.tamano_muestral_proporciones(0.10, 0.12)
@bench("potencia_contraste_medias", fijo=True)
def _(n): return lambda: co.potencia_contraste_medias(50, 0.4)
@bench("curva_potencia", fijo=True)
def _(n): return lambda: co.curva_potencia(0.4)
@bench("potencia_por_simulacion", fijo=True)
def _(n):
    gen = lambda rng, k: (rng.normal(0, 1, k), rng.normal(0.4, 1, k))
    return lambda: co.potencia_por_simulacion(gen, lambda a, b: stats.ttest_ind(a, b).pvalue, n=50, n_sim=500)
# gráficos
@bench("grafico_curva_roc")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return _cerrar(lambda: gr.grafico_curva_roc(y, p))
@bench("grafico_calibracion")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return _cerrar(lambda: gr.grafico_calibracion(y, p))
@bench("grafico_umbrales")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return _cerrar(lambda: gr.grafico_umbrales(y, p))
@bench("grafico_residuos_agrupados")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return _cerrar(lambda: gr.grafico_residuos_agrupados(y, p))
@bench("grafico_odds_ratios", fijo=True)
def _(n): _X, _y, s, m, _p = _logit_listo(datos(2000)); t = mo.tabla_odds_ratios(m, {"edad": 10}, s); return _cerrar(lambda: gr.grafico_odds_ratios(t))
@bench("grafico_vif", fijo=True)
def _(n): t = di.calcular_vif(sm.add_constant(_logit_listo(datos(2000))[0])); return _cerrar(lambda: gr.grafico_vif(t))
@bench("grafico_seleccion_k", fijo=True)
def _(n): X, _ = cl.preparar_matriz_clustering(datos(1000), ["edad", "precio", "valor"]); t = cl.buscar_k_silhouette(X, 2, 4); return _cerrar(lambda: gr.grafico_seleccion_k(t))
@bench("grafico_clusters_pca")
def _(n): X, _ = cl.preparar_matriz_clustering(datos(n), ["edad", "precio", "valor"]); e = cl.ajustar_kmeans(X, 3)["etiquetas"]; return _cerrar(lambda: gr.grafico_clusters_pca(X, e))
@bench("grafico_region_rechazo", fijo=True)
def _(n): return _cerrar(lambda: gr.grafico_region_rechazo(2.1, "t", gl=50))
@bench("grafico_potencia", fijo=True)
def _(n): return _cerrar(lambda: gr.grafico_potencia(0.5, 30))
@bench("grafico_curva_potencia", fijo=True)
def _(n): return _cerrar(lambda: gr.grafico_curva_potencia([0.3, 0.5]))
@bench("grafico_comparar_grupos")
def _(n): d = datos(n); return _cerrar(lambda: gr.grafico_comparar_grupos(d, "valor", "grupo"))
@bench("grafico_qq")
def _(n): x = np.random.default_rng(0).normal(size=n); return _cerrar(lambda: gr.grafico_qq(x))
@bench("grafico_fdr")
def _(n): p = np.random.default_rng(0).random(n); return _cerrar(lambda: gr.grafico_fdr(p))
@bench("grafico_regresion_simple")
def _(n): d = datos(n); return _cerrar(lambda: gr.grafico_regresion_simple(d, "edad", "valor"))
@bench("grafico_diagnostico_residuos")
def _(n): d = datos(n); m = sm.OLS(d.valor, sm.add_constant(d[["edad"]])).fit(); return _cerrar(lambda: gr.grafico_diagnostico_residuos(m))
@bench("grafico_kaplan_meier")
def _(n): d = datos(n); return _cerrar(lambda: gr.grafico_kaplan_meier(d, "meses", "baja", "grupo"))

# finanzas
@bench("var_tvar")
def _(n): x = stats.t.rvs(3, size=n, random_state=np.random.default_rng(0)); return lambda: fi.var_tvar(x)
@bench("ajustar_gpd")
def _(n): x = stats.t.rvs(3, size=n, random_state=np.random.default_rng(0)); return lambda: fi.ajustar_gpd(x)
@bench("estimador_hill")
def _(n): x = np.abs(stats.t.rvs(3, size=n, random_state=np.random.default_rng(0))); return lambda: fi.estimador_hill(x)
@bench("funcion_exceso_medio")
def _(n): x = np.random.default_rng(0).exponential(size=n); return lambda: fi.funcion_exceso_medio(x)
@bench("simular_copula")
def _(n): return lambda: fi.simular_copula("gumbel", 2, n)
@bench("matriz_covarianzas")
def _(n): R = pd.DataFrame(np.random.default_rng(0).normal(size=(n, 10))); return lambda: fi.matriz_covarianzas(R)
@bench("beta_capm")
def _(n): rng = np.random.default_rng(0); m = rng.normal(size=n); a = m + rng.normal(size=n); return lambda: fi.beta_capm(a, m)
@bench("modelo_factores")
def _(n):
    rng = np.random.default_rng(0); F = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("xyz")); R = pd.DataFrame({"a": F.x + rng.normal(size=n), "b": F.y + rng.normal(size=n)})
    return lambda: fi.modelo_factores(R, F)
@bench("dominancia_estocastica")
def _(n): rng = np.random.default_rng(0); a, b = rng.normal(size=n), rng.normal(.1, 1, n); return lambda: fi.dominancia_estocastica(a, b)
_S3 = pd.DataFrame([[.04, .006, .001], [.006, .01, .0005], [.001, .0005, .0025]], index=list("abc"), columns=list("abc"))
for _nombre, _fn in {
    "prueba_estres": lambda: fi.prueba_estres({"a": 1e6, "b": 5e6}, {"x": {"a": -.4}, "y": {"b": -.1}}),
    "frontera_eficiente": lambda: fi.frontera_eficiente(pd.Series([.08, .05, .03], index=list("abc")), _S3, 20, tasa_libre=.01),
    "black_litterman": lambda: fi.black_litterman(_S3, [.5, .3, .2], [[1, -1, 0]], [.05]),
    "equivalente_cierto": lambda: fi.equivalente_cierto([0, 100], [.5, .5], "exponencial", .02),
    "fraccion_anio": lambda: [fi.fraccion_anio("2020-01-15", "2026-03-31", c) for c in ("act/365", "act/360", "30/360", "act/act")],
    "precio_bono": lambda: fi.precio_bono(100, .05, 30, .04, 2), "tir_bono": lambda: fi.tir_bono(96.5, 100, .035, 7),
    "bootstrapping_etti": lambda: fi.bootstrapping_etti(range(1, 31), np.linspace(.02, .04, 30), np.full(30, 100.0)),
    "inmunizacion": lambda: fi.inmunizacion(5, 1e6, pd.DataFrame({"precio": [98, 105], "duracion": [2, 9]})),
    "valorar_swap": lambda: fi.valorar_swap(.03, np.linspace(.99, .7, 30)),
    "black_scholes": lambda: fi.black_scholes(100, 100, 1, .03, .2), "arbol_binomial": lambda: fi.arbol_binomial(100, 100, 1, .03, .2, 500, "put", True),
    "grafico_frontera_eficiente": lambda: gr.grafico_frontera_eficiente(fi.frontera_eficiente(pd.Series([.08, .05, .03], index=list("abc")), _S3, 15), pd.Series([.08, .05, .03], index=list("abc")), _S3),
    "grafico_copula": lambda: gr.grafico_copula(fi.simular_copula("clayton", 2, 3000)),
}.items():
    def _crear_fijo2(fn=_fn, graf=_nombre.startswith("grafico_")):
        def _(n): return _cerrar(fn) if graf else fn
        return _
    bench(_nombre, fijo=True)(_crear_fijo2())

# actuarial
_T = None
def _tabla():
    global _T
    if _T is None:
        _T = ac.tabla_mortalidad(ac.qx_ley(range(0, 121)))
    return _T
_TRI = [[357848, 766940, 610542, 482940, 527326, 574398, 146342, 139950, 227229, 67948], [352118, 884021, 933894, 1183289, 445745, 320996, 527804, 266172, 425046, None],
        [290507, 1001799, 926219, 1016654, 750816, 146923, 495992, 280405, None, None], [310608, 1108250, 776189, 1562400, 272482, 352053, 206286, None, None, None],
        [443160, 693190, 991983, 769488, 504851, 470639, None, None, None, None], [396132, 937085, 847498, 805037, 705960, None, None, None, None, None],
        [440832, 847631, 1131398, 1063269, None, None, None, None, None, None], [359480, 1061648, 1443370, None, None, None, None, None, None, None],
        [376686, 986608, None, None, None, None, None, None, None, None], [344014, None, None, None, None, None, None, None, None, None]]
def _tri(): return pd.DataFrame(_TRI, dtype=float).cumsum(axis=1).where(pd.DataFrame(_TRI).notna())
for _nombre, _fn in {
    "qx_ley": lambda: ac.qx_ley(range(0, 121)), "tabla_mortalidad": lambda: ac.tabla_mortalidad(ac.qx_ley(range(0, 121))),
    "probabilidad_supervivencia": lambda: ac.probabilidad_supervivencia(_tabla(), 40.3, 10.5), "vida_futura": lambda: ac.vida_futura(_tabla(), 65),
    "tabla_conjunta": lambda: ac.tabla_conjunta(_tabla(), 65, _tabla(), 62),
    "decrementos_multiples": lambda: ac.decrementos_multiples(q_independientes=pd.DataFrame({"a": np.full(80, .01), "b": np.full(80, .1)})),
    "conmutados": lambda: ac.conmutados(_tabla(), .03), "seguro_vida": lambda: ac.seguro_vida(_tabla(), 40, .03, "mixto", 20),
    "renta_actuarial": lambda: ac.renta_actuarial(_tabla(), 65, .03, fraccionamiento=12), "prima_neta": lambda: ac.prima_neta(_tabla(), 40, .03, "mixto", 20),
    "provision_matematica": lambda: ac.provision_matematica(_tabla(), 40, .03, "mixto", 20), "prima_tarifa": lambda: ac.prima_tarifa(_tabla(), 40, .03, "mixto", 20),
    "sensibilidad_longevidad": lambda: ac.sensibilidad_longevidad(_tabla(), 65, .03), "chain_ladder": lambda: ac.chain_ladder(_tri()),
    "bootstrap_chain_ladder": lambda: ac.bootstrap_chain_ladder(_tri(), 1000), "probabilidad_ruina": lambda: ac.probabilidad_ruina(10, .2, 1),
    "prima_por_principios": lambda: ac.prima_por_principios(distribucion=stats.gamma(2, scale=500), n_sim=100000),
    "tasas_especificas": lambda: ac.tasas_especificas(np.arange(1, 101), np.full(100, 1e4)),
    "estandarizar_tasas": lambda: ac.estandarizar_tasas(np.arange(1, 101), np.full(100, 1e4), np.linspace(1, 2, 100)),
    "indicadores_fecundidad": lambda: ac.indicadores_fecundidad(np.arange(1, 36) * 10, np.full(35, 1e4), np.arange(15, 50)),
    "proyeccion_leslie": lambda: ac.proyeccion_leslie(np.full(20, 100.0), np.full(19, .95), np.r_[np.zeros(4), np.full(8, .3), np.zeros(8)], 50),
    "grafico_tabla_mortalidad": lambda: gr.grafico_tabla_mortalidad(_tabla()),
}.items():
    def _crear_fijo(fn=_fn, graf=_nombre.startswith("grafico_")):
        def _(n): return _cerrar(fn) if graf else fn
        return _
    bench(_nombre, fijo=True)(_crear_fijo())
@bench("grafico_reserva_bootstrap", fijo=True)
def _(n): b = ac.bootstrap_chain_ladder(_tri(), 500); return _cerrar(lambda: gr.grafico_reserva_bootstrap(b))
@bench("ajustar_ley_mortalidad", fijo=True)
def _(n):
    x = np.arange(40, 95); E = np.full(len(x), 2e4); D = np.random.default_rng(0).poisson(E * (.0005 + .00007 * 1.1 ** (x + .5)))
    return lambda: ac.ajustar_ley_mortalidad(x, D, E)
@bench("recursion_panjer")
def _(n): return lambda: ac.recursion_panjer("poisson", {"lambda": 3}, [0, .25, .25, .25, .25], max_s=max(n // 20, 60))
@bench("simular_siniestralidad_agregada")
def _(n): return lambda: ac.simular_siniestralidad_agregada(stats.poisson(2), stats.lognorm(1, scale=1000), n_sim=n)
@bench("credibilidad_buhlmann")
def _(n):
    rng = np.random.default_rng(0); d = pd.DataFrame({"r": rng.integers(0, max(n // 20, 5), n)}); d["x"] = rng.gamma(2, 50, n)
    return lambda: ac.credibilidad_buhlmann(d, "r", "x")
@bench("exposicion_por_edad")
def _(n):
    rng = np.random.default_rng(0)
    nac = pd.Timestamp("1950-01-01") + pd.to_timedelta(rng.integers(0, 7000, n), unit="D"); ini = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    d = pd.DataFrame({"n": nac, "i": ini, "f": ini + pd.to_timedelta(rng.integers(30, 1500, n), unit="D"), "e": rng.random(n) < .03})
    return lambda: ac.exposicion_por_edad(d, "n", "i", "f", "e", 40, 90)

# machine learning
def _xy_ml(n, semilla=0):
    rng = np.random.default_rng(semilla)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.uniform(0, 10, n), "ruido": rng.normal(size=n),
                      "z": rng.choice(["p", "q", "r"], n)})
    eta = 1.5 * X.a * X.b + 1.2 * np.sin(X.c) + (X.z == "q") - .3
    return X, (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
for _nombre in ("ajustar_arbol_decision", "ajustar_random_forest", "ajustar_gradient_boosting", "ajustar_adaboost", "ajustar_knn",
                "ajustar_svm", "ajustar_naive_bayes", "ajustar_red_neuronal", "ajustar_stacking"):
    def _crear(nombre=_nombre):
        def _(n):
            X, y = _xy_ml(n); f = getattr(ml, nombre); return lambda: f(X, y)
        return _
    bench(_nombre)(_crear())
@bench("comparar_clasificadores")
def _(n): X, y = _xy_ml(n); return lambda: ml.comparar_clasificadores(X, y, ("logit", "gb"), cv=3)
@bench("importancia_permutacion")
def _(n): X, y = _xy_ml(n); r = ml.ajustar_gradient_boosting(X, y); return lambda: ml.importancia_permutacion(r, 3)
@bench("dependencia_parcial")
def _(n): X, y = _xy_ml(n); r = ml.ajustar_gradient_boosting(X, y); return lambda: ml.dependencia_parcial(r, "c", 20)
@bench("grafico_importancias", fijo=True)
def _(n): X, y = _xy_ml(2000); t = ml.ajustar_random_forest(X, y)["importancias"]; return _cerrar(lambda: gr.grafico_importancias(t))
@bench("grafico_dependencia_parcial", fijo=True)
def _(n): X, y = _xy_ml(2000); d = ml.dependencia_parcial(ml.ajustar_gradient_boosting(X, y), "c", 20, ice=20); return _cerrar(lambda: gr.grafico_dependencia_parcial(d))

# multivariante
def _num(n): return datos(n)[["edad", "precio", "valor", "ruido"]]
@bench("distancia_mahalanobis")
def _(n): X = _num(n); return lambda: mv.distancia_mahalanobis(X)
@bench("contraste_hotelling")
def _(n): d = datos(n); return lambda: mv.contraste_hotelling(d.loc[d.grupo == "A", ["edad", "valor"]], d.loc[d.grupo == "B", ["edad", "valor"]])
@bench("contraste_box_m")
def _(n): d = datos(n); return lambda: mv.contraste_box_m(d, ["edad", "valor"], "grupo")
@bench("manova")
def _(n): d = datos(n); return lambda: mv.manova(d, ["edad", "valor"], "grupo")
@bench("correlacion_canonica")
def _(n): d = datos(n); return lambda: mv.correlacion_canonica(d[["edad", "precio"]], d[["valor", "meses"]])
@bench("pca_completo")
def _(n): X = _num(n); return lambda: mv.pca_completo(X)
@bench("adecuacion_factorial")
def _(n): X = _num(n); return lambda: mv.adecuacion_factorial(X)
@bench("analisis_factorial")
def _(n): X = _num(n); return lambda: mv.analisis_factorial(X, 2)
@bench("alfa_cronbach")
def _(n): rng = np.random.default_rng(0); t = rng.normal(size=(n, 1)); I = pd.DataFrame(t + rng.normal(size=(n, 5))); return lambda: mv.alfa_cronbach(I)
@bench("escalamiento_multidimensional")
def _(n): X = _num(n); return lambda: mv.escalamiento_multidimensional(X=X)
@bench("analisis_correspondencias", fijo=True)
def _(n): t = pd.crosstab(datos(5000).zona, datos(5000).grupo); return lambda: mv.analisis_correspondencias(t)
@bench("clustering_jerarquico")
def _(n): X = _num(n); return lambda: mv.clustering_jerarquico(X, k=3)
@bench("mezclas_gaussianas")
def _(n): X = _num(n)[["edad", "valor"]]; return lambda: mv.mezclas_gaussianas(X, 3)
@bench("analisis_discriminante")
def _(n): d = datos(n); return lambda: mv.analisis_discriminante(d[["edad", "precio", "valor"]], d.grupo)
@bench("grafico_dendrograma", fijo=True)
def _(n): h = mv.clustering_jerarquico(_num(1000)); return _cerrar(lambda: gr.grafico_dendrograma(h, k=3))
@bench("grafico_biplot", fijo=True)
def _(n): p = mv.pca_completo(_num(3000)); return _cerrar(lambda: gr.grafico_biplot(p))

# preprocesado añadido
def _con_faltantes(n, semilla=0):
    rng = np.random.default_rng(semilla); d = datos(n)[["edad", "precio", "valor", "ruido", "zona"]].copy()
    d.loc[rng.random(n) < .15, "valor"] = np.nan; d.loc[rng.random(n) < .1, "precio"] = np.nan
    return d
@bench("resumen_faltantes")
def _(n): d = _con_faltantes(n); return lambda: pr.resumen_faltantes(d)
@bench("contraste_mcar_little")
def _(n): d = _con_faltantes(n).select_dtypes("number"); return lambda: pr.contraste_mcar_little(d)
@bench("imputar")
def _(n): d = _con_faltantes(n); return lambda: pr.imputar(d, "iterativa")
@bench("imputacion_multiple")
def _(n): d = _con_faltantes(n).drop(columns="zona"); return lambda: pr.imputacion_multiple(d, "valor ~ edad + precio", m=5)
@bench("agrupar_categorias_raras")
def _(n): z = pd.DataFrame({"k": np.random.default_rng(0).zipf(1.5, n) % 500}); return lambda: pr.agrupar_categorias_raras(z, "k", 0.01)
@bench("codificar_por_objetivo")
def _(n):
    rng = np.random.default_rng(0); z = pd.DataFrame({"k": rng.integers(0, 300, n)}); z["y"] = (rng.random(n) < .1).astype(int)
    return lambda: pr.codificar_por_objetivo(z, "k", "y")
@bench("smote")
def _(n): d = datos(n)[["edad", "precio", "valor", "compra"]]; return lambda: pr.smote(d, "compra", 1, ratio=1.0)

# selección y validación
@bench("seleccion_backward")
def _(n): d = datos(n); return lambda: se.seleccion_backward(d, "compra", ["edad", "precio", "zona", "ruido", "grupo"], categoricas=["zona", "grupo"], criterio="bic")
@bench("seleccion_por_pvalor")
def _(n): d = datos(n); return lambda: se.seleccion_por_pvalor(d, "compra", ["edad", "precio", "zona", "ruido"], categoricas=["zona"])
@bench("mejor_subconjunto")
def _(n): d = datos(n); return lambda: se.mejor_subconjunto(d, "valor", ["edad", "precio", "ruido", "meses"])
@bench("filtrar_varianza_casi_nula")
def _(n): d = datos(n); return lambda: se.filtrar_varianza_casi_nula(d)
@bench("filtrar_correlacion_alta")
def _(n): d = datos(n); return lambda: se.filtrar_correlacion_alta(d, 0.9)
@bench("eliminacion_recursiva")
def _(n): d = datos(n); return lambda: se.eliminacion_recursiva(d[["edad", "precio", "ruido", "valor"]], d.compra, cv=3)
@bench("validacion_cruzada")
def _(n): d = datos(n); return lambda: se.validacion_cruzada(d, "compra ~ edad + precio + C(zona)", k=5)
@bench("optimismo_bootstrap")
def _(n): d = datos(n); return lambda: se.optimismo_bootstrap(d, "compra ~ edad + precio", n_boot=20)
@bench("comparar_modelos_cv")
def _(n): d = datos(n); return lambda: se.comparar_modelos_cv(d, "compra ~ edad", "compra ~ edad + precio", k=5, repeticiones=1)
@bench("metricas_regresion")
def _(n): rng = np.random.default_rng(0); y = rng.normal(size=n); return lambda: se.metricas_regresion(y, y + rng.normal(size=n))
@bench("tabla_criterios", fijo=True)
def _(n):
    import statsmodels.formula.api as smf
    d = datos(2000); ms = {"a": smf.ols("valor ~ edad", d).fit(), "b": smf.ols("valor ~ edad + precio", d).fit()}
    return lambda: se.tabla_criterios(ms, ms["b"].scale)
@bench("tabla_ganancia_lift")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return lambda: di.tabla_ganancia_lift(y, p)
@bench("estadistico_ks_gini")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return lambda: di.estadistico_ks_gini(y, p)
@bench("curva_precision_recall")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return lambda: di.curva_precision_recall(y, p)
@bench("descomposicion_brier")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return lambda: di.descomposicion_brier(y, p)
@bench("calibrar_probabilidades")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return lambda: di.calibrar_probabilidades(y, np.clip(p * 1.5, 0, 1))
@bench("grafico_ganancia_lift")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return _cerrar(lambda: gr.grafico_ganancia_lift(y, p))
@bench("grafico_precision_recall")
def _(n): _X, y, _s, _m, p = _logit_listo(datos(n)); return _cerrar(lambda: gr.grafico_precision_recall(y, p))

# modelos añadidos
def _cart(n, semilla=0):
    rng = np.random.default_rng(semilla)
    d = pd.DataFrame({"zona": rng.choice(["A", "B", "C"], n), "edad": rng.uniform(18, 80, n), "expo": rng.uniform(.2, 1, n)})
    lam = .15 * np.exp(.5 * (d.zona == "B") - .4 * (d.zona == "C") - .01 * (d.edad - 40)) * d.expo
    d["nsin"] = rng.poisson(lam * rng.gamma(1.2, 1 / 1.2, n))
    d["coste"] = np.where(d.nsin > 0, rng.gamma(2, 600, n) * d.nsin, 0.0)
    return d
@bench("ajustar_glm_conteo")
def _(n): d = _cart(n); return lambda: mo.ajustar_glm_conteo("nsin ~ C(zona) + edad", d, "poisson", "expo")
@bench("contraste_sobredispersion")
def _(n): m = mo.ajustar_glm_conteo("nsin ~ C(zona) + edad", _cart(n), "poisson", "expo")["modelo"]; return lambda: mo.contraste_sobredispersion(m)
@bench("ajustar_glm_severidad")
def _(n): d = _cart(n * 4).query("coste > 0").head(n); return lambda: mo.ajustar_glm_severidad("coste ~ C(zona) + edad", d)
@bench("ajustar_tweedie")
def _(n): d = _cart(n); return lambda: mo.ajustar_tweedie("coste ~ C(zona) + edad", d, 1.5, "expo")
@bench("prima_pura")
def _(n):
    d = _cart(n); f = mo.ajustar_glm_conteo("nsin ~ C(zona)", d, "poisson", "expo"); s_ = mo.ajustar_glm_severidad("coste ~ C(zona)", d.query("coste > 0"))
    return lambda: mo.prima_pura(f, s_, d, "expo")
@bench("tabla_relatividades")
def _(n): f = mo.ajustar_glm_conteo("nsin ~ C(zona) + edad", _cart(n), "poisson", "expo"); return lambda: mo.tabla_relatividades(f, "zona")
@bench("ajustar_ols")
def _(n): d = datos(n); return lambda: mo.ajustar_ols("valor ~ edad + precio + C(zona)", d)
@bench("medidas_influencia")
def _(n): r = mo.ajustar_ols("valor ~ edad + precio", datos(n)); return lambda: mo.medidas_influencia(r)
@bench("contraste_f_parcial")
def _(n): d = datos(n); a, b = mo.ajustar_ols("valor ~ edad", d), mo.ajustar_ols("valor ~ edad + precio + ruido", d); return lambda: mo.contraste_f_parcial(a, b)
@bench("transformacion_box_cox")
def _(n): x = datos(n).meses; return lambda: mo.transformacion_box_cox(x)
@bench("ajustar_wls")
def _(n): d = datos(n); return lambda: mo.ajustar_wls("valor ~ edad + precio", d, estimar_pesos=True)
@bench("regresion_robusta")
def _(n): d = datos(n); return lambda: mo.regresion_robusta("valor ~ edad + precio", d)
@bench("regresion_no_lineal")
def _(n):
    rng = np.random.default_rng(0); t = rng.uniform(0, 10, n); y = 5 * (1 - np.exp(-.4 * t)) + rng.normal(0, .1, n)
    return lambda: mo.regresion_no_lineal(lambda t, a, b: a * (1 - np.exp(-b * t)), t, y, [1, 1])
@bench("contraste_falta_ajuste")
def _(n): rng = np.random.default_rng(0); d = pd.DataFrame({"x": rng.integers(1, 20, n)}); d["y"] = d.x + rng.normal(size=n); return lambda: mo.contraste_falta_ajuste(d, "x", "y")
@bench("ajustar_glm_splines")
def _(n): d = datos(n); return lambda: mo.ajustar_glm_splines(d, "compra", {"edad": 4}, "precio")
@bench("ajustar_modelo_mixto")
def _(n):
    rng = np.random.default_rng(0); g = pd.DataFrame({"g": rng.integers(0, max(n // 50, 5), n), "x": rng.normal(size=n)})
    g["y"] = g.x + rng.normal(size=n) + g.g.map(dict(enumerate(rng.normal(size=g.g.max() + 1))))
    return lambda: mo.ajustar_modelo_mixto("y ~ x", g, "g")
@bench("coeficiente_icc")
def _(n): d = datos(n); return lambda: mo.coeficiente_icc(d, "valor", "zona")
@bench("ajustar_logit_ordinal")
def _(n):
    d = datos(n).assign(nivel=lambda x: pd.cut(x.valor, [-np.inf, 9, 10.5, 12, np.inf], labels=["a", "b", "c", "d"]))
    return lambda: mo.ajustar_logit_ordinal(d, "nivel", ["edad", "ruido"], orden=["a", "b", "c", "d"])
@bench("ajustar_regularizado")
def _(n):
    rng = np.random.default_rng(0); X = pd.DataFrame(rng.normal(size=(n, 10)), columns=[f"v{i}" for i in range(10)])
    y = (rng.random(n) < 1 / (1 + np.exp(-X.v0))).astype(int)
    return lambda: mo.ajustar_regularizado(X, y, "lasso", cv=3)
@bench("contraste_schoenfeld")
def _(n): c = mo.ajustar_cox(datos(n), "meses", "baja", ["edad"], categoricas=["grupo"], refs={"grupo": "A"}); return lambda: mo.contraste_schoenfeld(c)
@bench("ajustar_supervivencia_parametrica")
def _(n): d = datos(n); return lambda: mo.ajustar_supervivencia_parametrica(d, "meses", "baja", ["edad"], "weibull", ["grupo"])
@bench("nelson_aalen")
def _(n): d = datos(n); return lambda: mo.nelson_aalen(d, "meses", "baja", "grupo")
@bench("incidencia_acumulada")
def _(n):
    d = datos(n); d["causa"] = np.where(d.baja == 1, np.random.default_rng(0).choice([1, 2], n), 0)
    return lambda: mo.incidencia_acumulada(d, "meses", "causa")
@bench("rmst")
def _(n): d = datos(n); return lambda: mo.rmst(d, "meses", "baja", 24, "grupo")
@bench("grafico_efecto_spline", fijo=True)
def _(n): r = mo.ajustar_glm_splines(datos(3000), "compra", {"edad": 4}); return _cerrar(lambda: gr.grafico_efecto_spline(r, "edad"))
@bench("grafico_regularizacion", fijo=True)
def _(n):
    rng = np.random.default_rng(0); X = pd.DataFrame(rng.normal(size=(500, 8))); y = X[0] + rng.normal(size=500)
    r = mo.ajustar_regularizado(X, y, "lasso", "gaussiana"); return _cerrar(lambda: gr.grafico_regularizacion(r))
@bench("grafico_relatividades", fijo=True)
def _(n): t = mo.tabla_relatividades(mo.ajustar_glm_conteo("nsin ~ C(zona)", _cart(5000), "poisson", "expo"), "zona"); return _cerrar(lambda: gr.grafico_relatividades(t))
@bench("grafico_incidencia_acumulada", fijo=True)
def _(n):
    d = datos(2000); d["causa"] = np.where(d.baja == 1, np.random.default_rng(0).choice([1, 2], 2000), 0)
    t = mo.incidencia_acumulada(d, "meses", "causa"); return _cerrar(lambda: gr.grafico_incidencia_acumulada(t))

# contrastes añadidos
@bench("contraste_apareado")
def _(n): rng = np.random.default_rng(0); a = rng.normal(size=n); return lambda: co.contraste_apareado(a, a + .1 + rng.normal(size=n))
@bench("contraste_friedman")
def _(n):
    k = max(n // 3, 2); rng = np.random.default_rng(0)
    d = pd.DataFrame({"s": np.repeat(np.arange(k), 3), "c": np.tile(["a", "b", "c"], k), "v": rng.normal(size=3 * k)})
    return lambda: co.contraste_friedman(d, "s", "c", "v")
@bench("contraste_mcnemar")
def _(n): rng = np.random.default_rng(0); a, b = (rng.random(n) < .3).astype(int), (rng.random(n) < .35).astype(int); return lambda: co.contraste_mcnemar(a, b)
@bench("contraste_permutacion")
def _(n): d = datos(n); return lambda: co.contraste_permutacion(d.valor[d.grupo == "A"], d.valor[d.grupo == "B"], n_perm=200)
@bench("contraste_jonckheere")
def _(n): d = datos(n); return lambda: co.contraste_jonckheere(d, "valor", "grupo")
@bench("posthoc_dunn")
def _(n): d = datos(n); return lambda: co.posthoc_dunn(d, "valor", "grupo")
@bench("games_howell")
def _(n): d = datos(n); return lambda: co.games_howell(d, "valor", "grupo")
@bench("intervalo_proporcion", fijo=True)
def _(n): return lambda: [co.intervalo_proporcion(7, 120, metodo=m) for m in ("wilson", "clopper_pearson", "jeffreys")]
@bench("bootstrap_ic")
def _(n): x = datos(n).valor.to_numpy(); return lambda: co.bootstrap_ic(x, np.mean, n_boot=200, metodo="percentil")
@bench("bondad_ajuste_multinomial", fijo=True)
def _(n): return lambda: co.bondad_ajuste_multinomial([180, 150, 148, 152, 160, 205, 210])

# descriptiva
@bench("resumen_descriptivo")
def _(n): d = datos(n); return lambda: de.resumen_descriptivo(d)
@bench("detectar_atipicos")
def _(n): x = datos(n).precio; return lambda: de.detectar_atipicos(x, "mad")
@bench("correlacion_con_ic")
def _(n): d = datos(n); return lambda: de.correlacion_con_ic(d.edad, d.valor, "spearman")
@bench("matriz_correlaciones")
def _(n): d = datos(n); return lambda: de.matriz_correlaciones(d, ["edad", "precio", "ruido", "valor", "meses"])
@bench("correlacion_parcial")
def _(n): d = datos(n); return lambda: de.correlacion_parcial(d, "edad", "valor", ["precio", "ruido"])
@bench("contraste_normalidad")
def _(n): x = datos(n).valor; return lambda: de.contraste_normalidad(x)
@bench("homogeneidad_varianzas")
def _(n): d = datos(n); return lambda: de.homogeneidad_varianzas(d, "valor", "grupo")
@bench("ajustar_distribuciones")
def _(n): x = datos(n).meses; return lambda: de.ajustar_distribuciones(x, candidatas=("gamma", "lognorm", "weibull_min", "expon"))
@bench("ajustar_distribucion_discreta")
def _(n): x = np.random.default_rng(0).negative_binomial(2, 0.4, n); return lambda: de.ajustar_distribucion_discreta(x)
@bench("indice_dispersion")
def _(n): x = np.random.default_rng(0).poisson(3, n); return lambda: de.indice_dispersion(x)
@bench("funcion_distribucion_empirica")
def _(n): x = datos(n).valor; return lambda: de.funcion_distribucion_empirica(x)
@bench("estimar_densidad")
def _(n): x = datos(n).valor; return lambda: de.estimar_densidad(x)
@bench("tabla_contingencia")
def _(n): d = datos(n); return lambda: de.tabla_contingencia(d, "zona", "grupo")
@bench("medidas_riesgo_2x2", fijo=True)
def _(n): return lambda: de.medidas_riesgo_2x2(30, 100, 15, 100)
@bench("grafico_distribucion")
def _(n): x = datos(n).valor; return _cerrar(lambda: gr.grafico_distribucion(x))
@bench("grafico_ajuste_distribuciones")
def _(n):
    x = datos(n).meses; t = de.ajustar_distribuciones(x, candidatas=("gamma", "lognorm", "expon"))
    return _cerrar(lambda: gr.grafico_ajuste_distribuciones(x, t))
@bench("grafico_matriz_correlaciones", fijo=True)
def _(n): r = de.matriz_correlaciones(datos(2000), ["edad", "precio", "ruido", "valor"])["r"]; return _cerrar(lambda: gr.grafico_matriz_correlaciones(r))


def _cronometrar(llamada, repeticiones=1):
    mejor = math.inf
    for _ in range(repeticiones):
        t0 = time.perf_counter(); llamada(); mejor = min(mejor, time.perf_counter() - t0)
    return mejor


# simulación y diseño (0.10)
from arbol_estadistica import diseno as ds  # noqa: E402
from arbol_estadistica import simulacion as si  # noqa: E402


def _r(n, s=0):
    return np.random.default_rng(s)


@bench("bayes_empirico_beta")
def _(n): r = _r(n); k = max(n // 50, 20); m = r.integers(5, 200, k); return lambda: si.bayes_empirico_beta(r.binomial(m, 0.2), m)
@bench("chequeo_predictivo")
def _(n): r = _r(n); y = r.poisson(3, n); lam = r.gamma(1 + y.sum(), 1 / n, 200); return lambda: si.chequeo_predictivo(y, lam, lambda l, k, g: g.poisson(l, k))
@bench("accion_bayes")
def _(n): x = _r(n).lognormal(0, 1, n); return lambda: si.accion_bayes(x, "absoluta")
@bench("diagnostico_mcmc")
def _(n): x = _r(n).normal(size=(4, n // 4, 2)); return lambda: si.diagnostico_mcmc(x)
@bench("simular_cadena_markov")
def _(n): return lambda: si.simular_cadena_markov([[0.9, 0.1], [0.5, 0.5]], n, 0, 10)
@bench("simular_proceso_poisson")
def _(n): return lambda: si.simular_proceso_poisson(1.0, n, 5)
@bench("contraste_proceso_poisson")
def _(n): t = si.simular_proceso_poisson(1.0, n)[0]; return lambda: si.contraste_proceso_poisson(t, n)
@bench("paseo_aleatorio")
def _(n): return lambda: si.paseo_aleatorio(n // 10, 0.5, 100)
@bench("simular_browniano")
def _(n): return lambda: si.simular_browniano(1, n // 10, 100)
@bench("generador_congruencial")
def _(n): return lambda: si.generador_congruencial(n)
@bench("contrastes_aleatoriedad")
def _(n): u = _r(n).random(n); return lambda: si.contrastes_aleatoriedad(u)
@bench("generar_por_inversion")
def _(n): return lambda: si.generar_por_inversion(n, cdf=lambda v: 1 - np.exp(-v), soporte=(0, 30))
@bench("generar_por_aceptacion_rechazo")
def _(n): return lambda: si.generar_por_aceptacion_rechazo(n, lambda v: stats.beta(2, 5).pdf(v), stats.uniform(0, 1), M=2.5)
@bench("generar_normal_multivariante")
def _(n): S = np.eye(5) * 0.5 + 0.5; return lambda: si.generar_normal_multivariante(np.zeros(5), S, n)
@bench("estimar_montecarlo")
def _(n): return lambda: si.estimar_montecarlo(np.exp, n, metodo="antiteticas")
@bench("comparar_estimadores")
def _(n): return lambda: si.comparar_estimadores({"m": np.mean, "md": np.median}, lambda k, g: g.normal(size=k), 0.0, n=30, n_sim=max(n // 10, 100))
@bench("metodo_momentos")
def _(n): x = stats.gamma(2).rvs(n, random_state=_r(n)); return lambda: si.metodo_momentos(x, "gamma")
@bench("informacion_fisher")
def _(n): x = _r(n).normal(5, 2, n); return lambda: si.informacion_fisher(lambda v, th: stats.norm.logpdf(v, th[0], th[1]), [5, 2], datos=x)
@bench("distribucion_muestral_simulada")
def _(n): return lambda: si.distribucion_muestral_simulada("t", 5, n_sim=n)
@bench("convergencia_media_muestral")
def _(n): return lambda: si.convergencia_media_muestral(stats.expon(), (1, 10, 30), max(n // 10, 200))
@bench("simular_regresion_a_la_media")
def _(n): return lambda: si.simular_regresion_a_la_media(0.5, n)
@bench("coste_de_dicotomizar")
def _(n): return lambda: si.coste_de_dicotomizar(0.3, n=100, n_sim=max(n // 20, 50))
@bench("extraer_muestra")
def _(n): d = pd.DataFrame({"e": _r(n).choice(list("abc"), n), "x": _r(n).normal(size=n)}); return lambda: si.extraer_muestra(d, n // 10, "estratificada", estrato="e")
@bench("estimar_mas")
def _(n): x = _r(n).gamma(2, 100, n); return lambda: si.estimar_mas(x, N=10 * n)
@bench("estimar_estratificado")
def _(n): d = pd.DataFrame({"e": _r(n).choice(list("abc"), n), "y": _r(n).normal(size=n)}); return lambda: si.estimar_estratificado(d, "y", "e", {"a": 10 * n, "b": 10 * n, "c": 10 * n})
@bench("estimador_razon")
def _(n): r = _r(n); x = r.gamma(2, 100, n); y = 1.1 * x + r.normal(0, 10, n); return lambda: si.estimador_razon(y, x, 200, N=10 * n)
@bench("ajuste_no_respuesta")
def _(n): r = _r(n); d = pd.DataFrame({"c": r.choice(list("abcd"), n), "resp": r.random(n) < 0.5}); return lambda: si.ajuste_no_respuesta(d, "resp", "c")
_FIJOS_010 = {
    "posterior_conjugado": lambda: si.posterior_conjugado("beta_binomial", (7, 20), (1, 1)),
    "metropolis": lambda: si.metropolis(lambda th: -0.5 * np.sum(th ** 2), [0.0, 0.0], n_iter=1000, cadenas=2),
    "ab_bayesiano": lambda: si.ab_bayesiano(100, 1000, 120, 1000, n_sim=50_000),
    "simular_bandido": lambda: si.simular_bandido([0.05, 0.06, 0.08], 2000),
    "cadena_markov": lambda: si.cadena_markov(np.full((10, 10), 0.1), inicial=0),
    "bonus_malus": lambda: si.bonus_malus([0.5, 0.7, 1.0, 1.3, 1.6], [[0, 2, 4], [0, 3, 4], [1, 4, 4], [2, 4, 4], [3, 4, 4]], 0.1),
    "cadena_markov_continua": lambda: si.cadena_markov_continua([[-0.2, 0.2], [1.0, -1.0]], t=[1, 5, 10]),
    "nacimiento_muerte": lambda: si.nacimiento_muerte(20, 6, servidores=5),
    "ruina_jugador": lambda: si.ruina_jugador(20, 100, 0.49),
    "teorema_bayes": lambda: si.teorema_bayes(0.01, (0.99, 0.95)),
    "analizar_distribucion_conjunta": lambda: si.analizar_distribucion_conjunta(np.arange(1, 26).reshape(5, 5)),
    "distribucion_estadistico_orden": lambda: si.distribucion_estadistico_orden(stats.expon(), 50, 50),
    "momentos_distribucion": lambda: si.momentos_distribucion(stats.gamma(2)),
    "metodo_delta": lambda: si.metodo_delta(lambda t: t[0] / t[1], [2, 4], [[0.04, 0], [0, 0.16]]),
    "tamano_muestra_encuesta": lambda: si.tamano_muestra_encuesta(0.03, N=20000, tasa_respuesta=0.4),
    "asignacion_estratos": lambda: si.asignacion_estratos([8000, 1800, 200], [300, 2000, 20000], 600),
    "diseno_factorial_2k": lambda: ds.diseno_factorial_2k(7, {"F": "ABCD", "G": "ABDE"}),
    "cuadrado_latino": lambda: ds.cuadrado_latino(8),
    "diseno_central_compuesto": lambda: ds.diseno_central_compuesto(4),
    "aleatorizar_ensayo": lambda: ds.aleatorizar_ensayo(1000, tamano_bloque=4),
    "tamano_muestral_modelo": lambda: mo.tamano_muestral_modelo(20, "binario", prevalencia=0.17, r2=0.25),
    "muestrear_siguiente": lambda: ml.muestrear_siguiente(pd.Series(np.full(1000, 1e-3)), 0.8, top_p=0.9),
    "autoconsistencia_votacion": lambda: ml.autoconsistencia_votacion(0.6),
    "arbol_black_derman_toy": lambda: fi.arbol_black_derman_toy(np.linspace(0.02, 0.04, 30), 0.15),
    "regresion_matriz_indicadora": lambda: mv.regresion_matriz_indicadora(pd.DataFrame({"x": np.r_[np.arange(300.0)]}), np.repeat(list("abc"), 100)),
}
for _nombre, _fn in _FIJOS_010.items():
    bench(_nombre, fijo=True)((lambda fn: (lambda n: fn))(_fn))


def _dg(n, s=0):
    r = _r(n, s)
    d = pd.DataFrame({"a": r.choice(list("xyz"), n), "b": r.choice(list("pq"), n), "x": r.normal(size=n), "w": r.normal(size=n)})
    d["y"] = (d.a == "z") + d.x + 0.5 * d.x * d.w + r.normal(size=n)
    d["t"] = (r.random(n) < 1 / (1 + np.exp(-d.x))).astype(int)
    return d


@bench("anova_factorial")
def _(n): d = _dg(n); return lambda: ds.anova_factorial(d, "y", ["a", "b"])
@bench("anova_bloques")
def _(n): d = _dg(n).assign(bl=lambda q: np.arange(n) % 20); return lambda: ds.anova_bloques(d, "y", "a", "bl")
@bench("ancova")
def _(n): d = _dg(n); return lambda: ds.ancova(d, "y", "a", "x")
@bench("anova_medidas_repetidas")
def _(n): r = _r(n); k = n // 4; d = pd.DataFrame({"s": np.repeat(np.arange(k), 4), "t": np.tile(list("abcd"), k), "g": np.repeat(r.choice(["u", "v"], k), 4), "y": r.normal(size=4 * k)}); return lambda: ds.anova_medidas_repetidas(d, "s", "t", "y", entre="g")
@bench("anova_anidado")
def _(n): r = _r(n); d = pd.DataFrame({"f": np.repeat(["a", "b"], n // 2), "u": np.arange(n) % 20, "y": r.normal(size=n // 2 * 2)}); return lambda: ds.anova_anidado(d, "y", "f", "u")
@bench("diagnostico_anova")
def _(n): d = _dg(n); return lambda: ds.diagnostico_anova(d, "y", "a")
@bench("efectos_factorial_2k")
def _(n):
    D = ds.diseno_factorial_2k(5, aleatorizar=False)["diseno"]; D = pd.concat([D] * max(1, n // 32), ignore_index=True)
    return lambda: ds.efectos_factorial_2k(D.assign(y=D.A + _r(n).normal(size=len(D))), "y", list("ABCDE"), orden_max=2)
@bench("anova_cuadrado_latino")
def _(n): L = ds.cuadrado_latino(max(3, min(int(np.sqrt(n)), 40))); return lambda: ds.anova_cuadrado_latino(L.assign(y=_r(n).normal(size=len(L))), "y")
@bench("superficie_respuesta")
def _(n): r = _r(n); d = pd.DataFrame(r.uniform(-1, 1, (n, 2)), columns=["u", "v"]); d["y"] = -d.u ** 2 - d.v ** 2 + r.normal(0, .1, n); return lambda: ds.superficie_respuesta(d, "y", ["u", "v"])
@bench("puntuacion_propension")
def _(n): d = _dg(n); return lambda: ds.puntuacion_propension(d, "t", "y", ["x", "w"], n_boot=20)
@bench("diferencias_en_diferencias")
def _(n): d = _dg(n).assign(post=lambda q: np.arange(n) % 2, u=lambda q: np.arange(n) // 2); return lambda: ds.diferencias_en_diferencias(d, "y", "t", "post", cluster="u")
@bench("analisis_intencion_tratar")
def _(n): r = _r(n); z = r.integers(0, 2, n); d = pd.DataFrame({"z": z, "t": np.where(r.random(n) < .8, z, 1 - z), "y": r.normal(size=n)}); return lambda: ds.analisis_intencion_tratar(d, "z", "t", "y")
@bench("mediacion")
def _(n): d = _dg(n); return lambda: ds.mediacion(d, "x", "w", "y", n_boot=100)
@bench("moderacion")
def _(n): d = _dg(n); return lambda: ds.moderacion(d, "y", "x", "w")
@bench("metaanalisis")
def _(n): r = _r(n); k = max(n // 100, 10); return lambda: ds.metaanalisis(r.normal(0.3, 0.2, k), r.uniform(0.05, 0.3, k), "reml")
@bench("bandas_confianza_regresion")
def _(n): x, y = _r(n).normal(size=n), _r(n, 1).normal(size=n); return lambda: mo.bandas_confianza_regresion(x, y)
@bench("regresion_inversa")
def _(n): x = _r(n).uniform(0, 10, n); y = 1 + 2 * x + _r(n, 1).normal(size=n); return lambda: mo.regresion_inversa(x, y, 7)
@bench("regresion_por_origen")
def _(n): x = _r(n).uniform(1, 10, n); y = 2 * x + _r(n, 1).normal(size=n); return lambda: mo.regresion_por_origen(x, y)
@bench("correccion_error_medida")
def _(n): x = _r(n).normal(size=n); y = x + _r(n, 1).normal(size=n); return lambda: mo.correccion_error_medida(x, y, var_error=0.3, n_sim=20)
@bench("regresion_polinomica")
def _(n): x = _r(n).uniform(-2, 2, n); y = x ** 2 + _r(n, 1).normal(size=n); return lambda: mo.regresion_polinomica(x, y, 5)
@bench("regresion_local")
def _(n): x = _r(n).uniform(0, 10, n); y = np.sin(x) + _r(n, 1).normal(0, .3, n); return lambda: mo.regresion_local(x, y, "nadaraya_watson")
@bench("funciones_escalonadas")
def _(n): d = _dg(n); return lambda: mo.funciones_escalonadas(d, "y", "x", 6)
@bench("ajustar_mars")
def _(n): d = _dg(n); return lambda: mo.ajustar_mars(d[["x", "w"]], d.y, max_terminos=7, max_nudos=15)
@bench("regresion_inversa_cortes")
def _(n): X = pd.DataFrame(_r(n).normal(size=(n, 6))); y = np.exp(X[0]) + _r(n, 1).normal(size=n); return lambda: mo.regresion_inversa_cortes(X, y)
@bench("regresion_pls_pcr")
def _(n): X = pd.DataFrame(_r(n).normal(size=(n, 20))); y = X[0] + _r(n, 1).normal(size=n); return lambda: mo.regresion_pls_pcr(X, y, max_componentes=5, cv=5)
@bench("modelo_loglineal")
def _(n): d = _dg(n); return lambda: mo.modelo_loglineal(d, ["a", "b", "t"], modelo="dobles")
@bench("tabla_nomograma")
def _(n):
    d = _dg(n); import statsmodels.formula.api as smf
    m = smf.glm("t ~ x + w + a", d, family=sm.families.Binomial()).fit(); return lambda: se.tabla_nomograma(m, d, ["x", "w", "a"])
@bench("aproximar_modelo")
def _(n): d = _dg(n); return lambda: se.aproximar_modelo(d[["x", "w", "a", "b"]], d.y)
@bench("dominio_aplicabilidad")
def _(n): r = _r(n); A = pd.DataFrame(r.normal(size=(n, 4))); B = pd.DataFrame(r.normal(size=(n // 10, 4))); return lambda: di.dominio_aplicabilidad(A, B)
@bench("centrar_por_grupo")
def _(n): d = _dg(n); return lambda: pr.centrar_por_grupo(d, ["x", "w"], "a")
@bench("diccionario_datos")
def _(n): d = datos(n); return lambda: de.diccionario_datos(d)
@bench("regresion_multivariante")
def _(n): d = _dg(n).assign(y2=lambda q: q.y + _r(n).normal(size=n)); return lambda: mv.regresion_multivariante(d, ["y", "y2"], ["x", "w"])
@bench("analisis_perfiles")
def _(n): r = _r(n); d = pd.DataFrame(r.normal(size=(n, 4)), columns=list("pqrs")).assign(g=r.choice(["a", "b"], n)); return lambda: mv.analisis_perfiles(d, list("pqrs"), "g")
@bench("control_t2_multivariante")
def _(n): r = _r(n); A = pd.DataFrame(r.normal(size=(n, 4))); return lambda: mv.control_t2_multivariante(A, A.iloc[: n // 10] * 1.01)
@bench("analisis_conjunto")
def _(n): r = _r(n); d = pd.DataFrame({"p": r.choice(list("abc"), n), "m": r.choice(list("xy"), n), "id": np.arange(n) % 50}); d["v"] = (d.p == "a") + r.normal(size=n); return lambda: mv.analisis_conjunto(d, "v", ["p", "m"])
@bench("modelo_grafico_gaussiano")
def _(n): X = pd.DataFrame(_r(n).normal(size=(n, 6))); return lambda: mv.modelo_grafico_gaussiano(X, alfa=0.05)
@bench("pca_funcional")
def _(n): Y = _r(n).normal(size=(max(n // 50, 20), 50)); return lambda: mv.pca_funcional(Y)
@bench("centroides_contraidos")
def _(n): r = _r(n); X = pd.DataFrame(r.normal(size=(n, 20))); y = r.integers(0, 3, n); return lambda: mv.centroides_contraidos(X, y, umbrales=[0, 0.5, 1], cv=3)
@bench("pls_da")
def _(n): r = _r(n); X = pd.DataFrame(r.normal(size=(n, 20))); y = r.integers(0, 3, n); return lambda: mv.pls_da(X, y, max_componentes=3, cv=3)
@bench("reglas_asociacion")
def _(n): r = _r(n); M = pd.DataFrame(r.random((n, 12)) < 0.2, columns=[f"p{i}" for i in range(12)]); return lambda: ml.reglas_asociacion(M, 0.02, 0.2)
@bench("mapa_autoorganizado")
def _(n): X = pd.DataFrame(_r(n).normal(size=(n, 4))); return lambda: ml.mapa_autoorganizado(X, 5, 5, 2000)
@bench("modelo_ngramas")
def _(n): r = _r(n); txt = [" ".join(r.choice(["a", "b", "c", "d", "e"], 10)) for _ in range(max(n // 10, 10))]; return lambda: ml.modelo_ngramas(txt, 2)
@bench("muestrear_texto")
def _(n): m = ml.modelo_ngramas(["el gato come", "el perro come pienso", "el gato duerme"], 2); return lambda: ml.muestrear_texto(m, "el", 20)
@bench("ajustar_bradley_terry")
def _(n): r = _r(n); g = r.integers(0, 20, n); p = (g + r.integers(1, 20, n)) % 20; return lambda: ml.ajustar_bradley_terry(pd.DataFrame({"ganador": g, "perdedor": p}))
@bench("recuperar_tfidf")
def _(n): r = _r(n); docs = [" ".join(r.choice(list("abcdefghij"), 12)) for _ in range(max(n // 10, 10))]; return lambda: ml.recuperar_tfidf(docs, "a b c", 5)
@bench("proceso_difusion")
def _(n): x = _r(n).normal(size=n); return lambda: ml.proceso_difusion(x)
for _nombre, _fn in {
    "grafico_trazas_mcmc": lambda: gr.grafico_trazas_mcmc(si.metropolis(lambda th: -0.5 * np.sum(th ** 2), [0.0, 0.0], n_iter=500, cadenas=2)),
    "grafico_bandido": lambda: gr.grafico_bandido({"t": si.simular_bandido([.05, .08], 500)}),
    "grafico_trayectorias": lambda: gr.grafico_trayectorias(si.paseo_aleatorio(200, .5, 200)["trayectorias"]),
    "grafico_interaccion": lambda: gr.grafico_interaccion(_dg(1000), "y", "a", "b"),
    "grafico_efectos_2k": lambda: gr.grafico_efectos_2k(ds.efectos_factorial_2k(ds.diseno_factorial_2k(4, aleatorizar=False)["diseno"].assign(y=np.arange(16.0)), "y", list("ABCD"))),
    "grafico_superficie_respuesta": lambda: gr.grafico_superficie_respuesta(ds.superficie_respuesta(ds.diseno_central_compuesto(2).assign(y=lambda q: -q.x1 ** 2 - q.x2 ** 2 + 0.01 * np.arange(len(q))), "y", ["x1", "x2"])),
    "grafico_balance": lambda: gr.grafico_balance(ds.puntuacion_propension(_dg(1000), "t", "y", ["x", "w"], n_boot=5)),
    "grafico_metaanalisis": lambda: gr.grafico_metaanalisis(ds.metaanalisis([0.1, 0.3, 0.5, 0.2], [0.1, 0.2, 0.15, 0.1])),
    "grafico_hexbin": lambda: gr.grafico_hexbin(_dg(5000), "x", "y"),
    "grafico_violin": lambda: gr.grafico_violin(_dg(2000), "y", "a"),
    "grafico_coordenadas_paralelas": lambda: gr.grafico_coordenadas_paralelas(_dg(1000), ["x", "w", "y"], "a"),
    "grafico_curvas_andrews": lambda: gr.grafico_curvas_andrews(_dg(500), ["x", "w", "y"], "a"),
    "grafico_caras_chernoff": lambda: gr.grafico_caras_chernoff(_dg(10), ["x", "w", "y"]),
    "grafico_variable_anadida": lambda: gr.grafico_variable_anadida(__import__("statsmodels.formula.api", fromlist=["ols"]).ols("y ~ x + w", _dg(1000)).fit(), "w"),
    "grafico_bandas_regresion": lambda: gr.grafico_bandas_regresion(mo.bandas_confianza_regresion(np.arange(50.0), np.arange(50.0) + np.sin(np.arange(50.0)))),
    "grafico_control_t2": lambda: gr.grafico_control_t2(mv.control_t2_multivariante(pd.DataFrame(np.random.default_rng(0).normal(size=(100, 3))))),
}.items():
    bench(_nombre, fijo=True)((lambda fn: (lambda n: _cerrar(fn)))(_fn))


def medir_codigo(nombre, rapido=False, limite_s=4.0) -> dict:
    preparar, fijo = BENCH[nombre]
    if fijo:
        llamada = preparar(0)
        llamada()                                        # calentamiento
        t = _cronometrar(llamada, 3)
        tracemalloc.start(); llamada(); pico = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
        return {"velocidad": {"nota": nota_velocidad(t), "detalle": f"{t * 1000:.1f} ms por llamada (no depende de n)"},
                "escalabilidad": {"nota": 10.0, "detalle": "no depende del tamaño de los datos"},
                "memoria": {"nota": nota_memoria(pico), "detalle": f"pico {pico:.1f} MB"}}
    tamanos = [1000, 4000, 16000] if rapido else [1000, 4000, 16000, 64000]
    tiempos, limite = {}, None
    for n in tamanos:
        try:
            llamada = preparar(n)
            if n == tamanos[0]:
                llamada()
            tiempos[n] = _cronometrar(llamada, 2 if n <= 4000 else 1)
        except (ValueError, MemoryError) as e:     # la función rechaza ese tamaño (p. ej. O(n²) en memoria)
            if not tiempos:
                raise
            limite = (n, str(e)[:80])
            break
        if tiempos[n] > limite_s:
            break
    ns = sorted(tiempos)
    # exponente entre los dos tamaños mayores (con n pequeño manda el coste fijo y el exponente sale bajo)
    b = float(math.log(tiempos[ns[-1]] / tiempos[ns[-2]]) / math.log(ns[-1] / ns[-2])) if len(ns) >= 2 else 1.0
    n_max = ns[-1]
    llamada = preparar(n_max)
    tracemalloc.start(); llamada(); pico = tracemalloc.get_traced_memory()[1] / 1e6; tracemalloc.stop()
    t_ref = tiempos.get(N_REF, tiempos[n_max] * (N_REF / n_max) ** b)
    det_t = ", ".join(f"n={k:,}: {v:.3g} s".replace(",", ".") for k, v in tiempos.items())
    nota_esc, det_esc = nota_escalabilidad(b), f"tiempo ∝ n^{b:.2f} ({det_t})"
    if limite:
        nota_esc = min(nota_esc, {4000: 2.0, 16000: 4.0, 64000: 6.0}.get(limite[0], 6.0))
        det_esc += f" · no admite n = {limite[0]:,} ({limite[1]})".replace(",", ".")
    return {"velocidad": {"nota": nota_velocidad(t_ref), "detalle": f"{t_ref:.3g} s con n = {N_REF:,}".replace(",", ".")},
            "escalabilidad": {"nota": nota_esc, "detalle": det_esc},
            "memoria": {"nota": nota_memoria(pico), "detalle": f"pico {pico:.1f} MB con n = {n_max:,}".replace(",", ".")}}


# ============================================================== Monte Carlo (propiedades estadísticas)
def _rechazo(gen, prueba, n_sim, semilla=1):
    rng = np.random.default_rng(semilla)
    return float(np.mean([prueba(*gen(rng)) < 0.05 for _ in range(n_sim)]))


def sim_contrastes(n_sim):
    out = {}
    # tamaño: dos grupos normales iguales (n = 25 por grupo) y lognormales iguales
    def p_elegir(a, b):
        d = pd.DataFrame({"v": np.r_[a, b], "g": ["a"] * len(a) + ["b"] * len(b)})
        return co.elegir_contraste(d, "v", "g")["p_valor"]
    alfa_n = _rechazo(lambda r: (r.normal(0, 1, 25), r.normal(0, 1, 25)), p_elegir, n_sim)
    alfa_l = _rechazo(lambda r: (r.lognormal(0, 1, 25), r.lognormal(0, 1, 25)), p_elegir, n_sim, 2)
    peor = max(abs(alfa_n - .05), abs(alfa_l - .05))
    out.setdefault("elegir_contraste", {})["tamano_alfa"] = {
        "nota": nota_desviacion(peor, n_sim), "detalle": f"α real {alfa_n:.3f} (normal) y {alfa_l:.3f} (lognormal), nominal 0.05, {n_sim} sim."}
    # potencia relativa: frente al mejor test «oráculo» (t si normal, Mann-Whitney si lognormal)
    pot_e = _rechazo(lambda r: (r.normal(0, 1, 25), r.normal(0.7, 1, 25)), p_elegir, n_sim, 3)
    pot_t = _rechazo(lambda r: (r.normal(0, 1, 25), r.normal(0.7, 1, 25)), lambda a, b: stats.ttest_ind(a, b).pvalue, n_sim, 3)
    pot_el = _rechazo(lambda r: (r.lognormal(0, 1, 25), r.lognormal(0.7, 1, 25)), p_elegir, n_sim, 4)
    pot_mw = _rechazo(lambda r: (r.lognormal(0, 1, 25), r.lognormal(0.7, 1, 25)), lambda a, b: stats.mannwhitneyu(a, b).pvalue, n_sim, 4)
    rel = min(pot_e / max(pot_t, 1e-9), pot_el / max(pot_mw, 1e-9))
    out["elegir_contraste"]["potencia"] = {"nota": _clip(10 * rel), "detalle":
        f"normal d=0.7: {pot_e:.2f} vs t-test {pot_t:.2f}; lognormal: {pot_el:.2f} vs Mann-Whitney {pot_mw:.2f} (n=25 por grupo)"}
    # chi² de una variable categórica independiente del objetivo (n = 300)
    def p_chi(z, y):
        return se.contraste_chi2_variable(pd.DataFrame({"z": z, "y": y}), "z", "y")["p_valor"]
    a = _rechazo(lambda r: (r.choice(["a", "b", "c"], 300), (r.random(300) < .3).astype(int)), p_chi, n_sim, 5)
    out["contraste_chi2_variable"] = {"tamano_alfa": {"nota": nota_desviacion(abs(a - .05), n_sim), "detalle": f"α real {a:.3f} (n=300, 3 niveles, {n_sim} sim.)"}}
    # Tukey: error familiar con 3 grupos iguales
    def p_tukey(x):
        d = pd.DataFrame({"v": x, "g": np.repeat(["a", "b", "c"], 20)})
        return float(co.tukey_entre_grupos(d, "v", "g")["p_ajustado"].min())
    a = _rechazo(lambda r: (r.normal(size=60),), p_tukey, n_sim, 6)
    out["tukey_entre_grupos"] = {"tamano_alfa": {"nota": nota_desviacion(abs(a - .05), n_sim), "detalle": f"error familiar real {a:.3f} (3 grupos de 20, {n_sim} sim.)"}}
    # BH: FDR real con 80 nulos y 20 efectos
    rng = np.random.default_rng(7); fdr, pot = [], []
    for _ in range(max(n_sim // 2, 100)):
        p = np.r_[rng.uniform(size=80), stats.norm.sf(rng.normal(3, 1, 20))]
        rech = co.ajustar_p_valores(p, "fdr_bh").sort_index()["rechaza"].to_numpy()
        fdr.append(rech[:80].sum() / max(rech.sum(), 1)); pot.append(rech[80:].mean())
    f = float(np.mean(fdr))
    out["ajustar_p_valores"] = {"tamano_alfa": {"nota": nota_desviacion(max(f - .05 * .8, 0), n_sim),
                                                "detalle": f"FDR real {f:.3f} (cota BH: 0.05·80/100 = 0.04); potencia media {np.mean(pot):.2f}"}}
    # Hosmer-Lemeshow con el modelo correcto (n = 500)
    def p_hl(rng):
        x = rng.normal(size=500); y = (rng.random(500) < 1 / (1 + np.exp(-(-.5 + x)))).astype(int)
        m = sm.Logit(y, sm.add_constant(x)).fit(disp=False)
        return (di.hosmer_lemeshow(y, m.predict())["p_valor"],)
    a = _rechazo(lambda r: (r,), lambda r: p_hl(r)[0], n_sim, 8)
    out["hosmer_lemeshow"] = {"tamano_alfa": {"nota": nota_desviacion(abs(a - .05), n_sim), "detalle": f"α real {a:.3f} con el modelo correcto (n=500, {n_sim} sim.)"}}
    return out


def sim_logistica(n_sim):
    out = {}
    rng = np.random.default_rng(11)
    b_true, cubre_or, rech0, b_ml, b_f, cubre_f = 1.0, [], [], [], [], []
    for _ in range(n_sim):
        n = 80
        x, z = rng.normal(size=n), rng.normal(size=n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(-1 + b_true * x)))).astype(int)
        X = pd.DataFrame({"x": x, "z": z})
        try:
            m = mo.ajustar_logit(pd.Series(y), X)
            t = mo.tabla_odds_ratios(m)
            cubre_or.append(t.loc["x", "IC_2.5%"] <= math.e <= t.loc["x", "IC_97.5%"])
            rech0.append(mo.tabla_parametros_wald(m).loc["z", "Pr>chi2"] < .05)
            b_ml.append(float(m.params["x"]))
        except Exception:
            pass
        r = mo.regresion_logistica_firth(sm.add_constant(X), y)
        bf, sf = float(np.asarray(r["beta"])[1]), float(np.asarray(r["se"])[1])
        b_f.append(bf); cubre_f.append(abs(bf - b_true) <= 1.96 * sf)
    c, a, c_f = float(np.mean(cubre_or)), float(np.mean(rech0)), float(np.mean(cubre_f))
    sesgo_ml, sesgo_f = float(np.mean(b_ml) - b_true), float(np.mean(b_f) - b_true)
    ecm_ml, ecm_f = float(np.mean((np.array(b_ml) - b_true) ** 2)), float(np.mean((np.array(b_f) - b_true) ** 2))
    det = f"n=80, β=1, {n_sim} sim."
    out["ajustar_logit"] = {"insesgadez": {"nota": nota_sesgo(sesgo_ml / b_true), "detalle": f"sesgo relativo {sesgo_ml / b_true:+.1%} ({det})"},
                            "ecm": {"nota": _clip(10 * min(ecm_f, ecm_ml) / ecm_ml), "detalle": f"ECM {ecm_ml:.3f} vs Firth {ecm_f:.3f} ({det})"}}
    out["tabla_odds_ratios"] = {"cobertura_ic": {"nota": nota_desviacion(abs(c - .95), n_sim), "detalle": f"cobertura real del IC 95 % del OR: {c:.3f} ({det})"}}
    out["tabla_parametros_wald"] = {"tamano_alfa": {"nota": nota_desviacion(abs(a - .05), n_sim), "detalle": f"α real del Wald para un coeficiente nulo: {a:.3f} ({det})"}}
    out["regresion_logistica_firth"] = {"insesgadez": {"nota": nota_sesgo(sesgo_f / b_true), "detalle": f"sesgo relativo {sesgo_f / b_true:+.1%} (logística normal: {sesgo_ml / b_true:+.1%}; {det})"},
                                        "ecm": {"nota": _clip(10 * min(ecm_f, ecm_ml) / ecm_f), "detalle": f"ECM {ecm_f:.3f} vs logística {ecm_ml:.3f} ({det})"},
                                        "cobertura_ic": {"nota": nota_desviacion(abs(c_f - .95), n_sim), "detalle": f"cobertura del IC de Wald: {c_f:.3f} ({det})"}}
    return out


def sim_supervivencia(n_sim):
    out = {}
    rng = np.random.default_rng(21)
    cubre_km, rech_lr, cubre_hr, sesgo_hr = [], [], [], []
    t0, s_true, hr = 10.0, math.exp(-0.05 * 10), 0.6
    for _ in range(n_sim):
        n = 200
        g = rng.choice(["A", "B"], n)
        tb = rng.exponential(1 / 0.05, n); tf = rng.uniform(0, 40, n)
        d = pd.DataFrame({"t": np.minimum(tb, tf) + 1e-6, "e": (tb <= tf).astype(int), "g": g})
        km = mo.kaplan_meier(d, "t", "e")["tabla"]
        fila = km[km.tiempo <= t0].iloc[-1]
        cubre_km.append(fila.ic_inf <= s_true <= fila.ic_sup)
        rech_lr.append(mo.contraste_log_rank(d, "t", "e", "g")["p_valor"] < .05)
        tb2 = rng.exponential(1 / (0.05 * np.where(g == "B", hr, 1)))
        d2 = d.assign(t=np.minimum(tb2, tf) + 1e-6, e=(tb2 <= tf).astype(int))
        tab = mo.ajustar_cox(d2, "t", "e", [], categoricas=["g"], refs={"g": "A"})["tabla"].iloc[0]
        cubre_hr.append(tab.IC_inf <= hr <= tab.IC_sup); sesgo_hr.append(tab.coef - math.log(hr))
    c, a, ch, sh = map(float, (np.mean(cubre_km), np.mean(rech_lr), np.mean(cubre_hr), np.mean(sesgo_hr)))
    det = f"n=200, censura uniforme, {n_sim} sim."
    out["kaplan_meier"] = {"cobertura_ic": {"nota": nota_desviacion(abs(c - .95), n_sim), "detalle": f"cobertura del IC de S(10): {c:.3f} ({det})"}}
    out["contraste_log_rank"] = {"tamano_alfa": {"nota": nota_desviacion(abs(a - .05), n_sim), "detalle": f"α real {a:.3f} ({det})"}}
    out["ajustar_cox"] = {"cobertura_ic": {"nota": nota_desviacion(abs(ch - .95), n_sim), "detalle": f"cobertura del IC del HR: {ch:.3f} ({det})"},
                          "insesgadez": {"nota": nota_sesgo(sh / math.log(hr)), "detalle": f"sesgo de log(HR): {sh:+.3f} ({det})"}}
    return out


def sim_contrastes_extra(n_sim):
    """α real (permutación, Jonckheere, McNemar, t apareado con diferencias exponenciales) y cobertura
    de Wilson frente a Wald (p = 0.05, n = 40) y del bootstrap BCa de la media lognormal (n = 40)."""
    rng = np.random.default_rng(41)
    r = {k: [] for k in ("perm", "jt", "mcn", "apa", "wil", "wald", "cp", "boot")}
    for i in range(n_sim):
        r["perm"].append(co.contraste_permutacion(rng.normal(size=15), rng.normal(size=15), n_perm=199, semilla=i)["p_valor"] < .05)
        d = pd.DataFrame({"g": np.repeat(["a", "b", "c"], 15), "v": rng.normal(size=45)})
        r["jt"].append(co.contraste_jonckheere(d, "v", "g")["p_valor"] < .05)
        base, disc, sentido = (rng.random(80) < .3).astype(int), rng.random(80) < .2, (rng.random(80) < .5).astype(int)
        a, b = np.where(disc, sentido, base), np.where(disc, 1 - sentido, base)      # H0: tantos 1→0 como 0→1
        r["mcn"].append(co.contraste_mcnemar(a, b)["p_valor"] < .05)
        x = rng.normal(size=25); r["apa"].append(co.contraste_apareado(x, x + rng.exponential(1, 25) - 1)["p_valor"] < .05)
        k = rng.binomial(40, .05)
        for m, key in (("wilson", "wil"), ("wald", "wald"), ("clopper_pearson", "cp")):
            ic = co.intervalo_proporcion(k, 40, metodo=m); r[key].append(ic["ic_inf"] <= .05 <= ic["ic_sup"])
        if i < n_sim // 4:
            bi = co.bootstrap_ic(rng.lognormal(0, 1, 40), np.mean, n_boot=500, metodo="bca", semilla=i)
            r["boot"].append(bi["ic_inf"] <= np.exp(.5) <= bi["ic_sup"])
    m = {k: float(np.mean(v)) for k, v in r.items()}
    nb = max(n_sim // 4, 1)
    return {"contraste_permutacion": {"tamano_alfa": {"nota": nota_desviacion(abs(m["perm"] - .05), n_sim), "detalle": f"α real {m['perm']:.3f} (n=15+15, {n_sim} sim.)"}},
            "contraste_jonckheere": {"tamano_alfa": {"nota": nota_desviacion(abs(m["jt"] - .05), n_sim), "detalle": f"α real {m['jt']:.3f} (3×15, {n_sim} sim.)"}},
            "contraste_mcnemar": {"tamano_alfa": {"nota": nota_desviacion(abs(m["mcn"] - .05), n_sim), "detalle": f"α real {m['mcn']:.3f} (n=80, {n_sim} sim.)"}},
            "contraste_apareado": {"tamano_alfa": {"nota": nota_desviacion(abs(m["apa"] - .05), n_sim), "detalle": f"α real {m['apa']:.3f} para H0 «media de las diferencias = 0» con diferencias asimétricas (exponencial centrada, n=25): el modo auto pasa a Wilcoxon, que contrasta la pseudomediana ({n_sim} sim.)"}},
            "intervalo_proporcion": {"cobertura_ic": {"nota": nota_desviacion(abs(m["wil"] - .95), n_sim),
                                     "detalle": f"cobertura con p=0.05, n=40: Wilson {m['wil']:.3f} · Clopper-Pearson {m['cp']:.3f} · Wald {m['wald']:.3f} ({n_sim} sim.)"}},
            "bootstrap_ic": {"cobertura_ic": {"nota": nota_desviacion(abs(m["boot"] - .95), nb), "detalle": f"cobertura BCa de la media lognormal (n=40): {m['boot']:.3f} ({nb} sim.)"}}}


def sim_modelos_extra(n_sim):
    """α real: GLM Poisson vs binomial negativa con sobredispersión; MCO clásico vs HC3 con heterocedasticidad;
    Schoenfeld con riesgos proporcionales. Cobertura: HR de Weibull AFT y RMST."""
    rng = np.random.default_rng(51)
    r = {k: [] for k in ("poi", "nb", "ols", "hc3", "sch", "wei", "rmst")}
    for i in range(n_sim):
        n = 400
        d = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
        d["y"] = rng.poisson(np.exp(.2 + .3 * d.x) * rng.gamma(.7, 1 / .7, n))
        if i < n_sim // 2:
            r["poi"].append(mo.ajustar_glm_conteo("y ~ x + z", d, "poisson")["tabla"].loc["z", "p_valor"] < .05)
            r["nb"].append(mo.ajustar_glm_conteo("y ~ x + z", d, "negbin")["tabla"].loc["z", "p_valor"] < .05)
        h = pd.DataFrame({"x": rng.uniform(0, 10, 100), "z": rng.normal(size=100)})
        h["y"] = 1 + h.x + rng.normal(0, .2, 100) * (h.x ** 1.5) * (1 + np.abs(h.z) * 2)
        r["ols"].append(mo.ajustar_ols("y ~ x + z", h, robusto=None)["tabla"].loc["z", "p_valor"] < .05)
        r["hc3"].append(mo.ajustar_ols("y ~ x + z", h)["tabla"].loc["z", "p_valor"] < .05)
        if i < n_sim // 4:
            g = rng.choice(["A", "B"], 300); lam = .05 * np.exp(.5 * (g == "B"))
            T = (-np.log(rng.random(300)) / lam) ** (1 / 1.5); C = rng.uniform(0, 15, 300)
            s = pd.DataFrame({"g": g, "t": np.minimum(T, C), "e": (T <= C).astype(int)})
            r["sch"].append(mo.contraste_schoenfeld(mo.ajustar_cox(s, "t", "e", [], ["g"], {"g": "A"})).loc["GLOBAL", "p_valor"] < .05)
            w = mo.ajustar_supervivencia_parametrica(s, "t", "e", [], "weibull", ["g"])
            b, se_ = w["tabla"].loc["g_B", "coef"], w["tabla"].loc["g_B", "SE"]
            r["wei"].append(abs(-b / w["sigma"] - .5) <= 1.96 * se_ / w["sigma"])
            sA = s[s.g == "A"]
            verdad = np.sum(np.exp(-.05 * np.linspace(0, 8, 8001) ** 1.5)) * (8 / 8000)
            rr = mo.rmst(sA, "t", "e", 8)
            r["rmst"].append(abs(rr["rmst"].iloc[0] - verdad) <= 1.96 * rr["se"].iloc[0])
    m = {k: float(np.mean(v)) for k, v in r.items()}
    n2, n4 = max(n_sim // 2, 1), max(n_sim // 4, 1)
    return {"ajustar_glm_conteo": {"tamano_alfa": {"nota": nota_desviacion(abs(m["nb"] - .05), n2),
                                   "detalle": f"α real con sobredispersión (coeficiente nulo, n=400): binomial negativa {m['nb']:.3f} · Poisson {m['poi']:.3f} ({n2} sim.)"}},
            "ajustar_ols": {"tamano_alfa": {"nota": nota_desviacion(abs(m["hc3"] - .05), n_sim),
                            "detalle": f"α real con heterocedasticidad fuerte (n=100): HC3 {m['hc3']:.3f} · SE clásicos {m['ols']:.3f} ({n_sim} sim.)"}},
            "contraste_schoenfeld": {"tamano_alfa": {"nota": nota_desviacion(abs(m["sch"] - .05), n4), "detalle": f"α real con riesgos proporcionales (Weibull, n=300): {m['sch']:.3f} ({n4} sim.)"}},
            "ajustar_supervivencia_parametrica": {"cobertura_ic": {"nota": nota_desviacion(abs(m["wei"] - .95), n4), "detalle": f"cobertura del IC del log-HR (Weibull, n=300): {m['wei']:.3f} ({n4} sim.)"}},
            "rmst": {"cobertura_ic": {"nota": nota_desviacion(abs(m["rmst"] - .95), n4), "detalle": f"cobertura del IC del RMST(8) (n≈150): {m['rmst']:.3f} ({n4} sim.)"}}}


def sim_validacion(n_sim):
    """α real de comparar_modelos_cv (variable de ruido añadida) con el t de Nadeau-Bengio frente al t pareado sin
    corregir, y tasa de inclusión de una variable de ruido en seleccion_backward con AIC y con BIC."""
    rng = np.random.default_rng(61)
    corr, sin, aic, bic = [], [], [], []
    k = max(n_sim // 5, 20)
    for i in range(k):
        d = pd.DataFrame({"a": rng.normal(size=300), "r": rng.normal(size=300)})
        d["y"] = (rng.random(300) < 1 / (1 + np.exp(-d.a))).astype(int)
        c = se.comparar_modelos_cv(d, "y ~ a", "y ~ a + r", k=5, repeticiones=2, semilla=i)
        corr.append(c["p_valor"] < .05); sin.append(c["p_valor_sin_corregir"] < .05)
        aic.append("r" in se.seleccion_backward(d, "y", ["a", "r"], criterio="aic")["seleccionadas"])
        bic.append("r" in se.seleccion_backward(d, "y", ["a", "r"], criterio="bic")["seleccionadas"])
    m = {k_: float(np.mean(v)) for k_, v in (("c", corr), ("s", sin), ("a", aic), ("b", bic))}
    return {"comparar_modelos_cv": {"tamano_alfa": {"nota": nota_desviacion(abs(m["c"] - .05), k),
                                    "detalle": f"α real con una variable de ruido: corregido {m['c']:.3f} · t pareado sin corregir {m['s']:.3f} ({k} sim.)"}},
            "seleccion_backward": {"consistencia": {"nota": _clip(10 - 50 * m["b"]),
                                   "detalle": f"una variable de ruido sobrevive: AIC {m['a']:.0%} · BIC {m['b']:.0%} (n=300, {k} sim.; la nota usa BIC)"}}}


def sim_ml(n_sim):
    """Mismo escenario no lineal (interacción + onda, n = 3000) con varias semillas: discriminación (AUC test relativo al
    mejor), calibración (fiabilidad de Brier) y generalización (diferencia de AUC train − test) de cada modelo."""
    from arbol_estadistica.diagnostico.negocio import descomposicion_brier
    modelos = ["ajustar_arbol_decision", "ajustar_random_forest", "ajustar_gradient_boosting", "ajustar_adaboost", "ajustar_knn",
               "ajustar_svm", "ajustar_naive_bayes", "ajustar_red_neuronal", "ajustar_stacking"]
    rep = max(n_sim // 200, 2)
    res = {m: {"auc": [], "fia": [], "gap": []} for m in modelos}
    for s_ in range(rep):
        X, y = _xy_ml(2500, 100 + s_)
        for m in modelos:
            r = getattr(ml, m)(X if m != "ajustar_svm" else X.head(1500), y if m != "ajustar_svm" else y[:1500], semilla=s_)
            p = r["modelo"].predict_proba(r["X_test"])[:, 1]
            res[m]["auc"].append(r["metricas_test"]["auc"]); res[m]["gap"].append(r["sobreajuste"])
            res[m]["fia"].append(descomposicion_brier(r["y_test"], p)["fiabilidad"])
    mejor = max(np.mean(v["auc"]) for v in res.values())
    out = {}
    for m, v in res.items():
        auc, fia, gap = np.mean(v["auc"]), np.mean(v["fia"]), np.mean(v["gap"])
        out[m] = {"discriminacion": {"nota": _clip(10 * (auc - .5) / (mejor - .5)), "detalle": f"AUC test {auc:.3f} (mejor del grupo {mejor:.3f}; escenario no lineal, {rep} semillas)"},
                  "calibracion": {"nota": _clip(10 - 1000 * fia), "detalle": f"fiabilidad de Brier {fia:.4f} (0 = perfecta)"},
                  "generalizacion": {"nota": _clip(10 - 40 * max(gap, 0)), "detalle": f"AUC train − test = {gap:.3f}"}}
    return out


def sim_finanzas(n_sim):
    """Cobertura real del VaR al 99 % (proporción de pérdidas futuras que lo superan; ideal 1 %) estimado con 1000
    observaciones de una t de Student con 4 gl: histórico, normal, Cornish-Fisher y GPD."""
    rng = np.random.default_rng(71)
    exc = {"historico": [], "normal": [], "cornish_fisher": [], "gpd": []}
    k = max(n_sim // 5, 50)
    real = stats.t.ppf(0.99, 4)
    for _ in range(k):
        x = stats.t.rvs(4, size=1000, random_state=rng)
        for m in ("historico", "normal", "cornish_fisher"):
            exc[m].append(stats.t.sf(fi.var_tvar(x, (0.99,), m).iloc[0]["var"], 4))
        exc["gpd"].append(stats.t.sf(fi.ajustar_gpd(x, cuantil_umbral=0.9, niveles=(0.99,))["tabla"].iloc[0]["var"], 4))
    m = {k_: float(np.mean(v)) for k_, v in exc.items()}
    nota = lambda p: _clip(10 - 1000 * abs(p - .01))         # noqa: E731  (1 pp de desviación = −10)
    det = f"prob. real de superar el VaR 99 % (ideal 0.010): histórico {m['historico']:.4f} · normal {m['normal']:.4f} · Cornish-Fisher {m['cornish_fisher']:.4f} · GPD {m['gpd']:.4f} (t4, n=1000, {k} sim.)"
    return {"var_tvar": {"cobertura_ic": {"nota": nota(m["historico"]), "detalle": det}},
            "ajustar_gpd": {"cobertura_ic": {"nota": nota(m["gpd"]), "detalle": det}}}


def sim_descriptiva(n_sim):
    """α real de Shapiro (n=50, normal), cobertura del IC de Pearson (ρ=0.5, n=40) y α real de Brown-Forsythe con datos asimétricos."""
    rng = np.random.default_rng(31)
    rech_sw, cubre, rech_bf, rech_bart = [], [], [], []
    S = np.array([[1, .5], [.5, 1]])
    for _ in range(n_sim):
        t = de.contraste_normalidad(rng.normal(size=50))["tabla"]
        rech_sw.append(t.loc["Shapiro-Wilk", "p_valor"] < .05)
        xy = rng.multivariate_normal([0, 0], S, 40)
        r = de.correlacion_con_ic(xy[:, 0], xy[:, 1]); cubre.append(r["ic_inf"] <= .5 <= r["ic_sup"])
        d = pd.DataFrame({"g": np.repeat(["a", "b", "c"], 30), "v": rng.lognormal(0, 1, 90)})
        h = de.homogeneidad_varianzas(d, "v", "g")
        rech_bf.append(h.loc["Brown-Forsythe (mediana)", "p_valor"] < .05); rech_bart.append(h.loc["Bartlett", "p_valor"] < .05)
    a, c, b, bt = map(float, (np.mean(rech_sw), np.mean(cubre), np.mean(rech_bf), np.mean(rech_bart)))
    return {"contraste_normalidad": {"tamano_alfa": {"nota": nota_desviacion(abs(a - .05), n_sim), "detalle": f"α real de Shapiro-Wilk {a:.3f} (n=50, normal, {n_sim} sim.)"}},
            "correlacion_con_ic": {"cobertura_ic": {"nota": nota_desviacion(abs(c - .95), n_sim), "detalle": f"cobertura del IC de Pearson {c:.3f} (ρ=0.5, n=40, {n_sim} sim.)"}},
            "homogeneidad_varianzas": {"tamano_alfa": {"nota": nota_desviacion(abs(b - .05), n_sim), "detalle": f"α real con lognormales iguales: Brown-Forsythe {b:.3f}, Bartlett {bt:.3f} ({n_sim} sim.)"}}}


def sim_diseno(n_sim):
    """Cobertura real del IC del metaanálisis (efectos fijos, DL y REML con heterogeneidad τ = 0.2, 8 estudios), α real
    del ANOVA de medidas repetidas sin corregir y con Greenhouse-Geisser (sin esfericidad), cobertura simultánea de la
    banda de Working-Hotelling frente a la puntual, cobertura del IC de la MAS con corrección por población finita y
    cobertura frecuentista del IC creíble beta con previa plana (n = 30, p = 0.1)."""
    rng = np.random.default_rng(81)
    k = max(n_sim // 2, 200)
    cub = {"fijo": [], "dl": [], "reml": [], "dl_hk": []}
    a_sin, a_gg, wh, pu, mas, cre = [], [], [], [], [], []
    pobl = rng.gamma(2, 100, 2000)
    xs = np.linspace(0, 10, 40)
    for _ in range(k):
        se_ = rng.uniform(0.1, 0.3, 8); y = rng.normal(0.3, 0.2, 8) + rng.normal(0, se_)
        for m in cub:
            lo, hi = ds.metaanalisis(y, se_, m.replace("_hk", ""), hartung_knapp=m.endswith("_hk"))["ic"]; cub[m].append(lo <= 0.3 <= hi)
        Y = rng.multivariate_normal(np.zeros(4), np.diag([1, 2, 4, 8]) + 0.5, 15)
        d = pd.DataFrame({"s": np.repeat(np.arange(15), 4), "t": np.tile(list("abcd"), 15), "y": Y.ravel()})
        t = ds.anova_medidas_repetidas(d, "s", "t", "y")["tabla"].loc["t"]
        a_sin.append(t.p_valor < .05); a_gg.append(t.p_greenhouse_geisser < .05)
        x = rng.uniform(0, 10, 30); yy = 1 + 0.5 * x + rng.normal(0, 1, 30); verdad = 1 + 0.5 * xs
        for lista, met in ((wh, "working_hotelling"), (pu, "puntual")):
            b = mo.bandas_confianza_regresion(x, yy, xs, metodo=met); lista.append(bool(((b.banda_inf <= verdad) & (verdad <= b.banda_sup)).all()))
        m = si.estimar_mas(rng.choice(pobl, 500, replace=False), N=2000).loc["media"]; mas.append(m.ic_inf <= pobl.mean() <= m.ic_sup)
        c = si.posterior_conjugado("beta_binomial", (int(rng.binomial(30, .1)), 30), (1, 1))["ic"]; cre.append(c[0] <= .1 <= c[1])
    r = {m_: float(np.mean(v)) for m_, v in cub.items()}
    det_m = f"cobertura IC 95 % con τ = 0.2: fijos {r['fijo']:.3f} · DL {r['dl']:.3f} · REML {r['reml']:.3f} · DL + Hartung-Knapp {r['dl_hk']:.3f} ({k} sim.)"
    asin, agg = float(np.mean(a_sin)), float(np.mean(a_gg))
    cwh, cpu = float(np.mean(wh)), float(np.mean(pu))
    return {"metaanalisis": {"cobertura_ic": {"nota": nota_desviacion(abs(r["dl_hk"] - .95), k), "detalle": det_m}},
            "anova_medidas_repetidas": {"tamano_alfa": {"nota": nota_desviacion(abs(agg - .05), k), "detalle": f"α real sin esfericidad: sin corregir {asin:.3f} · Greenhouse-Geisser {agg:.3f} ({k} sim.)"}},
            "bandas_confianza_regresion": {"cobertura_ic": {"nota": nota_desviacion(abs(cwh - .95), k), "detalle": f"cobertura SIMULTÁNEA de la recta (40 puntos): Working-Hotelling {cwh:.3f} · puntual {cpu:.3f} ({k} sim.)"}},
            "estimar_mas": {"cobertura_ic": {"nota": nota_desviacion(abs(float(np.mean(mas)) - .95), k), "detalle": f"cobertura del IC con n/N = 25 % y corrección finita {np.mean(mas):.3f} ({k} sim.)"}},
            "posterior_conjugado": {"cobertura_ic": {"nota": nota_desviacion(abs(float(np.mean(cre)) - .95), k), "detalle": f"cobertura frecuentista del IC creíble Beta(1,1), n = 30, p = 0.1: {np.mean(cre):.3f} ({k} sim.)"}}}


# Simulaciones adicionales (las registran los módulos nuevos): lista de funciones (n_sim) -> {función: {prop: medida}}
SIMULACIONES = [sim_contrastes, sim_logistica, sim_supervivencia, sim_descriptiva, sim_contrastes_extra, sim_modelos_extra, sim_validacion, sim_ml, sim_finanzas, sim_diseno]


def contar_tests(nombre: str) -> int:
    import re
    texto = " ".join(p.read_text(encoding="utf-8") for p in (CODIGO / "tests").glob("test_*.py"))
    return len(re.findall(r"\b" + re.escape(nombre) + r"\b", texto))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rapido = "--rapido" in argv
    solo = [a for a in argv if not a.startswith("--")]
    previo = json.loads(SALIDA.read_text(encoding="utf-8")) if SALIDA.exists() else {"funciones": {}}
    res = previo.get("funciones", {})
    for nombre in sorted(BENCH):
        if solo and nombre not in solo:
            continue
        t0 = time.perf_counter()
        try:
            res.setdefault(nombre, {}).update(medir_codigo(nombre, rapido))
            print(f"  {nombre:40s} {time.perf_counter() - t0:6.1f} s")
        except Exception as e:                       # una función rota no para la medición del resto
            print(f"  {nombre:40s} FALLO: {type(e).__name__}: {e}")
    if not solo or any(s in ("--sim",) for s in argv):
        n_sim = 300 if rapido else 1000
        for sim in SIMULACIONES:
            t0 = time.perf_counter()
            for f, props in sim(n_sim).items():
                res.setdefault(f, {}).update(props)
            print(f"  [simulación] {sim.__name__:28s} {time.perf_counter() - t0:6.1f} s")
    salida = {"generado": dt.date.today().isoformat(), "equipo": f"{platform.system()} · Python {platform.python_version()}",
              "reglas": NOTAS, "funciones": dict(sorted(res.items()))}
    SALIDA.write_text(json.dumps(salida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{SALIDA.relative_to(CODIGO.parent)}: {len(res)} funciones medidas")


if __name__ == "__main__":
    main()
