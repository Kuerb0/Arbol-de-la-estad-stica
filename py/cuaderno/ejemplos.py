"""Ejemplos ejecutables por celdas para CADA función del árbol, con datos simulados.

El visor (abierto desde la app) los muestra en «Probar» y deja ejecutarlos celda a celda, editar el
código y ver tablas y gráficos. Un test ejecuta todos para garantizar que funcionan.

ESCENARIOS: código que crea datos sintéticos (primera celda).  EJEMPLOS: función -> (escenario, celdas).
Al añadir una función al árbol, añade aquí su ejemplo (hay un test que lo exige).
"""

IMPORTS = """import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from arbol_estadistica.preprocesado import *
from arbol_estadistica.seleccion import *
from arbol_estadistica.modelos import *
from arbol_estadistica.diagnostico import *
from arbol_estadistica.clustering import *
from arbol_estadistica.contrastes import *
from arbol_estadistica.graficos import *
from arbol_estadistica.descriptiva import *
from arbol_estadistica.multivariante import *
from arbol_estadistica.ml import *
from arbol_estadistica.actuarial import *
from arbol_estadistica.finanzas import *
from arbol_estadistica.simulacion import *
from arbol_estadistica.diseno import *
rng = np.random.default_rng(42)"""

ESCENARIOS = {
    "ninguno": IMPORTS,
    "binario": IMPORTS + """
# Clientes simulados: la compra depende de edad (+), precio (-) y zona; 'ruido' no influye
n = 2000
df = pd.DataFrame({"edad": rng.normal(45, 12, n).round(), "precio": rng.normal(20000, 5000, n).round(-1),
                   "zona": rng.choice(["norte", "sur", "este"], n), "ruido": rng.normal(size=n)})
eta = -1 + 0.04 * (df.edad - 45) - 0.00012 * (df.precio - 20000) + df.zona.map({"norte": 0, "sur": 0.6, "este": -0.3})
df["compra"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
df.head()""",
    "logit": IMPORTS + """
n = 2000
df = pd.DataFrame({"edad": rng.normal(45, 12, n).round(), "precio": rng.normal(20000, 5000, n).round(-1),
                   "zona": rng.choice(["norte", "sur", "este"], n), "ruido": rng.normal(size=n)})
eta = -1 + 0.04 * (df.edad - 45) - 0.00012 * (df.precio - 20000) + df.zona.map({"norte": 0, "sur": 0.6, "este": -0.3})
df["compra"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
# Modelo ya ajustado (estilo PROC LOGISTIC) para las funciones de diagnóstico
X, y, std_map = preparar_matriz_modelo(df, "compra", ["edad", "precio", "zona"], ["zona"], ["edad", "precio"], refs={"zona": "norte"})
modelo = ajustar_logit(y, X)
p = modelo.predict(sm.add_constant(X))
print(f"n = {len(y)}, tasa de compra = {y.mean():.1%}")""",
    "grupos": IMPORTS + """
# Tres grupos con medias distintas (numérica) y una respuesta sí/no que también cambia por grupo
d = pd.DataFrame({"grupo": np.repeat(["A", "B", "C"], 60)})
d["valor"] = rng.normal(d.grupo.map({"A": 10, "B": 11, "C": 12.5}), 2.0)
d["exito"] = (rng.random(len(d)) < d.grupo.map({"A": 0.3, "B": 0.4, "C": 0.6})).astype(int)
d.groupby("grupo")[["valor", "exito"]].mean()""",
    "supervivencia": IMPORTS + """
# Tiempo hasta la baja (meses) con censura: la tarifa B dura más; la edad aumenta el riesgo
k = 500
surv = pd.DataFrame({"tarifa": rng.choice(["A", "B"], k), "edad": rng.normal(45, 10, k)})
riesgo = 0.08 * np.exp(-0.5 * (surv.tarifa == "B") + 0.03 * (surv.edad - 45))
t_baja, t_fin = rng.exponential(1 / riesgo), rng.uniform(0, 36, k)        # fin de seguimiento
surv["meses"] = np.minimum(t_baja, t_fin).round(2)
surv["baja"] = (t_baja <= t_fin).astype(int)                             # 0 = censurado
surv.head()""",
    "clustering": IMPORTS + """
# Tres segmentos de clientes en 3 variables + un precio para ordenar los clusters
seg = pd.DataFrame(np.vstack([rng.normal(c, 0.8, (150, 3)) for c in ([0, 0, 0], [5, 5, 0], [0, 6, 4])]), columns=["a", "b", "c"])
seg["precio"] = np.repeat([30000, 20000, 45000], 150) + rng.normal(0, 2000, 450)
seg["marca"] = rng.choice(["x", "y"], 450)
seg.describe().round(2)""",
    "multiclase": IMPORTS + """
n = 1500
dm = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "g": rng.choice(["a", "b"], n)})
e = np.column_stack([np.zeros(n), 1.2 * dm.x1 + 0.3, -1.0 * dm.x2 + (dm.g == "b") * 0.8])
prob = np.exp(e) / np.exp(e).sum(axis=1, keepdims=True)
dm["clase"] = np.array(["base", "media", "alta"])[(rng.random(n)[:, None] > prob.cumsum(axis=1)).sum(axis=1).clip(0, 2)]
dm.clase.value_counts()""",
    "pvalores": IMPORTS + """
# 200 contrastes: 180 sin efecto (p uniformes) y 20 con efecto real
p_valores = np.r_[rng.uniform(size=180), stats.norm.sf(rng.normal(3.2, 1, 20))]
print(f"«significativos» sin corregir: {(p_valores < 0.05).sum()} (y solo hay 20 efectos reales)")""",
    "cartera": IMPORTS + """
# Cartera de autos simulada: nº de siniestros (con exposición y heterogeneidad) y coste de cada siniestro
n = 8000
cart = pd.DataFrame({"zona": rng.choice(["A", "B", "C"], n), "edad": rng.uniform(18, 80, n).round(), "expo": rng.uniform(0.2, 1, n).round(2)})
lam = 0.15 * np.exp(0.5 * (cart.zona == "B") - 0.4 * (cart.zona == "C") - 0.01 * (cart.edad - 40)) * cart.expo
cart["nsin"] = rng.poisson(lam * rng.gamma(1.2, 1 / 1.2, n))
sini = cart.loc[cart.index.repeat(cart.nsin)].reset_index(names="poliza")
sini["coste"] = rng.gamma(2.0, 1200 * np.exp(0.3 * (sini.zona == "B")) / 2.0).round(2)
print(f"{n} pólizas · {cart.nsin.sum()} siniestros · frecuencia {cart.nsin.sum() / cart.expo.sum():.3f} por año")""",
    "superv2": IMPORTS + """
# Permanencia (meses) de pólizas con dos tarifas; Weibull con riesgos proporcionales
k = 800
sv = pd.DataFrame({"tarifa": rng.choice(["A", "B"], k), "x": rng.normal(size=k)})
lam = 0.05 * np.exp(0.5 * (sv.tarifa == "B") + 0.3 * sv.x)
T = (-np.log(rng.random(k)) / lam) ** (1 / 1.5); C = rng.uniform(0, 15, k)
sv["meses"], sv["baja"] = np.minimum(T, C).round(3), (T <= C).astype(int)
sv["motivo"] = np.where(sv.baja == 1, rng.choice([1, 2], k, p=[0.7, 0.3]), 0)     # 1 = cambio de compañía, 2 = impago
sv.head()""",
    "ml": IMPORTS + """
# Fuga de clientes con relaciones NO lineales (interacción a·b y efecto en onda de la antigüedad)
n = 3000
Xml = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "antig": rng.uniform(0, 10, n), "ruido": rng.normal(size=n),
                    "canal": rng.choice(["web", "oficina", "mediador"], n)})
eta = 1.5 * Xml.a * Xml.b + 1.2 * np.sin(Xml.antig) + (Xml.canal == "web") - 0.3
yml = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
print(f"tasa de fuga {yml.mean():.1%}")""",
    "vida": IMPORTS + """
# Tabla de mortalidad de ejemplo con la ley de Makeham (sustitúyela por PER2020, GKM… con tabla_mortalidad(qx))
tabla = tabla_mortalidad(qx_ley(range(0, 121), "makeham", A=0.0005, B=0.00007, c=1.10))
i = 0.025      # interés técnico
tabla.loc[[0, 40, 65, 90], ["qx", "lx", "e_completa"]].round(4)""",
    "triangulo": IMPORTS + """
# Triángulo de Taylor-Ashe (pagos incrementales; el ejemplo clásico de Mack, 1993)
inc = [[357848, 766940, 610542, 482940, 527326, 574398, 146342, 139950, 227229, 67948],
       [352118, 884021, 933894, 1183289, 445745, 320996, 527804, 266172, 425046, None],
       [290507, 1001799, 926219, 1016654, 750816, 146923, 495992, 280405, None, None],
       [310608, 1108250, 776189, 1562400, 272482, 352053, 206286, None, None, None],
       [443160, 693190, 991983, 769488, 504851, 470639, None, None, None, None],
       [396132, 937085, 847498, 805037, 705960, None, None, None, None, None],
       [440832, 847631, 1131398, 1063269, None, None, None, None, None, None],
       [359480, 1061648, 1443370, None, None, None, None, None, None, None],
       [376686, 986608, None, None, None, None, None, None, None, None],
       [344014, None, None, None, None, None, None, None, None, None]]
tri = pd.DataFrame(inc, dtype=float, index=range(2015, 2025)).cumsum(axis=1).where(pd.DataFrame(inc, index=range(2015, 2025)).notna())
tri.round(0)""",
    "precio": IMPORTS + """
# Ventas según precio en dos segmentos: uno muy sensible al precio y otro casi rígido
filas = []
for g, pend in {"sensible": -0.25, "rigido": -0.02}.items():
    precio = rng.normal(30000, 5000, 600)
    pv = np.clip(0.4 + pend * (precio - 30000) / 5000 + rng.normal(0, 0.05, 600), 0, 1)
    filas.append(pd.DataFrame({"seg": g, "precio": precio, "vende": (rng.random(600) < pv).astype(int)}))
ventas = pd.concat(filas, ignore_index=True)
ventas.groupby("seg").vende.mean()""",
    "regresion": IMPORTS + """
x = rng.uniform(0, 10, 150)
reg = pd.DataFrame({"x": x, "y": 3 + 1.5 * x + rng.normal(0, 1 + 0.3 * x)})   # varianza creciente a propósito
reg.head()""",
}

E = {
    # ---------------- preprocesado
    "balancear_clases": ("binario", ["print(df.compra.value_counts())\nbal = balancear_clases(df, 'compra', 'oversample')\nbal.compra.value_counts()",
                                     "pd.DataFrame({'original': df.compra.value_counts(), 'balanceado': bal.compra.value_counts()}).plot.bar(rot=0, title='Clases antes y después');"]),
    "pesos_por_clase": ("binario", ["pesos_por_clase(df.compra)"]),
    "submuestreo_por_ratio": ("binario", ["sub = submuestreo_por_ratio(df, 'compra', ratio_neg_pos=1.0)\nsub.compra.value_counts()"]),
    "corregir_probabilidades_por_balanceo": ("binario", [
        "# Entrenar en datos balanceados infla las probabilidades; se corrigen a la prevalencia real\nbal = submuestreo_por_ratio(df, 'compra', 1)\nm = sm.Logit(bal.compra, sm.add_constant(bal[['edad', 'precio']])).fit(disp=False)\np_bal = m.predict(sm.add_constant(df[['edad', 'precio']]))\np_ok = corregir_probabilidades_por_balanceo(p_bal, df.compra.mean(), bal.compra.mean())\nprint(f'media predicha: balanceada {p_bal.mean():.3f} · corregida {p_ok.mean():.3f} · real {df.compra.mean():.3f}')",
        "grafico_calibracion(df.compra, p_ok);"]),
    "categorizar_por_cuantiles": ("binario", ["out, creadas = categorizar_por_cuantiles(df, ['edad', 'precio'], q=5)\nprint(creadas)\nout.groupby('edad_cat', observed=True).compra.mean().plot.bar(rot=0, title='Tasa de compra por quintil de edad');"]),
    "codificar_ordinal": ("ninguno", ["t = pd.DataFrame({'talla': ['S', 'M', 'L', 'M', None, 'XL']})\nout, mapa = codificar_ordinal(t, 'talla', ['S', 'M', 'L'])\nprint(mapa)\nout"]),
    "duracion_hasta_evento": ("ninguno", ["h = pd.DataFrame({'id': [1, 1, 1, 2, 2, 3, 3, 3], 'anio': [1, 2, 3, 1, 2, 1, 2, 3], 'renueva': [0, 0, 1, 1, 0, 0, 0, 0]})\nduracion_hasta_evento(h, 'id', 'renueva', 'anio')   # OJO: el id 3 nunca renueva (censura) y cuenta 3"]),
    # ---------------- selección
    "es_posible_fuga": ("ninguno", ["{c: es_posible_fuga(c) for c in ['edad', 'y_pred', 'prob_baja', 'precio']}"]),
    "contraste_chi2_variable": ("binario", ["contraste_chi2_variable(df, 'zona', 'compra')"]),
    "cribar_variables": ("binario", ["cribar_variables(df.assign(y_pred=df.compra), 'compra')"]),
    "seleccion_forward": ("binario", ["r = seleccion_forward(df, 'compra', ['edad', 'precio', 'zona', 'ruido'], categoricas=['zona'], criterio='bic')\nprint(r['formula'])\nr['historial']"]),
    # ---------------- modelos
    "preparar_matriz_modelo": ("binario", ["X, y, std_map = preparar_matriz_modelo(df, 'compra', ['edad', 'precio', 'zona'], ['zona'], ['edad', 'precio'], refs={'zona': 'norte'})\nprint(std_map)\nX.head()"]),
    "ajustar_logit": ("binario", ["X, y, std_map = preparar_matriz_modelo(df, 'compra', ['edad', 'precio', 'zona'], ['zona'], ['edad', 'precio'], refs={'zona': 'norte'})\nmodelo = ajustar_logit(y, X)\nmodelo.params",
                                  "grafico_odds_ratios(tabla_odds_ratios(modelo, {'edad': 10, 'precio': 1000}, std_map));"]),
    "tabla_odds_ratios": ("logit", ["tabla_odds_ratios(modelo, unidades={'edad': 10, 'precio': 1000}, std_map=std_map)"]),
    "tabla_parametros_wald": ("logit", ["tabla_parametros_wald(modelo)"]),
    "perfil_respuesta": ("binario", ["perfil_respuesta(df.compra, {0: 'no compra', 1: 'compra'})"]),
    "regresion_logistica_firth": ("ninguno", ["# Separación perfecta: Logit normal diverge, Firth da coeficientes finitos\nXs = sm.add_constant(pd.DataFrame({'a': [-3, -2, -1, 1, 2, 3.0]}))\nys = np.array([0, 0, 0, 1, 1, 1])\nregresion_logistica_firth(Xs, ys)['beta']"]),
    "comparar_tecnicas_estimacion": ("binario", ["tr, te = dividir_train_test(df, 'compra')\nXtr, Xte = sm.add_constant(tr[['edad', 'precio']]), sm.add_constant(te[['edad', 'precio']])\ntabla, delta = comparar_tecnicas_estimacion(tr.compra, Xtr, te.compra, Xte)\ntabla"]),
    "comparar_enlaces": ("binario", ["tr, te = dividir_train_test(df, 'compra')\nXtr, Xte = sm.add_constant(tr[['edad', 'precio']]), sm.add_constant(te[['edad', 'precio']])\ncomparar_enlaces(tr.compra, Xtr, te.compra, Xte)"]),
    "dividir_train_test": ("binario", ["tr, te = dividir_train_test(df, 'compra', train=0.7)\nprint(len(tr), len(te), round(tr.compra.mean(), 3), round(te.compra.mean(), 3))   # estratificado: misma tasa"]),
    "ajustar_glm_binomial": ("binario", ["tr, te = dividir_train_test(df, 'compra')\nr = ajustar_glm_binomial('compra ~ edad + precio + C(zona)', tr, te, 'compra')\nprint(f\"AUC train {r['auc_train']:.3f} · test {r['auc_test']:.3f} · AIC {r['aic']:.0f}\")\nr['tabla']"]),
    "tabla_coeficientes": ("binario", ["m = sm.GLM.from_formula('compra ~ edad + precio + C(zona)', df, family=sm.families.Binomial()).fit()\ntabla_coeficientes(m)"]),
    "logit_multinomial_sas": ("multiclase", ["r = logit_multinomial_sas(dm, 'clase', x_num=['x1', 'x2'], x_cat=['g'], base_class='base')\nr['coeficientes'].head()", "r['lsmeans_like']('g')"]),
    "sensibilidad_por_grupo": ("precio", ["sensibilidad_por_grupo(ventas, 'vende', 'precio', 'seg')"]),
    "ganancia_por_bajada_precio": ("precio", ["ganancia_por_bajada_precio(ventas, 'vende', 'precio', 'seg', bajada=0.05)"]),
    "tendencia_polinomica_ponderada": ("ninguno", [
        "# Optimización: ¿qué precio maximiza la tasa? (parábola ponderada por volumen)\nx = np.linspace(10, 50, 40)\ny = -0.001 * (x - 30) ** 2 + 0.5 + rng.normal(0, 0.01, 40)\ncoef, f = tendencia_polinomica_ponderada(x, y, pesos=np.linspace(1, 5, 40), grado=2)\noptimo = -coef[1] / (2 * coef[0])\nprint(f'precio óptimo ≈ {optimo:.1f}')",
        "plt.scatter(x, y, s=12); xs = np.linspace(10, 50, 200); plt.plot(xs, f(xs)); plt.axvline(optimo, color='gray'); plt.title(f'Óptimo en {optimo:.1f}');"]),
    "kaplan_meier": ("supervivencia", ["km = kaplan_meier(surv, 'meses', 'baja', 'tarifa')\nkm['medianas']", "grafico_kaplan_meier(surv, 'meses', 'baja', 'tarifa');"]),
    "contraste_log_rank": ("supervivencia", ["contraste_log_rank(surv, 'meses', 'baja', 'tarifa')['interpretacion']"]),
    "ajustar_cox": ("supervivencia", ["cox = ajustar_cox(surv, 'meses', 'baja', ['edad'], categoricas=['tarifa'], refs={'tarifa': 'A'})\ncox['tabla']   # verdad simulada: HR tarifa B = 0.61, HR edad = 1.03"]),
    # ---------------- diagnóstico
    "hosmer_lemeshow": ("logit", ["hl = hosmer_lemeshow(y, p)\nprint(f\"chi2 = {hl['estadistico']:.2f}, p = {hl['p_valor']:.3f}\")\nhl['tabla']", "grafico_calibracion(y, p);"]),
    "estadisticos_asociacion": ("logit", ["estadisticos_asociacion(y, p)"]),
    "calcular_vif": ("logit", ["calcular_vif(sm.add_constant(X))", "grafico_vif(calcular_vif(sm.add_constant(X)));"]),
    "filtrar_vif_iterativo": ("logit", ["Xc = sm.add_constant(X.assign(edad_copia=X.edad * 2 + rng.normal(0, 0.01, len(X))))\nfiltrar_vif_iterativo(Xc, umbral=8, verbose=True)"]),
    "auc_train_test": ("binario", ["tr, te = dividir_train_test(df, 'compra')\nm = sm.GLM.from_formula('compra ~ edad + precio + C(zona)', tr, family=sm.families.Binomial()).fit()\nauc_train_test(m, tr, te, 'compra')"]),
    "metricas_binarias": ("logit", ["metricas_binarias(y, p, 'logística')"]),
    "tabla_umbrales": ("logit", ["tabla_umbrales(y, p)", "grafico_umbrales(y, p);"]),
    "umbral_optimo_youden": ("logit", ["u = umbral_optimo_youden(y, p)\nprint(f'umbral óptimo (Youden) = {u:.3f}')", "grafico_curva_roc(y, p);"]),
    "resumen_auc_multiclase": ("multiclase", ["r = logit_multinomial_sas(dm, 'clase', x_num=['x1', 'x2'], x_cat=['g'], base_class='base')\nr['auc']['por_clase_ovr']"]),
    "matrices_confusion": ("ninguno", ["cm, cm_filas = matrices_confusion(['a', 'b', 'a', 'b', 'a'], ['a', 'a', 'a', 'b', 'b'], ['a', 'b'])\ncm_filas"]),
    # ---------------- clustering
    "preparar_matriz_clustering": ("clustering", ["Xc, nombres = preparar_matriz_clustering(seg, ['a', 'b', 'c'], ['marca'], escalado='standard')\nprint(nombres)\nXc[:5].round(2)"]),
    "buscar_k_silhouette": ("clustering", ["Xc, _ = preparar_matriz_clustering(seg, ['a', 'b', 'c'], escalado='standard')\nt = buscar_k_silhouette(Xc, 2, 7)\nt", "grafico_seleccion_k(t);"]),
    "ajustar_kmeans": ("clustering", ["Xc, _ = preparar_matriz_clustering(seg, ['a', 'b', 'c'], escalado='standard')\nr = ajustar_kmeans(Xc, 3, ordenar_por=seg.precio)\nprint(f\"pseudo-R² = {r['pseudo_r2']:.3f}\")\nseg.groupby(r['etiquetas']).precio.mean().round()", "grafico_clusters_pca(Xc, r['etiquetas']);"]),
    "proyeccion_pca": ("clustering", ["Xc, _ = preparar_matriz_clustering(seg, ['a', 'b', 'c'], escalado='standard')\nz, var = proyeccion_pca(Xc)\nprint(var.round(3))\nplt.scatter(z.PC1, z.PC2, s=8); plt.title('PC1 vs PC2');"]),
    # ---------------- contrastes
    "elegir_contraste": ("grupos", ["r = elegir_contraste(d, 'valor', 'grupo')\nprint(r['interpretacion'])\nr['supuestos']", "grafico_comparar_grupos(d, 'valor', 'grupo');"]),
    "tukey_entre_grupos": ("grupos", ["tukey_entre_grupos(d, 'valor', 'grupo')"]),
    "ajustar_p_valores": ("pvalores", ["t = ajustar_p_valores(p_valores, 'fdr_bh')\nprint('BH:', t.rechaza.sum(), '· Bonferroni:', ajustar_p_valores(p_valores, 'bonferroni').rechaza.sum())\nt.sort_values('rango').head(10)", "grafico_fdr(p_valores);"]),
    "tamano_muestral_medias": ("ninguno", ["tamano_muestral_medias(efecto=0.5)", "grafico_curva_potencia([0.2, 0.5, 0.8]);"]),
    "tamano_muestral_proporciones": ("ninguno", ["# A/B test: conversión 10 % -> 12 %\ntamano_muestral_proporciones(0.10, 0.12)"]),
    "potencia_contraste_medias": ("ninguno", ["potencia_contraste_medias(n_grupo1=40, efecto=0.5)", "grafico_potencia(0.5, 40);"]),
    "curva_potencia": ("ninguno", ["c = curva_potencia(0.5)\nc.plot(x='n_grupo1', y='potencia', title='Potencia vs n (d = 0.5)');"]),
    "potencia_por_simulacion": ("ninguno", ["# Potencia de Mann-Whitney con datos asimétricos (sin fórmula): Monte Carlo\ngen = lambda rng, n: (rng.exponential(1, n), rng.exponential(1.5, n))\npotencia_por_simulacion(gen, lambda a, b: stats.mannwhitneyu(a, b).pvalue, n=40, n_sim=500)"]),
    # ---------------- gráficos
    "grafico_regresion_simple": ("regresion", ["grafico_regresion_simple(reg, 'x', 'y');"]),
    "grafico_diagnostico_residuos": ("regresion", ["ols = sm.OLS(reg.y, sm.add_constant(reg[['x']])).fit()\ngrafico_diagnostico_residuos(ols);   # escala-localización creciente = heterocedasticidad"]),
    "grafico_curva_roc": ("logit", ["grafico_curva_roc(y, {'completo': p, 'solo edad': 1 / (1 + np.exp(-0.04 * (X.edad)))});"]),
    "grafico_calibracion": ("logit", ["grafico_calibracion(y, p);"]),
    "grafico_odds_ratios": ("logit", ["grafico_odds_ratios(tabla_odds_ratios(modelo, {'edad': 10, 'precio': 1000}, std_map));"]),
    "grafico_umbrales": ("logit", ["grafico_umbrales(y, p);"]),
    "grafico_residuos_agrupados": ("logit", ["grafico_residuos_agrupados(y, p);"]),
    "grafico_region_rechazo": ("grupos", ["t = stats.ttest_ind(d.valor[d.grupo == 'A'], d.valor[d.grupo == 'B'])\ngrafico_region_rechazo(t.statistic, 't', gl=118);"]),
    "grafico_potencia": ("ninguno", ["grafico_potencia(efecto=0.5, n_grupo1=30);"]),
    "grafico_curva_potencia": ("ninguno", ["grafico_curva_potencia([0.3, 0.5, 0.8], n_max=250);"]),
    "grafico_comparar_grupos": ("grupos", ["grafico_comparar_grupos(d, 'valor', 'grupo');", "grafico_comparar_grupos(d, 'exito', 'grupo');"]),
    "grafico_qq": ("ninguno", ["grafico_qq(rng.exponential(1, 200));   # asimétrica: curva en U"]),
    "grafico_fdr": ("pvalores", ["grafico_fdr(p_valores);"]),
    "grafico_kaplan_meier": ("supervivencia", ["grafico_kaplan_meier(surv, 'meses', 'baja', 'tarifa');"]),
    "grafico_seleccion_k": ("clustering", ["Xc, _ = preparar_matriz_clustering(seg, ['a', 'b', 'c'], escalado='standard')\ngrafico_seleccion_k(buscar_k_silhouette(Xc, 2, 7));"]),
    "grafico_clusters_pca": ("clustering", ["Xc, _ = preparar_matriz_clustering(seg, ['a', 'b', 'c'], escalado='standard')\ngrafico_clusters_pca(Xc, ajustar_kmeans(Xc, 3)['etiquetas']);"]),
    "grafico_vif": ("logit", ["grafico_vif(calcular_vif(sm.add_constant(X)));"]),
    # ---------------- finanzas (0.9)
    "var_tvar": ("ninguno", ["perdidas = stats.t.rvs(3, size=20000, random_state=rng) * 1e4     # colas pesadas\npd.concat({m: var_tvar(perdidas, metodo=m) for m in ['historico', 'normal', 'cornish_fisher']}, axis=1).round(0)"]),
    "ajustar_gpd": ("ninguno", ["siniestros = stats.lomax(2.5, scale=2000).rvs(5000, random_state=rng)\ng = ajustar_gpd(siniestros, cuantil_umbral=0.9)\nprint(f\"ξ = {g['xi']:.3f} (verdadero 0.4) · umbral {g['umbral']:.0f} · {g['n_excesos']} excesos\")\ng['tabla'].round(0)"]),
    "estimador_hill": ("ninguno", ["siniestros = stats.lomax(2.5, scale=2000).rvs(5000, random_state=rng)\nh = estimador_hill(siniestros)\nh.xi.plot(); plt.axhline(0.4, color='gray'); plt.title('Hill plot: ξ̂ según k (verdadero 0.4)');"]),
    "funcion_exceso_medio": ("ninguno", ["e1 = funcion_exceso_medio(stats.lomax(2.5, scale=2000).rvs(5000, random_state=rng)); e2 = funcion_exceso_medio(rng.exponential(1300, 5000))\nplt.plot(e1.umbral, e1.exceso_medio, label='Pareto (creciente)'); plt.plot(e2.umbral, e2.exceso_medio, label='exponencial (plana)'); plt.legend(); plt.title('Función de exceso medio');"]),
    "simular_copula": ("ninguno", ["c = simular_copula('gumbel', 2.0, 5000, marginales=(stats.lognorm(1, scale=1000), stats.lognorm(1, scale=500)))\nprint('τ de Kendall', round(stats.kendalltau(c.x, c.y)[0], 3), '· teórica', c.attrs['tau_kendall'])", "grafico_copula(c);"]),
    "prueba_estres": ("ninguno", ["prueba_estres({'renta variable': 2e6, 'renta fija': 8e6, 'inmuebles': 1e6},\n              {'choque SII': {'renta variable': -0.39, 'inmuebles': -0.25}, 'subida tipos': {'renta fija': -0.08}, 'combinado': {'renta variable': -0.3, 'renta fija': -0.05, 'inmuebles': -0.15}})"]),
    "matriz_covarianzas": ("ninguno", ["R = pd.DataFrame(rng.multivariate_normal(np.zeros(20), np.eye(20) * 0.04 + 0.01, 60))   # 20 activos, solo 60 meses\nm, l = matriz_covarianzas(R, 'muestral'), matriz_covarianzas(R, 'ledoit_wolf')\nprint('contracción', round(l['contraccion'], 3), '· nº de condición muestral', round(np.linalg.cond(m['covarianza'])), 'vs LW', round(np.linalg.cond(l['covarianza'])))"]),
    "frontera_eficiente": ("ninguno", ["R = pd.DataFrame(rng.multivariate_normal([.08, .05, .03, .06], [[.04, .006, .001, .01], [.006, .01, .0005, .002], [.001, .0005, .0025, .001], [.01, .002, .001, .02]], 500), columns=['acciones', 'crédito', 'monetario', 'inmuebles'])\nmc = matriz_covarianzas(R)\nfe = frontera_eficiente(mc['medias'], mc['covarianza'], 25, tasa_libre=0.01)\nprint(fe['tangente'].round(3).to_dict())", "grafico_frontera_eficiente(fe, mc['medias'], mc['covarianza']);"]),
    "beta_capm": ("ninguno", ["mercado = rng.normal(.006, .04, 120); activo = .001 + 1.25 * mercado + rng.normal(0, .025, 120)\nbeta_capm(activo, mercado, tasa_libre=0.002)"]),
    "modelo_factores": ("ninguno", ["F = pd.DataFrame({'mercado': rng.normal(.006, .04, 240), 'tamano': rng.normal(0, .02, 240), 'valor': rng.normal(0, .02, 240)})\nR = pd.DataFrame({'fondo_A': 1.0 * F.mercado + .5 * F.tamano + rng.normal(0, .01, 240), 'fondo_B': .8 * F.mercado - .4 * F.valor + rng.normal(0, .01, 240)})\nmodelo_factores(R, F).round(3)"]),
    "black_litterman": ("ninguno", ["S = pd.DataFrame([[.04, .006, .001], [.006, .01, .0005], [.001, .0005, .0025]], index=['acc', 'cred', 'mon'], columns=['acc', 'cred', 'mon'])\n# opinión: las acciones batirán al crédito en un 5 %\nbl = black_litterman(S, [.5, .3, .2], [[1, -1, 0]], [0.05])\npd.DataFrame({'equilibrio': bl['equilibrio'], 'posterior': bl['posterior'], 'pesos': bl['pesos']}).round(4)"]),
    "dominancia_estocastica": ("ninguno", ["a, b = rng.normal(0.05, 0.10, 5000), rng.normal(0.05, 0.25, 5000)\ndominancia_estocastica(a - a.mean() + .05, b - b.mean() + .05)   # misma media, A menos arriesgada: SSD"]),
    "equivalente_cierto": ("ninguno", ["# Pérdida de 0 o 10 000 € al 50 %: ¿cuánto pagarías por un seguro? (riqueza inicial 20 000)\nfor a in [0.00005, 0.0001, 0.0002]:\n    r = equivalente_cierto([20000, 10000], [.5, .5], 'exponencial', a)\n    print(f\"aversión {a}: prima de riesgo {r['prima_riesgo']:.0f} €\")"]),
    "fraccion_anio": ("ninguno", ["{c: round(fraccion_anio('2024-01-31', '2024-07-31', c), 6) for c in ['act/365', 'act/360', '30/360', 'act/act']}"]),
    "precio_bono": ("ninguno", ["pd.DataFrame({y: precio_bono(100, .04, 10, y) for y in [.02, .03, .04, .05, .06]}).T.round(3)"]),
    "tir_bono": ("ninguno", ["tir_bono(precio=96.5, nominal=100, cupon=.035, anios=7)"]),
    "bootstrapping_etti": ("ninguno", ["e = bootstrapping_etti([1, 2, 3, 4, 5], [.02, .025, .03, .032, .035], [100.2, 100.1, 100.4, 100.0, 99.8])\ne.round(5)", "e.tipo_cero.plot(marker='o', label='cupón cero'); e.forward_1a.plot(marker='s', label='forward'); plt.legend(); plt.title('ETTI por bootstrapping');"]),
    "inmunizacion": ("ninguno", ["bonos = pd.DataFrame({'precio': [99.0, 103.0], 'duracion': [2.0, 12.0], 'convexidad': [5.0, 160.0]}, index=['bono 2a', 'bono 15a'])\ninmunizacion(duracion_pasivo=7, valor_pasivo=5e6, bonos=bonos)"]),
    "valorar_swap": ("ninguno", ["df = bootstrapping_etti([1, 2, 3, 4, 5], [.02, .025, .03, .032, .035], [100.2, 100.1, 100.4, 100.0, 99.8]).factor_descuento\nvalorar_swap(0.03, df, nominal=1e7)"]),
    "black_scholes": ("ninguno", ["pd.DataFrame({K: black_scholes(100, K, 1, .03, .2) for K in [80, 100, 120]}).round(4)"]),
    "arbol_binomial": ("ninguno", ["bs = black_scholes(100, 100, 1, .03, .25, 'put')['precio']\nfor n in [10, 50, 200, 1000]:\n    print(n, round(arbol_binomial(100, 100, 1, .03, .25, n, 'put')['precio'], 4), '· americana', round(arbol_binomial(100, 100, 1, .03, .25, n, 'put', americana=True)['precio'], 4), '· BS', round(bs, 4))"]),
    "grafico_frontera_eficiente": ("ninguno", ["S = pd.DataFrame([[.04, .006, .001], [.006, .01, .0005], [.001, .0005, .0025]], index=list('abc'), columns=list('abc')); mu = pd.Series([.08, .05, .03], index=list('abc'))\ngrafico_frontera_eficiente(frontera_eficiente(mu, S, 20, tasa_libre=.01), mu, S);"]),
    "grafico_copula": ("ninguno", ["grafico_copula(simular_copula('clayton', 3, 5000));"]),
    # ---------------- actuarial (0.9)
    "qx_ley": ("ninguno", ["q = qx_ley(range(20, 101), 'makeham')\nplt.semilogy(q.index, q); plt.title('qx de Makeham (log): casi una recta = crecimiento exponencial');"]),
    "tabla_mortalidad": ("vida", ["tabla.loc[60:66]", "grafico_tabla_mortalidad(tabla, tabla_mortalidad(tabla.qx * 0.8), nombres=['base', 'mortalidad −20 %']);"]),
    "ajustar_ley_mortalidad": ("ninguno", ["x = np.arange(40, 95); E = np.full(len(x), 15000.0)\nD = rng.poisson(E * (0.0005 + 0.00007 * 1.1 ** (x + 0.5)))\nr = ajustar_ley_mortalidad(x, D, E, ley='makeham')\nprint(r['parametros'], '· χ² =', round(r['chi2'], 1))\nplt.semilogy(x, D / E, 'o', ms=3, label='bruta'); plt.semilogy(x, r['qx_ajustadas'], label='Makeham'); plt.legend();"]),
    "probabilidad_supervivencia": ("vida", ["pd.Series({h: probabilidad_supervivencia(tabla, 80.25, 0.5, h) for h in ['udd', 'constante', 'balducci']})"]),
    "vida_futura": ("vida", ["vf = vida_futura(tabla, 65)\nprint(f\"E[K65] = {vf['esperanza']:.2f} · mediana {vf['cuantiles'][0.5]} años\")\nvf['funcion_probabilidad'].plot.bar(figsize=(9, 3), title='P(K65 = k)');"]),
    "tabla_conjunta": ("vida", ["tc = tabla_conjunta(tabla, 67, tabla, 63)\nprint(tc.attrs)\ntc[['tpx', 'tpy', 'tpxy', 'tp_ultimo']].plot(title='Vida conjunta y último superviviente');"]),
    "decrementos_multiples": ("ninguno", ["qi = pd.DataFrame({'muerte': [0.005, 0.01, 0.02], 'rescate': [0.12, 0.09, 0.06]}, index=[40, 50, 60])\nr = decrementos_multiples(q_independientes=qi)\nr['dependientes'].round(5)"]),
    "conmutados": ("vida", ["conmutados(tabla, i).loc[[40, 65]].round(2)"]),
    "seguro_vida": ("vida", ["pd.DataFrame({t: seguro_vida(tabla, 40, i, t, n=20, capital=100000) for t in ['vida_entera', 'temporal', 'dotal_puro', 'mixto']}).T.round(2)"]),
    "renta_actuarial": ("vida", ["print('ä65 anual :', round(renta_actuarial(tabla, 65, i), 4))\nprint('ä65 mensual:', round(renta_actuarial(tabla, 65, i, fraccionamiento=12), 4))\nprint('renta diferida 25 años a los 40:', round(renta_actuarial(tabla, 40, i, diferido=25), 4))"]),
    "prima_neta": ("vida", ["prima_neta(tabla, 40, i, 'mixto', n=20, capital=50000)"]),
    "provision_matematica": ("vida", ["pm = provision_matematica(tabla, 40, i, 'mixto', n=20, capital=50000)\nprint('prima anual', round(pm.attrs['prima_anual'], 2))\npm.provision.plot(title='Provisión matemática del mixto a 20 años');"]),
    "prima_tarifa": ("vida", ["prima_tarifa(tabla, 40, i, 'mixto', n=20, capital=50000, alfa=0.03, beta=0.05, gamma=0.002)"]),
    "sensibilidad_longevidad": ("vida", ["sensibilidad_longevidad(tabla, 65, i, choque=-0.20)   # choque de longevidad de Solvencia II"]),
    "chain_ladder": ("triangulo", ["cl = chain_ladder(tri)\nprint('factores', cl['factores'].round(3).tolist())\nprint(f\"reserva {cl['reserva_total']:,.0f} · error de Mack {cl['se_total']:,.0f}\")\ncl['tabla'].round(0)"]),
    "bootstrap_chain_ladder": ("triangulo", ["b = bootstrap_chain_ladder(tri, 3000)\nprint({k: round(v) for k, v in b['percentiles'].items()})", "grafico_reserva_bootstrap(b);"]),
    "recursion_panjer": ("ninguno", ["# Severidad discretizada en unidades de 1000 €: 1, 2, 3 o 4 con igual probabilidad; N ~ Poisson(3)\nS = recursion_panjer('poisson', {'lambda': 3}, [0, .25, .25, .25, .25], max_s=60)\nprint('media', S.attrs['media'])\nS.probabilidad.plot.bar(figsize=(10, 3), title='Distribución exacta de la siniestralidad agregada');"]),
    "simular_siniestralidad_agregada": ("ninguno", ["r = simular_siniestralidad_agregada(stats.poisson(2.5), stats.lognorm(1.1, scale=1500), 200000)\nprint({k: round(v, 1) for k, v in r.items() if isinstance(v, float)}); print('VaR', {k: round(v) for k, v in r['var'].items()}); print('TVaR', {k: round(v) for k, v in r['tvar'].items()})"]),
    "probabilidad_ruina": ("ninguno", ["pd.DataFrame([probabilidad_ruina(u, 0.2, 1.0) for u in [0, 5, 10, 20]]).round(4)"]),
    "prima_por_principios": ("ninguno", ["prima_por_principios(distribucion=stats.gamma(2, scale=500)).round(1)"]),
    "credibilidad_buhlmann": ("ninguno", ["# 25 flotas con 6 años de siniestralidad cada una: prima de credibilidad\nd2 = pd.DataFrame({'flota': np.repeat(range(25), 6)}); d2['coste'] = np.repeat(rng.gamma(4, 25, 25), 6) + rng.normal(0, 40, 150)\nc = credibilidad_buhlmann(d2, 'flota', 'coste')\nprint(f\"k = {c['k']:.2f} · media colectiva {c['media_colectiva']:.1f}\")\nc['tabla'].head().round(2)"]),
    "tasas_especificas": ("ninguno", ["tasas_especificas([12, 40, 160, 610], [52000, 48000, 30000, 12000], ['0-39', '40-59', '60-74', '75+'])"]),
    "estandarizar_tasas": ("ninguno", ["defs, pobl = [12, 40, 160, 610], [52000, 48000, 30000, 12000]\nprint(estandarizar_tasas(defs, pobl, [60000, 50000, 25000, 8000], 'directa'))\nprint(estandarizar_tasas(defs, pobl, [0.0002, 0.0009, 0.005, 0.045], 'indirecta'))"]),
    "exposicion_por_edad": ("ninguno", ["n = 3000\nnac = pd.Timestamp('1945-01-01') + pd.to_timedelta(rng.integers(0, 365 * 25, n), unit='D')\nini = pd.Timestamp('2018-01-01') + pd.to_timedelta(rng.integers(0, 365, n), unit='D')\nfin = ini + pd.to_timedelta(rng.integers(60, 365 * 5, n), unit='D')\ncartera = pd.DataFrame({'nac': nac, 'alta': ini, 'baja': fin, 'fallece': rng.random(n) < 0.04})\ne = exposicion_por_edad(cartera, 'nac', 'alta', 'baja', 'fallece')\ne.loc[70:75].round(4)"]),
    "indicadores_fecundidad": ("ninguno", ["indicadores_fecundidad([4000, 18000, 52000, 98000, 70000, 18000, 1500], [1.1e6, 1.2e6, 1.3e6, 1.5e6, 1.6e6, 1.7e6, 1.8e6], range(15, 50, 5), amplitud=5)"]),
    "proyeccion_leslie": ("ninguno", ["r = proyeccion_leslie([1000, 900, 800, 600], [0.95, 0.9, 0.7], [0, 0.5, 0.6, 0.1], pasos=15)\nprint('λ =', round(r['lambda'], 4))\nr['poblaciones'].plot(title='Proyección de Leslie por grupos');"]),
    "grafico_tabla_mortalidad": ("vida", ["grafico_tabla_mortalidad(tabla, tabla_mortalidad(qx_ley(range(0, 121), 'gompertz', B=0.0001, c=1.095)), nombres=['Makeham', 'Gompertz']);"]),
    "grafico_reserva_bootstrap": ("triangulo", ["grafico_reserva_bootstrap(bootstrap_chain_ladder(tri, 2000));"]),
    # ---------------- ml (0.9)
    "ajustar_arbol_decision": ("ml", ["r = ajustar_arbol_decision(Xml, yml)\nprint(r['metricas_test'], '· hojas', r['hojas'])\nprint(r['reglas'][:600])"]),
    "ajustar_random_forest": ("ml", ["r = ajustar_random_forest(Xml, yml)\nprint('test', r['metricas_test'], '· OOB', round(r['oob'], 3), '· sobreajuste', round(r['sobreajuste'], 3))", "grafico_importancias(r['importancias']);"]),
    "ajustar_gradient_boosting": ("ml", ["r = ajustar_gradient_boosting(Xml, yml)\nprint('test', r['metricas_test'], '· iteraciones', r['iteraciones'])"]),
    "ajustar_adaboost": ("ml", ["ajustar_adaboost(Xml, yml)['metricas_test']"]),
    "ajustar_knn": ("ml", ["r = ajustar_knn(Xml, yml)\nprint('k =', r['k'], r['metricas_test'])"]),
    "ajustar_svm": ("ml", ["ajustar_svm(Xml.head(1500), yml[:1500])['metricas_test']"]),
    "ajustar_naive_bayes": ("ml", ["ajustar_naive_bayes(Xml, yml)['metricas_test']   # independencia: no ve la interacción a·b"]),
    "ajustar_red_neuronal": ("ml", ["ajustar_red_neuronal(Xml, yml, capas=(32, 16))['metricas_test']"]),
    "ajustar_stacking": ("ml", ["r = ajustar_stacking(Xml, yml)\nprint(r['metricas_test']); r['pesos_meta']"]),
    "comparar_clasificadores": ("ml", ["comparar_clasificadores(Xml, yml, ('logit', 'arbol', 'rf', 'gb', 'knn', 'nb'), cv=5).round(3)"]),
    "importancia_permutacion": ("ml", ["r = ajustar_gradient_boosting(Xml, yml)\nimp = importancia_permutacion(r)\nimp.round(3)", "grafico_importancias(imp);"]),
    "dependencia_parcial": ("ml", ["r = ajustar_gradient_boosting(Xml, yml)\ndp = dependencia_parcial(r, 'antig', 30, ice=40)", "grafico_dependencia_parcial(dp);"]),
    "grafico_importancias": ("ml", ["grafico_importancias(ajustar_random_forest(Xml, yml)['importancias']);"]),
    "grafico_dependencia_parcial": ("ml", ["grafico_dependencia_parcial(dependencia_parcial(ajustar_gradient_boosting(Xml, yml), 'a', 25, ice=30));"]),
    # ---------------- multivariante (0.9)
    "distancia_mahalanobis": ("ninguno", ["X = pd.DataFrame(rng.multivariate_normal([0, 0], [[1, .9], [.9, 1]], 300), columns=['importe', 'horas'])\nX.iloc[:10] = [[2.5, -2.5]] * 10    # atípicos que van CONTRA la correlación (cada variable por separado parece normal)\nr = distancia_mahalanobis(X)\nprint('atípicos robustos:', r.atipico.sum(), '· clásicos:', distancia_mahalanobis(X, robusta=False).atipico.sum())\nplt.scatter(X.importe, X.horas, c=r.atipico, cmap='coolwarm', s=10); plt.title('Atípicos multivariantes (MCD)');"]),
    "contraste_hotelling": ("ninguno", ["A = pd.DataFrame(rng.normal(size=(80, 3)), columns=list('xyz')); B = pd.DataFrame(rng.normal(size=(80, 3)), columns=list('xyz')) + [0.3, 0.3, 0]\nr = contraste_hotelling(A, B)\nprint(f\"T² = {r['T2']:.2f}, p = {r['p_valor']:.4f}\"); r['diferencia_medias']"]),
    "contraste_box_m": ("grupos", ["d2 = d.assign(otra=rng.normal(size=len(d)))\ncontraste_box_m(d2, ['valor', 'otra'], 'grupo')"]),
    "manova": ("grupos", ["d2 = d.assign(otra=rng.normal(size=len(d)) + (d.grupo == 'C') * 0.5)\nmanova(d2, ['valor', 'otra'], 'grupo').round(4)"]),
    "correlacion_canonica": ("ninguno", ["z = rng.normal(size=500)\nX = pd.DataFrame({'renta': z + rng.normal(size=500), 'edad': rng.normal(size=500)})\nY = pd.DataFrame({'gasto': z + rng.normal(size=500), 'visitas': rng.normal(size=500)})\nr = correlacion_canonica(X, Y)\nprint(r['cargas_x'].round(2)); r['tabla'].round(4)"]),
    "pca_completo": ("clustering", ["p = pca_completo(seg[['a', 'b', 'c', 'precio']])\nprint(p['cargas'].round(2)); p['varianza'].round(2)", "grafico_biplot(p);"]),
    "adecuacion_factorial": ("ninguno", ["L = rng.normal(size=(400, 2))\nitems = pd.DataFrame(np.c_[L[:, [0]] + rng.normal(0, .6, (400, 3)), L[:, [1]] + rng.normal(0, .6, (400, 3))], columns=[f'p{i}' for i in range(1, 7)])\nr = adecuacion_factorial(items)\nprint(f\"KMO = {r['kmo']:.2f} · Bartlett p = {r['bartlett_p']:.2g}\")"]),
    "analisis_factorial": ("ninguno", ["L = rng.normal(size=(400, 2))\nitems = pd.DataFrame(np.c_[L[:, [0]] + rng.normal(0, .6, (400, 3)), L[:, [1]] + rng.normal(0, .6, (400, 3))], columns=['precio', 'cuota', 'franquicia', 'atencion', 'rapidez', 'trato'])\nfa = analisis_factorial(items, 2)\nfa['cargas'].round(2)   # dos factores: «coste» y «servicio»"]),
    "alfa_cronbach": ("ninguno", ["t = rng.normal(size=(250, 1))\nencuesta = pd.DataFrame(np.c_[t + rng.normal(0, .8, (250, 4)), rng.normal(size=(250, 1))], columns=[f'item{i}' for i in range(1, 6)])\nr = alfa_cronbach(encuesta)\nprint(f\"α = {r['alfa']:.3f} [{r['ic_inf']:.3f}, {r['ic_sup']:.3f}]\")\nr['por_item'].round(3)   # item5 no mide lo mismo"]),
    "escalamiento_multidimensional": ("clustering", ["m = escalamiento_multidimensional(X=seg[['a', 'b', 'c']].sample(120, random_state=1))\nprint('bondad de ajuste', round(m['bondad_ajuste'], 3))\nplt.scatter(m['coordenadas'].D1, m['coordenadas'].D2, s=10); plt.title('MDS clásico');"]),
    "analisis_correspondencias": ("ninguno", ["tab = pd.DataFrame([[120, 30, 10], [40, 80, 30], [10, 20, 90]], index=['joven', 'medio', 'mayor'], columns=['terceros', 'ampliado', 'todo riesgo'])\nca = analisis_correspondencias(tab)\nprint(ca['pct_inercia'].round(1).to_dict())\nf, c = ca['filas'], ca['columnas']\nplt.scatter(f.Dim1, f.Dim2); plt.scatter(c.Dim1, c.Dim2, marker='s')\nfor n, r in pd.concat([f, c]).iterrows(): plt.annotate(n, (r.Dim1, r.Dim2))\nplt.title('Correspondencias: edad × cobertura');"]),
    "clustering_jerarquico": ("clustering", ["h = clustering_jerarquico(seg[['a', 'b', 'c']], k=3)\nprint('cofenética', round(h['cofenetica'], 3)); print(h['etiquetas'].value_counts())", "grafico_dendrograma(h, k=3);"]),
    "mezclas_gaussianas": ("clustering", ["m = mezclas_gaussianas(seg[['a', 'b', 'c']], k_max=6)\nprint('k elegido por BIC:', m['k']); m['tabla'].round(1)"]),
    "analisis_discriminante": ("multiclase", ["r = analisis_discriminante(dm[['x1', 'x2']], dm.clase, 'lda')\nprint('exactitud CV', round(r['exactitud_cv'], 3)); r['confusion_cv']"]),
    "grafico_dendrograma": ("clustering", ["grafico_dendrograma(clustering_jerarquico(seg[['a', 'b', 'c']]), k=3);"]),
    "grafico_biplot": ("clustering", ["grafico_biplot(pca_completo(seg[['a', 'b', 'c', 'precio']]));"]),
    # ---------------- preprocesado (0.9)
    "resumen_faltantes": ("ninguno", ["d2 = pd.DataFrame({'edad': rng.normal(45, 12, 800), 'renta': rng.lognormal(10, .5, 800)})\nd2['gasto'] = 0.02 * d2.renta + rng.normal(0, 100, 800)\nd2.loc[(d2.edad > 55) & (rng.random(800) < .5), 'renta'] = np.nan     # los mayores no dan la renta (MAR)\nr = resumen_faltantes(d2)\nprint(r['little'])\nr['patrones']"]),
    "contraste_mcar_little": ("ninguno", ["d2 = pd.DataFrame(rng.normal(size=(500, 3)), columns=list('abc'))\nd2.loc[rng.random(500) < .2, 'c'] = np.nan\nprint('al azar  :', contraste_mcar_little(d2))\nd3 = d2.copy(); d3.loc[d3.a > .8, 'b'] = np.nan\nprint('depende de a:', contraste_mcar_little(d3))"]),
    "imputar": ("ninguno", ["d2 = pd.DataFrame({'a': rng.normal(size=600), 'b': rng.normal(size=600)}); d2['c'] = d2.a + d2.b + rng.normal(0, .5, 600)\nfalt = d2.copy(); falt.loc[(d2.a > .5) & (rng.random(600) < .6), 'c'] = np.nan\n{m: round(np.corrcoef(imputar(falt, m).c, d2.c)[0, 1], 3) for m in ['mediana', 'iterativa', 'knn']}"]),
    "imputacion_multiple": ("ninguno", ["d2 = pd.DataFrame({'a': rng.normal(size=600), 'b': rng.normal(size=600)}); d2['c'] = d2.a + d2.b + rng.normal(0, .5, 600)\nd2.loc[(d2.a > .5) & (rng.random(600) < .6), 'c'] = np.nan\nimputacion_multiple(d2, 'c ~ a + b', m=10)['tabla'].round(3)   # fmi = fracción de información perdida"]),
    "agrupar_categorias_raras": ("ninguno", ["z = pd.DataFrame({'marca': rng.choice(list('ABCDEFGH'), 2000, p=[.35, .3, .2, .1, .02, .01, .01, .01])})\nout, raras = agrupar_categorias_raras(z, 'marca', 0.03)\nprint('agrupadas:', raras); out.marca.value_counts()"]),
    "codificar_por_objetivo": ("ninguno", ["z = pd.DataFrame({'cp': rng.choice([f'CP{i:03d}' for i in range(200)], 5000)})\nz['siniestro'] = (rng.random(5000) < 0.05 + 0.1 * (z.cp.str[-1].astype(int) / 9)).astype(int)\nenc, tabla = codificar_por_objetivo(z, 'cp', 'siniestro', suavizado=20)\ntabla.sort_values('n').head(6).round(3)   # los CP con pocos casos se acercan a la media global"]),
    "smote": ("ninguno", ["dd = pd.DataFrame({'x1': rng.normal(size=600), 'x2': rng.normal(size=600)}); dd['fraude'] = (rng.random(600) < 0.05 + 0.2 * (dd.x1 > 1)).astype(int)\ns = smote(dd, 'fraude')\nprint(s.fraude.value_counts().to_dict())\nplt.scatter(s.x1, s.x2, c=s.sintetico, s=6, cmap='coolwarm'); plt.title('Casos sintéticos (rojo) de la clase minoritaria');"]),
    # ---------------- selección y validación (0.9)
    "seleccion_backward": ("binario", ["r = seleccion_backward(df, 'compra', ['edad', 'precio', 'zona', 'ruido'], categoricas=['zona'], criterio='bic')\nprint(r['formula'])\nr['historial']"]),
    "seleccion_por_pvalor": ("binario", ["r = seleccion_por_pvalor(df, 'compra', ['edad', 'precio', 'zona', 'ruido'], categoricas=['zona'], sle=0.05, sls=0.05)\nprint(r['formula'])\nr['historial']"]),
    "mejor_subconjunto": ("ninguno", ["d2 = pd.DataFrame(rng.normal(size=(500, 6)), columns=list('abcdef')); d2['y'] = 2 * d2.a - d2.c + rng.normal(size=500)\nmejor_subconjunto(d2, 'y', list('abcdef')).round(2)"]),
    "filtrar_varianza_casi_nula": ("binario", ["d2 = df.assign(casi_constante=np.where(rng.random(len(df)) < 0.003, 1, 0), constante=7)\nfiltrar_varianza_casi_nula(d2).round(2)"]),
    "filtrar_correlacion_alta": ("binario", ["d2 = df.assign(precio_con_iva=df.precio * 1.21 + rng.normal(0, 50, len(df)))\nr = filtrar_correlacion_alta(d2, umbral=0.9)\nprint('quitar:', r['quitar'])"]),
    "eliminacion_recursiva": ("binario", ["r = eliminacion_recursiva(df[['edad', 'precio', 'ruido']], df.compra)\nprint(r['seleccionadas']); r['curva_cv']"]),
    "validacion_cruzada": ("binario", ["v = validacion_cruzada(df, 'compra ~ edad + precio + C(zona)', 'binomial', k=5, repeticiones=3)\nv['resumen'].round(4)"]),
    "optimismo_bootstrap": ("binario", ["optimismo_bootstrap(df, 'compra ~ edad + precio + C(zona) + ruido', n_boot=100).round(4)"]),
    "comparar_modelos_cv": ("binario", ["comparar_modelos_cv(df, 'compra ~ edad + precio', 'compra ~ edad + precio + C(zona)', k=5, repeticiones=3)"]),
    "metricas_regresion": ("regresion", ["m = ajustar_ols('y ~ x', reg)['modelo']\nmetricas_regresion(reg.y, m.fittedvalues)"]),
    "tabla_criterios": ("regresion", ["import statsmodels.formula.api as smf\nms = {'lineal': smf.ols('y ~ x', reg).fit(), 'cuadrático': smf.ols('y ~ x + I(x**2)', reg).fit(), 'cúbico': smf.ols('y ~ x + I(x**2) + I(x**3)', reg).fit()}\ntabla_criterios(ms, ms['cúbico'].scale).round(3)"]),
    "tabla_ganancia_lift": ("logit", ["tabla_ganancia_lift(y, p).round(3)", "grafico_ganancia_lift(y, p);"]),
    "estadistico_ks_gini": ("logit", ["estadistico_ks_gini(y, p)"]),
    "curva_precision_recall": ("logit", ["r = curva_precision_recall(y, p)\nprint({k: v for k, v in r.items() if k != 'tabla'})", "grafico_precision_recall(y, p);"]),
    "descomposicion_brier": ("logit", ["pd.DataFrame({'modelo': descomposicion_brier(y, p), 'descalibrado (×1.6)': descomposicion_brier(y, np.clip(p * 1.6, 0, 1))}).round(4)"]),
    "calibrar_probabilidades": ("logit", ["mitad = len(y) // 2\np_mal = np.clip(p * 1.6, 0, 1)     # probabilidades infladas (p. ej. por entrenar balanceado)\nc = calibrar_probabilidades(y[:mitad], p_mal[:mitad], p_mal[mitad:], metodo='platt')\nprint(f\"media real {y[mitad:].mean():.3f} · inflada {p_mal[mitad:].mean():.3f} · recalibrada {c['probabilidades'].mean():.3f}\")", "grafico_calibracion(y[mitad:], c['probabilidades']);"]),
    "grafico_ganancia_lift": ("logit", ["grafico_ganancia_lift(y, p);"]),
    "grafico_precision_recall": ("logit", ["grafico_precision_recall(y, p);"]),
    # ---------------- modelos (0.9)
    "ajustar_glm_conteo": ("cartera", ["fr = ajustar_glm_conteo('nsin ~ C(zona) + edad', cart, 'poisson', exposicion='expo')\nprint('dispersión de Pearson:', round(fr['dispersion'], 3), '· Cameron-Trivedi p =', round(fr['sobredispersion']['p_valor'], 4))\nfr['tabla'].round(3)",
                                       "nb = ajustar_glm_conteo('nsin ~ C(zona) + edad', cart, 'negbin', exposicion='expo')\nprint(f\"AIC Poisson {fr['aic']:.0f} · binomial negativa {nb['aic']:.0f} (alpha = {nb['alpha_nb']:.2f})\")"]),
    "contraste_sobredispersion": ("cartera", ["fr = ajustar_glm_conteo('nsin ~ C(zona) + edad', cart, 'poisson', exposicion='expo')\ncontraste_sobredispersion(fr['modelo'])"]),
    "ajustar_glm_severidad": ("cartera", ["sev = ajustar_glm_severidad('coste ~ C(zona) + edad', sini, 'gamma')\nprint('dispersión (≈ 1/forma):', round(sev['dispersion'], 3))\nsev['tabla'].round(3)"]),
    "prima_pura": ("cartera", ["nb = ajustar_glm_conteo('nsin ~ C(zona) + edad', cart, 'negbin', exposicion='expo')\nsev = ajustar_glm_severidad('coste ~ C(zona)', sini)\ntarifa = pd.DataFrame({'zona': ['A', 'B', 'C'], 'edad': 40, 'expo': 1.0})\ntarifa.join(prima_pura(nb, sev, tarifa, 'expo')).round(2)"]),
    "tabla_relatividades": ("cartera", ["nb = ajustar_glm_conteo('nsin ~ C(zona) + edad', cart, 'negbin', exposicion='expo')\nr = tabla_relatividades(nb, 'zona')\nr.round(3)", "grafico_relatividades(r, 'Frecuencia por zona');"]),
    "ajustar_tweedie": ("cartera", ["cart['coste_total'] = sini.groupby('poliza').coste.sum().reindex(cart.index, fill_value=0)\ntw = ajustar_tweedie('coste_total ~ C(zona) + edad', cart, 1.5, exposicion='expo')\nprint(f\"{(cart.coste_total == 0).mean():.0%} de pólizas sin coste\")\ntw['tabla'].round(3)"]),
    "ajustar_ols": ("regresion", ["r = ajustar_ols('y ~ x', reg)\nprint(r['avisos']); print(r['diagnosticos'])\nr['tabla'].round(3)"]),
    "medidas_influencia": ("regresion", ["r = ajustar_ols('y ~ x', reg.assign(y=reg.y.where(reg.index != 0, reg.y.max() * 3)))\ninf = medidas_influencia(r)\ninf.sort_values('cook', ascending=False).head().round(3)"]),
    "contraste_f_parcial": ("binario", ["d2 = df.assign(gasto=0.002 * df.precio + 0.05 * df.edad + rng.normal(0, 3, len(df)))\ncontraste_f_parcial(ajustar_ols('gasto ~ precio', d2), ajustar_ols('gasto ~ precio + edad + ruido', d2))"]),
    "transformacion_box_cox": ("ninguno", ["y = rng.lognormal(3, 0.8, 400)\nbc = transformacion_box_cox(y)\nprint({k: v for k, v in bc.items() if k != 'transformada'})\ngrafico_qq(bc['transformada']);"]),
    "ajustar_wls": ("regresion", ["w = ajustar_wls('y ~ x', reg, estimar_pesos=True)\nprint('R² ponderado', round(w['r2'], 3))\nw['tabla']"]),
    "regresion_robusta": ("ninguno", ["d2 = pd.DataFrame({'x': rng.uniform(0, 10, 200)}); d2['y'] = 2 + d2.x + rng.standard_t(1.5, 200)   # colas pesadas\npd.DataFrame({'MCO': ajustar_ols('y ~ x', d2, robusto=None)['tabla'].coef, 'Huber': regresion_robusta('y ~ x', d2)['tabla'].coef,\n              'mediana (LAD)': regresion_robusta('y ~ x', d2, 'cuantil')['tabla'].coef, 'cuantil 90 %': regresion_robusta('y ~ x', d2, 'cuantil', 0.9)['tabla'].coef}).round(3)"]),
    "regresion_no_lineal": ("ninguno", ["# Curva de desarrollo de siniestros: % pagado = a·(1 − e^(−b·t))\nt = np.arange(1, 11); pagado = 0.98 * (1 - np.exp(-0.45 * t)) + rng.normal(0, 0.01, 10)\nr = regresion_no_lineal(lambda t, a, b: a * (1 - np.exp(-b * t)), t, pagado, [1, 0.5], ['a', 'b'])\nprint('R² =', round(r['r2'], 4)); r['tabla']",
                                            "plt.plot(t, pagado, 'o'); plt.plot(t, r['predichos']); plt.title('Ajuste no lineal');"]),
    "contraste_falta_ajuste": ("ninguno", ["rep = pd.DataFrame({'dosis': np.repeat(np.arange(1, 8), 5)}); rep['y'] = (rep.dosis - 4) ** 2 + rng.normal(0, 1, len(rep))\nprint('recta     :', contraste_falta_ajuste(rep, 'dosis', 'y', 1)['p_valor'])\nprint('parábola  :', contraste_falta_ajuste(rep, 'dosis', 'y', 2)['p_valor'])"]),
    "ajustar_glm_splines": ("ninguno", ["d2 = pd.DataFrame({'edad': rng.uniform(18, 80, 3000), 'z': rng.normal(size=3000)})\nd2['sin'] = (rng.random(3000) < 1 / (1 + np.exp(-(-1.5 + 0.002 * (d2.edad - 45) ** 2)))).astype(int)   # forma de U\nr = ajustar_glm_splines(d2, 'sin', {'edad': 4}, 'z')\nprint(f\"AIC spline {r['aic']:.0f} vs lineal {r['aic_lineal']:.0f} · p no linealidad {r['p_no_linealidad']:.2g}\")",
                                            "grafico_efecto_spline(r, 'edad');"]),
    "ajustar_modelo_mixto": ("ninguno", ["# Talleres (grupos) con su propio nivel de coste: intercepto aleatorio\ng = pd.DataFrame({'taller': np.repeat(np.arange(30), 20)}); g['horas'] = rng.normal(5, 1, 600)\ng['coste'] = 200 + 40 * g.horas + np.repeat(rng.normal(0, 60, 30), 20) + rng.normal(0, 50, 600)\nm = ajustar_modelo_mixto('coste ~ horas', g, 'taller')\nprint(f\"ICC = {m['icc']:.2f}\")\nm['efectos_fijos'].round(2)"]),
    "coeficiente_icc": ("ninguno", ["g = pd.DataFrame({'taller': np.repeat(np.arange(30), 20)}); g['coste'] = np.repeat(rng.normal(0, 60, 30), 20) + rng.normal(0, 50, 600)\ncoeficiente_icc(g, 'coste', 'taller')"]),
    "ajustar_logit_ordinal": ("ninguno", ["o = pd.DataFrame({'edad': rng.uniform(20, 70, 1200), 'canal': rng.choice(['web', 'oficina'], 1200)})\nlat = 0.04 * (o.edad - 45) + 0.5 * (o.canal == 'web') + rng.logistic(size=1200)\no['satisfaccion'] = pd.cut(lat, [-np.inf, -1, 0.5, 2, np.inf], labels=['baja', 'media', 'alta', 'muy alta'])\nr = ajustar_logit_ordinal(o, 'satisfaccion', ['edad', 'canal'], orden=['baja', 'media', 'alta', 'muy alta'])\nprint(r['coef_por_corte'].round(2))   # parecidos entre cortes = odds proporcionales\nr['tabla'].round(3)"]),
    "ajustar_regularizado": ("ninguno", ["X = pd.DataFrame(rng.normal(size=(1000, 20)), columns=[f'v{i}' for i in range(20)])\ny = (rng.random(1000) < 1 / (1 + np.exp(-(1.2 * X.v0 - X.v1 + 0.5 * X.v2)))).astype(int)\nr = ajustar_regularizado(X, y, 'lasso')\nprint('variables que quedan:', r['tabla'].query('seleccionada').index.tolist())", "grafico_regularizacion(r);"]),
    "contraste_schoenfeld": ("superv2", ["cx = ajustar_cox(sv, 'meses', 'baja', ['x'], ['tarifa'], {'tarifa': 'A'})\ncontraste_schoenfeld(cx)"]),
    "ajustar_supervivencia_parametrica": ("superv2", ["pd.DataFrame({d: {'AIC': ajustar_supervivencia_parametrica(sv, 'meses', 'baja', ['x'], d, ['tarifa'])['aic']} for d in ['weibull', 'exponencial', 'lognormal', 'loglogistica']}).T.round(1)",
                                                         "w = ajustar_supervivencia_parametrica(sv, 'meses', 'baja', ['x'], 'weibull', ['tarifa'])\nprint('forma Weibull =', round(w['forma_weibull'], 2))\nw['tabla'].round(3)"]),
    "nelson_aalen": ("superv2", ["na = nelson_aalen(sv, 'meses', 'baja', 'tarifa')\nfor g, s in na.groupby('grupo'): plt.step(s.tiempo, s.riesgo_acumulado, where='post', label=g)\nplt.legend(); plt.title('Riesgo acumulado de Nelson-Aalen');"]),
    "incidencia_acumulada": ("superv2", ["c = incidencia_acumulada(sv, 'meses', 'motivo')\nc.tail(3)", "grafico_incidencia_acumulada(c);"]),
    "rmst": ("superv2", ["r = rmst(sv, 'meses', 'baja', tau=12, grupo='tarifa')\nprint(r.attrs)\nr"]),
    "grafico_efecto_spline": ("ninguno", ["d2 = pd.DataFrame({'x': rng.uniform(0, 10, 2000)}); d2['y'] = rng.poisson(np.exp(0.5 + np.sin(d2.x / 2)))\ngrafico_efecto_spline(ajustar_glm_splines(d2, 'y', {'x': 5}, familia='poisson'), 'x');"]),
    "grafico_regularizacion": ("ninguno", ["X = pd.DataFrame(rng.normal(size=(500, 12)), columns=[f'v{i}' for i in range(12)]); y = 2 * X.v0 - X.v3 + rng.normal(size=500)\ngrafico_regularizacion(ajustar_regularizado(X, y, 'lasso', 'gaussiana'));"]),
    "grafico_relatividades": ("cartera", ["grafico_relatividades(tabla_relatividades(ajustar_glm_conteo('nsin ~ C(zona)', cart, 'poisson', 'expo'), 'zona'));"]),
    "grafico_incidencia_acumulada": ("superv2", ["grafico_incidencia_acumulada(incidencia_acumulada(sv, 'meses', 'motivo', 'tarifa'));"]),
    # ---------------- contrastes (0.9)
    "contraste_apareado": ("ninguno", ["# Mismo cliente antes y después de una campaña (pares)\nantes = rng.normal(100, 15, 30); despues = antes + 4 + rng.normal(0, 5, 30)\nr = contraste_apareado(antes, despues)\nprint(r['interpretacion'])\nprint('t de grupos independientes (MAL aquí): p =', round(stats.ttest_ind(antes, despues).pvalue, 3))"]),
    "contraste_friedman": ("ninguno", ["# 20 tasadores valoran 3 siniestros: ¿alguno se valora sistemáticamente más alto?\nd = pd.DataFrame({'tasador': np.repeat(range(20), 3), 'siniestro': np.tile(['A', 'B', 'C'], 20)})\nd['valor'] = d.siniestro.map({'A': 10, 'B': 10.5, 'C': 12}) + np.repeat(rng.normal(0, 2, 20), 3) + rng.normal(0, 1, 60)\ncontraste_friedman(d, 'tasador', 'siniestro', 'valor')"]),
    "contraste_mcnemar": ("ninguno", ["# ¿Cambia la tasa de impago tras una medida? (mismos clientes, 0/1)\nantes = (rng.random(300) < 0.20).astype(int)\ndespues = np.where(antes == 1, (rng.random(300) < 0.6).astype(int), (rng.random(300) < 0.05).astype(int))\ncontraste_mcnemar(antes, despues)"]),
    "contraste_permutacion": ("ninguno", ["a, b = rng.exponential(1, 25), rng.exponential(1.8, 25)\nr = contraste_permutacion(a, b, 'mediana', n_perm=5000)\nprint(f\"diferencia de medianas {r['estadistico']:.3f} · p = {r['p_valor']:.4f}\")\nplt.hist(r['distribucion_nula'], bins=60, alpha=.6); plt.axvline(r['estadistico'], color='C1'); plt.axvline(-r['estadistico'], color='C1'); plt.title('Distribución bajo H0 (permutaciones)');"]),
    "contraste_jonckheere": ("ninguno", ["# Frecuencia de siniestros por tramo de antigüedad del carné (¿tendencia decreciente?)\nd = pd.DataFrame({'tramo': np.repeat(['0-2', '3-5', '6-10', '>10'], 80)})\nd['frecuencia'] = rng.gamma(2, d.tramo.map({'0-2': .12, '3-5': .10, '6-10': .085, '>10': .08}))\ncontraste_jonckheere(d, 'frecuencia', 'tramo', orden=['0-2', '3-5', '6-10', '>10'], alternativa='decreasing')"]),
    "posthoc_dunn": ("grupos", ["print(elegir_contraste(d.assign(valor=np.exp(d.valor / 3)), 'valor', 'grupo')['contraste'])\nposthoc_dunn(d, 'valor', 'grupo')"]),
    "games_howell": ("ninguno", ["d2 = pd.DataFrame({'g': np.repeat(['a', 'b', 'c'], 40), 'v': np.r_[rng.normal(10, 1, 40), rng.normal(10.5, 1, 40), rng.normal(12, 5, 40)]})\nprint(homogeneidad_varianzas(d2, 'v', 'g')['p_valor'].round(4).to_dict())\ngames_howell(d2, 'v', 'g')"]),
    "intervalo_proporcion": ("ninguno", ["# 3 siniestros graves en 120 pólizas: los métodos discrepan con proporciones pequeñas\npd.DataFrame([intervalo_proporcion(3, 120, metodo=m) for m in ['wald', 'wilson', 'agresti_coull', 'jeffreys', 'clopper_pearson']]).set_index('metodo').round(4)"]),
    "bootstrap_ic": ("ninguno", ["importes = rng.lognormal(7, 1.1, 150)\nr = bootstrap_ic(importes, np.mean, metodo='bca')\nprint({k: round(v, 1) for k, v in r.items() if k != 'replicas' and isinstance(v, float)})\nplt.hist(r['replicas'], bins=60, alpha=.6); plt.axvline(r['ic_inf'], color='C1'); plt.axvline(r['ic_sup'], color='C1'); plt.title('Bootstrap BCa de la media (asimétrica)');"]),
    "bondad_ajuste_multinomial": ("ninguno", ["# ¿Los siniestros se reparten igual entre los días de la semana?\nobs = [180, 150, 148, 152, 160, 205, 210]\nr = bondad_ajuste_multinomial(obs, nombres=['L', 'M', 'X', 'J', 'V', 'S', 'D'])\nprint(f\"chi2 = {r['chi2']:.1f}, p = {r['p_valor']:.3g}\")\nr['tabla'].round(2)"]),
    # ---------------- descriptiva
    "resumen_descriptivo": ("binario", ["resumen_descriptivo(df).round(2).T"]),
    "detectar_atipicos": ("ninguno", ["x = np.r_[rng.normal(50, 5, 200), [95, 110, 4]]\nfor m in ['iqr', 'mad', 'z']:\n    a = detectar_atipicos(x, m)\n    print(f\"{m:>3}: {a.atipico.sum()} atípicos · límites {np.round(a.attrs['limites'], 1)}\")"]),
    "correlacion_con_ic": ("ninguno", ["x = rng.normal(size=300); y = 0.4 * x + rng.normal(size=300)\n{m: round(correlacion_con_ic(x, y, m)['r'], 3) for m in ['pearson', 'spearman', 'kendall']}",
                                        "correlacion_con_ic(x, y)"]),
    "matriz_correlaciones": ("binario", ["m = matriz_correlaciones(df, ['edad', 'precio', 'ruido', 'compra'], 'spearman')\nm['r'].round(3)",
                                          "grafico_matriz_correlaciones(m['r'], m['p_valor']);"]),
    "correlacion_parcial": ("ninguno", ["# x e y solo se parecen porque ambas dependen de z (confusión)\nz = rng.normal(size=500)\nd = pd.DataFrame({'z': z, 'x': z + rng.normal(size=500), 'y': z + rng.normal(size=500)})\ncorrelacion_parcial(d, 'x', 'y', ['z'])"]),
    "contraste_normalidad": ("ninguno", ["r = contraste_normalidad(rng.lognormal(0, 0.5, 200))\nprint(r['avisos'])\nr['tabla']", "grafico_qq(rng.lognormal(0, 0.5, 200));"]),
    "homogeneidad_varianzas": ("grupos", ["homogeneidad_varianzas(d, 'valor', 'grupo')"]),
    "ajustar_distribuciones": ("ninguno", ["# Importes de siniestros simulados (lognormales): ¿qué distribución encaja?\nimportes = rng.lognormal(7, 0.9, 800)\nt = ajustar_distribuciones(importes)\nt[['distribucion', 'AIC', 'delta_AIC', 'p_KS']]", "grafico_ajuste_distribuciones(importes, t);"]),
    "ajustar_distribucion_discreta": ("ninguno", ["# Nº de siniestros por póliza con heterogeneidad (binomial negativa)\nn_sin = rng.negative_binomial(1.5, 0.6, 3000)\nt = ajustar_distribucion_discreta(n_sin)\nprint(t.attrs)\nt"]),
    "indice_dispersion": ("ninguno", ["print(indice_dispersion(rng.poisson(2, 500))['interpretacion'])\nprint(indice_dispersion(rng.negative_binomial(1, 1 / 3, 500))['interpretacion'])"]),
    "funcion_distribucion_empirica": ("ninguno", ["e = funcion_distribucion_empirica(rng.exponential(2, 150))\nplt.step(e.x, e.F, where='post', label='F empírica'); plt.fill_between(e.x, e.banda_inf, e.banda_sup, step='post', alpha=0.25, label='banda DKW 95 %')\nxs = np.linspace(0, e.x.max(), 200); plt.plot(xs, stats.expon(scale=2).cdf(xs), label='verdadera'); plt.legend();"]),
    "estimar_densidad": ("ninguno", ["x = np.r_[rng.normal(0, 1, 300), rng.normal(4, 0.7, 150)]\nfor a in [0.1, 'scott', 1.0]:\n    d = estimar_densidad(x, ancho=a); plt.plot(d.x, d.densidad, label=f'ancho {a}')\nplt.legend(); plt.title('El ancho de banda decide si ves la bimodalidad');"]),
    "tabla_contingencia": ("binario", ["t = tabla_contingencia(df, 'zona', 'compra')\nprint(f\"chi2 = {t['chi2']:.1f}, p = {t['p_valor']:.2g}, V = {t['cramer_v']:.3f}\")\nt['residuos_ajustados'].round(2)"]),
    "medidas_riesgo_2x2": ("ninguno", ["# 30 de 100 expuestos frente a 15 de 100 no expuestos\nmedidas_riesgo_2x2(30, 100, 15, 100)"]),
    "grafico_distribucion": ("ninguno", ["grafico_distribucion(rng.gamma(2, 1.5, 600));"]),
    "grafico_ajuste_distribuciones": ("ninguno", ["grafico_ajuste_distribuciones(rng.weibull(1.5, 500) * 3);"]),
    "grafico_matriz_correlaciones": ("binario", ["m = matriz_correlaciones(df, ['edad', 'precio', 'ruido', 'compra'])\ngrafico_matriz_correlaciones(m['r'], m['p_valor']);"]),
    # ---------------- simulación: bayes (0.10)
    "posterior_conjugado": ("ninguno", ["# 7 compras de 20 visitas; previa Beta(2, 8) (≈ 20 % de conversión histórica)\nb = posterior_conjugado('beta_binomial', (7, 20), (2, 8))\nprint(f\"media posterior {b['media']:.3f} · IC {b['ic'][0]:.3f}-{b['ic'][1]:.3f} · peso de los datos Z = {b['credibilidad_Z']:.2f}\")\nx = np.linspace(0, 1, 300); plt.plot(x, stats.beta(2, 8).pdf(x), label='previa'); plt.plot(x, b['distribucion'].pdf(x), label='posterior'); plt.legend();"]),
    "bayes_empirico_beta": ("ninguno", ["n = rng.integers(5, 300, 25); x = rng.binomial(n, rng.beta(10, 40, 25))   # 25 oficinas\nr = bayes_empirico_beta(x, n)\nprint(r['previa'])\nr['tabla'].sort_values('ensayos').head(8).round(3)"]),
    "metropolis": ("ninguno", ["# Posterior de (μ, log σ) de una normal con previas planas\ny = rng.normal(10, 3, 50)\nlp = lambda th: stats.norm.logpdf(y, th[0], np.exp(th[1])).sum()\nr = metropolis(lp, [0.0, 0.0], n_iter=3000, nombres=['mu', 'log_sigma'])\nprint('aceptación', round(r['aceptacion'], 2), r['avisos'])\nr['resumen'].round(3)", "grafico_trazas_mcmc(r);"]),
    "diagnostico_mcmc": ("ninguno", ["buenas = rng.normal(0, 1, (4, 1000))\nmalas = np.stack([rng.normal(k, 1, 1000) for k in range(4)])   # cada cadena en un sitio\npd.concat({'buenas': diagnostico_mcmc(buenas), 'malas': diagnostico_mcmc(malas)}).round(3)"]),
    "chequeo_predictivo": ("ninguno", ["# ¿Vale un Poisson para conteos sobredispersos? Chequeo predictivo posterior\ny = rng.negative_binomial(1, 0.2, 300)\nlam = rng.gamma(1 + y.sum(), 1 / len(y), 400)               # posterior de λ\nchequeo_predictivo(y, lam, lambda l, n, r: r.poisson(l, n)).round(3)"]),
    "accion_bayes": ("ninguno", ["post = rng.lognormal(10, 0.5, 20000)         # posterior de la pérdida esperada\npd.Series({p if isinstance(p, str) else 'asimétrica k=3': accion_bayes(post, p)['accion'] for p in ['cuadratica', 'absoluta', ('asimetrica', 3)]}).round(0)",
                                       "L = pd.DataFrame([[0, 50], [10, 10]], index=['aceptar', 'rechazar'], columns=['bueno', 'malo'])\naccion_bayes([0.85, 0.15], L)"]),
    "ab_bayesiano": ("ninguno", ["ab_bayesiano(120, 2400, 151, 2380)"]),
    "simular_bandido": ("ninguno", ["P = [0.04, 0.05, 0.07]\nres = {e: simular_bandido(P, 3000, e, repeticiones=5) for e in ['uniforme', 'epsilon', 'ucb', 'thompson']}\nprint({e: r['jugadas'].round(0).tolist() for e, r in res.items()})", "grafico_bandido(res);"]),
    # ---------------- simulación: procesos
    "cadena_markov": ("ninguno", ["P = pd.DataFrame([[0.9, 0.08, 0.02], [0.2, 0.7, 0.1], [0, 0, 1]], index=['al día', 'impago', 'baja'], columns=['al día', 'impago', 'baja'])\nr = cadena_markov(P, inicial='al día', pasos=12)\nprint(r['clases'])\nprint(r['prob_absorcion']); print(r['tiempo_hasta_absorcion'])"]),
    "simular_cadena_markov": ("ninguno", ["P = pd.DataFrame([[0.9, 0.1], [0.5, 0.5]], index=['sol', 'lluvia'], columns=['sol', 'lluvia'])\ns = simular_cadena_markov(P, 5000, 'sol', n_trayectorias=3)\ns.apply(lambda c: c.value_counts(normalize=True)).round(3)   # → (5/6, 1/6)"]),
    "bonus_malus": ("ninguno", ["coef = [0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5]\nreglas = [[max(i - 1, 0), min(i + 2, 6), min(i + 4, 6), 6] for i in range(7)]   # −1 sin siniestros, +2 por siniestro\nfor lam in [0.05, 0.1, 0.2]:\n    r = bonus_malus(coef, reglas, lam)\n    print(f\"λ = {lam}: coeficiente medio {r['coeficiente_medio']:.3f} · eficiencia de Loimaranta {r['eficiencia_loimaranta']:.2f}\")\nr['evolucion'].coeficiente_medio.plot(title='Convergencia al estacionario');"]),
    "cadena_markov_continua": ("ninguno", ["# Sano ↔ inválido → fallecido (tasas anuales)\nQ = pd.DataFrame([[-0.12, 0.10, 0.02], [0.30, -0.40, 0.10], [0, 0, 0]], index=['sano', 'inválido', 'muerto'], columns=['sano', 'inválido', 'muerto'])\ncadena_markov_continua(Q, t=5)['P_t'].round(4)"]),
    "nacimiento_muerte": ("ninguno", ["# Centro de llamadas: 20 llamadas/h, 6 por agente/h\npd.DataFrame({c: {k: v for k, v in nacimiento_muerte(20, 6, servidores=c).items() if k != 'distribucion'} for c in [4, 5, 6]}).round(3)"]),
    "simular_proceso_poisson": ("ninguno", ["t = simular_proceso_poisson(lambda s: 5 + 4 * np.sin(2 * np.pi * s / 24), 72, tasa_maxima=9)[0]\nplt.hist(t % 24, bins=24); plt.title(f'{len(t)} llegadas en 72 h: patrón diario (no homogéneo)'); plt.xlabel('hora');"]),
    "contraste_proceso_poisson": ("ninguno", ["t = simular_proceso_poisson(2.0, 200)[0]\ncontraste_proceso_poisson(t, 200).round(3)"]),
    "ruina_jugador": ("ninguno", ["pd.DataFrame({p: ruina_jugador(20, 100, p) for p in [0.45, 0.49, 0.5, 0.51]}).round(4)"]),
    "paseo_aleatorio": ("ninguno", ["r = paseo_aleatorio(200, 0.52, 3000)\nr['martingalas'].plot(title='Medias de las martingalas: constantes (E[M_n] = M_0)');"]),
    "simular_browniano": ("ninguno", ["b = simular_browniano(1, 250, 2000, mu=0.08, sigma=0.25, inicio=100, geometrico=True)\nprint(b['comprobacion'].round(3))", "grafico_trayectorias(b['trayectorias'], titulo='Browniano geométrico');"]),
    # ---------------- simulación: Monte Carlo y probabilidad
    "generador_congruencial": ("ninguno", ["u = generador_congruencial(20000)\nr = generador_congruencial(20000, 1, 65539, 0, 2 ** 31)   # RANDU\npd.concat({'LCG bueno': contrastes_aleatoriedad(u).p_valor, 'RANDU': contrastes_aleatoriedad(r).p_valor}, axis=1).round(4)"]),
    "contrastes_aleatoriedad": ("ninguno", ["contrastes_aleatoriedad(rng.random(30000)).round(4)"]),
    "generar_por_inversion": ("ninguno", ["# Pareto por inversión: F⁻¹(u) = x_m (1 − u)^(−1/α)\nx = generar_por_inversion(20000, cuantil=lambda u: 1000 * (1 - u) ** (-1 / 2.5))\nprint(stats.kstest(x, stats.pareto(2.5, scale=1000).cdf))"]),
    "generar_por_aceptacion_rechazo": ("ninguno", ["f = lambda x: np.exp(-x ** 4)        # densidad sin normalizar\nr = generar_por_aceptacion_rechazo(20000, f, stats.norm(0, 0.8))\nprint('aceptación', round(r['tasa_aceptacion'], 3))\nplt.hist(r['muestra'], bins=60, density=True);"]),
    "generar_normal_multivariante": ("ninguno", ["S = pd.DataFrame([[1, .8, .3], [.8, 1, .5], [.3, .5, 1]])\nX = generar_normal_multivariante(pd.Series([0, 1, 2], index=['a', 'b', 'c']), S, 10000)\nX.corr().round(3)"]),
    "estimar_montecarlo": ("ninguno", ["f = lambda u: np.exp(-u ** 2)        # ∫₀¹ e^{−u²} du\npd.DataFrame({m: estimar_montecarlo(f, 10000, metodo=m, control=lambda u: 1 - u ** 2, media_control=2 / 3) for m in ['simple', 'antiteticas', 'control']}).T"]),
    "teorema_bayes": ("ninguno", ["teorema_bayes(0.005, (0.98, 0.97)).round(4)    # cribado de una enfermedad rara"]),
    "analizar_distribucion_conjunta": ("ninguno", ["# nº de siniestros (filas) × nº de reclamaciones (columnas)\nT = pd.DataFrame([[300, 20, 5], [80, 60, 15], [10, 25, 30]], index=[0, 1, 2], columns=[0, 1, 2])\nr = analizar_distribucion_conjunta(T)\nprint(r['esperanza_y_dado_x'].round(3)); print('cov', round(r['covarianza'], 3), '·', r['chi2_independencia'])"]),
    "distribucion_estadistico_orden": ("ninguno", ["# El mayor de 50 siniestros exponenciales de media 1000\nr = distribucion_estadistico_orden(stats.expon(scale=1000), 50, 50, n_sim=5000)\nprint(round(r['media']), round(r['media_simulada']), [round(v) for v in r['ic95']])"]),
    "momentos_distribucion": ("ninguno", ["pd.DataFrame({n: {k: v for k, v in momentos_distribucion(d).items() if k in ('media', 'varianza', 'asimetria', 'curtosis_exceso', 'fgm')} for n, d in [('gamma(2,1)', stats.gamma(2)), ('lognormal(0,1)', stats.lognorm(1)), ('Poisson(3)', stats.poisson(3))]})"]),
    "convergencia_media_muestral": ("ninguno", ["r = convergencia_media_muestral(stats.lognorm(1.2), (1, 5, 30, 200, 1000), 3000)\nprint(r['tabla'].round(3))\nr['media_acumulada'].plot(logx=True); plt.axhline(r['media_teorica'], color='gray'); plt.title('Ley de los grandes números');"]),
    # ---------------- simulación: inferencia
    "metodo_momentos": ("ninguno", ["x = stats.gamma(2.5, scale=400).rvs(500, random_state=rng)\nmetodo_momentos(x, 'gamma')"]),
    "informacion_fisher": ("ninguno", ["ld = lambda x, l: stats.poisson.logpmf(x, l)\ninformacion_fisher(ld, 3.0, distribucion=stats.poisson(3), n=100)   # I(λ) = 1/λ"]),
    "comparar_estimadores": ("ninguno", ["# Uniforme(0, θ): 2·media frente al máximo corregido (n+1)/n·max (usa el suficiente)\ncomparar_estimadores({'2·media': lambda x: 2 * x.mean(), '(n+1)/n·max': lambda x: (len(x) + 1) / len(x) * x.max(), 'max': np.max},\n                     lambda n, r: r.uniform(0, 10, n), 10.0, n=20, n_sim=4000).round(4)"]),
    "metodo_delta": ("ninguno", ["# EE de un odds ratio a partir del coeficiente logit β = 0.4 (EE 0.1)\nmetodo_delta(np.exp, 0.4, [[0.01]])"]),
    "distribucion_muestral_simulada": ("ninguno", ["r = distribucion_muestral_simulada('t', 5)\nprint(r['cuantiles'].round(3)); print(distribucion_muestral_simulada('varianza', 5)['tabla'].round(3))"]),
    "simular_regresion_a_la_media": ("ninguno", ["simular_regresion_a_la_media(correlacion=0.4, cuantil_seleccion=0.95)"]),
    "coste_de_dicotomizar": ("ninguno", ["coste_de_dicotomizar(0.25, n=150, n_sim=500)"]),
    # ---------------- simulación: muestreo
    "extraer_muestra": ("ninguno", ["marco = pd.DataFrame({'zona': rng.choice(['norte', 'sur', 'centro'], 10000, p=[.5, .3, .2]), 'prima': rng.gamma(2, 300, 10000)})\nm = extraer_muestra(marco, 500, 'estratificada', estrato='zona')\nm.groupby('zona').agg(n=('prima', 'size'), peso=('peso', 'first'))"]),
    "estimar_mas": ("ninguno", ["muestra = rng.gamma(2, 300, 400)\nestimar_mas(muestra, N=5000).round(2)"]),
    "tamano_muestra_encuesta": ("ninguno", ["pd.DataFrame({e: tamano_muestra_encuesta(e, N=20000, tasa_respuesta=0.35) for e in [0.05, 0.03, 0.02]})"]),
    "asignacion_estratos": ("ninguno", ["t = asignacion_estratos({'particulares': 8000, 'pymes': 1800, 'grandes': 200}, {'particulares': 300, 'pymes': 2000, 'grandes': 20000}, 600)\nprint(t.attrs)\nt.round(1)"]),
    "estimar_estratificado": ("ninguno", ["N = {'a': 6000, 'b': 3000, 'c': 1000}\nmuestra = pd.concat([pd.DataFrame({'estrato': h, 'y': rng.gamma(2, m, k)}) for h, m, k in [('a', 100, 100), ('b', 400, 100), ('c', 2000, 100)]])\nr = estimar_estratificado(muestra, 'y', 'estrato', N)\n{k: r[k] for k in ['media', 'error_estandar', 'ic', 'efecto_diseno']}"]),
    "estimador_razon": ("ninguno", ["x = rng.gamma(2, 500, 300); y = 1.08 * x + rng.normal(0, 60, 300)   # prima de este año vs la del pasado\nestimador_razon(y, x, media_x_poblacional=1010, N=20000)"]),
    "ajuste_no_respuesta": ("ninguno", ["d = pd.DataFrame({'edad': rng.choice(['<35', '35-60', '>60'], 3000)})\nd['satisfaccion'] = d.edad.map({'<35': 6, '35-60': 7, '>60': 8.5}) + rng.normal(0, 1, 3000)\nd['responde'] = rng.random(3000) < d.edad.map({'<35': 0.15, '35-60': 0.4, '>60': 0.7})\nr = ajuste_no_respuesta(d, 'responde', 'edad')\nR = r['respondentes']\nprint(f\"real {d.satisfaccion.mean():.2f} · sin ajustar {R.satisfaccion.mean():.2f} · ajustada {np.average(R.satisfaccion, weights=R.peso_ajustado):.2f}\")\nr['por_celda']"]),
    # ---------------- diseño (0.10)
    "anova_factorial": ("ninguno", ["d = pd.DataFrame([(c, t) for c in ['web', 'tienda', 'tel'] for t in ['A', 'B'] for _ in range(25)], columns=['canal', 'tarifa'])\nd['venta'] = 10 + (d.canal == 'web') * 2 + (d.tarifa == 'B') * 1 + ((d.canal == 'web') & (d.tarifa == 'B')) * 2 + rng.normal(0, 2, len(d))\nr = anova_factorial(d, 'venta', ['canal', 'tarifa'])\nr['tabla'].round(4)", "grafico_interaccion(d, 'venta', 'canal', 'tarifa');"]),
    "anova_bloques": ("ninguno", ["d = pd.DataFrame([(s, t) for s in range(8) for t in ['a', 'b', 'c']], columns=['sucursal', 'campaña'])\nd['ventas'] = d.sucursal * 3 + d['campaña'].map({'a': 0, 'b': 1, 'c': 2.5}) + rng.normal(0, 1, len(d))\nr = anova_bloques(d, 'ventas', 'campaña', 'sucursal')\nprint('eficiencia relativa de bloquear:', round(r['eficiencia_relativa'], 1))\nr['tabla'].round(4)"]),
    "ancova": ("ninguno", ["n = 200; g = rng.choice(['control', 'curso'], n); pre = rng.normal(60, 10, n)\npost = 10 + 0.8 * pre + 4 * (g == 'curso') + rng.normal(0, 4, n)\nr = ancova(pd.DataFrame({'g': g, 'pre': pre, 'post': post}), 'post', 'g', 'pre')\nprint('p homogeneidad de pendientes', round(r['p_homogeneidad_pendientes'], 3))\nr['medias_ajustadas'].round(2)"]),
    "anova_medidas_repetidas": ("ninguno", ["filas = [{'cliente': i, 'trimestre': f'T{k}', 'grupo': 'A' if i < 15 else 'B', 'gasto': u + 2 * k * (i >= 15) + rng.normal(0, 1 + k / 2)}\n         for i, u in enumerate(rng.normal(50, 5, 30)) for k in range(4)]\nr = anova_medidas_repetidas(pd.DataFrame(filas), 'cliente', 'trimestre', 'gasto', entre='grupo')\nprint(r['esfericidad'], r['avisos'])\nr['tabla'].round(4)"]),
    "anova_anidado": ("ninguno", ["filas = [{'region': r_, 'oficina': o, 'tiempo': u + (r_ == 'sur') * 1 + rng.normal(0, 1)} for r_ in ['norte', 'sur'] for o in range(5) for u in [rng.normal(0, 2)] for _ in range(12)]\nr = anova_anidado(pd.DataFrame(filas), 'tiempo', 'region', 'oficina')\nprint(r['tabla'].round(4)); r['componentes_varianza'].round(3)"]),
    "diagnostico_anova": ("ninguno", ["d = pd.DataFrame({'g': np.repeat(list('abcd'), 30)}); d['y'] = rng.lognormal(d.g.map({'a': 1, 'b': 1.4, 'c': 1.8, 'd': 2.2}), 0.5)\ndiagnostico_anova(d, 'y', 'g')"]),
    "diseno_factorial_2k": ("ninguno", ["f = diseno_factorial_2k(5, {'E': 'ABCD'}, aleatorizar=False)\nprint('resolución', f['resolucion'], '·', f['n_ensayos'], 'ensayos'); print(f['alias'].head(8))\nf['diseno'].head()"]),
    "efectos_factorial_2k": ("ninguno", ["D = diseno_factorial_2k(4, aleatorizar=False)['diseno']\ny = 60 + 5 * D.A + 3 * D.C - 2.5 * D.A * D.C + rng.normal(0, 0.8, len(D))\nr = efectos_factorial_2k(D.assign(y=y), 'y', list('ABCD'))\nprint('PSE de Lenth', round(r['pse_lenth'], 3), '· ME', round(r['me'], 2))\nr['tabla'].head(6).round(3)", "grafico_efectos_2k(r);"]),
    "cuadrado_latino": ("ninguno", ["cuadrado_latino(4, ['A', 'B', 'C', 'D']).pivot(index='fila', columns='columna', values='tratamiento')"]),
    "anova_cuadrado_latino": ("ninguno", ["L = cuadrado_latino(5)\nL['y'] = L.fila + 0.5 * L.columna + L.tratamiento.map(dict(zip('ABCDE', [0, 0, 1, 2, 3]))) + rng.normal(0, 0.5, 25)\nanova_cuadrado_latino(L, 'y')['tabla'].round(4)"]),
    "diseno_central_compuesto": ("ninguno", ["diseno_central_compuesto(2, centros=5).round(3)"]),
    "superficie_respuesta": ("ninguno", ["D = diseno_central_compuesto(2, centros=5, nombres=['temperatura', 'tiempo'])\ny = 90 - 3 * (D.temperatura - 0.5) ** 2 - 2 * (D.tiempo + 0.2) ** 2 + rng.normal(0, 0.3, len(D))\nr = superficie_respuesta(D.assign(rendimiento=y), 'rendimiento', ['temperatura', 'tiempo'])\nprint(r['tipo'], r['punto_estacionario'].round(3).to_dict(), round(r['respuesta_estacionaria'], 2))", "grafico_superficie_respuesta(r);"]),
    "puntuacion_propension": ("ninguno", ["n = 3000; edad = rng.normal(45, 12, n); renta = rng.normal(30, 8, n)\ne = 1 / (1 + np.exp(-(-2 + 0.04 * edad + 0.03 * renta))); t = (rng.random(n) < e).astype(int)\nd = pd.DataFrame({'t': t, 'edad': edad, 'renta': renta, 'y': 5 + 2 * t + 0.1 * edad + 0.2 * renta + rng.normal(0, 2, n)})\nr = puntuacion_propension(d, 't', 'y', ['edad', 'renta'], n_boot=100)\nprint(f\"ingenuo {r['efecto_ingenuo']:.2f} · IPW {r['efecto']:.2f} IC {r['ic'][0]:.2f}-{r['ic'][1]:.2f} (verdadero 2)\")", "grafico_balance(r);"]),
    "diferencias_en_diferencias": ("ninguno", ["filas = [{'tienda': u, 'tratada': int(u < 30), 'semana': s, 'post': int(s >= 6), 'ventas': 100 + 5 * (u < 30) + s + 8 * (u < 30) * (s >= 6) + rng.normal(0, 3)} for u in range(60) for s in range(10)]\nr = diferencias_en_diferencias(pd.DataFrame(filas), 'ventas', 'tratada', 'post', cluster='tienda', periodo='semana')\nprint(round(r['efecto'], 2), [round(v, 2) for v in r['ic']], r['tendencias_previas'])\nr['medias_2x2'].round(1)"]),
    "aleatorizar_ensayo": ("ninguno", ["aleatorizar_ensayo(12, ('placebo', 'fármaco'), tamano_bloque=4)"]),
    "analisis_intencion_tratar": ("ninguno", ["n = 4000; z = rng.integers(0, 2, n); sano = rng.normal(size=n)\nt = np.where(rng.random(n) < 0.7, z, (sano > 0).astype(int))      # 30 % no cumple y elige según su salud\ny = 1.0 * t + sano + rng.normal(size=n)\nanalisis_intencion_tratar(pd.DataFrame({'asignado': z, 'recibido': t, 'y': y}), 'asignado', 'recibido', 'y').round(3)"]),
    "mediacion": ("ninguno", ["n = 1000; formacion = rng.normal(size=n); confianza = 0.6 * formacion + rng.normal(size=n)\nventas = 0.5 * confianza + 0.2 * formacion + rng.normal(size=n)\nmediacion(pd.DataFrame({'formacion': formacion, 'confianza': confianza, 'ventas': ventas}), 'formacion', 'confianza', 'ventas', n_boot=500)"]),
    "moderacion": ("ninguno", ["n = 800; precio = rng.normal(size=n); renta = rng.normal(size=n)\ncompra = 2 - 0.8 * precio + 0.3 * renta + 0.4 * precio * renta + rng.normal(size=n)\nr = moderacion(pd.DataFrame({'precio': precio, 'renta': renta, 'compra': compra}), 'compra', 'precio', 'renta')\nprint('Johnson-Neyman:', np.round(r['johnson_neyman'], 2))\nr['pendientes_simples'].round(3)"]),
    "metaanalisis": ("ninguno", ["efectos = [-0.89, -1.59, -1.35, -1.44, -0.22, -0.79, -1.62, 0.01, -0.47, -1.37, -0.34, 0.45, -0.02]\nee = np.sqrt([0.33, 0.19, 0.42, 0.02, 0.05, 0.01, 0.22, 0.004, 0.06, 0.07, 0.01, 0.53, 0.07])\nm = metaanalisis(efectos, ee, 'reml')\nprint({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k in ('efecto', 'ic', 'I2', 'tau2', 'intervalo_prediccion', 'egger')})", "grafico_metaanalisis(m);"]),
    # ---------------- regresión y modelos (0.10)
    "bandas_confianza_regresion": ("ninguno", ["x = rng.uniform(0, 10, 60); y = 3 + 0.7 * x + rng.normal(0, 1.5, 60)\nb = bandas_confianza_regresion(x, y)\nprint('multiplicador W-H', round(b.multiplicador.iloc[0], 3), 'vs t', round(stats.t.ppf(.975, 58), 3))", "grafico_bandas_regresion(b, x, y);"]),
    "regresion_inversa": ("ninguno", ["patron = np.repeat([0, 2, 4, 6, 8, 10], 3); lectura = 0.5 + 1.9 * patron + rng.normal(0, 0.4, 18)\nregresion_inversa(patron, lectura, y0=12.3, m_replicas=2)"]),
    "regresion_por_origen": ("ninguno", ["x = rng.uniform(1, 20, 50); y = 2.5 * x + rng.normal(0, 2, 50)\nregresion_por_origen(x, y)"]),
    "correccion_error_medida": ("ninguno", ["X = rng.normal(0, 1, 3000); Xobs = X + rng.normal(0, 0.8, 3000); y = 1.5 * X + rng.normal(0, 1, 3000)\nr = correccion_error_medida(Xobs, y, var_error=0.64)\nprint(round(r['pendiente_ingenua'], 3), '→ SIMEX', round(r['pendiente_corregida'], 3), '· calibración', round(r['pendiente_calibracion'], 3), '(verdadera 1.5)')\nr['curva_simex'].plot(marker='o', title='SIMEX: pendiente según error añadido');"]),
    "regresion_polinomica": ("ninguno", ["x = rng.uniform(-3, 3, 200); y = 2 + x - 0.5 * x ** 2 + rng.normal(0, 1, 200)\nr = regresion_polinomica(x, y)\nprint('grado elegido', r['grado'])\nr['tabla'].round(4)"]),
    "tamano_muestral_modelo": ("ninguno", ["t = tamano_muestral_modelo(n_parametros=15, tipo='binario', prevalencia=0.08, r2=0.10)\nprint(t.attrs)\nt"]),
    "regresion_local": ("ninguno", ["x = rng.uniform(18, 80, 500); y = 0.02 * (x - 45) ** 2 + 5 * (x > 60) + rng.normal(0, 3, 500)\nr = regresion_local(x, y)\nprint('fracción elegida por CV', r['parametro'])\nplt.scatter(x, y, s=5, alpha=0.4); plt.plot(r['x'], r['ajuste'], color='C1', lw=2); plt.title('LOESS');"]),
    "funciones_escalonadas": ("ninguno", ["d = pd.DataFrame({'edad': rng.uniform(18, 80, 2000)}); d['siniestro'] = (rng.random(2000) < 0.05 + 0.1 * (d.edad < 25)).astype(int)\nr = funciones_escalonadas(d, 'siniestro', 'edad', [18, 25, 35, 50, 65, 80], familia='binomial')\nprint('AIC escalonado', round(r['aic_escalonado'], 1), 'vs lineal', round(r['aic_lineal'], 1))\nr['tabla'].round(3)"]),
    "ajustar_mars": ("ninguno", ["X = pd.DataFrame({'edad': rng.uniform(18, 80, 800), 'km': rng.uniform(0, 40, 800)})\ny = 3 * np.maximum(0, 25 - X.edad) + 0.5 * np.maximum(0, X.km - 20) + rng.normal(0, 2, 800)\nm = ajustar_mars(X, y, max_terminos=9)\nprint('R²', round(m['r2'], 3))\nm['terminos']"]),
    "regresion_inversa_cortes": ("ninguno", ["X = pd.DataFrame(rng.normal(size=(800, 5)), columns=list('abcde'))\ny = np.exp(0.8 * X.a - 0.6 * X.c) + rng.normal(0, 0.2, 800)\nr = regresion_inversa_cortes(X, y, n_direcciones=2)\nprint(r['autovalores'].round(3)); r['direcciones'].round(3)"]),
    "regresion_pls_pcr": ("ninguno", ["lat = rng.normal(size=(200, 3)); X = pd.DataFrame(lat @ rng.normal(size=(3, 40)) + rng.normal(0, 0.3, (200, 40)))\ny = lat @ [1, -0.5, 0.2] + rng.normal(0, 0.3, 200)\np, q = regresion_pls_pcr(X, y, 'pls'), regresion_pls_pcr(X, y, 'pcr')\npd.DataFrame({'PLS': p['curva_cv'], 'PCR': q['curva_cv']}).head(10).plot(marker='o', title='RMSE de CV por nº de componentes');"]),
    "modelo_loglineal": ("ninguno", ["d = pd.DataFrame({'canal': rng.choice(['web', 'oficina'], 2000)})\nd['producto'] = np.where(rng.random(2000) < np.where(d.canal == 'web', 0.7, 0.4), 'auto', 'hogar'); d['zona'] = rng.choice(['n', 's'], 2000)\nfor m in ['independencia', 'canal*producto + zona']:\n    r = modelo_loglineal(d, ['canal', 'producto', 'zona'], modelo=m)\n    print(f\"{m}: G² = {r['G2']:.1f} con {r['gl']} gl, p = {r['p_valor']:.3g}\")"]),
    "tabla_nomograma": ("logit", ["import statsmodels.formula.api as smf\nm = smf.glm('compra ~ edad + precio + zona', df, family=sm.families.Binomial()).fit()\nnom = tabla_nomograma(m, df, ['edad', 'precio', 'zona'])\nprint(nom['importancia_puntos'].round(1)); nom['conversion'].round(3)"]),
    "aproximar_modelo": ("logit", ["import statsmodels.formula.api as smf\nm = smf.glm('compra ~ edad + precio + zona + ruido', df, family=sm.families.Binomial()).fit()\na = aproximar_modelo(df[['edad', 'precio', 'zona', 'ruido']], m.predict(df, which='linear'), 0.95)\nprint(a['variables']); a['historial']"]),
    "dominio_aplicabilidad": ("ninguno", ["train = pd.DataFrame(rng.normal(size=(1000, 3)), columns=['edad_z', 'renta_z', 'km_z'])\nnuevos = pd.DataFrame([[0, 0, 0], [3.5, 0, 0], [2, -2, 2]], columns=train.columns)\ndominio_aplicabilidad(train, nuevos).round(3)"]),
    "centrar_por_grupo": ("ninguno", ["d = pd.DataFrame({'oficina': np.repeat(['a', 'b', 'c'], 4), 'antiguedad': rng.integers(1, 20, 12)})\ncentrar_por_grupo(d, 'antiguedad', 'oficina')"]),
    "diccionario_datos": ("binario", ["diccionario_datos(df.assign(id=range(len(df)), prob_compra=0.5), {'precio': 'precio del vehículo (€)'})"]),
    "regresion_multivariante": ("ninguno", ["n = 300; x = rng.normal(size=n); E = rng.multivariate_normal([0, 0], [[1, .6], [.6, 1]], n)\nd = pd.DataFrame({'x': x, 'r': rng.normal(size=n), 'satisfaccion': 1 + .5 * x + E[:, 0], 'fidelidad': 2 + .3 * x + E[:, 1]})\nregresion_multivariante(d, ['satisfaccion', 'fidelidad'], ['x', 'r'])['contrastes'].round(4)"]),
    "analisis_perfiles": ("ninguno", ["d = pd.DataFrame({'grupo': np.repeat(['A', 'B'], 60)}); b = rng.normal(size=120)\nfor k in range(4):\n    d[f'mes{k}'] = b + k * 0.5 + (d.grupo == 'B') * (0.3 * k) + rng.normal(0, 0.5, 120)\nanalisis_perfiles(d, ['mes0', 'mes1', 'mes2', 'mes3'], 'grupo').round(4)"]),
    "control_t2_multivariante": ("ninguno", ["S = [[1, .85], [.85, 1]]\nf1 = pd.DataFrame(rng.multivariate_normal([0, 0], S, 80), columns=['peso', 'altura'])\nf2 = pd.DataFrame(np.r_[rng.multivariate_normal([0, 0], S, 30), [[1.5, -1.5]]], columns=['peso', 'altura'])\nc = control_t2_multivariante(f1, f2)\nprint(c['fase2'][c['fase2'].fuera])", "grafico_control_t2(c);"]),
    "analisis_conjunto": ("ninguno", ["perfiles = pd.DataFrame([(p, f, c) for p in ['300 €', '400 €', '500 €'] for f in ['0 €', '300 €'] for c in ['web', 'oficina']], columns=['precio', 'franquicia', 'canal'])\nfilas = [{**r.to_dict(), 'id': e, 'nota': 5 + {'300 €': 2, '400 €': 0, '500 €': -2}[r.precio] + (0.8 if r.franquicia == '0 €' else -0.8) + rng.normal(0, .5)} for e in range(20) for _, r in perfiles.iterrows()]\ncj = analisis_conjunto(pd.DataFrame(filas), 'nota', ['precio', 'franquicia', 'canal'])\nprint(cj['importancia'].round(1)); cj['utilidades'].round(2)"]),
    "modelo_grafico_gaussiano": ("ninguno", ["z = rng.normal(size=(1500, 4))\nX = pd.DataFrame({'a': z[:, 0], 'b': z[:, 0] + .6 * z[:, 1], 'c': z[:, 0] + .6 * z[:, 1] + .6 * z[:, 2], 'd': z[:, 3]})\ng = modelo_grafico_gaussiano(X)\ng['aristas'].round(3)", "plt.imshow(g['correlaciones_parciales'], cmap='RdBu_r', vmin=-1, vmax=1); plt.colorbar(); plt.xticks(range(4), list('abcd')); plt.yticks(range(4), list('abcd')); plt.title('Correlaciones parciales');"]),
    "pca_funcional": ("ninguno", ["t = np.linspace(0, 24, 48); a, b = rng.normal(0, 2, 150), rng.normal(0, .7, 150)\nY = 10 + np.outer(a, np.sin(2 * np.pi * t / 24)) + np.outer(b, np.cos(4 * np.pi * t / 24)) + rng.normal(0, .3, (150, 48))\nf = pca_funcional(Y, t)\nprint(f['varianza_explicada'].round(3))\nf['funciones_propias'].plot(title='Funciones propias (modos de variación del perfil diario)');"]),
    "regresion_matriz_indicadora": ("ninguno", ["X = pd.DataFrame({'x': np.r_[rng.normal(-3, .5, 100), rng.normal(0, .5, 100), rng.normal(3, .5, 100)]})\nr = regresion_matriz_indicadora(X, np.repeat(['bajo', 'medio', 'alto'], 100))\nprint('exactitud', round(r['exactitud'], 3), '· clases que nunca gana:', r['clases_enmascaradas'])"]),
    "centroides_contraidos": ("ninguno", ["X = pd.DataFrame(rng.normal(size=(90, 300))); y = np.repeat([0, 1, 2], 30)\nX.iloc[:30, :5] += 2; X.iloc[30:60, 5:10] += 2\nr = centroides_contraidos(X, y)\nprint('umbral', r['umbral'], '· exactitud CV', round(r['exactitud_cv'], 3), '·', len(r['variables_activas']), 'variables activas de 300')"]),
    "pls_da": ("ninguno", ["X = pd.DataFrame(rng.normal(size=(90, 300))); y = np.repeat(['a', 'b', 'c'], 30)\nX.iloc[:30, :5] += 2; X.iloc[30:60, 5:10] += 2\nr = pls_da(X, y, max_componentes=5)\nprint(r['n_componentes'], 'componentes · exactitud CV', round(r['exactitud_cv'], 3)); r['vip'].head(10).round(2)"]),
    # ---------------- ML (0.10)
    "reglas_asociacion": ("ninguno", ["cestas = []\nfor _ in range(3000):\n    c = set(rng.choice(['auto', 'hogar', 'vida', 'salud', 'mascotas'], rng.integers(1, 3), replace=False))\n    if 'auto' in c and rng.random() < 0.6: c.add('asistencia')\n    cestas.append(sorted(c))\nreglas_asociacion(cestas, 0.02, 0.3)['reglas'].head(8).round(3)"]),
    "mapa_autoorganizado": ("ninguno", ["X = pd.DataFrame(np.r_[rng.normal(0, .4, (200, 3)), rng.normal(3, .4, (200, 3)), rng.normal([0, 3, 0], .4, (200, 3))], columns=['a', 'b', 'c'])\ns = mapa_autoorganizado(X, 8, 8, 4000)\nprint('error de cuantización', round(s['error_cuantizacion'], 3))\nplt.imshow(s['matriz_u'], cmap='viridis'); plt.colorbar(); plt.title('Matriz U (fronteras = valores altos)');"]),
    "modelo_ngramas": ("ninguno", ["textos = ['el asegurado declara el siniestro', 'el asegurado paga la prima', 'la prima sube con el siniestro', 'el perito valora el siniestro']\nm = modelo_ngramas(textos, n=2)\nprint('perplejidad', round(m['perplejidad'], 2))\nm['distribucion']('el').sort_values(ascending=False).head(5).round(3)"]),
    "muestrear_siguiente": ("ninguno", ["p = pd.Series({'siniestro': .45, 'asegurado': .3, 'perito': .15, 'prima': .1})\npd.DataFrame({f'T={t}': muestrear_siguiente(p, t)['distribucion'] for t in [0.2, 1, 2]}).round(3)"]),
    "muestrear_texto": ("ninguno", ["m = modelo_ngramas(['el asegurado declara el siniestro', 'el asegurado paga la prima', 'la prima sube con el siniestro'], n=2)\nfor t in [0.1, 1.0, 2.0]:\n    print(t, '→', muestrear_texto(m, 'el', 8, temperatura=t, semilla=3))"]),
    "ajustar_bradley_terry": ("ninguno", ["# Preferencias humanas entre pares de respuestas (como en RLHF)\nfuerza = {'respuesta A': 1.0, 'respuesta B': 0.3, 'respuesta C': -0.5}\nfilas = []\nfor _ in range(500):\n    i, j = rng.choice(list(fuerza), 2, replace=False)\n    g = i if rng.random() < 1 / (1 + np.exp(-(fuerza[i] - fuerza[j]))) else j\n    filas.append({'ganador': g, 'perdedor': j if g == i else i})\najustar_bradley_terry(pd.DataFrame(filas)).round(3)"]),
    "recuperar_tfidf": ("ninguno", ["docs = ['La franquicia es la parte del daño que paga el asegurado.', 'El siniestro debe declararse en un plazo de 7 días.', 'La prima se fracciona sin recargo en pagos mensuales.']\nrecuperar_tfidf(docs, '¿en cuántos días hay que declarar un siniestro?', k=2)"]),
    "autoconsistencia_votacion": ("ninguno", ["pd.concat({f'p = {p}': autoconsistencia_votacion(p, n_respuestas_erroneas=3).precision for p in [0.3, 0.5, 0.7]}, axis=1).round(3)"]),
    "proceso_difusion": ("ninguno", ["x0 = rng.choice([-3.0, 0.0, 3.0], 5000)\nr = proceso_difusion(x0)\nfig, axs = plt.subplots(1, len(r['muestras']), figsize=(12, 2.5), sharey=True)\nfor ax, (t, v) in zip(axs, r['muestras'].items()):\n    ax.hist(v, bins=60); ax.set_title(f't = {t}')"]),
    "arbol_black_derman_toy": ("ninguno", ["r = arbol_black_derman_toy([0.030, 0.034, 0.037, 0.039, 0.040], 0.18)\n(r['arbol_tipos'] * 100).round(3)"]),
    # ---------------- gráficos (0.10)
    "grafico_trazas_mcmc": ("ninguno", ["r = metropolis(lambda th: stats.norm.logpdf(th[0], 2, 1) + stats.norm.logpdf(th[1], -1, .5), [0, 0], n_iter=2000)\ngrafico_trazas_mcmc(r);"]),
    "grafico_bandido": ("ninguno", ["grafico_bandido({e: simular_bandido([.05, .06, .09], 2000, e) for e in ['uniforme', 'thompson']});"]),
    "grafico_trayectorias": ("ninguno", ["grafico_trayectorias(paseo_aleatorio(300, 0.5, 500)['trayectorias'], titulo='Paseo aleatorio simétrico');"]),
    "grafico_interaccion": ("grupos", ["d['sexo'] = rng.choice(['h', 'm'], len(d)); d['valor'] += (d.sexo == 'm') * (d.grupo == 'C') * 2\ngrafico_interaccion(d, 'valor', 'grupo', 'sexo');"]),
    "grafico_efectos_2k": ("ninguno", ["D = diseno_factorial_2k(4, aleatorizar=False)['diseno']\ngrafico_efectos_2k(efectos_factorial_2k(D.assign(y=10 + 3 * D.B - 2 * D.D + rng.normal(0, .5, 16)), 'y', list('ABCD')));"]),
    "grafico_superficie_respuesta": ("ninguno", ["D = diseno_central_compuesto(2)\ngrafico_superficie_respuesta(superficie_respuesta(D.assign(y=50 - (D.x1 - .3) ** 2 - 2 * D.x2 ** 2 + rng.normal(0, .1, len(D))), 'y', ['x1', 'x2']));"]),
    "grafico_balance": ("ninguno", ["n = 2000; x = rng.normal(size=(n, 3)); t = (rng.random(n) < 1 / (1 + np.exp(-x @ [1, -.5, .3]))).astype(int)\nd = pd.DataFrame(x, columns=['a', 'b', 'c']).assign(t=t, y=t + x.sum(1) + rng.normal(size=n))\ngrafico_balance(puntuacion_propension(d, 't', 'y', ['a', 'b', 'c'], n_boot=20));"]),
    "grafico_metaanalisis": ("ninguno", ["grafico_metaanalisis(metaanalisis(rng.normal(0.3, 0.2, 8), rng.uniform(0.08, 0.3, 8)));"]),
    "grafico_hexbin": ("ninguno", ["d = pd.DataFrame({'edad': rng.normal(45, 12, 50000)}); d['prima'] = 300 + 5 * (d.edad - 45) ** 2 / 10 + rng.gamma(2, 50, 50000)\ngrafico_hexbin(d, 'edad', 'prima');"]),
    "grafico_violin": ("grupos", ["grafico_violin(d, 'valor', 'grupo');"]),
    "grafico_coordenadas_paralelas": ("ninguno", ["from sklearn.datasets import load_iris\nir = load_iris(as_frame=True); d = ir.data.assign(especie=ir.target.map(dict(enumerate(ir.target_names))))\ngrafico_coordenadas_paralelas(d, ir.data.columns, 'especie');"]),
    "grafico_curvas_andrews": ("ninguno", ["from sklearn.datasets import load_iris\nir = load_iris(as_frame=True); d = ir.data.assign(especie=ir.target.map(dict(enumerate(ir.target_names))))\ngrafico_curvas_andrews(d, ir.data.columns, 'especie');"]),
    "grafico_caras_chernoff": ("ninguno", ["seg = pd.DataFrame(rng.normal(size=(6, 6)), columns=['prima', 'antigüedad', 'siniestros', 'edad', 'productos', 'quejas'], index=[f'segmento {i}' for i in range(1, 7)])\ngrafico_caras_chernoff(seg, seg.columns, columnas=3);"]),
    "grafico_variable_anadida": ("ninguno", ["import statsmodels.formula.api as smf\nd = pd.DataFrame(rng.normal(size=(200, 3)), columns=['a', 'b', 'c']); d['y'] = d.a + 0.5 * d.b + d.a * 0.8 + rng.normal(size=200)\ngrafico_variable_anadida(smf.ols('y ~ a + b + c', d).fit(), 'b');"]),
    "grafico_bandas_regresion": ("ninguno", ["x = rng.uniform(0, 10, 40); y = 1 + 0.5 * x + rng.normal(0, 1, 40)\ngrafico_bandas_regresion(bandas_confianza_regresion(x, y), x, y);"]),
    "grafico_control_t2": ("ninguno", ["f1 = pd.DataFrame(rng.multivariate_normal([0, 0], [[1, .8], [.8, 1]], 60), columns=['a', 'b'])\ngrafico_control_t2(control_t2_multivariante(f1, f1.sample(20, random_state=1) + np.array([0.5, -0.5])));"]),
}


def celdas_de(funcion: str) -> list[str] | None:
    """Celdas listas para ejecutar: [escenario, ...celdas de la función]."""
    if funcion not in E:
        return None
    escenario, celdas = E[funcion]
    return [ESCENARIOS[escenario]] + list(celdas)


# ---------------------------------------------------------------- conceptos sin función propia
# Ejemplos con numpy/scipy para que también los «huecos» se puedan ver y probar. Los conceptos que sí
# tienen función usan el ejemplo de su primera función (ver celdas_concepto).
def _dist(nombre, crear, comentario=""):
    return ("ninguno", [f"""# {comentario}
d = {crear}
media, var, asim = (float(v) for v in d.stats(moments="mvs"))
print(f"media = {{media:.3f}} · varianza = {{var:.3f}} · asimetría = {{asim:.3f}}")
muestra = d.rvs(size=5000, random_state=42)
pd.Series(muestra).describe().round(3)""",
        f"""fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
x = np.linspace(d.ppf(0.001), d.ppf(0.999), 400) if hasattr(d.dist, "pdf") else np.arange(d.ppf(0.001), d.ppf(0.999) + 1)
if hasattr(d.dist, "pdf"):
    ax[0].hist(muestra, bins=50, density=True, alpha=0.4, label="muestra simulada"); ax[0].plot(x, d.pdf(x), lw=2, label="densidad")
else:
    ax[0].bar(x, d.pmf(x), alpha=0.6, label="P(X = k)")
ax[0].set_title("{nombre}"); ax[0].legend()
ax[1].plot(x, d.cdf(x), lw=2); ax[1].set_title("Función de distribución P(X ≤ x)")
plt.tight_layout()""", "grafico_qq(muestra);   # ¿se parece a una normal? (colas y asimetría)"])


CONCEPTOS = {
    "c_distribucion_normal": _dist("Normal(10, 2)", "stats.norm(loc=10, scale=2)", "Normal: simétrica; el 95 % cae en media ± 1.96 σ"),
    "c_distribucion_t_de_student": _dist("t de Student (3 gl)", "stats.t(df=3)", "t: como la normal pero con colas más pesadas (pocos grados de libertad)"),
    "c_distribucion_lognormal": _dist("Lognormal(0, 0.8)", "stats.lognorm(s=0.8, scale=np.exp(0))", "Lognormal: importes y costes; su logaritmo es normal"),
    "c_distribucion_gamma": _dist("Gamma(k=2, θ=1500)", "stats.gamma(a=2, scale=1500)", "Gamma: coste medio de un siniestro (severidad)"),
    "c_distribucion_inversa_gaussiana": _dist("Inversa gaussiana", "stats.invgauss(mu=0.5, scale=2000)", "Inversa gaussiana: severidad con cola más pesada que la gamma"),
    "c_distribucion_binomial": _dist("Binomial(20, 0.3)", "stats.binom(n=20, p=0.3)", "Binomial: nº de éxitos en n ensayos independientes"),
    "c_distribucion_de_poisson": _dist("Poisson(λ = 2)", "stats.poisson(mu=2)", "Poisson: nº de siniestros por póliza; media = varianza"),
    "c_distribucion_binomial_negativa": _dist("Binomial negativa (media 2, r = 1.5)", "stats.nbinom(n=1.5, p=1.5 / (1.5 + 2))", "Binomial negativa: conteos con sobredispersión (varianza > media)"),
    "c_distribucion_compuesta_poisson_gamma": ("ninguno", ["""# Poisson con λ ~ Gamma  ->  Binomial negativa (heterogeneidad entre asegurados)
lam = rng.gamma(shape=1.5, scale=2 / 1.5, size=20000)
n_sin = rng.poisson(lam)
print(f"media = {n_sin.mean():.3f} · varianza = {n_sin.var():.3f} (> media: sobredispersión)")
k = np.arange(0, 12)
nb = stats.nbinom(n=1.5, p=1.5 / (1.5 + 2))
plt.bar(k - 0.2, np.bincount(n_sin, minlength=12)[:12] / len(n_sin), width=0.4, label="simulado Poisson-Gamma")
plt.bar(k + 0.2, nb.pmf(k), width=0.4, label="Binomial negativa teórica"); plt.legend(); plt.title("Mezcla Poisson-Gamma = Binomial negativa");"""]),
    "c_distribuciones_de_la_clase_a_b_0_y_a_b_1": ("ninguno", ["""# Clase (a,b,0): p_k / p_{k-1} = a + b / k  ->  Poisson (a=0), binomial (a<0), binomial negativa (a>0)
k = np.arange(1, 10)
for nombre, d in {"Poisson": stats.poisson(2), "Binomial": stats.binom(10, 0.2), "Bin. negativa": stats.nbinom(2, 0.5)}.items():
    plt.plot(k, k * d.pmf(k) / d.pmf(k - 1), "o-", label=nombre)
plt.xlabel("k"); plt.ylabel("k · p_k / p_(k-1)  =  a·k + b"); plt.title("Recta de la clase (a,b,0)"); plt.legend();"""]),
    "c_distribuciones_con_colas_pesadas": ("ninguno", ["""# Cola pesada: P(X > x) cae como una potencia, no exponencialmente
x = np.linspace(1, 50, 300)
for nombre, d in {"Exponencial": stats.expon(scale=2), "Lognormal": stats.lognorm(s=1, scale=1), "Pareto α=1.5": stats.pareto(b=1.5)}.items():
    plt.loglog(x, d.sf(x), label=nombre)
plt.title("P(X > x) en escala log-log"); plt.legend();"""]),
    "c_familias_parametricas": ("ninguno", ["""# Una familia = una forma que cambia con pocos parámetros (aquí la gamma con varias formas k)
x = np.linspace(0, 15, 400)
for k in (0.8, 1, 2, 4, 8):
    plt.plot(x, stats.gamma(a=k).pdf(x), label=f"k = {k}")
plt.ylim(0, 0.8); plt.title("Familia gamma: misma fórmula, distintos parámetros"); plt.legend();"""]),
    "c_variables_aleatorias_y_distribuciones": ("ninguno", ["""# Variable aleatoria: el resultado de un experimento antes de hacerlo. Simulamos la suma de 2 dados
dados = rng.integers(1, 7, (100000, 2)).sum(axis=1)
frec = pd.Series(dados).value_counts(normalize=True).sort_index()
frec.plot.bar(rot=0, title="Distribución de la suma de dos dados (simulada)");"""]),
    "c_teorema_central_del_limite": ("ninguno", ["""# Medias de muestras de una exponencial (muy asimétrica) para varios n
fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for a, n in zip(ax, (1, 5, 30, 200)):
    medias = rng.exponential(1, (5000, n)).mean(axis=1)
    a.hist(medias, bins=50, density=True, alpha=0.6); a.set_title(f"n = {n} · asimetría {stats.skew(medias):.2f}")
plt.tight_layout()"""]),
    "c_axiomas_de_kolmogorov_y_bayes": ("ninguno", ["""# Bayes: prueba con 95 % de sensibilidad y especificidad, prevalencia 1 %
prev, sens, espec = 0.01, 0.95, 0.95
p_pos = sens * prev + (1 - espec) * (1 - prev)
print(f"P(enfermo | positivo) = {sens * prev / p_pos:.1%}   (¡no el 95 %!)")
prevs = np.linspace(0.001, 0.5, 200)
plt.plot(prevs, sens * prevs / (sens * prevs + (1 - espec) * (1 - prevs))); plt.xlabel("prevalencia"); plt.ylabel("P(enfermo | positivo)");"""]),
    "c_copulas": ("ninguno", ["""# Cópula gaussiana: mismas marginales (exponenciales), distinta dependencia
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for a, rho in zip(ax, (0, 0.6, 0.95)):
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 2000)
    u = stats.norm.cdf(z)                                    # cópula: uniformes dependientes
    x = stats.expon.ppf(u)                                   # marginales exponenciales
    a.scatter(x[:, 0], x[:, 1], s=3, alpha=0.4); a.set_title(f"ρ = {rho}")
plt.tight_layout()"""]),
    "c_descomposicion_de_cholesky": ("ninguno", ["""# Cholesky: de normales independientes a normales correlacionadas
S = np.array([[1, 0.8], [0.8, 1]])
L = np.linalg.cholesky(S)
z = rng.normal(size=(3000, 2)) @ L.T
print("L =\\n", L.round(3)); print("correlación simulada:", np.corrcoef(z.T)[0, 1].round(3))
plt.scatter(z[:, 0], z[:, 1], s=3, alpha=0.4); plt.title("Normales con ρ = 0.8 vía Cholesky");"""]),
    "c_esperanza_condicionada": ("ninguno", ["""# E[Y | X]: la media de Y en cada valor de X (aquí, la curva que mejor predice Y)
x = rng.uniform(0, 10, 3000); y = np.sin(x) + rng.normal(0, 0.4, 3000)
tr = pd.cut(x, 20)
m = pd.Series(y).groupby(tr, observed=True).mean()
plt.scatter(x, y, s=2, alpha=0.2); plt.plot([i.mid for i in m.index], m.values, "o-", lw=2); plt.title("E[Y | X] estimada por tramos");"""]),
    "c_generadores_de_numeros_aleatorios": ("ninguno", ["""# Un generador con semilla da siempre la misma secuencia (reproducibilidad)
a = np.random.default_rng(123).uniform(size=5); b = np.random.default_rng(123).uniform(size=5)
print(a.round(4), "\\n", b.round(4), "\\niguales:", np.allclose(a, b))
u = np.random.default_rng(1).uniform(size=(5000, 2))
plt.scatter(u[:, 0], u[:, 1], s=2); plt.title("Pares (u_i, u_i+1): sin patrones = buen generador");"""]),
    "c_generacion_de_variables_aleatorias": ("ninguno", ["""# Método de la inversa: X = F^-1(U) con U uniforme (exponencial de media 2)
u = rng.uniform(size=10000)
x = -2 * np.log(1 - u)
xs = np.linspace(0, 15, 200)
plt.hist(x, bins=60, density=True, alpha=0.5, label="F^-1(U)"); plt.plot(xs, stats.expon(scale=2).pdf(xs), label="exponencial teórica"); plt.legend();"""]),
    "c_generacion_de_variables_normales_y_multivariantes": ("ninguno", ["""# Box-Muller: dos uniformes -> dos normales independientes
u1, u2 = rng.uniform(size=(2, 10000))
z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
print(f"media {z.mean():.3f} · desv {z.std():.3f}")
grafico_qq(z);"""]),
    "c_metodo_de_monte_carlo": ("ninguno", ["""# Monte Carlo: estimar π y ver cómo el error baja con √n
n = np.logspace(2, 6, 20).astype(int)
est = [4 * (rng.uniform(size=(k, 2)) ** 2).sum(axis=1).__le__(1).mean() for k in n]
plt.semilogx(n, est, "o-"); plt.axhline(np.pi, color="gray"); plt.title("Estimación de π por Monte Carlo"); plt.xlabel("n simulaciones");"""]),
    "c_bootstrap": ("ninguno", ["""# Bootstrap: IC de la mediana remuestreando con reemplazo
datos = rng.lognormal(0, 0.8, 80)
medianas = np.array([np.median(rng.choice(datos, len(datos))) for _ in range(4000)])
lo, hi = np.percentile(medianas, [2.5, 97.5])
print(f"mediana = {np.median(datos):.3f} · IC 95 % bootstrap [{lo:.3f}, {hi:.3f}]")
plt.hist(medianas, bins=50, alpha=0.6); plt.axvline(lo, color="gray"); plt.axvline(hi, color="gray"); plt.title("Distribución bootstrap de la mediana");"""]),
    "c_intervalos_de_confianza": ("ninguno", ["""# 100 intervalos del 95 %: ~5 no contienen la media verdadera (50)
fuera = 0
for i in range(100):
    x = rng.normal(50, 10, 25); m = x.mean(); e = stats.t.ppf(0.975, 24) * x.std(ddof=1) / 5
    ok = m - e <= 50 <= m + e; fuera += not ok
    plt.plot([m - e, m + e], [i, i], color="C0" if ok else "C1")
plt.axvline(50, color="gray"); plt.title(f"{100 - fuera} de 100 contienen μ");"""]),
}


def celdas_concepto(concepto: str, funciones: list[str]) -> list[str] | None:
    """Ejemplo del concepto si lo tiene; si no, el de su primera función con ejemplo."""
    if concepto in CONCEPTOS:
        escenario, celdas = CONCEPTOS[concepto]
        return [ESCENARIOS[escenario]] + list(celdas)
    for f in funciones:
        if f in E:
            return celdas_de(f)
    return None
