"""Reorganiza el catálogo de conceptos en TEMAS (ramas del mapa) > ÁREAS (módulos) > conceptos.

Migración única de la versión 0.6.0. Conserva los ids de los conceptos y todo su contenido
(funciones, sinónimos, fuentes). Solo cambia el campo `area`, añade `temas`/áreas nuevas,
`ambito` por área y `prioridad` en unos pocos conceptos.

    python herramientas/reorganizar_catalogo.py          # aplica a conceptos/catalogo_base.json y catalogo.json
Es idempotente: si ya está reorganizado, no hace nada.
"""
from __future__ import annotations

import colorsys
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

# tema: (id, nombre, descripción, color oscuro)
TEMAS = [
    ("t_prob", "Probabilidad y distribuciones", "Fundamentos de probabilidad, familias de distribuciones, colas y dependencia.", "#F08CC0"),
    ("t_inf", "Inferencia y contrastes", "Estimación, intervalos y contrastes de hipótesis, paramétricos y no paramétricos.", "#A8D84A"),
    ("t_mod", "Regresión y GLM", "Modelos lineales, GLM para respuestas categóricas, de conteo y continuas, y extensiones.", "#FF9F5A"),
    ("t_ml", "Validación y machine learning", "Selección y validación de modelos, métricas de clasificación, preprocesado, clustering y árboles.", "#5ED6A0"),
    ("t_sim", "Simulación, Bayes y diseño", "Monte Carlo y bootstrap, estadística bayesiana, diseño de estudios y causalidad.", "#D98CF5"),
    ("t_pob", "Demografía y supervivencia", "Tasas y tablas demográficas, análisis de supervivencia y procesos estocásticos.", "#E8708F"),
    ("t_seg", "Seguros y riesgo", "Seguros de vida y no vida, medidas de riesgo, Solvencia II y capital.", "#7AB8FF"),
    ("t_fin", "Finanzas y normativa", "Carteras, renta fija y derivados, contabilidad, regulación y previsión social.", "#E0C98F"),
]

# área: (id, tema, nombre, ámbito, descripción)
AREAS = [
    ("prob_fundamentos", "t_prob", "Fundamentos de probabilidad", "metodologico", "Axiomas, variables aleatorias, esperanza condicionada y teorema central del límite."),
    ("dist_univariantes", "t_prob", "Distribuciones univariantes", "metodologico", "Normal, binomial, Poisson, gamma, lognormal, t y familias relacionadas."),
    ("dependencia_colas", "t_prob", "Colas pesadas y dependencia", "metodologico", "Colas pesadas, cópulas y descomposición de Cholesky."),
    ("inf_fundamentos", "t_inf", "Fundamentos de inferencia", "metodologico", "Población y muestra, estimadores, modelo estadístico, exploración y robustez."),
    ("estimacion", "t_inf", "Estimación e intervalos", "metodologico", "Máxima verosimilitud, intervalos de confianza, Wald y razón de verosimilitudes, EM."),
    ("contrastes_base", "t_inf", "Lógica del contraste", "metodologico", "Hipótesis, p-valor, errores, potencia, tamaño del efecto y control de falsos descubrimientos."),
    ("contrastes_tests", "t_inf", "Tests concretos", "metodologico", "Test t, ANOVA, comparaciones múltiples, no paramétricos, chi-cuadrado y Fisher."),
    ("regresion_lineal", "t_mod", "Regresión lineal", "metodologico", "Modelo lineal, multicolinealidad, predictores categóricos y residuos."),
    ("glm_categoricos", "t_mod", "GLM de respuesta categórica", "metodologico", "GLM en general, logística, odds ratio, enlaces, multinomial y ordinal."),
    ("glm_conteo_continuo", "t_mod", "GLM de conteo y continua", "metodologico", "Poisson, binomial negativa, gamma, sobredispersión y elasticidad."),
    ("modelos_avanzados", "t_mod", "Modelos avanzados", "metodologico", "Efectos mixtos, datos longitudinales, splines y datos funcionales."),
    ("seleccion_validacion", "t_ml", "Selección y validación de modelos", "metodologico", "AIC/BIC, validación cruzada, leakage, regularización y cribado de variables."),
    ("evaluacion_clasificadores", "t_ml", "Evaluación de clasificadores", "metodologico", "ROC/AUC, calibración, umbral de decisión, matriz de confusión y desbalanceo."),
    ("preproceso", "t_ml", "Preprocesado", "metodologico", "Transformaciones, discretización y preparación de matrices."),
    ("no_supervisado", "t_ml", "Aprendizaje no supervisado", "metodologico", "K-Means, elección de K y componentes principales."),
    ("arboles_ensembles", "t_ml", "Árboles y ensembles", "metodologico", "Árboles de decisión, random forest y boosting."),
    ("simulacion", "t_sim", "Simulación", "metodologico", "Generación de números y variables aleatorias, Monte Carlo, bootstrap y potencia."),
    ("bayes", "t_sim", "Estadística bayesiana", "metodologico", "Previas conjugadas, MCMC, chequeos predictivos y modelos jerárquicos."),
    ("diseno_causal", "t_sim", "Diseño y causalidad", "metodologico", "Aleatorización, tamaño muestral, confusión, causalidad y ensayos."),
    ("demografia_tasas", "t_pob", "Demografía y tasas", "actuarial", "Lexis, tasas, estandarización, mortalidad, fecundidad y proyecciones."),
    ("tablas_vida", "t_pob", "Tablas de vida y supervivencia actuarial", "actuarial", "Tablas de mortalidad, leyes de supervivencia, decrementos y longevidad."),
    ("supervivencia", "t_pob", "Análisis de supervivencia", "metodologico", "Censura, Kaplan-Meier y modelo de Cox."),
    ("procesos", "t_pob", "Procesos estocásticos", "metodologico", "Cadenas de Markov, Poisson, martingalas y movimiento browniano."),
    ("no_vida", "t_seg", "Seguros no vida", "actuarial", "Siniestralidad agregada, Panjer, primas, ruina y provisiones."),
    ("vida", "t_seg", "Seguros de vida", "actuarial", "Base técnica, primas, rentas y provisión matemática."),
    ("riesgo_medidas", "t_seg", "Medidas de riesgo", "actuarial", "VaR, TVaR, valores extremos y tests de estrés."),
    ("solvencia", "t_seg", "Solvencia y capital", "normativo", "Solvencia II, SCR, Best Estimate, riesgos por módulos y Basilea."),
    ("carteras", "t_fin", "Carteras", "financiero", "Markowitz, CAPM, factores, utilidad y Black-Litterman."),
    ("renta_fija", "t_fin", "Renta fija y derivados", "financiero", "Bonos, ETTI, duración, derivados y valoración neutral al riesgo."),
    ("marco_contable", "t_fin", "Marco contable", "normativo", "NIIF, estados consolidados, instrumentos financieros y provisiones."),
    ("marco_seguros", "t_fin", "Marco legal de seguros", "normativo", "Contrato de seguro, fuentes del Derecho, transparencia y supervisión."),
    ("prevision_social", "t_fin", "Previsión social", "normativo", "Sistema complementario, financiación, productos y pensiones."),
]

ASIGNACION = {
    "prob_fundamentos": ["axiomas_de_kolmogorov_y_bayes", "teorema_central_del_limite", "esperanza_condicionada",
                         "variables_aleatorias_y_distribuciones", "familias_parametricas"],
    "dist_univariantes": ["distribucion_normal", "distribucion_binomial", "distribucion_de_poisson",
                          "distribucion_binomial_negativa", "distribucion_lognormal", "distribucion_gamma",
                          "distribucion_inversa_gaussiana", "distribucion_t_de_student",
                          "distribuciones_de_la_clase_a_b_0_y_a_b_1", "distribucion_compuesta_poisson_gamma"],
    "dependencia_colas": ["distribuciones_con_colas_pesadas", "copulas", "descomposicion_de_cholesky"],
    "inf_fundamentos": ["poblacion_muestra_y_parametros", "estadisticos_y_estimadores", "modelo_estadistico",
                        "analisis_exploratorio_de_datos", "estadistico_robusto_mediana", "inferencia_robusta",
                        "panorama_de_la_estadistica_vision_general"],
    "estimacion": ["estimacion_por_maxima_verosimilitud", "intervalos_de_confianza",
                   "tests_de_wald_y_razon_de_verosimilitudes", "datos_faltantes_y_algoritmo_em"],
    "contrastes_base": ["contraste_de_hipotesis", "significacion_estadistica_y_p_valor", "error_tipo_i_y_potencia",
                        "tamano_del_efecto", "tasa_de_falsos_descubrimientos_fdr",
                        "comparaciones_multiples_tukey"],
    "contrastes_tests": ["test_t_y_test_t_de_welch", "anova_y_welch_anova",
                         "tests_no_parametricos_mann_whitney_kruskal_wallis", "test_chi_cuadrado_y_fisher",
                         "v_de_cramer"],
    "regresion_lineal": ["regresion_lineal", "multicolinealidad_y_vif", "predictores_categoricos_y_referencia",
                         "diagnostico_de_residuos"],
    "glm_categoricos": ["modelos_lineales_generalizados_glm", "regresion_logistica_binaria", "odds_ratio",
                        "funcion_de_enlace_logit_probit_y_cloglog", "separacion_perfecta_y_firth",
                        "regresion_multinomial_nominal", "regresion_ordinal",
                        "probabilidad_ante_datos_desbalanceados_correccion_de_prior"],
    "glm_conteo_continuo": ["glm_de_conteo_poisson_y_binomial_negativa",
                            "glm_de_variable_continua_gamma_e_inversa_gaussiana", "sobredispersion",
                            "elasticidad_y_sensibilidad_al_precio"],
    "modelos_avanzados": ["modelos_de_efectos_mixtos", "datos_longitudinales_y_medidas_repetidas",
                          "splines_de_suavizado", "analisis_de_datos_funcionales"],
    "seleccion_validacion": ["seleccion_de_variables_stepwise", "criterios_de_informacion_aic_y_bic",
                             "validacion_cruzada", "particion_train_test", "fuga_de_informacion_leakage",
                             "regularizacion_ridge_lasso_y_elastic_net", "modelos_sobreparametrizados",
                             "cribado_de_variables"],
    "evaluacion_clasificadores": ["curva_roc_y_auc", "estadisticos_de_asociacion_c_somers_d_gamma_tau_a",
                                  "calibracion_y_hosmer_lemeshow", "umbral_de_decision_youden_ctable",
                                  "matriz_de_confusion_y_metricas_de_clasificacion", "desbalanceo_de_clases"],
    "preproceso": ["discretizacion_en_tramos_cuantiles", "preprocesamiento_y_transformaciones",
                   "preparacion_de_matrices_para_segmentar"],
    "no_supervisado": ["k_means", "eleccion_de_k_silhouette_davies_bouldin_y_calinski_harabasz",
                       "analisis_de_componentes_principales_pca"],
    "arboles_ensembles": ["arboles_de_decision", "random_forest", "boosting"],
    "simulacion": ["generadores_de_numeros_aleatorios", "generacion_de_variables_aleatorias",
                   "generacion_de_variables_normales_y_multivariantes", "metodo_de_monte_carlo", "bootstrap",
                   "simulacion_de_potencia"],
    "bayes": ["estadistica_bayesiana", "familias_conjugadas", "algoritmo_de_metropolis_y_mcmc",
              "chequeos_predictivos_previo_y_posterior", "modelos_jerarquicos_y_multinivel_bayesianos",
              "analisis_de_decision_adaptativo"],
    "diseno_causal": ["inferencia_causal_contrafactica", "ensayos_clinicos", "ensayos_n_of_1_y_de_cesta_basket",
                      "metaanalisis", "switchers_cambio_de_tratamiento", "tamano_muestral",
                      "aleatorizacion_rct_y_tests_a_b", "confusion", "tasa_de_respuesta_encuestas"],
    "demografia_tasas": ["diagrama_de_lexis", "analisis_longitudinal_y_transversal",
                         "tasas_ratios_y_probabilidades_demograficas", "tasas_brutas_y_especificas_de_mortalidad",
                         "estandarizacion_de_tasas", "mortalidad_infantil_y_por_causa",
                         "proyecciones_de_poblacion", "fecundidad_y_migracion"],
    "tablas_vida": ["tablas_de_mortalidad", "funciones_biometricas",
                    "supuestos_de_homogeneidad_independencia_y_estacionariedad",
                    "esperanza_de_vida_completa_e_incompleta", "vida_futura_como_variable_aleatoria_t_x",
                    "hipotesis_para_fracciones_de_ano", "leyes_de_supervivencia_de_moivre_gompertz_makeham",
                    "grupos_de_varias_cabezas", "multiples_decrementos", "riesgo_de_longevidad"],
    "supervivencia": ["duracion_hasta_el_evento", "analisis_de_supervivencia_y_censura",
                      "estimador_de_kaplan_meier", "modelo_de_riesgos_proporcionales_de_cox"],
    "procesos": ["procesos_estocasticos_definicion_y_clasificacion", "problema_de_la_ruina_del_jugador",
                 "cadenas_de_markov_en_tiempo_discreto", "cadenas_de_markov_regulares_y_modelo_bonus_malus",
                 "cadenas_de_markov_en_tiempo_continuo", "procesos_de_nacimiento_y_muerte", "proceso_de_poisson",
                 "martingalas_y_recorridos_aleatorios", "movimiento_browniano_y_calculo_estocastico"],
    "no_vida": ["proceso_del_riesgo_asegurador", "siniestralidad_agregada", "convolucion_y_recursion_de_panjer",
                "componentes_de_la_prima_de_no_vida", "tarificacion_por_glm", "teoria_de_la_ruina",
                "provisiones_tecnicas_de_no_vida", "chain_ladder_y_reservas_por_bootstrap"],
    "vida": ["base_tecnica", "seguros_de_vida_capital_diferido_temporal_vida_entera",
             "rentas_vinculadas_a_la_supervivencia", "principio_de_equivalencia_y_prima_neta",
             "prima_de_inventario_y_de_tarifa", "provision_matematica",
             "valoracion_financiero_actuarial_de_prestaciones_complementar"],
    "riesgo_medidas": ["value_at_risk_var", "tvar_y_medidas_coherentes_de_riesgo", "teoria_de_valores_extremos",
                       "test_de_estres"],
    "solvencia": ["solvencia_ii_y_balance_economico", "scr_mcr_y_formula_estandar", "best_estimate_y_risk_margin",
                  "curva_smith_wilson", "riesgo_de_suscripcion_vida_salud_y_no_vida", "riesgo_de_mercado_y_alm",
                  "riesgo_de_credito_y_contraparte", "riesgo_operacional_orsa_y_riesgos_emergentes",
                  "analisis_y_medida_de_riesgos_en_prevision_social", "basilea_iii_y_requerimientos_de_capital",
                  "escenarios_economicos_esg"],
    "carteras": ["modelo_de_markowitz_media_varianza", "matriz_de_covarianzas_y_estimacion_de_parametros",
                 "capm_beta_y_linea_del_mercado_de_valores", "apt_y_modelos_de_factores", "dominancia_estocastica",
                 "aversion_al_riesgo_y_funciones_de_utilidad", "black_litterman"],
    "renta_fija": ["valoracion_de_bonos", "estructura_temporal_de_tipos_de_interes_etti", "duracion_e_inmunizacion",
                   "modelos_estocasticos_de_tipos_black_derman_toy", "fras_swaps_caps_y_floors",
                   "valoracion_de_opciones_black_scholes", "convencion_de_calculo_de_dias",
                   "escenario_neutral_al_riesgo"],
    "marco_contable": ["normas_de_contabilidad_niif_y_us_gaap", "estados_financieros_consolidados",
                       "instrumentos_financieros_y_cobertura", "coste_amortizado_y_valor_razonable",
                       "inversion_crediticia_deterioro_y_titulizacion", "provisiones_por_contratos_de_seguros",
                       "cuenta_de_perdidas_y_ganancias_de_entidades_financieras"],
    "marco_seguros": ["contrato_de_seguro_y_condiciones", "fuentes_del_derecho_de_seguros",
                      "transparencia_y_distribucion_de_seguros", "supervision_prudencial"],
    "prevision_social": ["sistema_de_prevision_social_complementaria",
                         "modelos_de_financiacion_de_la_prevision_social", "productos_de_prevision_social",
                         "seguridad_social_y_pensiones"],
}

PRIORIDAD_ALTA = ["regresion_lineal", "regresion_logistica_binaria", "contraste_de_hipotesis",
                  "intervalos_de_confianza", "validacion_cruzada", "curva_roc_y_auc",
                  "estimacion_por_maxima_verosimilitud"]


def _claro(hexa: str) -> str:
    """Versión más oscura del color para el tema claro."""
    r, g, b = (int(hexa[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, 0.38, min(1.0, s * 0.95))
    return "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))


def reorganizar(cat: dict) -> dict:
    """Devuelve un catálogo nuevo con temas/áreas nuevos. Lanza AssertionError si algo no cuadra."""
    if cat.get("temas"):
        return cat
    por_id = {c["id"]: c for c in cat["conceptos"]}
    destino: dict[str, str] = {}
    for area, sufijos in ASIGNACION.items():
        for s in sufijos:
            cid = "c_" + s
            assert cid in por_id, f"concepto inexistente en la asignación: {cid}"
            assert cid not in destino, f"concepto asignado dos veces: {cid}"
            destino[cid] = area
    faltan = sorted(set(por_id) - set(destino))
    assert not faltan, f"conceptos sin asignar: {faltan}"
    assert {a[0] for a in AREAS} == set(ASIGNACION), "áreas y asignación no coinciden"
    assert {a[1] for a in AREAS} <= {t[0] for t in TEMAS}
    nuevo = {
        "temas": [{"id": t, "nombre": n, "desc": d, "color": {"claro": _claro(c), "oscuro": c}}
                  for t, n, d, c in TEMAS],
        "areas": [{"id": i, "tema": t, "nombre": n, "ambito": am, "desc": d} for i, t, n, am, d in AREAS],
        "conceptos": [],
    }
    for c in cat["conceptos"]:
        c = dict(c)
        c["area"] = destino[c["id"]]
        if c["id"][2:] in PRIORIDAD_ALTA:
            c["prioridad"] = "alta"
        nuevo["conceptos"].append(c)
    assert len(nuevo["conceptos"]) == len(cat["conceptos"])
    assert [c["id"] for c in nuevo["conceptos"]] == [c["id"] for c in cat["conceptos"]]
    return nuevo


def main(raiz: Path = RAIZ) -> int:
    for nombre in ("catalogo_base.json", "catalogo.json"):
        p = raiz / "conceptos" / nombre
        if not p.exists():
            continue
        cat = json.loads(p.read_text(encoding="utf-8"))
        if cat.get("temas"):
            print(f"{nombre}: ya reorganizado")
            continue
        nuevo = reorganizar(cat)
        p.write_text(json.dumps(nuevo, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{nombre}: {len(nuevo['temas'])} temas, {len(nuevo['areas'])} áreas, {len(nuevo['conceptos'])} conceptos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
