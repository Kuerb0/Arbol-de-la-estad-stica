"""Rama GRAFICOS: gráficos de matplotlib listos para notebooks (devuelven la Figure).

Importar aparte (no se carga con `import arbol_estadistica`, para no pagar matplotlib si no se usa):
    from arbol_estadistica.graficos import grafico_curva_roc
"""
from .clasificacion import (grafico_calibracion, grafico_curva_roc, grafico_ganancia_lift, grafico_odds_ratios,
                            grafico_precision_recall,
                            grafico_residuos_agrupados, grafico_umbrales)
from .clustering import grafico_clusters_pca, grafico_seleccion_k, grafico_vif
from .finanzas import grafico_copula, grafico_frontera_eficiente
from .diseno import (grafico_balance, grafico_efectos_2k, grafico_interaccion, grafico_metaanalisis,
                     grafico_superficie_respuesta)
from .exploracion import (grafico_bandas_regresion, grafico_caras_chernoff, grafico_control_t2, grafico_coordenadas_paralelas,
                          grafico_curvas_andrews, grafico_hexbin, grafico_variable_anadida, grafico_violin)
from .simulacion import grafico_bandido, grafico_trayectorias, grafico_trazas_mcmc
from .actuarial import grafico_reserva_bootstrap, grafico_tabla_mortalidad
from .ml import grafico_dependencia_parcial, grafico_importancias
from .multivariante import grafico_biplot, grafico_dendrograma
from .modelos_extra import (grafico_efecto_spline, grafico_incidencia_acumulada, grafico_regularizacion,
                            grafico_relatividades)
from .descriptiva import grafico_ajuste_distribuciones, grafico_distribucion, grafico_matriz_correlaciones
from .contrastes import (grafico_comparar_grupos, grafico_curva_potencia, grafico_fdr, grafico_potencia,
                         grafico_qq, grafico_region_rechazo)
from .regresion import grafico_diagnostico_residuos, grafico_regresion_simple
from .supervivencia import grafico_kaplan_meier

__all__ = [
    "grafico_bandas_regresion", "grafico_caras_chernoff", "grafico_control_t2", "grafico_coordenadas_paralelas",
    "grafico_curvas_andrews", "grafico_hexbin", "grafico_variable_anadida", "grafico_violin",
    "grafico_balance", "grafico_efectos_2k", "grafico_interaccion", "grafico_metaanalisis", "grafico_superficie_respuesta",
    "grafico_bandido", "grafico_trayectorias", "grafico_trazas_mcmc",
    "grafico_copula", "grafico_frontera_eficiente",
    "grafico_reserva_bootstrap", "grafico_tabla_mortalidad",
    "grafico_dependencia_parcial", "grafico_importancias",
    "grafico_biplot", "grafico_dendrograma",
    "grafico_ganancia_lift", "grafico_precision_recall",
    "grafico_efecto_spline", "grafico_incidencia_acumulada", "grafico_regularizacion", "grafico_relatividades",
    "grafico_ajuste_distribuciones", "grafico_distribucion", "grafico_matriz_correlaciones",
    "grafico_calibracion", "grafico_clusters_pca", "grafico_comparar_grupos", "grafico_curva_potencia",
    "grafico_curva_roc", "grafico_diagnostico_residuos", "grafico_fdr", "grafico_kaplan_meier",
    "grafico_odds_ratios", "grafico_potencia", "grafico_qq", "grafico_regresion_simple",
    "grafico_region_rechazo", "grafico_residuos_agrupados", "grafico_seleccion_k", "grafico_umbrales",
    "grafico_vif",
]
