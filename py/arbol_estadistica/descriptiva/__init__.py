"""Descriptiva: resumen numérico, atípicos, correlaciones, supuestos, distribuciones y tablas de contingencia."""
from .correlacion import correlacion_con_ic, correlacion_parcial, matriz_correlaciones
from .distribuciones import (ajustar_distribucion_discreta, ajustar_distribuciones, estimar_densidad,
                             funcion_distribucion_empirica, indice_dispersion)
from .resumen import detectar_atipicos, resumen_descriptivo
from .supuestos import contraste_normalidad, homogeneidad_varianzas
from .tablas import medidas_riesgo_2x2, tabla_contingencia
from .diccionario import diccionario_datos

__all__ = ["ajustar_distribucion_discreta", "ajustar_distribuciones", "contraste_normalidad", "correlacion_con_ic",
           "correlacion_parcial", "detectar_atipicos", "estimar_densidad", "funcion_distribucion_empirica",
           "homogeneidad_varianzas", "indice_dispersion", "matriz_correlaciones", "medidas_riesgo_2x2",
           "resumen_descriptivo", "tabla_contingencia", "diccionario_datos"]
