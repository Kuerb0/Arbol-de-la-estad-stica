"""Rama DISENO: ANOVA y diseños (factorial, bloques, ANCOVA, medidas repetidas y mixto, anidado, diagnóstico),
diseño de experimentos (2^k y fraccionados, cuadrado latino, superficie de respuesta) e inferencia causal
(propensión, diferencias en diferencias, ensayos, mediación, moderación, metaanálisis)."""
from .anova import (ancova, anova_anidado, anova_bloques, anova_factorial, anova_medidas_repetidas, diagnostico_anova)
from .causal import (aleatorizar_ensayo, analisis_intencion_tratar, diferencias_en_diferencias, mediacion, metaanalisis,
                     moderacion, puntuacion_propension)
from .experimentos import (anova_cuadrado_latino, cuadrado_latino, diseno_central_compuesto, diseno_factorial_2k,
                           efectos_factorial_2k, superficie_respuesta)

__all__ = [
    "aleatorizar_ensayo",
    "analisis_intencion_tratar",
    "ancova",
    "anova_anidado",
    "anova_bloques",
    "anova_cuadrado_latino",
    "anova_factorial",
    "anova_medidas_repetidas",
    "cuadrado_latino",
    "diagnostico_anova",
    "diferencias_en_diferencias",
    "diseno_central_compuesto",
    "diseno_factorial_2k",
    "efectos_factorial_2k",
    "mediacion",
    "metaanalisis",
    "moderacion",
    "puntuacion_propension",
    "superficie_respuesta",
]
