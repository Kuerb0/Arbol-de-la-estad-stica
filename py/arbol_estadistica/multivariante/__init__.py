"""Multivariante: Mahalanobis, Hotelling, Box, MANOVA, correlación canónica, PCA, factorial, Cronbach, MDS,
correspondencias, clustering jerárquico, mezclas gaussianas y discriminante."""
from .clasificacion import analisis_discriminante, clustering_jerarquico, mezclas_gaussianas
from .inferencia import contraste_box_m, contraste_hotelling, correlacion_canonica, distancia_mahalanobis, manova
from .reduccion import (adecuacion_factorial, alfa_cronbach, analisis_correspondencias, analisis_factorial,
                        escalamiento_multidimensional, pca_completo)
from .extra import (analisis_conjunto, analisis_perfiles, centroides_contraidos, control_t2_multivariante,
                    modelo_grafico_gaussiano, pca_funcional, pls_da, regresion_matriz_indicadora, regresion_multivariante)

__all__ = ["adecuacion_factorial", "alfa_cronbach", "analisis_correspondencias", "analisis_discriminante",
           "analisis_factorial", "clustering_jerarquico", "contraste_box_m", "contraste_hotelling", "correlacion_canonica",
           "distancia_mahalanobis", "escalamiento_multidimensional", "manova", "mezclas_gaussianas", "pca_completo", "analisis_conjunto", "analisis_perfiles", "centroides_contraidos", "control_t2_multivariante", "modelo_grafico_gaussiano", "pca_funcional", "pls_da", "regresion_matriz_indicadora", "regresion_multivariante"]
