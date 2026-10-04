# Equivalencias SAS ↔ Python

Qué función del árbol (o de librería) sustituye a cada procedimiento SAS, y **en qué se diferencian**. "≈" = equivalente práctico, no idéntico.

| SAS | Python / árbol | Diferencias que importan |
|---|---|---|
| `PROC LOGISTIC` (binaria) | `modelos.ajustar_logit` (+ `tabla_odds_ratios`, `tabla_parametros_wald`) | SAS modela `EVENT='1'` (usa `DESCENDING`); aquí el evento es el 1 |
| `CLASS x(REF='a') / PARAM=REF` | `preparar_matriz_modelo(refs={'x': 'a'})`; en fórmulas `C(x, Treatment(reference='a'))` | Sin `REF=`, SAS usa la última categoría; pandas/patsy la primera |
| `PROC LOGISTIC … LINK=GLOGIT` | `modelos.logit_multinomial_sas` (`sm.MNLogit`) | Referencia por defecto distinta (SAS: última categoría): fijar `base_class` |
| `ODDSRATIO x / UNITS=(u)` | `tabla_odds_ratios(unidades={'x': u}, std_map=…)` | Con variables estandarizadas hay que reescalar: exp(β·u/σ) |
| `MODEL … / LACKFIT` | `diagnostico.hosmer_lemeshow` | Mismos deciles de riesgo; gl = grupos − 2 |
| "Association of Predicted Probabilities" | `diagnostico.estadisticos_asociacion` | Aquí exactos y sin muestreo |
| `CTABLE PPROB=` | `diagnostico.tabla_umbrales` | — |
| `ROC` / `ROCCONTRAST` | `sklearn.metrics.roc_curve`, `roc_auc_score` | El contraste entre ROC (DeLong) no está en el árbol |
| `MODEL … / FIRTH` | `modelos.regresion_logistica_firth` | Wald para el p-valor (SAS ofrece también perfil de verosimilitud penalizada) |
| `SELECTION=FORWARD` (SLENTRY) | `seleccion.seleccion_forward` | SAS entra por p-valor del score test; aquí por AIC/BIC |
| `LSMEANS` | `logit_multinomial_sas(...)["lsmeans_like"]` ≈ | SAS evalúa en las **medias de las covariables**; aquí estandarización marginal (promedio de predicciones) |
| `ESTIMATE` / `LSMESTIMATE` | `modelo.t_test`, `modelo.wald_test` (statsmodels) | Sintaxis de contrastes distinta |
| `PROC GENMOD` (dist/link) | `smf.glm(formula, family=…)`; `modelos.ajustar_glm_binomial` | **Siempre pasar `family`**: por defecto es gaussiana |
| `PROC REG / VIF` | `diagnostico.calcular_vif` | SAS no incluye el intercepto en el VIF "centrado"; statsmodels lo incluye en la matriz |
| `PROC FREQ / CHISQ` | `scipy.stats.chi2_contingency`; `seleccion.contraste_chi2_variable` | Yates solo en 2×2 (scipy lo aplica por defecto) |
| `PROC TTEST` | `elegir_contraste` (Student / Welch según Levene) | SAS muestra los dos y el test de igualdad de varianzas |
| `PROC NPAR1WAY` | `elegir_contraste` (Mann-Whitney / Kruskal) | — |
| `PROC ANOVA` / `PROC GLM` + `MEANS / TUKEY` | `elegir_contraste` + `contrastes.tukey_entre_grupos` | — |
| `PROC RANK GROUPS=n` | `preprocesado.categorizar_por_cuantiles` | `GROUPS=` numera desde 0; aquí etiquetas Q1..Qn |
| `PROC STDIZE` | `StandardScaler` / `MinMaxScaler` (`preparar_matriz_clustering`) | — |
| `PROC FASTCLUS` | `clustering.ajustar_kmeans` | Inicialización distinta (k-means++); fijar semilla |
| `PROC PRINCOMP` | `clustering.proyeccion_pca` | Signo de las componentes arbitrario |
| `PROC SURVEYSELECT` (estratos) | `dividir_train_test`, `balancear_clases`, `submuestreo_por_ratio` | — |
| `PROC SURVEYSELECT` muestreo simple | `df.sample(n, random_state=…)` | — |
| `PROC SGPLOT` | matplotlib / seaborn | — |
