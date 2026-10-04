# GLM: guía rápida

Guía propia de GLM, con ejemplos de renovación, frecuencia y severidad.

## Estructura

`g(E[Y]) = Xβ` con tres piezas: **distribución** de Y (familia exponencial), **enlace** g y **predictor lineal** Xβ.
Estimación por máxima verosimilitud con **IRLS** (mínimos cuadrados reponderados iterativos); alternativas: Newton-Raphson, BFGS, Firth (penalizado), bayesiano (MCMC).

## Qué familia y enlace uso

| Variable respuesta | Familia | Enlace habitual | Interpretación de exp(β) | En el árbol |
|---|---|---|---|---|
| Binaria (renueva / no renueva, vende en < 30 d) | Binomial | logit | odds ratio | `ajustar_logit`, `ajustar_glm_binomial` |
| Categórica sin orden (>2 clases) | Multinomial | logit generalizado | OR de cada clase vs base | `logit_multinomial_sas` |
| Conteo (nº de siniestros) | Poisson / Binomial Negativa | log | razón de tasas | *pendiente* |
| Positiva continua asimétrica (coste de siniestro) | Gamma | log | razón de medias | *pendiente* |
| Continua ~ normal | Gaussiana | identidad | cambio en la media | OLS |
| Evento raro / asimétrico | Binomial | cloglog | razón de riesgos (instantánea) | `comparar_enlaces` |

Prima pura = frecuencia esperada × severidad esperada (modelos separados).

## Flujo de trabajo habitual

1. **ETL** y variables derivadas → `preprocesado`.
2. **Cribado** de candidatas (chi², V de Cramér, fuga) → `seleccion.cribar_variables`.
3. **Selección** forward por AIC/BIC → `seleccion.seleccion_forward`.
4. **Colinealidad** (VIF < 5 ideal, 5–10 vigilar, ≥ 10 problema) → `diagnostico.calcular_vif`.
5. **Ajuste** y **odds ratios** en unidades de negocio → `modelos.tabla_odds_ratios`.
6. **Validación**: AUC train vs test (gap > 0.02 = alerta), Hosmer-Lemeshow, c/Somers' D, CTABLE, CV 5-fold.
7. **Robustez**: Newton vs IRLS vs Firth; logit vs probit vs cloglog.

## Interpretaciones que hay que saber decir en voz alta

- **Odds ratio 1.5**: las odds de renovar son 1.5 veces las de la categoría de referencia (no "50 % más probable").
- **Coeficiente estandarizado**: efecto de subir 1 desviación típica; sirve para **ordenar** impulsores y frenos, no para explicar unidades.
- **AUC**: probabilidad de que un positivo reciba mayor puntuación que un negativo (0.5 = azar). **c = AUC**; **Somers' D = 2c − 1**.
- **Pseudo-R² de McFadden**: 0.2–0.4 ya es un ajuste muy bueno; no se compara con el R² lineal.
- **AIC/BIC**: menor es mejor; solo comparables entre modelos sobre las **mismas filas**. BIC penaliza más (log n), AIC admite más variables.
- **Hosmer-Lemeshow** p > 0.05 = calibración aceptable (pero con n grande suele rechazar).

## Problemas típicos y remedio

| Problema | Síntoma | Remedio |
|---|---|---|
| Multicolinealidad | VIF alto, signos raros, SE enormes | `filtrar_vif_iterativo`, agrupar, Ridge/Lasso |
| Separación perfecta | Coeficientes → ±∞, no converge | `regresion_logistica_firth` |
| Clases desbalanceadas | AUC bien pero recall de la minoritaria bajo | `submuestreo_por_ratio`, `pesos_por_clase`, ajustar umbral con `umbral_optimo_youden` |
| Sobredispersión (conteos) | deviance/gl ≫ 1 | Binomial Negativa |
| Sobreajuste | gap AUC train–test > 0.02 | menos variables, BIC, regularización, CV |
| Fuga de información | AUC ≈ 1 sospechoso | `es_posible_fuga`, revisar variables derivadas del objetivo |
| Variables en distinta escala | OR ilegibles | `preparar_matriz_modelo` + `unidades` en `tabla_odds_ratios` |
