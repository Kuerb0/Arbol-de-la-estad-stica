"""Fuente legible de py/propiedades/fichas.json (notas estimadas, pros/contras, cuándo usar).

    python herramientas/fichas_fuente.py      # reescribe py/propiedades/fichas.json

Notas 0-10 con abreviaturas (ver ABREV). Lo que no aparece = no aplica (n/a).
Las propiedades medibles (velocidad, memoria, escalabilidad, cobertura de tests y algunas estadísticas)
las sobrescribe py/medir_propiedades.py en py/propiedades/medidas.json: aquí son la estimación de partida.
"""
import json
from pathlib import Path

ABREV = {
    "ins": "insesgadez", "con": "consistencia", "efi": "eficiencia", "ecm": "ecm", "nas": "normalidad_asintotica",
    "ide": "identificabilidad", "alf": "tamano_alfa", "pot": "potencia", "cob": "cobertura_ic",
    "inc": "cuantificacion_incertidumbre", "peq": "validez_muestras_pequenas", "sel": "validez_tras_seleccion",
    "gen": "generalizacion", "cal": "calibracion", "dis": "discriminacion", "par": "parsimonia",
    "int": "interpretabilidad", "est": "estabilidad", "rob": "robustez", "sup": "supuestos", "cen": "censura",
    "des": "desbalanceo", "dep": "dependencia", "esc": "escalabilidad", "vel": "velocidad", "mem": "memoria",
    "num": "estabilidad_numerica", "suc": "tolerancia_datos_sucios", "rep": "reproducibilidad",
    "cfg": "configurabilidad", "aju": "coste_ajuste", "sas": "trazabilidad_sas",
}

# Código común de los gráficos (matplotlib): rápidos, deterministas, sin propiedades estadísticas propias
G = "esc=7 vel=7 mem=7 num=8 rep=10 aju=9 "

F = {}


def f(nombre, grupo, notas, pros, contras, usar, evitar):
    F[nombre] = {"grupo": grupo, "notas": notas, "pros": pros, "contras": contras, "usar_si": usar, "evitar_si": evitar}


# ------------------------------------------------------------------ contrastes
f("elegir_contraste", "comparar_grupos",
  "alf=8 pot=8 inc=9 peq=7 rob=8 sup=8 int=9 esc=8 vel=8 mem=8 num=9 suc=8 rep=10 cfg=7 aju=9 sas=8",
  ["Elige t / Welch / Mann-Whitney / ANOVA / Kruskal / chi² / Fisher según los datos", "Devuelve tamaño del efecto y avisos de supuestos"],
  ["Elegir el test mirando los mismos datos infla un poco el α real", "No cubre muestras apareadas"],
  "Quieres comparar una variable entre 2 o más grupos independientes sin pensar qué test toca.",
  "Los datos están apareados (mismo individuo antes/después) o hay covariables que ajustar (usa un modelo).")
f("tukey_entre_grupos", "multiples",
  "alf=9 pot=7 cob=9 inc=10 peq=7 rob=4 sup=4 int=9 esc=7 vel=8 mem=8 num=9 suc=6 rep=10 cfg=4 aju=10 sas=9",
  ["Controla el error familiar en todas las parejas", "IC simultáneos de cada diferencia"],
  ["Supone normalidad y varianzas iguales", "Con varianzas muy distintas el α real se desvía"],
  "Tras un ANOVA significativo, quieres saber qué pares de grupos difieren.",
  "Las varianzas son muy distintas o los datos muy asimétricos (usa Games-Howell o Dunn).")
f("ajustar_p_valores", "multiples",
  "alf=9 pot=8 inc=6 peq=10 rob=8 sup=8 dep=5 int=8 esc=10 vel=10 mem=10 num=10 suc=8 rep=10 cfg=8 aju=8 sas=9",
  ["BH controla la tasa de falsos descubrimientos con mucha más potencia que Bonferroni", "Holm/Bonferroni si un falso positivo es caro"],
  ["BH supone independencia o dependencia positiva (BY si no)", "Hay que elegir el criterio antes de ver los resultados"],
  "Haces decenas o cientos de contrastes a la vez (variables, segmentos, A/B).",
  "Solo haces uno o dos contrastes planificados.")
f("tamano_muestral_medias", "potencia_n",
  "ins=9 con=9 inc=5 peq=8 sup=4 int=10 esc=10 vel=10 mem=10 num=9 suc=7 rep=10 cfg=8 aju=7 sas=9",
  ["Fórmula exacta con la t no central", "Acepta d de Cohen o diferencia + desviación"],
  ["Supone normalidad y varianzas iguales", "El resultado depende mucho del efecto que supongas"],
  "Planificas un estudio o un A/B con una variable continua.",
  "La respuesta es muy asimétrica o binaria (usa tamano_muestral_proporciones o simulación).")
f("tamano_muestral_proporciones", "potencia_n",
  "ins=8 con=9 peq=6 sup=6 int=10 esc=10 vel=10 mem=10 num=9 suc=7 rep=10 cfg=8 aju=7 sas=9",
  ["Directo para tasas de conversión o siniestralidad", "Admite grupos de tamaño distinto"],
  ["Aproximación normal: con tasas muy pequeñas (<1 %) se queda corta"],
  "Comparas dos tasas (conversión, fuga, siniestro sí/no).",
  "Las tasas son rarísimas o hay muy pocos eventos (simula con potencia_por_simulacion).")
f("potencia_contraste_medias", "potencia_n",
  "ins=9 con=9 peq=8 sup=4 int=10 esc=10 vel=10 mem=10 num=9 suc=7 rep=10 cfg=7 aju=8 sas=9",
  ["Potencia exacta del t-test para un n dado"], ["Mismos supuestos que el t-test"],
  "Ya tienes n fijado y quieres saber qué probabilidad tienes de detectar el efecto.",
  "El test que vas a usar no es un t-test.")
f("curva_potencia", "potencia_n",
  "ins=9 con=9 peq=8 sup=4 int=10 esc=9 vel=9 mem=10 num=9 suc=7 rep=10 cfg=7 aju=8 sas=9",
  ["Tabla n → potencia para elegir n de un vistazo"], ["Mismos supuestos que el t-test"],
  "Quieres ver cuánto ganas de potencia al subir n.", "El test no es un t-test.")
f("potencia_por_simulacion", "potencia_n",
  "ins=9 con=10 inc=7 peq=10 rob=9 sup=10 cen=8 des=8 dep=8 int=8 esc=4 vel=3 mem=8 num=9 suc=6 rep=10 cfg=10 aju=6",
  ["Vale para CUALQUIER test y cualquier generador de datos", "También mide el α real de un test"],
  ["Lenta (miles de simulaciones)", "Error de Monte Carlo: ±1.4 pp con 1000 simulaciones"],
  "El test o los datos se salen de las fórmulas (censura, conteos, clases raras, test no paramétrico).",
  "Existe fórmula cerrada y basta (t-test o proporciones estándar).")

# ------------------------------------------------------------------ selección
f("cribar_variables", "cribado",
  "alf=5 pot=7 inc=7 sel=3 rob=7 sup=8 des=6 int=9 par=7 est=6 esc=7 vel=7 mem=7 num=8 suc=9 rep=10 cfg=7 aju=7 sas=8",
  ["Revisa TODAS las columnas: chi², V de Cramér, fuga, baja variabilidad, cardinalidad", "Rápido para un primer filtro"],
  ["Univariante: no ve interacciones ni confusión", "Con n grande todo sale significativo: mira la V de Cramér"],
  "Empiezas con muchas columnas y quieres descartar las inútiles o peligrosas (fuga).",
  "Ya tienes un conjunto pequeño de variables con sentido de negocio.")
f("contraste_chi2_variable", "cribado",
  "alf=8 pot=7 inc=8 peq=5 rob=7 sup=8 int=9 esc=9 vel=9 mem=9 num=9 suc=7 rep=10 cfg=5 aju=9 sas=9",
  ["Chi² + V de Cramér + diferencia máxima de tasas"], ["Con celdas esperadas < 5 la aproximación falla (usa Fisher)"],
  "Quieres el detalle de una variable categórica frente al objetivo binario.",
  "La variable es numérica continua (categorízala o usa un modelo).")
f("es_posible_fuga", "cribado",
  "rob=5 sup=10 int=10 esc=10 vel=10 mem=10 num=10 suc=10 rep=10 cfg=8 aju=9",
  ["Detecta y_*, prob_*, *_pred antes de que contaminen el modelo"], ["Solo mira el nombre: una fuga con nombre inocente pasa"],
  "Antes de modelar, para limpiar columnas derivadas del objetivo.", "Nunca está de más; complementa con cribar_variables.")
f("seleccion_forward", "seleccion_modelo",
  "con=5 ide=7 alf=3 inc=6 sel=2 gen=6 par=7 int=8 est=3 rob=5 sup=5 esc=3 vel=3 mem=7 num=7 suc=6 rep=10 cfg=8 aju=7 sas=8",
  ["Familia correcta (Binomial por defecto)", "AIC para predecir, BIC para parsimonia", "Devuelve el historial de pasos"],
  ["AIC no es consistente en la selección (una variable de ruido entra ~16 % de las veces); BIC sí",
   "Los p-valores del modelo final están sesgados (selección con los mismos datos)", "Inestable: cambia con pequeñas variaciones de los datos",
   "Ajusta O(p²) modelos"],
  "Tienes ≤ 30 candidatas y quieres un modelo interpretable.",
  "Hay muchas variables (usa Lasso) o vas a reportar p-valores del modelo final.")

# ------------------------------------------------------------------ modelos: logística
f("ajustar_logit", "logistica",
  "ins=7 con=10 efi=10 ecm=7 nas=9 ide=5 alf=8 cob=8 inc=10 peq=5 gen=7 cal=9 dis=8 par=8 int=10 est=7 rob=4 sup=5 des=5 "
  "esc=9 vel=8 mem=7 num=6 suc=4 rep=10 cfg=8 aju=10 sas=10",
  ["Máxima verosimilitud: consistente y eficiente", "Equivale a PROC LOGISTIC; pesos bien tratados (GLM)", "Coeficientes → odds ratios"],
  ["Sesgo hacia fuera con pocos eventos por variable", "Falla con separación perfecta (usa Firth)", "Requiere X sin NaN"],
  "Objetivo 0/1 con suficientes eventos (≥ 10-20 por parámetro) y quieres interpretar.",
  "Hay separación o muy pocos eventos (Firth) o lo que importa es predecir con relaciones no lineales.")
f("regresion_logistica_firth", "logistica",
  "ins=9 con=10 efi=9 ecm=9 nas=8 ide=10 alf=8 cob=8 inc=9 peq=9 gen=7 cal=8 dis=8 par=8 int=10 est=8 rob=5 sup=5 des=9 "
  "esc=6 vel=5 mem=6 num=9 suc=4 rep=10 cfg=4 aju=10 sas=9",
  ["Siempre da estimaciones finitas (aunque haya separación)", "Reduce el sesgo de muestra pequeña", "Ideal con eventos raros"],
  ["Más lenta que la logística normal", "IC de Wald; SAS usa perfil de verosimilitud penalizada"],
  "Pocos eventos, separación perfecta o casi, o clase rara.",
  "n y eventos son grandes: da lo mismo que ajustar_logit y es más lenta.")
f("ajustar_glm_binomial", "logistica",
  "ins=7 con=10 efi=10 ecm=7 nas=9 ide=5 alf=8 cob=8 inc=10 peq=5 gen=8 cal=9 dis=8 par=8 int=10 est=7 rob=4 sup=5 des=5 "
  "esc=8 vel=7 mem=7 num=7 suc=6 rep=10 cfg=9 aju=10 sas=9",
  ["Fórmula estilo R/SAS con categóricas automáticas", "AIC, BIC y AUC train/test en un paso"],
  ["Mismos límites que la logística (separación, pocos eventos)"],
  "Quieres ajustar y validar rápido con una fórmula.", "Separación o pocos eventos (Firth).")
f("dividir_train_test", "validacion",
  "gen=8 des=8 est=6 esc=9 vel=9 mem=7 num=10 suc=8 rep=10 cfg=6 aju=8 sas=8",
  ["Estratificado: misma tasa en train y test", "Semilla fija"],
  ["Un solo split: la estimación del error tiene varianza (mejor validación cruzada si n es pequeño)"],
  "Tienes datos suficientes para separar un test.", "n es pequeño: usa validación cruzada.")
f("comparar_tecnicas_estimacion", "justificar",
  "con=9 inc=7 gen=7 dis=8 int=8 rob=4 sup=5 esc=5 vel=5 mem=6 num=7 suc=4 rep=10 cfg=3 aju=10 sas=8",
  ["Justifica la técnica: Newton vs IRLS vs Firth lado a lado"], ["Ajusta tres modelos (3× el coste)"],
  "Tienes que documentar por qué usas una técnica de estimación.", "Solo necesitas el modelo final.")
f("comparar_enlaces", "justificar",
  "con=9 inc=7 gen=7 cal=8 dis=8 int=8 rob=4 sup=6 des=7 esc=5 vel=5 mem=6 num=7 suc=4 rep=10 cfg=3 aju=10 sas=8",
  ["Logit vs probit vs cloglog con AIC y AUC", "cloglog para eventos raros o asimétricos"],
  ["Las diferencias suelen ser pequeñas; el logit gana en interpretación (OR)"],
  "Dudas del enlace o los eventos son muy raros.", "Necesitas odds ratios: quédate con logit.")
f("preparar_matriz_modelo", "preparar",
  "ide=8 int=9 esc=8 vel=8 mem=6 num=9 suc=6 rep=10 cfg=9 aju=8 sas=10",
  ["Dummies con referencia como CLASS … PARAM=REF", "Guarda el z-score para reescalar los OR"],
  ["Elimina filas con NaN (avisa del número)", "Muchas categorías → muchas columnas"],
  "Antes de ajustar_logit, para tener la matriz estilo SAS.", "Usas fórmulas (ajustar_glm_binomial ya lo hace).")
f("tabla_odds_ratios", "tablas_modelo",
  "nas=9 cob=8 inc=10 peq=5 int=10 esc=10 vel=10 mem=10 num=9 suc=7 rep=10 cfg=9 aju=10 sas=10",
  ["OR en unidades de negocio (UNITS=)", "IC 95 % y p-valor"], ["IC de Wald: con pocos eventos son optimistas"],
  "Explicar el efecto de cada variable a negocio.", "Hay separación (los OR salen absurdos: usa Firth).")
f("tabla_parametros_wald", "tablas_modelo",
  "nas=9 alf=7 cob=8 inc=10 peq=5 int=8 esc=10 vel=10 mem=10 num=9 suc=7 rep=10 cfg=3 aju=10 sas=10",
  ["Idéntica a 'Analysis of Maximum Likelihood Estimates'"], ["Wald es el contraste menos fiable con n pequeño (mejor razón de verosimilitudes)"],
  "Reproducir la tabla de SAS.", "Muestra pequeña o coeficientes enormes (efecto Hauck-Donner).")
f("tabla_coeficientes", "tablas_modelo",
  "nas=9 cob=8 inc=10 peq=5 int=9 esc=10 vel=10 mem=10 num=9 suc=7 rep=10 cfg=7 aju=10 sas=8",
  ["Coeficientes, SE, p e IC (y OR) de cualquier modelo de statsmodels"], ["IC de Wald"],
  "Resumir un GLM con fórmula.", "Necesitas el formato exacto de SAS (tabla_parametros_wald).")
f("perfil_respuesta", "tablas_modelo",
  "int=10 des=7 esc=10 vel=10 mem=10 num=10 suc=8 rep=10 cfg=6 aju=10 sas=10",
  ["'Response Profile' de SAS: frecuencias del objetivo"], ["Solo descriptiva"],
  "Siempre al empezar: ver la prevalencia.", "—")
f("logit_multinomial_sas", "multiclase",
  "ins=7 con=10 efi=9 nas=9 ide=5 alf=8 cob=8 inc=10 peq=4 gen=7 cal=8 dis=8 par=5 int=7 est=6 rob=4 sup=4 des=5 "
  "esc=5 vel=4 mem=5 num=6 suc=7 rep=10 cfg=9 aju=8 sas=9",
  ["Todo lo de PROC LOGISTIC GLOGIT: betas, AUC, confusión, VIF, efectos marginales", "base_class explícita (evita el error de SAS)"],
  ["Supone independencia de alternativas irrelevantes (IIA)", "Muchos parámetros: (K−1)·p", "lsmeans_like ≠ LSMEANS de SAS"],
  "Objetivo con 3 o más categorías sin orden.", "Las categorías están ordenadas (logit ordinal) o hay muy pocos casos por clase.")

# ------------------------------------------------------------------ modelos: precio
f("sensibilidad_por_grupo", "precio",
  "ins=6 con=8 inc=6 int=9 rob=4 sup=4 dep=3 esc=8 vel=8 mem=8 num=8 suc=5 rep=10 cfg=6 aju=8",
  ["Compara la sensibilidad al precio entre segmentos en la misma escala"],
  ["Lineal y sin controlar otras variables: es correlación, no causalidad"],
  "Primera exploración de qué segmentos reaccionan más al precio.", "Necesitas una elasticidad causal (usa un diseño o un modelo con controles).")
f("ganancia_por_bajada_precio", "precio",
  "ins=5 con=7 int=10 rob=4 sup=3 esc=8 vel=8 mem=8 num=8 suc=5 rep=10 cfg=6 aju=7",
  ["Traduce la pendiente a puntos de tasa ganados"], ["Extrapola linealmente; hereda los límites de sensibilidad_por_grupo"],
  "Comunicar a negocio el efecto de una bajada.", "Bajadas grandes (fuera del rango observado).")
f("tendencia_polinomica_ponderada", "precio",
  "ins=7 con=8 efi=8 inc=3 gen=5 par=7 int=8 rob=3 sup=4 esc=10 vel=10 mem=10 num=7 suc=4 rep=10 cfg=6 aju=6 sas=7",
  ["'Sweet spot' cuadrático ponderado por volumen"], ["Polinomios: malos extremos y sensibles a atípicos", "No da incertidumbre"],
  "Buscar el punto óptimo de una curva agregada.", "Quieres extrapolar fuera del rango.")

# ------------------------------------------------------------------ supervivencia
f("kaplan_meier", "superv_curvas",
  "ins=9 con=10 efi=8 nas=9 cob=8 inc=10 peq=7 int=10 rob=8 sup=9 cen=10 esc=8 vel=8 mem=8 num=9 suc=6 rep=10 cfg=6 aju=10 sas=9",
  ["No paramétrico: no supone ninguna distribución", "Usa bien la censura", "IC log(−log) y medianas"],
  ["Supone censura no informativa", "No ajusta por covariables"],
  "Describir la supervivencia (duración, permanencia) con datos censurados.", "Quieres el efecto de varias covariables a la vez (Cox).")
f("contraste_log_rank", "superv_contrastes",
  "alf=9 pot=8 inc=8 peq=7 int=8 rob=7 sup=8 cen=10 esc=8 vel=8 mem=8 num=9 suc=6 rep=10 cfg=4 aju=10 sas=9",
  ["Óptimo si los riesgos son proporcionales", "Usa la censura"], ["Pierde potencia si las curvas se cruzan"],
  "Comparar curvas de supervivencia entre grupos.", "Las curvas se cruzan (usa Wilcoxon/Peto o RMST).")
f("ajustar_cox", "superv_modelos",
  "ins=8 con=10 efi=9 nas=9 ide=7 alf=8 cob=8 inc=10 peq=6 gen=7 dis=8 par=7 int=9 est=7 rob=5 sup=7 cen=10 "
  "esc=6 vel=6 mem=6 num=8 suc=6 rep=10 cfg=7 aju=10 sas=9",
  ["Semiparamétrico: no hace falta la forma del riesgo base", "Hazard ratios con IC"],
  ["Supone riesgos proporcionales (compruébalo)", "Empates: aproximación de Efron"],
  "Efecto de covariables sobre el tiempo hasta un evento con censura.", "Los riesgos no son proporcionales o necesitas predecir la supervivencia absoluta.")

# ------------------------------------------------------------------ diagnóstico
f("calcular_vif", "colinealidad",
  "int=9 rob=6 sup=8 esc=6 vel=6 mem=6 num=8 suc=5 rep=10 cfg=6 aju=9 sas=9",
  ["Diagnóstico estándar con etiquetas OK / Moderado / ALTO"], ["Un VIF por columna: no ve qué variables forman el grupo colineal", "Hace p regresiones"],
  "Antes de interpretar coeficientes.", "Solo te importa predecir (la colinealidad no daña la predicción).")
f("filtrar_vif_iterativo", "colinealidad",
  "par=8 int=8 est=5 rob=5 sup=8 esc=3 vel=3 mem=6 num=8 suc=5 rep=10 cfg=8 aju=6",
  ["Automatiza la limpieza de colinealidad respetando las protegidas"], ["Bucle O(p) × VIF: lento con muchas columnas", "Puede quitar la variable que te interesa (protégela)"],
  "Muchas variables numéricas correlacionadas.", "Variables con sentido de negocio que quieres conservar (usa Ridge).")
f("hosmer_lemeshow", "calibracion",
  "alf=6 pot=5 inc=8 peq=5 cal=9 int=7 rob=5 sup=7 esc=9 vel=9 mem=9 num=9 suc=5 rep=10 cfg=5 aju=7 sas=10",
  ["Contraste clásico de calibración (LACKFIT)", "Tabla observado/esperado por grupos"],
  ["Con n grande casi siempre rechaza", "Depende del número de grupos g"],
  "Comprobar la calibración en n moderado y acompañarlo de la gráfica.", "n muy grande: mira la tabla O/E y grafico_calibracion.")
f("estadisticos_asociacion", "discriminacion",
  "ins=10 con=10 inc=4 dis=10 int=8 rob=8 sup=10 des=8 esc=9 vel=8 mem=7 num=10 suc=5 rep=10 cfg=3 aju=10 sas=10",
  ["c, Somers' D, Gamma y Tau-a EXACTOS en O(n log n)"], ["Sin IC"],
  "Reproducir la tabla de asociación de PROC LOGISTIC.", "Necesitas el IC del AUC (bootstrap).")
f("auc_train_test", "validacion",
  "con=9 inc=4 gen=9 dis=10 int=8 rob=8 sup=10 des=7 esc=8 vel=8 mem=7 num=9 suc=7 rep=10 cfg=6 aju=8",
  ["Alerta de sobreajuste si el AUC cae más de 0.02 en test"], ["Un solo split", "El AUC no mide calibración"],
  "Validar un modelo con fórmula.", "n pequeño (usa validación cruzada).")
f("metricas_binarias", "discriminacion",
  "con=9 inc=3 cal=8 dis=9 int=7 rob=7 sup=10 des=6 esc=9 vel=9 mem=8 num=8 suc=5 rep=10 cfg=4 aju=10",
  ["AUC + LogLoss + Brier: discriminación y calibración"], ["LogLoss explota con probabilidades 0 o 1"],
  "Comparar modelos de clasificación en un conjunto.", "—")
f("tabla_umbrales", "discriminacion",
  "con=9 inc=3 dis=9 int=10 rob=8 sup=10 des=7 esc=8 vel=8 mem=8 num=10 suc=5 rep=10 cfg=7 aju=8 sas=10",
  ["CTABLE: sensibilidad, especificidad, PPV, NPV y Youden"], ["Elegir el umbral sobre el test sesga a favor"],
  "Decidir un umbral de corte.", "Lo que necesitas son probabilidades, no decisiones.")
f("umbral_optimo_youden", "discriminacion",
  "con=8 inc=2 dis=9 int=9 rob=7 sup=10 des=7 est=5 esc=9 vel=9 mem=8 num=10 suc=5 rep=10 cfg=3 aju=10",
  ["Umbral exacto que maximiza sensibilidad + especificidad"], ["Supone costes iguales de error", "El óptimo es ruidoso con n pequeño"],
  "Sin información de costes, como punto de partida.", "Los errores tienen costes distintos (pondera).")
f("resumen_auc_multiclase", "discriminacion",
  "con=9 inc=3 dis=9 int=7 rob=7 sup=10 des=7 esc=8 vel=8 mem=7 num=9 suc=5 rep=10 cfg=4 aju=10",
  ["AUC OvR y OvO, macro y ponderado"], ["Varios AUC distintos: hay que elegir uno y justificarlo"],
  "Evaluar un multinomial.", "—")
f("matrices_confusion", "discriminacion",
  "dis=7 int=10 des=7 esc=9 vel=9 mem=8 num=10 suc=6 rep=10 cfg=4 aju=10",
  ["Conteos y recall por clase"], ["Depende del umbral o de la regla de asignación"],
  "Ver qué clases se confunden.", "—")

# ------------------------------------------------------------------ preprocesado
f("balancear_clases", "desbalanceo",
  "cal=2 dis=6 des=8 est=5 int=8 esc=7 vel=7 mem=4 num=10 suc=6 rep=10 cfg=7 aju=7 sas=7",
  ["Iguala clases por sobremuestreo o submuestreo"], ["Descalibra las probabilidades (corrígelas después)", "Sobremuestreo = duplicados: riesgo de sobreajuste"],
  "Solo en train y si el algoritmo sufre con clases raras.", "Necesitas probabilidades calibradas sin corregirlas; o en el test (nunca).")
f("submuestreo_por_ratio", "desbalanceo",
  "cal=3 dis=7 des=8 est=5 int=8 esc=9 vel=9 mem=8 num=10 suc=6 rep=10 cfg=8 aju=7 sas=7",
  ["Ratio controlado N:1; reduce el tamaño (rápido)"], ["Tira información de la clase mayoritaria", "Descalibra"],
  "Datos enormes con clase rara: entrenar más rápido.", "n es pequeño (perderías información).")
f("pesos_por_clase", "desbalanceo",
  "cal=4 dis=7 des=8 est=8 int=8 esc=10 vel=10 mem=10 num=10 suc=7 rep=10 cfg=4 aju=9 sas=8",
  ["Usa todos los datos (nada se tira ni se duplica)"], ["También descalibra; los errores estándar con pesos de frecuencia son engañosos"],
  "Preferible a remuestrear si el modelo acepta pesos.", "Quieres inferencia (p-valores) además de predicción.")
f("corregir_probabilidades_por_balanceo", "desbalanceo",
  "ins=8 con=9 cal=10 des=9 int=8 esc=10 vel=10 mem=10 num=9 suc=6 rep=10 cfg=6 aju=9",
  ["Recupera la escala real de las probabilidades (corrección de prior)"], ["Necesitas conocer la prevalencia real"],
  "Siempre que hayas entrenado con datos balanceados y vayas a usar las probabilidades.", "No balanceaste.")
f("categorizar_por_cuantiles", "preparar",
  "rob=8 sup=9 int=9 est=7 esc=8 vel=8 mem=6 num=9 suc=8 rep=10 cfg=8 aju=6 sas=9",
  ["Tramos robustos a atípicos; más tramos para importes"], ["Pierde información (escalones)", "Los cortes dependen de la muestra"],
  "Tarificación por tramos o relaciones no lineales sin splines.", "El efecto es suave y quieres eficiencia (usa la variable continua o splines).")
f("codificar_ordinal", "preparar",
  "int=9 esc=10 vel=10 mem=9 num=10 suc=8 rep=10 cfg=7 aju=9 sas=8",
  ["Mapping invertible y nulo explícito"], ["Impone distancias iguales entre niveles"],
  "Categóricas con orden natural.", "Los niveles no tienen orden (usa dummies).")
f("duracion_hasta_evento", "preparar",
  "cen=2 int=8 esc=7 vel=6 mem=7 num=10 suc=5 rep=10 cfg=5 aju=9",
  ["Construye la duración desde datos de panel"], ["NO distingue censura de evento (error conocido nº 7)"],
  "Preparar datos de panel para supervivencia (añadiendo tú la marca de evento).", "Directamente en Kaplan-Meier sin la marca de censura.")

# ------------------------------------------------------------------ clustering
f("preparar_matriz_clustering", "segmentar",
  "int=7 rob=5 esc=8 vel=8 mem=6 num=9 suc=6 rep=10 cfg=9 aju=6",
  ["Escalado + one-hot + ordinal + pesos por variable"], ["Los pesos y el escalado cambian los clusters: decisión subjetiva"],
  "Antes de K-Means con variables mixtas.", "—")
f("buscar_k_silhouette", "segmentar",
  "con=6 inc=3 int=7 est=6 rob=4 sup=5 esc=5 vel=4 mem=6 num=8 suc=4 rep=10 cfg=7 aju=6",
  ["Tres índices por K; silhouette con muestra (evita O(n²))"], ["Los índices pueden no coincidir", "Favorece clusters esféricos"],
  "Elegir K para K-Means.", "Esperas clusters no esféricos o de densidad variable.")
f("ajustar_kmeans", "segmentar",
  "con=6 int=8 est=6 par=7 rob=3 sup=4 esc=8 vel=7 mem=7 num=8 suc=4 rep=10 cfg=6 aju=5",
  ["Rápido y escalable; etiquetas ordenadas por una variable de negocio"], ["Sensible a atípicos y a la escala", "Clusters esféricos y de tamaño parecido"],
  "Segmentar con muchas filas y variables escaladas.", "Hay atípicos fuertes o formas raras (usa K-medoids o DBSCAN).")
f("proyeccion_pca", "segmentar",
  "con=9 efi=8 int=6 rob=3 sup=6 esc=8 vel=8 mem=7 num=9 suc=4 rep=10 cfg=4 aju=8 sas=8",
  ["Visualizar en 2D la estructura"], ["Lineal; solo 2 componentes pueden explicar poco"],
  "Dibujar clusters o ver la estructura.", "Relaciones no lineales fuertes.")

# ------------------------------------------------------------------ gráficos
for nombre, grupo, extra, pro, contra in [
    ("grafico_curva_roc", "graf_clasificacion", "dis=9 int=8 des=6 cfg=7 sas=9", "ROC con AUC y punto de Youden; compara modelos", "No muestra calibración"),
    ("grafico_calibracion", "graf_clasificacion", "cal=10 inc=8 int=9 cfg=6 sas=8", "Observado vs predicho con IC de Wilson", "Depende del nº de grupos"),
    ("grafico_odds_ratios", "graf_clasificacion", "inc=9 int=10 cfg=6 sas=9", "Forest plot listo para informe", "Escala log: explícalo al lector"),
    ("grafico_umbrales", "graf_clasificacion", "dis=8 int=9 cfg=4 sas=9", "Sensibilidad/especificidad según umbral", "Supone costes iguales en Youden"),
    ("grafico_residuos_agrupados", "graf_modelos", "cal=8 int=7 rob=6 cfg=5", "Residuos útiles en modelos 0/1", "Necesita n moderado"),
    ("grafico_seleccion_k", "segmentar", "int=8 cfg=3", "Silhouette y codo en un vistazo", "Solo dibuja la tabla"),
    ("grafico_clusters_pca", "segmentar", "int=7 cfg=6", "Clusters en PC1-PC2 con centros", "Proyección lineal: puede solapar clusters separados"),
    ("grafico_vif", "colinealidad", "int=9 cfg=3 sas=8", "VIF con cortes 5 y 10", "—"),
    ("grafico_region_rechazo", "graf_contrastes", "int=10 cfg=8", "Explica el p-valor y la región de rechazo", "Didáctico, no analítico"),
    ("grafico_potencia", "graf_contrastes", "pot=8 int=10 cfg=6 sas=8", "α, β y potencia como áreas", "Solo para el t-test"),
    ("grafico_curva_potencia", "graf_contrastes", "pot=8 int=9 cfg=6 sas=8", "Potencia frente a n con el n objetivo", "Solo para el t-test"),
    ("grafico_comparar_grupos", "comparar_grupos", "alf=8 rob=8 int=10 cfg=5 sas=8", "Caja + puntos con el test correcto en el título", "Mismos límites que elegir_contraste"),
    ("grafico_qq", "graf_modelos", "pot=6 int=8 sup=8 cfg=3 sas=8", "Q-Q con Shapiro-Wilk", "Con n grande Shapiro rechaza siempre"),
    ("grafico_fdr", "multiples", "alf=9 int=9 cfg=4 sas=8", "Visualiza BH frente a Bonferroni", "—"),
    ("grafico_regresion_simple", "graf_modelos", "inc=9 cob=8 int=10 rob=3 sup=3 cfg=5 sas=8", "Recta con bandas de confianza y predicción", "Solo una X; MCO sensible a atípicos"),
    ("grafico_diagnostico_residuos", "graf_modelos", "rob=8 int=8 cfg=5 sas=9", "Los 4 paneles clásicos + Cook", "Hay que saber leerlos"),
    ("grafico_kaplan_meier", "superv_curvas", "cen=10 inc=9 int=10 cfg=7 sas=9", "Curvas KM con censuras, IC y log-rank", "—"),
]:
    f(nombre, grupo, G + extra, [pro], [contra] if contra != "—" else [],
      "Necesitas la figura para un notebook o un informe.", "—")


# ------------------------------------------------------------------ finanzas (0.9)
A = "esc=10 vel=10 mem=10 num=9 rep=10 aju=10 "
f("var_tvar", "medir_riesgo", "ins=6 con=9 inc=3 int=9 rob=5 sup=7 esc=10 vel=10 mem=9 num=9 suc=6 rep=10 cfg=7 aju=10 sas=8",
  ["Histórico, normal o Cornish-Fisher; TVaR (coherente) además del VaR"], ["El normal subestima colas pesadas; Cornish-Fisher se dispara con curtosis alta"],
  "Riesgo de mercado o de suscripción con datos suficientes.", "Niveles extremos (≥ 99.5 %) con pocos datos: usa EVT.")
f("ajustar_gpd", "medir_riesgo", "con=8 efi=7 nas=7 inc=6 peq=4 int=7 rob=6 sup=7 esc=9 vel=8 mem=9 num=7 suc=6 rep=10 cfg=7 aju=4",
  ["EVT: extrapola la cola más allá de los datos", "VaR/TVaR extremos"], ["Muy sensible al umbral", "Pocos excesos = mucha incertidumbre"],
  "Cuantiles extremos de siniestros grandes o pérdidas de mercado.", "Pocos datos en la cola (< 50 excesos).")
f("estimador_hill", "medir_riesgo", "con=7 inc=6 peq=4 int=7 sup=7 esc=8 vel=9 mem=9 num=9 suc=6 rep=10 cfg=4 aju=4",
  ["Índice de cola sin ajustar un modelo completo"], ["Solo colas tipo Pareto (ξ > 0)", "Elegir k es delicado"], "Diagnosticar lo pesada que es la cola.", "—")
f("funcion_exceso_medio", "medir_riesgo", A + "int=8 sup=10 suc=6 cfg=3",
  ["Diagnóstico visual del tipo de cola y del umbral"], ["Ruidosa en umbrales altos"], "Antes del POT/GPD.", "—")
f("simular_copula", "medir_riesgo", "con=9 int=7 sup=8 esc=9 vel=9 mem=8 num=8 suc=6 rep=10 cfg=9 aju=7",
  ["Cuatro familias: gaussiana, t, Clayton (cola inferior), Gumbel (cola superior)", "Marginales a elección"], ["Elegir familia y parámetro (no ajusta)"],
  "Agregar riesgos con dependencia en colas (capital).", "—")
f("prueba_estres", "medir_riesgo", A + "int=10 sup=10 suc=6 cfg=7",
  ["Pérdida por escenario y factor, ordenada"], ["Lineal (no recalcula valoraciones)"], "Choques estándar de Solvencia II o escenarios propios.", "Instrumentos no lineales (opciones): revalora.")
f("matriz_covarianzas", "carteras", "ins=6 con=9 efi=8 ecm=9 int=7 est=9 rob=5 sup=6 esc=9 vel=9 mem=8 num=9 suc=5 rep=10 cfg=4 aju=10",
  ["Ledoit-Wolf: covarianza estable con muchos activos"], ["Sesgada a propósito (contracción)"], "Cualquier optimización de carteras.", "—")
f("frontera_eficiente", "carteras", "con=6 inc=2 int=8 est=2 rob=2 sup=3 esc=7 vel=6 mem=9 num=7 suc=5 rep=10 cfg=8 aju=8",
  ["Frontera, mínima varianza y tangente con o sin cortos"], ["Pesos MUY sensibles a las medias estimadas", "Solo media-varianza"],
  "Ilustrar el compromiso riesgo-rentabilidad.", "Decidir pesos reales con medias históricas (Black-Litterman).")
f("beta_capm", "carteras", "ins=8 con=9 efi=9 nas=9 alf=8 cob=8 inc=10 int=10 rob=4 sup=5 esc=10 vel=10 mem=10 num=9 suc=6 rep=10 cfg=3 aju=10",
  ["β, α con p-valor y SE robustos"], ["β no es estable en el tiempo"], "Riesgo sistemático de un activo o fondo.", "—")
f("modelo_factores", "carteras", "ins=8 con=9 efi=9 alf=8 inc=10 int=9 rob=4 sup=5 esc=9 vel=9 mem=9 num=8 suc=6 rep=10 cfg=6 aju=10",
  ["Exposiciones a varios factores (APT, Fama-French)"], ["Hay que proporcionar los factores"], "Atribuir el riesgo de una cartera.", "—")
f("black_litterman", "carteras", "con=7 inc=4 int=8 est=8 rob=7 sup=5 esc=10 vel=10 mem=10 num=8 suc=5 rep=10 cfg=8 aju=5",
  ["Pesos razonables combinando equilibrio y opiniones"], ["Hay que fijar τ, aversión y confianza"], "Construir carteras con opiniones del gestor.", "—")
f("dominancia_estocastica", "carteras", A + "inc=2 int=8 rob=8 sup=10 suc=6 cfg=3",
  ["Comparación sin elegir función de utilidad"], ["A menudo ninguna domina a la otra"], "Ordenar alternativas de inversión o reaseguro.", "—")
f("equivalente_cierto", "carteras", A + "int=10 sup=6 suc=6 cfg=7",
  ["Prima de riesgo: por qué existe el seguro"], ["Depende de la utilidad y la aversión elegidas"], "Docencia de utilidad y seguro.", "—")
f("fraccion_anio", "renta_fija", A + "int=10 sup=10 suc=6 cfg=8",
  ["Cuatro convenciones (act/365, act/360, 30/360, act/act)"], ["30/360 en versión europea"], "Devengos y valoración por fechas.", "—")
f("precio_bono", "renta_fija", A + "ins=10 int=10 sup=6 suc=6 cfg=6",
  ["Precio, duraciones y convexidad"], ["TIR plana (sin curva)"], "Valorar y medir la sensibilidad a tipos de un bono.", "—")
f("tir_bono", "renta_fija", A + "int=10 sup=6 suc=5 cfg=5", ["TIR por raíz numérica"], ["Supone reinversión a la propia TIR"], "Rentabilidad al vencimiento.", "—")
f("bootstrapping_etti", "renta_fija", A + "ins=10 int=9 sup=6 suc=4 cfg=3",
  ["Curva cupón cero y forwards desde bonos"], ["Necesita un bono por plazo anual (sin interpolación)"], "Construir la curva para descontar.", "Plazos irregulares (Nelson-Siegel).")
f("inmunizacion", "renta_fija", A + "int=9 sup=5 suc=5 cfg=4",
  ["Redington con dos bonos: valor y duración casados"], ["Solo protege ante movimientos paralelos pequeños"], "Casar activos y pasivos (vida, pensiones).", "—")
f("valorar_swap", "renta_fija", A + "ins=10 int=9 sup=6 suc=5 cfg=5",
  ["Valor y tipo swap par desde la curva"], ["Pagos anuales/regulares; sin margen ni CVA"], "Valorar un IRS o un FRA.", "—")
f("black_scholes", "opciones", A + "ins=10 int=9 sup=3 suc=6 cfg=7 sas=6",
  ["Precio y griegas cerradas"], ["Volatilidad constante y log-normalidad (sonrisa de volatilidad ignorada)"], "Opciones europeas y coberturas.", "Opciones americanas o exóticas (árbol o Monte Carlo).")
f("arbol_binomial", "opciones", "ins=9 con=10 int=9 sup=4 esc=8 vel=8 mem=9 num=9 suc=6 rep=10 cfg=8 aju=8",
  ["Valoración neutral al riesgo paso a paso", "Ejercicio americano"], ["Converge despacio (oscila)"], "Opciones americanas y docencia del neutral al riesgo.", "—")
for nombre, grupo, extra, pro in [
    ("grafico_frontera_eficiente", "carteras", "int=9 cfg=3", "Frontera con mínima varianza, tangente y activos"),
    ("grafico_copula", "medir_riesgo", "int=8 cfg=2", "Nube (u, v) de la cópula"),
]:
    f(nombre, grupo, G + extra, [pro], [], "Necesitas la figura para un notebook o un informe.", "—")

# ------------------------------------------------------------------ actuarial (0.9)
A = "esc=10 vel=10 mem=10 num=9 rep=10 aju=10 "
f("qx_ley", "biometria", A + "int=9 par=10 sup=3 suc=6 cfg=6",
  ["qx de Gompertz/Makeham para ejemplos o suavizar"], ["Una ley de 2-3 parámetros no capta la joroba de accidentes ni las edades muy altas"],
  "Tabla de ejemplo o extrapolar a edades extremas.", "Necesitas la tabla oficial (PER2020, PASEM…): úsala directamente.")
f("tabla_mortalidad", "biometria", A + "ins=10 con=10 int=10 sup=9 suc=7 cfg=5 sas=7",
  ["Tabla completa desde qx: lx, dx, Lx, Tx, esperanzas completa y abreviada"], ["Lx con UDD"], "Cualquier cálculo de vida.", "—")
f("ajustar_ley_mortalidad", "biometria", "con=9 efi=9 nas=8 inc=6 peq=6 int=9 par=10 rob=4 sup=4 esc=10 vel=8 mem=10 num=6 suc=5 rep=10 cfg=6 aju=8",
  ["MV de Poisson con exposiciones (o MC sobre qx)", "Suaviza la experiencia propia"], ["La ley impone forma: mal en edades jóvenes", "Sensible al punto inicial"],
  "Graduar la mortalidad observada de una cartera.", "Edades con muy pocos datos sin comprobar el ajuste.")
f("probabilidad_supervivencia", "biometria", A + "int=9 sup=6 suc=5 cfg=7",
  ["Edades y plazos fraccionarios con UDD, fuerza constante o Balducci"], ["Las tres hipótesis difieren poco salvo en edades altas"], "Pagos fraccionados o fechas exactas.", "—")
f("vida_futura", "biometria", A + "ins=10 con=10 inc=8 int=9 sup=9 suc=6 cfg=2",
  ["K_x como variable aleatoria: probabilidades, esperanza, varianza y cuantiles"], ["Abreviada (años completos)"], "Explicar la incertidumbre de la duración de vida.", "—")
f("tabla_conjunta", "biometria", A + "int=9 sup=3 suc=6 cfg=4",
  ["Vida conjunta y último superviviente"], ["Supone vidas INDEPENDIENTES (subestima la dependencia real de parejas)"], "Pensiones de viudedad y seguros sobre dos vidas.", "—")
f("decrementos_multiples", "biometria", A + "int=8 sup=5 suc=6 cfg=6",
  ["Pasa de tasas dependientes a independientes y viceversa (UDD)"], ["Hipótesis UDD en cada decremento"], "Muerte, invalidez y rescate a la vez.", "—")
f("conmutados", "vida", A + "ins=10 int=8 sup=9 suc=6 cfg=3 sas=7",
  ["Dx, Nx, Cx, Mx, Sx, Rx en una tabla"], ["Herramienta de cálculo manual: las funciones directas son más claras"], "Comprobar a mano o reproducir apuntes.", "—")
f("seguro_vida", "vida", A + "ins=10 inc=8 int=10 sup=8 suc=7 cfg=9",
  ["Vida entera, temporal, dotal puro, mixto, diferido; final del año o momento de la muerte", "Varianza de la pérdida"],
  ["Interés técnico constante"], "Prima única pura de cualquier seguro de vida.", "Interés estocástico (necesitas escenarios).")
f("renta_actuarial", "vida", A + "ins=10 int=10 sup=8 suc=7 cfg=10",
  ["Anticipada/vencida, diferida, creciente y fraccionada"], ["Fraccionamiento aproximado (Woolhouse 2 términos)"], "Pensiones y rentas vitalicias.", "—")
f("prima_neta", "vida", A + "ins=10 int=10 sup=8 suc=7 cfg=7",
  ["Principio de equivalencia: prima única y anual"], ["Sin gastos (ver prima_tarifa)"], "Prima pura nivelada.", "—")
f("provision_matematica", "vida", A + "ins=10 int=10 sup=8 suc=7 cfg=7 sas=7",
  ["Provisión prospectiva año a año"], ["Bases técnicas de primas = de provisión"], "Provisiones de un producto de vida.", "—")
f("prima_tarifa", "vida", A + "int=10 sup=7 suc=7 cfg=9",
  ["Prima comercial con gastos α, β, γ y prima de inventario"], ["Estructura de gastos simplificada"], "Tarifa comercial de un producto de vida.", "—")
f("sensibilidad_longevidad", "vida", A + "int=10 rob=6 sup=7 suc=7 cfg=6",
  ["Choque de longevidad tipo Solvencia II en un paso"], ["Choque uniforme en todas las edades"], "Medir el riesgo de longevidad de una cartera de rentas.", "—")
f("chain_ladder", "reservas", "ins=7 con=8 cob=7 inc=10 peq=5 int=10 par=8 est=6 rob=3 sup=4 esc=9 vel=10 mem=10 num=8 suc=6 rep=10 cfg=5 aju=10 sas=9",
  ["Reserva por año de origen con el error de Mack (reproduce Taylor-Ashe)"], ["Supone patrón de desarrollo estable", "Sensible a factores con pocos datos"],
  "Provisiones de siniestros (IBNR) con triángulos.", "Cambios de cartera, inflación o gestión de siniestros (ajusta antes).")
f("bootstrap_chain_ladder", "reservas", "ins=7 con=8 cob=7 inc=10 peq=5 int=8 est=6 rob=4 sup=5 esc=8 vel=8 mem=8 num=8 suc=5 rep=10 cfg=6 aju=8",
  ["Distribución completa de la reserva: percentil 99.5 %", "Incluye error de proceso (ODP)"], ["Los residuos deben ser intercambiables", "Pocos años = pocos residuos"],
  "Capital de reservas (Solvencia II) o rangos de reserva.", "Triángulos con incrementales negativos frecuentes.")
f("recursion_panjer", "agregada", "ins=10 con=10 inc=7 int=8 sup=7 esc=6 vel=8 mem=9 num=8 suc=5 rep=10 cfg=8 aju=10",
  ["Distribución EXACTA de la siniestralidad agregada"], ["Necesita discretizar la severidad", "Coste O(s²)"],
  "Primas stop-loss o VaR exacto con frecuencia de la clase (a,b,0).", "Severidad continua sin discretizar (simula).")
f("simular_siniestralidad_agregada", "agregada", "ins=9 con=10 inc=8 int=9 rob=8 sup=10 esc=8 vel=7 mem=6 num=9 suc=6 rep=10 cfg=10 aju=9",
  ["Cualquier frecuencia y severidad; VaR y TVaR"], ["Error de Monte Carlo en las colas"], "Riesgo de suscripción, reaseguro, capital.", "—")
f("probabilidad_ruina", "agregada", A + "inc=6 int=8 sup=3 suc=5 cfg=7",
  ["ψ(u) exacta (exponencial), cota de Lundberg y ruina en horizonte finito"], ["Modelo clásico muy simplificado", "Sin coeficiente de ajuste con colas pesadas"],
  "Docencia y orden de magnitud del capital necesario.", "Cálculo de capital real (simula la cartera completa).")
f("prima_por_principios", "agregada", A + "int=9 rob=5 sup=8 suc=6 cfg=9",
  ["Siete principios de prima a la vez"], ["Exponencial y Esscher no existen con colas pesadas"], "Comparar recargos de seguridad.", "—")
f("credibilidad_buhlmann", "credibilidad", "ins=8 con=9 efi=8 ecm=9 inc=6 int=9 est=8 rob=5 sup=7 dep=9 esc=9 vel=9 mem=9 num=9 suc=6 rep=10 cfg=6 aju=10",
  ["Bühlmann-Straub con exposiciones: Z, k y prima de credibilidad"], ["Supone riesgos intercambiables y varianzas homogéneas"],
  "Tarificar por experiencia (flotas, colectivos).", "Muy pocos riesgos (la VHM se estima mal).")
f("tasas_especificas", "demografia", A + "ins=10 con=10 cob=9 inc=10 peq=9 int=10 sup=8 suc=6 cfg=4",
  ["Tasas con IC exacto de Poisson"], ["Sin estandarizar no son comparables entre poblaciones"], "Mortalidad o incidencia por edad o grupo.", "—")
f("estandarizar_tasas", "demografia", A + "ins=8 con=9 cob=8 inc=10 peq=7 int=10 sup=7 suc=6 cfg=5",
  ["Directa (tasa con IC) e indirecta (SMR con IC exacto)"], ["La directa necesita tasas por edad estables"], "Comparar la mortalidad de una cartera con una tabla o región.", "—")
f("exposicion_por_edad", "demografia", "ins=10 con=10 int=9 sup=9 cen=9 esc=7 vel=7 mem=8 num=10 suc=6 rep=10 cfg=5 aju=10",
  ["Exposición central exacta por edad (Lexis)", "Base para la experiencia propia de mortalidad"], ["Asigna el evento a la edad cumplida"],
  "Construir qx brutas de tu cartera.", "—")
f("indicadores_fecundidad", "demografia", A + "int=10 sup=9 suc=6 cfg=4",
  ["ISF, tasa bruta de reproducción y edad media"], ["Medidas transversales (de un año)"], "Demografía de fecundidad.", "—")
f("proyeccion_leslie", "demografia", A + "int=8 sup=4 suc=5 cfg=5",
  ["Proyección por cohortes y crecimiento a largo plazo"], ["Tasas constantes y sin migraciones"], "Proyecciones de población sencillas.", "Escenarios con migración o tasas cambiantes.")
for nombre, grupo, extra, pro in [
    ("grafico_tabla_mortalidad", "biometria", "int=9 cfg=4", "qx en log y supervivientes de varias tablas"),
    ("grafico_reserva_bootstrap", "reservas", "int=9 cfg=2", "Distribución de la reserva con el 99.5 %"),
]:
    f(nombre, grupo, G + extra, [pro], [], "Necesitas la figura para un notebook o un informe.", "—")

# ------------------------------------------------------------------ ml (0.9)
f("ajustar_arbol_decision", "modelo_ml",
  "con=7 inc=1 gen=6 cal=6 dis=6 par=7 int=9 est=2 rob=7 sup=10 des=6 esc=8 vel=8 mem=8 num=10 suc=7 rep=10 cfg=7 aju=6 sas=8",
  ["Reglas legibles; capta interacciones sin especificarlas", "Poda por CV"], ["Muy inestable (otra muestra, otro árbol)", "Peor predictor que los conjuntos"],
  "Explicar segmentos con reglas sencillas.", "Lo que quieres es la mejor predicción (boosting o random forest).")
f("ajustar_random_forest", "modelo_ml",
  "con=8 inc=2 gen=7 cal=7 dis=9 par=3 int=4 est=8 rob=8 sup=10 des=6 esc=6 vel=5 mem=5 num=10 suc=7 rep=10 cfg=7 aju=8",
  ["Muy buen rendimiento sin apenas ajuste", "Error OOB gratis"], ["Caja negra", "Importancias por impureza sesgadas", "Memoria con muchos árboles"],
  "Predicción robusta con poco tuning.", "Necesitas explicar coeficientes o extrapolar.")
f("ajustar_gradient_boosting", "modelo_ml",
  "con=8 inc=2 gen=8 cal=7 dis=10 par=4 int=4 est=7 rob=7 sup=10 des=7 esc=9 vel=9 mem=8 num=9 suc=9 rep=10 cfg=8 aju=6",
  ["Suele ser el mejor en datos tabulares", "Rápido (histogramas) y admite NaN", "Parada temprana"], ["Caja negra", "Más hiperparámetros"],
  "Máxima capacidad predictiva en tablas.", "Pocos datos o necesidad regulatoria de explicar cada coeficiente.")
f("ajustar_adaboost", "modelo_ml",
  "con=7 inc=2 gen=7 cal=4 dis=7 par=5 int=4 est=6 rob=3 sup=10 esc=7 vel=7 mem=8 num=10 suc=5 rep=10 cfg=5 aju=6",
  ["Histórico y sencillo (tocones)"], ["Sensible a etiquetas erróneas y atípicos", "Probabilidades mal calibradas"],
  "Docencia o comparación.", "Datos con ruido en la etiqueta (gradient boosting).")
f("ajustar_knn", "modelo_ml",
  "con=7 inc=1 gen=7 cal=6 dis=7 int=5 est=6 rob=6 sup=10 esc=4 vel=6 mem=6 num=10 suc=5 rep=10 cfg=5 aju=6",
  ["Sin supuestos; k por CV", "Escala las X"], ["Lento al predecir con n grande", "Sufre con muchas variables (dimensión)"],
  "Pocas variables numéricas y n moderado.", "Muchas variables o n muy grande.")
f("ajustar_svm", "modelo_ml",
  "con=8 inc=1 gen=7 cal=6 dis=8 int=3 est=7 rob=6 sup=9 esc=2 vel=4 mem=5 num=9 suc=4 rep=10 cfg=7 aju=4",
  ["Fronteras no lineales con kernel", "Bueno con n moderado y p grande"], ["O(n²)-O(n³): no para n grande", "Sensible a C y gamma"],
  "n ≤ 20 000 con fronteras complejas.", "n grande o necesitas probabilidades bien calibradas.")
f("ajustar_naive_bayes", "modelo_ml",
  "con=6 inc=1 gen=8 cal=3 dis=5 par=9 int=6 est=9 rob=6 sup=4 esc=10 vel=10 mem=10 num=10 suc=6 rep=10 cfg=2 aju=10",
  ["Rapidísimo y estable con pocos datos"], ["Supone independencia: no ve interacciones", "Probabilidades extremas"],
  "Línea base rápida o texto/variables casi independientes.", "Variables correlacionadas o con interacciones.")
f("ajustar_red_neuronal", "modelo_ml",
  "con=7 inc=1 gen=6 cal=6 dis=8 par=2 int=1 est=5 rob=5 sup=9 esc=7 vel=6 mem=7 num=7 suc=4 rep=8 cfg=8 aju=3",
  ["Muy flexible", "Parada temprana"], ["Caja negra total", "Sensible a hiperparámetros y escala", "Rara vez supera al boosting en tablas"],
  "Muchas interacciones suaves o como base de stacking.", "Datos tabulares pequeños o necesidad de explicar.")
f("ajustar_stacking", "modelo_ml",
  "con=8 inc=1 gen=8 cal=8 dis=9 par=1 int=2 est=7 rob=7 sup=10 esc=4 vel=3 mem=5 num=9 suc=6 rep=10 cfg=5 aju=5",
  ["Combina logística, bosque y boosting fuera de fold"], ["Coste alto", "Ganancia pequeña si un modelo domina"],
  "Exprimir las últimas décimas de AUC.", "Necesitas simplicidad o explicabilidad.")
f("comparar_clasificadores", "evaluar_ml",
  "con=8 inc=6 gen=9 cal=7 dis=9 int=8 sup=10 esc=4 vel=3 mem=6 num=9 suc=6 rep=10 cfg=8 aju=8",
  ["Misma CV para todos: AUC, log-loss, Brier y tiempo"], ["Hiperparámetros por defecto (razonables, no óptimos)"],
  "Elegir familia de modelo antes de afinar.", "—")
f("importancia_permutacion", "explicar_ml",
  "con=8 inc=6 int=9 est=6 rob=7 sup=10 esc=5 vel=4 mem=7 num=10 suc=6 rep=10 cfg=5 aju=9",
  ["Vale para cualquier modelo; mide en test", "Con desviación entre repeticiones"], ["Variables correlacionadas se reparten la importancia"],
  "Explicar qué usa un modelo de caja negra.", "Variables muy correlacionadas (agrupa o usa SHAP).")
f("dependencia_parcial", "explicar_ml",
  "con=7 inc=2 int=9 rob=6 sup=8 esc=5 vel=5 mem=7 num=10 suc=6 rep=10 cfg=6 aju=9",
  ["Efecto marginal de una variable", "Curvas ICE: revela interacciones"], ["Extrapola si las variables están correlacionadas"],
  "Enseñar a negocio la forma del efecto.", "—")
for nombre, grupo, extra, pro in [
    ("grafico_importancias", "explicar_ml", "int=9 cfg=3", "Barras de importancia con sd"),
    ("grafico_dependencia_parcial", "explicar_ml", "int=9 cfg=2", "DP con curvas ICE"),
]:
    f(nombre, grupo, G + extra, [pro], [], "Necesitas la figura para un notebook o un informe.", "—")

# ------------------------------------------------------------------ multivariante (0.9)
f("distancia_mahalanobis", "atipicos_mv",
  "alf=7 inc=7 rob=9 sup=5 int=7 esc=7 vel=7 mem=7 num=8 suc=6 rep=10 cfg=6 aju=8 sas=8",
  ["Detecta atípicos que ninguna variable por separado muestra", "MCD robusto: no se enmascaran"], ["Supone forma elíptica (normal)", "MCD lento con muchas variables"],
  "Fraude o errores de datos en varias variables a la vez.", "Variables muy asimétricas sin transformar.")
f("contraste_hotelling", "comparar_multivariante",
  "alf=8 pot=8 inc=8 peq=7 int=7 rob=3 sup=4 esc=10 vel=10 mem=10 num=8 suc=6 rep=10 cfg=2 aju=10 sas=9",
  ["Compara vectores de medias controlando el error tipo I", "Ve diferencias conjuntas"], ["Normalidad multivariante y covarianza común"],
  "Dos grupos medidos en varias variables.", "Muchas variables y pocos datos (p cerca de n).")
f("contraste_box_m", "comparar_multivariante",
  "alf=5 pot=8 inc=7 int=6 sup=2 esc=9 vel=9 mem=9 num=7 suc=6 rep=10 cfg=2 aju=10 sas=9",
  ["Comprueba el supuesto de covarianzas iguales"], ["Hipersensible a la no normalidad", "Con n grande rechaza siempre"],
  "Antes de LDA o MANOVA.", "Datos no normales (ignóralo y usa Pillai o QDA).")
f("manova", "comparar_multivariante",
  "alf=8 pot=8 inc=8 peq=6 int=7 rob=5 sup=4 esc=8 vel=8 mem=8 num=8 suc=6 rep=10 cfg=4 aju=10 sas=10",
  ["Cuatro estadísticos (Pillai el más robusto)"], ["Normalidad multivariante", "Interpretar exige análisis posteriores"],
  "Varias respuestas correlacionadas y un factor.", "Una sola respuesta (ANOVA).")
f("correlacion_canonica", "reduccion",
  "con=9 alf=8 inc=8 int=6 sup=5 esc=8 vel=9 mem=9 num=8 suc=6 rep=10 cfg=2 aju=10 sas=9",
  ["Relación entre dos bloques de variables con contrastes por dimensión"], ["Difícil de interpretar más allá de la primera"],
  "¿Cómo se relaciona un bloque (perfil) con otro (consumo)?", "Un bloque es una sola variable (regresión).")
f("pca_completo", "reduccion",
  "con=9 efi=8 inc=2 int=7 par=8 est=7 rob=3 sup=7 esc=8 vel=9 mem=8 num=9 suc=5 rep=10 cfg=6 aju=8 sas=10",
  ["Varianza, cargas, puntuaciones y criterio de Kaiser", "Contribución de cada variable"], ["Lineal y sensible a atípicos y a la escala"],
  "Resumir muchas variables correlacionadas o visualizar.", "Quieres factores interpretables (factorial con rotación).")
f("adecuacion_factorial", "escalas",
  "alf=7 inc=7 int=8 sup=5 esc=9 vel=9 mem=9 num=7 suc=6 rep=10 cfg=1 aju=10 sas=9",
  ["KMO y Bartlett: ¿merece la pena factorizar?"], ["Bartlett rechaza casi siempre con n grande"], "Antes de un análisis factorial.", "—")
f("analisis_factorial", "reduccion",
  "con=8 efi=8 nas=7 ide=5 inc=4 int=8 par=8 est=6 rob=4 sup=4 esc=7 vel=7 mem=8 num=7 suc=5 rep=10 cfg=6 aju=5 sas=9",
  ["Factores latentes con rotación varimax", "Comunalidades y adecuación incluidas"], ["Elegir nº de factores y rotación es subjetivo", "No identificable sin rotación"],
  "Encuestas, escalas o variables que miden constructos.", "Solo quieres reducir dimensión (PCA).")
f("alfa_cronbach", "escalas",
  "ins=7 con=9 cob=8 inc=9 int=9 sup=5 esc=10 vel=10 mem=9 num=9 suc=6 rep=10 cfg=2 aju=10 sas=10",
  ["Fiabilidad con IC, α si se elimina e ítem-total"], ["Supone unidimensionalidad y tau-equivalencia"], "Validar una escala de encuesta.", "La escala tiene varias dimensiones (α por subescala).")
f("escalamiento_multidimensional", "reduccion",
  "con=8 inc=2 int=7 sup=8 esc=2 vel=5 mem=3 num=8 suc=5 rep=10 cfg=4 aju=9 sas=8",
  ["Mapa a partir de cualquier matriz de distancias"], ["O(n²) memoria y O(n³) tiempo"], "Visualizar similitudes (marcas, productos).", "n grande (usa PCA).")
f("analisis_correspondencias", "reduccion",
  "con=9 inc=4 int=8 sup=9 esc=10 vel=10 mem=10 num=9 suc=6 rep=10 cfg=2 aju=10 sas=9",
  ["Mapa de dos categóricas: qué niveles van juntos"], ["Las distancias fila-columna no se interpretan directamente"],
  "Explorar una tabla de contingencia grande.", "—")
f("clustering_jerarquico", "segmentar",
  "con=6 inc=2 int=8 est=6 rob=4 sup=6 esc=2 vel=4 mem=2 num=9 suc=4 rep=10 cfg=7 aju=6 sas=9",
  ["No hay que fijar k antes; dendrograma", "Correlación cofenética"], ["O(n²) memoria: no para n grande", "Ward favorece grupos esféricos"],
  "n moderado y quieres ver la estructura a varios niveles.", "Más de ~10 000 filas.")
f("mezclas_gaussianas", "segmentar",
  "con=8 efi=8 inc=7 int=7 par=7 est=6 rob=3 sup=5 esc=6 vel=5 mem=7 num=7 suc=4 rep=10 cfg=8 aju=7",
  ["Elige k por BIC", "Probabilidades de pertenencia y clusters elípticos"], ["Supone normalidad dentro de cada grupo", "Varios inicios (lento)"],
  "Segmentos de forma o tamaño distintos y quieres incertidumbre.", "Atípicos fuertes o variables muy asimétricas.")
f("analisis_discriminante", "multiclase",
  "ins=7 con=9 efi=9 inc=5 gen=7 cal=6 dis=8 par=8 int=8 est=8 rob=4 sup=4 esc=9 vel=9 mem=8 num=8 suc=5 rep=10 cfg=4 aju=10 sas=10",
  ["LDA estable con clases bien separadas (donde la logística falla)", "Funciones discriminantes interpretables"], ["Normalidad y covarianzas iguales (LDA)"],
  "Clasificar en varios grupos con predictores continuos.", "Predictores categóricos o muy asimétricos (logística).")
for nombre, grupo, extra, pro in [
    ("grafico_dendrograma", "segmentar", "int=8 cfg=3 sas=9", "Dendrograma con línea de corte"),
    ("grafico_biplot", "reduccion", "int=8 cfg=3 sas=8", "Observaciones y cargas en el mismo plano"),
]:
    f(nombre, grupo, G + extra, [pro], [], "Necesitas la figura para un notebook o un informe.", "—")

# ------------------------------------------------------------------ preprocesado añadido (0.9)
f("resumen_faltantes", "faltantes",
  "alf=7 inc=6 int=9 sup=6 esc=7 vel=7 mem=7 num=8 suc=10 rep=10 cfg=3 aju=10 sas=8",
  ["Porcentajes, patrones y test MCAR de Little de una vez"], ["Little supone normalidad multivariante", "No distingue MAR de MNAR"],
  "Antes de decidir cómo tratar los faltantes.", "—")
f("contraste_mcar_little", "faltantes",
  "alf=7 pot=7 inc=7 peq=5 int=7 sup=4 esc=6 vel=7 mem=7 num=7 suc=10 rep=10 cfg=2 aju=10 sas=8",
  ["Contraste global MCAR con EM"], ["Supone normalidad", "No rechazar no prueba MCAR"], "Documentar el mecanismo de faltantes.", "Variables categóricas (no entran).")
f("imputar", "faltantes",
  "ins=5 con=6 inc=1 cal=6 int=8 rob=6 sup=6 esc=7 vel=7 mem=7 num=8 suc=10 rep=10 cfg=9 aju=8",
  ["Mediana, iterativa (MICE de una pasada) o k-NN", "Indicadores de faltante opcionales"],
  ["La simple subestima la varianza", "k-NN es lento con n grande"], "Predicción o preparación rápida de datos.", "Inferencia (p-valores, IC): imputación múltiple.")
f("imputacion_multiple", "faltantes",
  "ins=8 con=9 cob=8 inc=10 peq=6 int=8 rob=5 sup=5 esc=4 vel=3 mem=6 num=8 suc=10 rep=10 cfg=6 aju=7 sas=9",
  ["Reglas de Rubin: incertidumbre correcta", "Fracción de información faltante por coeficiente"],
  ["m modelos (lento)", "Supone MAR y un modelo de imputación razonable"], "Inferencia con faltantes no despreciables.", "Faltantes MNAR graves.")
f("agrupar_categorias_raras", "preparar",
  "est=8 int=9 par=8 rob=8 sup=10 esc=10 vel=10 mem=9 num=10 suc=9 rep=10 cfg=7 aju=7",
  ["Evita dummies con casi ningún caso"], ["Mezcla niveles que quizá no se parecen"], "Categóricas con cola larga de niveles.", "—")
f("codificar_por_objetivo", "preparar",
  "ins=6 con=8 gen=8 dis=8 int=6 est=7 rob=7 sup=8 esc=9 vel=9 mem=8 num=10 suc=8 rep=10 cfg=7 aju=6",
  ["Alta cardinalidad en una sola columna", "Suavizado bayesiano y fuera de fold (sin fuga)"], ["Pierde interpretabilidad por nivel", "Elegir el suavizado"],
  "Códigos postales, modelos de vehículo, profesiones.", "Pocas categorías (dummies son más claras).")
f("smote", "desbalanceo",
  "cal=2 gen=6 dis=7 des=9 int=6 est=5 rob=4 sup=7 esc=6 vel=7 mem=6 num=9 suc=4 rep=10 cfg=7 aju=6",
  ["Casos sintéticos en vez de duplicados"], ["Descalibra", "Solo variables numéricas; puede crear casos imposibles"],
  "Clasificadores que sufren con clases muy raras (solo en train).", "Necesitas probabilidades calibradas o el modelo acepta pesos.")

# ------------------------------------------------------------------ selección y validación (0.9)
f("seleccion_backward", "seleccion_modelo",
  "con=6 ide=6 alf=3 inc=6 sel=2 gen=6 par=7 int=8 est=3 rob=5 sup=5 esc=3 vel=3 mem=7 num=6 suc=6 rep=10 cfg=8 aju=7 sas=8",
  ["Parte del modelo completo: no se pierde variables que solo sirven juntas", "Protegidas que nunca salen"],
  ["El modelo completo es inestable con muchas variables y pocos eventos", "p-valores finales sesgados", "O(p²) ajustes"],
  "Pocas candidatas con sentido y bastantes datos.", "p grande o pocos eventos (Lasso).")
f("seleccion_por_pvalor", "seleccion_modelo",
  "con=4 alf=2 inc=6 sel=1 gen=5 par=7 int=8 est=3 rob=5 sup=5 esc=3 vel=3 mem=7 num=6 suc=6 rep=10 cfg=7 aju=6 sas=10",
  ["Replica SELECTION=STEPWISE SLENTRY/SLSTAY de SAS"], ["El criterio por p-valor es el menos defendible estadísticamente", "p-valores del modelo final inválidos"],
  "Tienes que reproducir un modelo hecho en SAS.", "Modelo nuevo: usa BIC, Lasso o validación.")
f("mejor_subconjunto", "seleccion_modelo",
  "con=8 alf=3 inc=7 sel=2 gen=6 par=9 int=9 est=4 sup=5 esc=1 vel=2 mem=8 num=8 suc=6 rep=10 cfg=6 aju=8 sas=9",
  ["Explora TODOS los modelos: el mejor de cada tamaño con AIC, BIC, R² ajustado y Cp"], ["2^p modelos: solo p ≤ 15", "Muy optimista (selección)"],
  "Pocas candidatas y quieres ver el compromiso tamaño-ajuste.", "Más de 15 candidatas.")
f("filtrar_varianza_casi_nula", "preparar",
  "rob=8 sup=10 int=9 esc=9 vel=9 mem=9 num=10 suc=9 rep=10 cfg=7 aju=7",
  ["Detecta constantes y casi constantes (criterio de Kuhn)"], ["Una variable rara pero muy predictiva puede marcarse"], "Antes de modelar o validar con folds.", "—")
f("filtrar_correlacion_alta", "colinealidad",
  "par=8 int=8 est=6 rob=6 sup=8 esc=7 vel=8 mem=7 num=9 suc=7 rep=10 cfg=7 aju=6",
  ["Rápido y sin ajustar modelos (solo correlaciones)", "Respeta las protegidas"], ["Solo ve pares (para combinaciones usa VIF)"],
  "Muchas numéricas redundantes antes de modelar.", "Te importa interpretar una variable concreta (protégela).")
f("eliminacion_recursiva", "seleccion_modelo",
  "con=6 gen=7 dis=7 par=7 int=6 est=4 sup=6 esc=5 vel=5 mem=7 num=8 suc=4 rep=10 cfg=5 aju=6",
  ["Elige el nº de variables por validación cruzada"], ["El ranking depende de la colinealidad", "Inestable"],
  "Reducir variables con criterio predictivo.", "Necesitas inferencia.")
f("validacion_cruzada", "validacion",
  "ins=8 con=9 inc=8 peq=8 gen=10 cal=8 dis=9 int=8 est=7 sup=9 des=8 esc=6 vel=5 mem=7 num=8 suc=6 rep=10 cfg=8 aju=8",
  ["Repetida y estratificada: estimación del error con su IC", "Usa todos los datos"], ["Coste k×repeticiones ajustes", "Lo que se aprende de los datos debe ir dentro de cada fold"],
  "Estimar el rendimiento fuera de muestra con n moderado.", "Datos temporales (usa validación por bloques de tiempo).")
f("optimismo_bootstrap", "validacion",
  "ins=8 con=9 inc=6 peq=9 gen=10 cal=8 dis=9 int=8 est=8 sup=9 esc=4 vel=4 mem=7 num=7 suc=6 rep=10 cfg=6 aju=8",
  ["Validación interna de Harrell: corrige el optimismo usando TODOS los datos"], ["Lento (B reajustes)", "No sustituye a una validación externa"],
  "n pequeño donde partir en train/test desperdicia datos.", "n muy grande (un test separado basta).")
f("comparar_modelos_cv", "validacion",
  "alf=9 pot=7 inc=9 peq=7 gen=9 int=8 sup=8 esc=4 vel=3 mem=7 num=8 suc=6 rep=10 cfg=6 aju=8",
  ["Mismos folds para los dos modelos", "t corregido de Nadeau-Bengio (el pareado normal es optimista)"], ["Coste: 2 × k × repeticiones ajustes"],
  "Decidir si un modelo más complejo mejora de verdad.", "Modelos anidados con n grande (basta un LR test).")
f("metricas_regresion", "validacion",
  "inc=2 gen=8 int=10 rob=5 sup=10 esc=10 vel=10 mem=10 num=10 suc=8 rep=10 aju=10",
  ["RMSE, MAE, MAPE, sesgo y R²"], ["MAPE explota con valores cerca de 0"], "Evaluar predicciones numéricas.", "—")
f("tabla_criterios", "validacion",
  "con=7 inc=5 par=9 int=9 sel=3 esc=10 vel=10 mem=10 num=10 suc=6 rep=10 cfg=5 aju=10 sas=9",
  ["AIC, BIC, R² ajustado, Cp y pesos de Akaike en una tabla"], ["Solo compara modelos ajustados a las mismas filas"],
  "Elegir entre varios modelos candidatos.", "—")
f("tabla_ganancia_lift", "discriminacion",
  "dis=9 int=10 des=8 rob=8 sup=10 esc=9 vel=9 mem=8 num=10 suc=6 rep=10 cfg=5 aju=9 sas=8",
  ["La tabla de campañas: lift y % de eventos capturados por decil"], ["No mide calibración"], "Planificar campañas o cortes por score.", "—")
f("estadistico_ks_gini", "discriminacion",
  "alf=8 inc=6 dis=9 int=9 des=8 rob=8 sup=10 esc=9 vel=9 mem=8 num=10 suc=6 rep=10 cfg=2 aju=10 sas=9",
  ["KS y Gini: las métricas estándar de scoring de crédito"], ["KS depende del umbral; ambos ignoran la calibración"], "Informar un modelo de scoring.", "—")
f("curva_precision_recall", "discriminacion",
  "dis=9 int=8 des=10 rob=8 sup=10 esc=9 vel=9 mem=8 num=10 suc=6 rep=10 cfg=2 aju=10",
  ["Más informativa que la ROC con eventos raros", "Umbral de F1 máximo"], ["Depende de la prevalencia (no comparable entre carteras)"],
  "Fraude, siniestros graves u otros eventos raros.", "—")
f("descomposicion_brier", "calibracion",
  "ins=8 con=9 cal=10 dis=8 int=8 sup=10 esc=9 vel=9 mem=8 num=9 suc=6 rep=10 cfg=3 aju=8",
  ["Separa calibración (fiabilidad) de discriminación (resolución)"], ["Depende de la agrupación"], "Entender por qué un Brier es malo.", "—")
f("calibrar_probabilidades", "calibracion",
  "ins=7 con=9 cal=10 gen=7 int=7 sup=8 des=8 esc=9 vel=9 mem=8 num=9 suc=6 rep=10 cfg=6 aju=7",
  ["Platt (2 parámetros) o isotónica", "Arregla probabilidades infladas sin reentrenar"], ["Necesita un conjunto de calibración aparte", "Isotónica sobreajusta con pocos datos"],
  "Probabilidades descalibradas (balanceo, árboles, cambio de cartera).", "No tienes datos de calibración independientes.")
for nombre, grupo, extra, pro in [
    ("grafico_ganancia_lift", "discriminacion", "dis=9 int=10 cfg=3", "Ganancia acumulada y lift por decil"),
    ("grafico_precision_recall", "discriminacion", "dis=9 des=10 int=9 cfg=2", "Precisión-exhaustividad con AP y F1 máximo"),
]:
    f(nombre, grupo, G + extra, [pro], [], "Necesitas la figura para un notebook o un informe.", "—")

# ------------------------------------------------------------------ modelos añadidos (0.9)
f("ajustar_glm_conteo", "tarificacion",
  "ins=8 con=10 efi=9 nas=9 ide=7 alf=8 cob=8 inc=10 peq=6 gen=8 cal=9 dis=7 par=8 int=10 est=8 rob=5 sup=6 dep=4 "
  "esc=8 vel=7 mem=7 num=8 suc=6 rep=10 cfg=9 aju=9 sas=10",
  ["Frecuencia con offset log(exposición): relatividades multiplicativas", "Poisson o binomial negativa con test de sobredispersión"],
  ["Poisson con sobredispersión da SE optimistas", "Supone efectos multiplicativos (enlace log)"],
  "Modelar el número de siniestros por póliza (tarificación).", "Hay exceso de ceros estructurales (modelos ZIP/hurdle).")
f("contraste_sobredispersion", "usar_tarifa",
  "alf=7 pot=8 inc=8 peq=6 int=8 sup=7 esc=9 vel=9 mem=9 num=9 suc=6 rep=10 cfg=2 aju=10",
  ["Cameron-Trivedi: contraste directo de α = 0"], ["Solo detecta sobredispersión lineal o cuadrática en μ"],
  "Tras un GLM Poisson, para decidir si pasar a binomial negativa.", "—")
f("ajustar_glm_severidad", "tarificacion",
  "ins=8 con=10 efi=8 nas=8 alf=7 cob=7 inc=10 peq=6 gen=7 cal=8 par=8 int=10 rob=4 sup=5 esc=8 vel=8 mem=7 num=8 suc=7 rep=10 cfg=8 aju=10 sas=10",
  ["Gamma o inversa gaussiana con enlace log", "Pesos por nº de siniestros"], ["Los siniestros punta distorsionan la media: tratarlos aparte (capar o EVT)"],
  "Coste medio por siniestro en función de las variables de tarifa.", "Cola muy pesada (mejor separar grandes siniestros).")
f("ajustar_tweedie", "tarificacion",
  "ins=7 con=9 efi=7 nas=8 inc=9 peq=5 gen=7 cal=8 par=9 int=9 rob=4 sup=5 esc=8 vel=7 mem=7 num=7 suc=6 rep=10 cfg=6 aju=6 sas=9",
  ["Un solo modelo para la prima pura (ceros + importes)"], ["Supone las mismas relatividades en frecuencia y severidad", "Hay que elegir la potencia p"],
  "Modelo rápido de prima pura o variable objetivo con muchos ceros.", "Quieres entender por separado frecuencia y severidad.")
f("prima_pura", "usar_tarifa",
  "int=10 cal=8 esc=10 vel=10 mem=9 num=9 suc=6 rep=10 cfg=6 aju=10 sas=9",
  ["Combina frecuencia × severidad por póliza con exposición"], ["Supone independencia frecuencia-severidad dadas las X"],
  "Construir la tarifa a partir de dos GLM.", "—")
f("tabla_relatividades", "usar_tarifa",
  "inc=10 int=10 esc=10 vel=10 mem=10 num=10 suc=7 rep=10 cfg=4 aju=10 sas=10",
  ["Tabla tipo Emblem/Radar con la base = 1 e IC"], ["Solo variables categóricas"], "Presentar la tarifa por factor.", "—")
f("ajustar_ols", "lineal",
  "ins=9 con=10 efi=10 ecm=8 nas=9 ide=7 alf=9 cob=9 inc=10 peq=8 gen=7 par=8 int=10 est=7 rob=3 sup=4 dep=3 "
  "esc=10 vel=9 mem=8 num=8 suc=6 rep=10 cfg=7 aju=10 sas=10",
  ["MCO con SE robustos HC3 por defecto", "Diagnósticos (Breusch-Pagan, Durbin-Watson, JB, condición) con avisos", "Coeficientes estandarizados"],
  ["Sensible a atípicos e influyentes", "Linealidad supuesta"], "Respuesta continua y relaciones aproximadamente lineales.", "Respuesta binaria, conteo o importes muy asimétricos (GLM).")
f("medidas_influencia", "lineal_diagnostico",
  "rob=7 int=9 esc=7 vel=8 mem=7 num=9 suc=6 rep=10 cfg=3 aju=10 sas=10",
  ["Apalancamiento, Cook, DFFITS y estudentizados con cortes clásicos"], ["Los cortes son orientativos"],
  "Revisar qué observaciones mueven el modelo.", "—")
f("contraste_f_parcial", "lineal_diagnostico",
  "alf=9 pot=9 inc=8 peq=9 sel=4 int=8 sup=4 esc=10 vel=10 mem=10 num=9 suc=6 rep=10 cfg=3 aju=10 sas=10",
  ["Contrasta un bloque de variables a la vez", "R² parcial"], ["Modelos anidados y con las mismas filas"],
  "¿Aporta un grupo de variables (dummies de un factor, polinomio)?", "Modelos no anidados (compara AIC).")
f("transformacion_box_cox", "lineal_diagnostico",
  "con=8 efi=8 inc=7 int=7 rob=5 sup=7 esc=9 vel=9 mem=9 num=9 suc=6 rep=10 cfg=3 aju=8 sas=9",
  ["λ por máxima verosimilitud con IC y sugerencia (log, raíz…)", "Yeo-Johnson si hay valores ≤ 0"],
  ["Interpretar en la escala transformada es más difícil", "Para medias en la escala original mejor un GLM"],
  "Estabilizar varianza o simetrizar residuos en un modelo lineal.", "Quieres predecir la media de Y (retransformar sesga: usa GLM log).")
f("ajustar_wls", "lineal",
  "ins=9 con=10 efi=10 cob=8 inc=9 int=9 rob=4 sup=5 esc=10 vel=9 mem=8 num=8 suc=6 rep=10 cfg=7 aju=8 sas=10",
  ["Recupera eficiencia con heterocedasticidad", "Pesos dados o estimados (MCP factible)"], ["Pesos mal estimados pueden empeorar"],
  "Varianza conocida o que crece con X (exposición, nº de unidades).", "Basta con SE robustos (HC3) para la inferencia.")
f("regresion_robusta", "lineal",
  "ins=8 con=9 efi=7 inc=8 rob=9 sup=7 int=9 est=8 esc=7 vel=7 mem=7 num=8 suc=6 rep=10 cfg=8 aju=8 sas=9",
  ["Huber/bisquare resisten atípicos en Y", "Regresión cuantílica: modela colas (p. ej. P90 del coste)"],
  ["No protege de puntos de alto apalancamiento en X", "Menos eficiente que MCO si los errores son normales"],
  "Colas pesadas o atípicos en la respuesta; interés por cuantiles.", "Errores normales sin atípicos (MCO es óptimo).")
f("regresion_no_lineal", "lineal",
  "con=9 efi=9 nas=7 ide=5 cob=7 inc=9 peq=5 int=9 par=9 rob=3 sup=4 esc=8 vel=8 mem=9 num=5 suc=4 rep=10 cfg=9 aju=4 sas=9",
  ["Cualquier forma funcional con parámetros interpretables"], ["Depende del punto inicial", "IC asintóticos"],
  "Curvas de desarrollo, saturación, leyes de mortalidad.", "La forma es desconocida (splines).")
f("contraste_falta_ajuste", "lineal_diagnostico",
  "alf=9 pot=8 inc=7 peq=8 int=8 sup=5 esc=10 vel=10 mem=9 num=9 suc=6 rep=10 cfg=4 aju=10 sas=10",
  ["Separa error puro y falta de ajuste"], ["Necesita réplicas en x"], "Experimentos con varias observaciones por nivel de x.", "No hay réplicas.")
f("ajustar_glm_splines", "flexibles",
  "con=9 efi=8 ecm=8 inc=9 alf=8 gen=7 cal=8 dis=8 par=6 int=7 est=7 rob=5 sup=8 esc=8 vel=7 mem=7 num=8 suc=6 rep=10 cfg=8 aju=6 sas=8",
  ["Relaciones no lineales sin elegir la forma", "Test de no linealidad y curva parcial"], ["Elegir los grados de libertad", "Más parámetros que la versión lineal"],
  "Edad, antigüedad o precio con efecto en U o con saturación.", "La relación es claramente lineal.")
f("ajustar_modelo_mixto", "flexibles",
  "ins=8 con=9 efi=9 nas=8 alf=8 cob=8 inc=10 peq=5 gen=8 int=8 est=8 rob=4 sup=4 dep=10 esc=5 vel=5 mem=6 num=6 suc=6 rep=10 cfg=8 aju=8 sas=9",
  ["Respeta el agrupamiento (talleres, oficinas, asegurados con varias pólizas)", "ICC y BLUP (encogimiento tipo credibilidad)"],
  ["Pocos grupos → varianza mal estimada", "Supone efectos aleatorios normales"], "Datos agrupados o medidas repetidas.", "Grupos muy pocos (< 10): usa efectos fijos.")
f("coeficiente_icc", "flexibles",
  "ins=7 con=9 cob=8 inc=9 int=9 sup=5 dep=9 esc=9 vel=9 mem=9 num=9 suc=7 rep=10 cfg=3 aju=10 sas=8",
  ["ICC con IC y efecto diseño"], ["ANOVA de un factor: sin covariables"], "¿Cuánto se parecen las observaciones del mismo grupo?", "—")
f("ajustar_logit_ordinal", "multiclase",
  "ins=7 con=10 efi=9 nas=9 alf=8 cob=8 inc=10 peq=5 gen=7 cal=7 dis=7 par=9 int=9 est=7 rob=4 sup=5 esc=7 vel=6 mem=7 num=7 suc=6 rep=10 cfg=6 aju=10 sas=9",
  ["Usa el orden de la respuesta: un solo OR por variable", "Comprueba odds proporcionales por corte"],
  ["Si los OR cambian entre cortes, el supuesto falla (usa multinomial)"], "Respuesta ordenada (gravedad, satisfacción, tramo).", "Categorías sin orden.")
f("ajustar_regularizado", "seleccion_modelo",
  "ins=4 con=7 ecm=9 ide=10 inc=3 sel=3 gen=9 cal=7 dis=8 par=9 int=7 est=6 rob=5 sup=6 des=6 "
  "esc=6 vel=5 mem=7 num=9 suc=5 rep=10 cfg=9 aju=6 sas=8",
  ["Lasso selecciona, Ridge estabiliza con colinealidad", "λ por validación cruzada con regla 1-SE", "Funciona con p grande"],
  ["Coeficientes sesgados hacia 0: sin p-valores válidos", "Hay que estandarizar (lo hace dentro)"],
  "Muchas variables candidatas o colinealidad; interés predictivo.", "Necesitas inferencia clásica sobre los coeficientes.")
f("contraste_schoenfeld", "superv_contrastes",
  "alf=7 pot=7 inc=8 peq=5 int=8 sup=7 cen=9 esc=8 vel=8 mem=8 num=8 suc=6 rep=10 cfg=4 aju=10 sas=8",
  ["Comprueba riesgos proporcionales por variable y global"], ["Aproximación sin escalar (orientativa)", "Con n grande detecta desviaciones irrelevantes"],
  "Siempre después de un Cox.", "—")
f("ajustar_supervivencia_parametrica", "superv_modelos",
  "ins=8 con=10 efi=10 nas=9 cob=8 inc=10 peq=6 gen=8 par=9 int=9 rob=4 sup=4 cen=10 esc=7 vel=6 mem=8 num=7 suc=6 rep=10 cfg=8 aju=9 sas=10",
  ["Weibull, exponencial, lognormal, log-logística (AFT)", "Más eficiente que Cox si la forma es correcta y permite extrapolar"],
  ["Si la distribución es incorrecta, sesgo", "Solo censura por la derecha"], "Necesitas curvas completas, extrapolar o calcular primas.", "No sabes la forma del riesgo (Cox).")
f("nelson_aalen", "superv_curvas",
  "ins=9 con=10 efi=8 inc=9 peq=7 int=8 rob=8 sup=10 cen=10 esc=7 vel=7 mem=8 num=9 suc=6 rep=10 cfg=4 aju=10 sas=9",
  ["Riesgo acumulado no paramétrico con SE"], ["Sin covariables"], "Ver la forma del riesgo (creciente, constante).", "—")
f("incidencia_acumulada", "superv_curvas",
  "ins=9 con=10 inc=4 peq=7 int=9 rob=8 sup=10 cen=10 esc=8 vel=8 mem=8 num=9 suc=6 rep=10 cfg=4 aju=10 sas=8",
  ["Riesgos competitivos correctos (Aalen-Johansen)"], ["Sin IC (de momento)"],
  "Varias causas de salida que se excluyen (rescate, muerte, impago).", "Solo hay una causa (Kaplan-Meier).")
f("rmst", "superv_curvas",
  "ins=9 con=10 cob=8 inc=10 peq=7 int=10 rob=8 sup=10 cen=10 esc=8 vel=8 mem=8 num=9 suc=6 rep=10 cfg=4 aju=8 sas=9",
  ["Resumen en unidades de tiempo («3 meses más»)", "Válido aunque no haya riesgos proporcionales"], ["Depende del τ elegido"],
  "Comparar grupos con curvas que se cruzan o comunicar a negocio.", "—")
for nombre, grupo, extra, pro in [
    ("grafico_efecto_spline", "flexibles", "int=9 cfg=3", "Curva parcial con el p de no linealidad"),
    ("grafico_regularizacion", "seleccion_modelo", "int=8 cfg=3", "Error de CV frente a λ con el elegido"),
    ("grafico_relatividades", "usar_tarifa", "inc=8 int=10 cfg=3 sas=8", "Relatividades con IC tipo Emblem"),
    ("grafico_incidencia_acumulada", "superv_curvas", "cen=10 int=9 cfg=3", "CIF por causa y grupo"),
]:
    f(nombre, grupo, G + extra, [pro], [], "Necesitas la figura para un notebook o un informe.", "—")

# ------------------------------------------------------------------ contrastes añadidos (0.9)
f("contraste_apareado", "apareados",
  "alf=8 pot=9 cob=8 inc=10 peq=7 rob=7 sup=7 dep=9 int=9 esc=10 vel=9 mem=9 num=9 suc=8 rep=10 cfg=6 aju=9 sas=9",
  ["Usa la correlación de los pares (mucha más potencia que grupos independientes)", "Elige t o Wilcoxon según las diferencias"],
  ["Elegir el test mirando los datos infla algo el α"], "Mismo individuo medido dos veces (antes/después, dos tasadores).", "Los grupos son independientes.")
f("contraste_friedman", "apareados",
  "alf=8 pot=7 inc=7 peq=7 rob=9 sup=9 dep=9 int=8 esc=8 vel=8 mem=8 num=9 suc=6 rep=10 cfg=2 aju=10 sas=8",
  ["No paramétrico para medidas repetidas", "W de Kendall como tamaño del efecto"], ["Descarta sujetos incompletos", "Sin post-hoc incluido"],
  "3+ condiciones en los mismos sujetos con datos no normales.", "Datos normales y completos (ANOVA de medidas repetidas es más potente).")
f("contraste_mcnemar", "apareados",
  "alf=9 pot=7 inc=7 peq=9 rob=9 sup=10 dep=9 int=9 esc=10 vel=10 mem=9 num=10 suc=7 rep=10 cfg=2 aju=10 sas=9",
  ["Exacto con pocos discordantes"], ["Solo usa los pares que cambian"], "Proporciones 0/1 antes y después en los mismos individuos.", "Grupos independientes (χ² o Fisher).")
f("contraste_permutacion", "comparar_grupos",
  "alf=10 pot=8 inc=7 peq=10 rob=8 sup=10 int=8 esc=4 vel=4 mem=7 num=10 suc=7 rep=10 cfg=10 aju=7",
  ["Exacto sin suponer distribución", "Cualquier estadístico (media, mediana, propio)"], ["Lento: n_perm repeticiones", "Supone intercambiabilidad (misma forma bajo H0)"],
  "Muestras pequeñas o estadísticos raros.", "n grande y supuestos razonables (el t o Mann-Whitney dan lo mismo, más rápido).")
f("contraste_jonckheere", "comparar_grupos",
  "alf=8 pot=9 inc=7 peq=6 rob=9 sup=9 int=8 esc=4 vel=6 mem=5 num=9 suc=7 rep=10 cfg=6 aju=9 sas=8",
  ["Más potente que Kruskal-Wallis si la alternativa es una tendencia ordenada"], ["Hay que fijar el orden ANTES de mirar los datos", "Compara todos los pares: O(n²) en memoria"],
  "Grupos con orden natural (tramos, dosis, niveles bonus-malus).", "Los grupos no tienen orden.")
f("posthoc_dunn", "multiples",
  "alf=8 pot=7 inc=8 peq=6 rob=9 sup=9 int=8 esc=8 vel=8 mem=8 num=9 suc=7 rep=10 cfg=7 aju=9 sas=8",
  ["Post-hoc no paramétrico coherente con Kruskal-Wallis", "Ajuste Holm/BH configurable"], ["Menos potente que Tukey con datos normales"],
  "Tras un Kruskal-Wallis significativo.", "Datos normales con varianzas iguales (Tukey).")
f("games_howell", "multiples",
  "alf=8 pot=7 cob=8 inc=10 peq=6 rob=6 sup=7 int=9 esc=9 vel=9 mem=9 num=9 suc=7 rep=10 cfg=3 aju=10 sas=7",
  ["No supone varianzas iguales", "IC de cada diferencia"], ["Algo liberal con n muy pequeños"],
  "Tras un ANOVA de Welch o con varianzas distintas.", "Varianzas iguales (Tukey es algo más potente).")
f("intervalo_proporcion", "intervalos",
  "ins=7 con=10 cob=9 inc=10 peq=9 int=10 sup=10 des=8 esc=10 vel=10 mem=10 num=10 suc=7 rep=10 cfg=9 aju=10 sas=10",
  ["Cinco métodos; Wilson y Jeffreys aciertan la cobertura incluso con pocos casos"], ["Wald (el de libro) falla con p cerca de 0 o 1"],
  "Cualquier tasa con su incertidumbre (siniestralidad, conversión, impago).", "—")
f("bootstrap_ic", "intervalos",
  "ins=7 con=9 cob=8 inc=10 peq=6 rob=6 sup=9 int=8 esc=5 vel=5 mem=5 num=9 suc=7 rep=10 cfg=10 aju=7",
  ["IC de CUALQUIER estadístico sin fórmula", "BCa corrige sesgo y asimetría"], ["Coste: miles de réplicas", "Falla con estadísticos de extremos o n muy pequeño"],
  "Medianas, ratios, cuantiles o estadísticos sin error estándar conocido.", "Datos dependientes (necesitas bootstrap por bloques).")
f("bondad_ajuste_multinomial", "ajuste_distribucion",
  "alf=9 pot=7 inc=8 peq=6 int=9 sup=9 esc=10 vel=10 mem=10 num=10 suc=7 rep=10 cfg=6 aju=10 sas=9",
  ["χ² de bondad de ajuste con residuos por categoría"], ["Esperados < 5 invalidan la aproximación"],
  "¿Las frecuencias siguen un reparto teórico (días, categorías, dígitos)?", "Variables continuas (usa KS o Anderson-Darling).")

# ------------------------------------------------------------------ descriptiva (0.9)
f("resumen_descriptivo", "describir",
  "rob=7 sup=10 int=10 inc=2 esc=9 vel=9 mem=8 num=10 suc=9 rep=10 cfg=6 aju=9 sas=9",
  ["Posición, dispersión y forma en una tabla (media recortada, MAD, IQR robustos)", "Cuenta faltantes"],
  ["Sin IC de cada medida"], "Primer vistazo a cualquier tabla de datos.", "—")
f("detectar_atipicos", "describir",
  "rob=8 sup=8 int=9 est=7 esc=10 vel=10 mem=9 num=9 suc=8 rep=10 cfg=7 aju=6 sas=7",
  ["Tres criterios: IQR (Tukey), MAD (robusto) y z", "Devuelve los límites usados"],
  ["Univariante: no ve atípicos multivariantes (usa Mahalanobis)", "z no es robusto: los atípicos inflan la desviación"],
  "Revisar una variable antes de modelar.", "Datos con colas pesadas legítimas (siniestros grandes): no los borres, modélalos.")
f("correlacion_con_ic", "correlacion",
  "ins=7 con=10 efi=9 nas=9 alf=9 cob=8 inc=10 peq=6 rob=6 sup=7 int=9 esc=9 vel=9 mem=9 num=9 suc=8 rep=10 cfg=7 aju=9 sas=9",
  ["Pearson, Spearman o Kendall con IC (z de Fisher)", "Quita pares incompletos"],
  ["Pearson: solo relación lineal y sensible a atípicos", "Correlación ≠ causalidad"],
  "Medir la fuerza de una relación con su incertidumbre.", "La relación no es monótona (mira un gráfico).")
f("matriz_correlaciones", "correlacion",
  "con=9 alf=5 inc=7 rob=6 sup=7 int=9 esc=5 vel=6 mem=8 num=9 suc=8 rep=10 cfg=6 aju=9 sas=9",
  ["Matriz de r, p-valores y n por pares completos"], ["Muchos p-valores: ajusta por comparaciones múltiples", "Bucle O(p²)"],
  "Explorar relaciones entre muchas variables numéricas.", "Quieres efectos ajustados (usa correlación parcial o un modelo).")
f("correlacion_parcial", "correlacion",
  "con=9 efi=8 alf=8 inc=8 rob=4 sup=5 int=8 esc=9 vel=9 mem=9 num=8 suc=7 rep=10 cfg=6 aju=9 sas=9",
  ["Quita el efecto de variables de confusión", "Da parcial y semiparcial"], ["Supone relaciones lineales con los controles"],
  "Sospechas que una tercera variable explica la correlación.", "Los controles actúan de forma no lineal.")
f("contraste_normalidad", "supuestos",
  "alf=8 pot=7 inc=8 peq=6 int=8 sup=9 esc=8 vel=8 mem=8 num=9 suc=8 rep=10 cfg=4 aju=9 sas=9",
  ["Cinco contrastes a la vez y avisos según n"], ["Con n grande rechaza siempre; con n pequeño no detecta nada"],
  "Documentar un supuesto de normalidad (junto al Q-Q).", "n > 5000: decide por el gráfico y la asimetría.")
f("homogeneidad_varianzas", "supuestos",
  "alf=8 pot=7 inc=8 peq=7 rob=8 sup=8 int=8 esc=9 vel=9 mem=9 num=9 suc=8 rep=10 cfg=4 aju=9 sas=10",
  ["Levene, Brown-Forsythe, Bartlett y Fligner juntos"], ["Bartlett se equivoca con datos no normales"],
  "Antes de un ANOVA o un t clásico.", "Vas a usar Welch de todos modos (no hace falta contrastar).")
f("ajustar_distribuciones", "ajuste_distribucion",
  "con=9 efi=9 nas=8 inc=6 peq=6 int=8 par=8 rob=4 sup=6 esc=7 vel=5 mem=8 num=7 suc=7 rep=10 cfg=8 aju=8 sas=8",
  ["MLE de 8 distribuciones ordenadas por AIC", "Fija el origen en 0 para importes"],
  ["p-valor KS optimista con parámetros estimados", "Las colas mandan en seguros: valida con P-P/Q-Q y EVT"],
  "Elegir la distribución de importes, tiempos o severidades.", "Te importan solo las colas extremas (usa teoría de valores extremos).")
f("ajustar_distribucion_discreta", "ajuste_distribucion",
  "con=9 efi=9 nas=8 alf=8 inc=7 int=9 par=8 sup=6 esc=9 vel=9 mem=9 num=8 suc=6 rep=10 cfg=3 aju=10 sas=7",
  ["Poisson vs binomial negativa con test LR", "Parametrización de seguros (media, r)"], ["Sin covariables (para eso, GLM de conteo)"],
  "Modelar el número de siniestros por póliza.", "La frecuencia depende de covariables o de la exposición (GLM con offset).")
f("indice_dispersion", "ajuste_distribucion",
  "alf=8 pot=7 inc=7 peq=7 int=9 sup=6 esc=10 vel=10 mem=10 num=10 suc=6 rep=10 cfg=2 aju=10",
  ["Contraste rápido de sobredispersión"], ["Sin covariables: la heterogeneidad explicada por variables también cuenta"],
  "Comprobar si un conteo es Poisson.", "—")
f("funcion_distribucion_empirica", "ajuste_distribucion",
  "ins=10 con=10 cob=9 inc=9 peq=9 int=9 rob=9 sup=10 esc=9 vel=9 mem=8 num=10 suc=8 rep=10 aju=10 sas=8",
  ["F empírica con banda simultánea DKW (válida para cualquier n)"], ["Banda conservadora (más ancha que la puntual)"],
  "Ver la distribución sin suponer forma y comparar con una teórica.", "—")
f("estimar_densidad", "describir",
  "con=8 ecm=7 inc=2 int=8 rob=7 sup=9 esc=7 vel=8 mem=8 num=9 suc=8 rep=10 cfg=6 aju=5",
  ["Densidad suave sin suponer forma"], ["El ancho de banda decide lo que ves", "Pone masa fuera del soporte (p. ej. < 0)"],
  "Ver la forma (multimodalidad) de una variable continua.", "Datos discretos o con soporte acotado sin transformar.")
f("tabla_contingencia", "tablas",
  "alf=9 pot=8 inc=9 peq=7 int=10 sup=9 esc=9 vel=9 mem=9 num=9 suc=8 rep=10 cfg=4 aju=10 sas=10",
  ["Todo PROC FREQ: χ², G², Fisher, V de Cramér y residuos ajustados", "Avisa de celdas con esperado < 5"],
  ["Solo dos variables"], "Relación entre dos categóricas.", "Quieres ajustar por terceras variables (logística o loglineal).")
f("medidas_riesgo_2x2", "tablas",
  "ins=7 con=10 cob=8 inc=10 peq=6 int=10 sup=9 esc=10 vel=10 mem=10 num=8 suc=7 rep=10 cfg=4 aju=10 sas=10",
  ["RR, DR, OR y NNT con IC", "Corrección de Haldane con ceros"], ["IC de Wald: con pocos eventos son aproximados"],
  "Comparar el riesgo entre expuestos y no expuestos.", "Hay confusión: usa un modelo (Mantel-Haenszel o logística).")
for nombre, grupo, extra, pro in [
    ("grafico_distribucion", "describir", "int=10 rob=8 cfg=3", "Histograma + KDE + caja con media y mediana"),
    ("grafico_ajuste_distribuciones", "ajuste_distribucion", "int=9 cfg=5 sas=8", "Mejores ajustes superpuestos y P-P"),
    ("grafico_matriz_correlaciones", "correlacion", "int=10 cfg=5 sas=8", "Heatmap con valores y no significativas en gris"),
]:
    f(nombre, grupo, G + extra, [pro], [], "Necesitas la figura para un notebook o un informe.", "—")

# ------------------------------------------------------------------ simulación (0.10)
f("posterior_conjugado", "bayes", A + "ins=7 con=10 efi=9 cob=9 inc=10 peq=9 int=9 sup=6 suc=5 cfg=7 sas=7",
  ["Exacta, sin simulación", "IC creíble y predictiva", "Factor de credibilidad Z explícito"], ["Solo para las parejas conjugadas", "La previa influye mucho con pocos datos"],
  "Tasas, frecuencias o medias con información previa (credibilidad, experiencia de otra cartera).", "El modelo no es conjugado o la previa no se puede justificar: usa Metropolis o un análisis de sensibilidad.")
f("bayes_empirico_beta", "bayes", "ins=6 con=8 efi=8 ecm=9 cob=6 inc=7 peq=7 int=8 est=8 sup=6 esc=9 vel=9 mem=10 num=8 suc=6 rep=10 cfg=5 aju=9",
  ["Contrae las tasas pequeñas hacia la media: menos error total", "La previa sale de los datos"], ["IC algo estrechos (no propaga la incertidumbre de la previa)", "Con < 5 grupos es inestable"],
  "Muchas tasas por grupo con tamaños muy distintos (oficinas, agentes, cestas de un ensayo).", "Pocos grupos o grupos que no son intercambiables.")
f("metropolis", "mcmc", "con=9 efi=5 nas=8 cob=8 inc=10 peq=8 int=6 est=6 sup=8 dep=4 esc=6 vel=4 mem=7 num=7 suc=4 rep=10 cfg=8 aju=4 sas=7",
  ["Cualquier posterior que sepas escribir", "Varias cadenas + R-hat y ESS automáticos", "Escala adaptativa"], ["Lento y mezcla mal con parámetros muy correlados o muchos", "Hay que vigilar la convergencia"],
  "Modelos bayesianos pequeños sin solución cerrada.", "Muchos parámetros (> 20) o geometría difícil: usa Stan/PyMC (HMC).")
f("diagnostico_mcmc", "mcmc", A + "int=8 sup=8 suc=5 cfg=3 sas=7", ["R-hat dividido y ESS como en Stan/ArviZ"], ["No garantiza convergencia: solo detecta fallos"],
  "Tras cualquier MCMC, antes de usar las medias.", "—")
f("chequeo_predictivo", "mcmc", "con=8 inc=8 int=9 rob=7 sup=8 esc=7 vel=7 mem=8 num=9 suc=6 rep=10 cfg=9 aju=7",
  ["Comprueba aspectos concretos (ceros, colas, dispersión)", "Vale para previa y posterior"], ["Usa los datos dos veces: p bayesianos conservadores"],
  "Validar si el modelo reproduce lo que importa de los datos.", "Necesitas un contraste formal de bondad de ajuste con α controlado.")
f("accion_bayes", "decision_adaptativa", A + "int=9 sup=8 suc=6 cfg=8", ["Une posterior y coste de equivocarse", "Pérdidas asimétricas (provisiones prudentes)"], ["Requiere fijar la función de pérdida"],
  "Decidir una cifra o una acción con incertidumbre (reservas, aceptar/rechazar).", "No puedes cuantificar las pérdidas: informa del IC.")
f("ab_bayesiano", "decision_adaptativa", "ins=7 con=9 cob=8 inc=10 peq=8 int=10 sup=7 dep=5 esc=9 vel=9 mem=8 num=9 suc=5 rep=10 cfg=7 aju=10",
  ["P(B > A) y pérdida esperada: fácil de comunicar", "Regla de parada adaptativa"], ["Mirar a cada rato rompe las garantías frecuentistas de α"],
  "Tests A/B de conversión en producto o marketing.", "Necesitas error tipo I controlado para un regulador: diseño secuencial frecuentista.")
f("simular_bandido", "decision_adaptativa", "con=8 int=8 sup=6 dep=4 esc=6 vel=5 mem=8 num=9 suc=5 rep=10 cfg=8 aju=8",
  ["Compara Thompson, UCB, ε-greedy y reparto uniforme", "Mide el coste de explorar (arrepentimiento)"], ["Simulación con probabilidades conocidas; recompensas Bernoulli"],
  "Decidir si un bandido merece la pena frente a un A/B clásico.", "El efecto cambia en el tiempo o hay retardo en la respuesta.")
f("cadena_markov", "markov", A + "con=10 int=9 sup=6 suc=5 cfg=7 sas=6", ["Clasificación de estados, estacionaria, absorción y tiempos medios"], ["Supone homogeneidad en el tiempo y memoria de un paso"],
  "Morosidad, migración de clientes entre estados, bonus-malus.", "Las probabilidades cambian con el tiempo o dependen del historial largo.")
f("simular_cadena_markov", "markov", "con=9 int=8 sup=6 esc=8 vel=8 mem=8 num=10 suc=5 rep=10 cfg=6 aju=10", ["Trayectorias para cualquier indicador por simulación"], ["Error de Monte Carlo"],
  "Calcular por simulación lo que no tiene fórmula (beneficio acumulado, tiempos).", "Tienes la fórmula exacta (usa cadena_markov).")
f("bonus_malus", "markov", A + "int=9 sup=6 suc=5 cfg=8 sas=5", ["Estacionaria, prima media y eficiencia de Loimaranta", "Evolución año a año"], ["Frecuencia Poisson homogénea por asegurado"],
  "Diseñar o evaluar un sistema bonus-malus.", "La frecuencia individual varía mucho: mezcla Poisson-gamma por perfiles.")
f("cadena_markov_continua", "procesos_continuos", A + "con=10 int=8 sup=6 suc=5 cfg=6", ["P(t) = exp(Qt) exacto", "Modelos multiestado"], ["Tiempos de permanencia exponenciales"],
  "Modelos sano-inválido-fallecido, fiabilidad.", "Las tasas dependen de la duración en el estado (semi-Markov).")
f("nacimiento_muerte", "procesos_continuos", A + "int=9 sup=5 suc=5 cfg=7", ["Erlang C, L, W, Lq, Wq y rechazo con capacidad"], ["Llegadas Poisson y servicio exponencial"],
  "Dimensionar un centro de llamadas, una oficina o un servidor.", "Servicios muy regulares o llegadas en ráfagas: simula.")
f("simular_proceso_poisson", "procesos_continuos", "con=10 int=9 sup=7 esc=9 vel=9 mem=9 num=10 suc=6 rep=10 cfg=8 aju=10", ["Homogéneo y no homogéneo (adelgazamiento)"], ["Hay que dar la tasa máxima si λ(t) es irregular"],
  "Simular llegadas de siniestros o clientes.", "Hay contagio entre llegadas (procesos de Hawkes).")
f("contraste_proceso_poisson", "procesos_continuos", "alf=7 pot=6 inc=7 int=8 sup=7 esc=9 vel=9 mem=9 num=9 suc=6 rep=10 cfg=4 aju=10",
  ["Tres comprobaciones: uniformidad, exponencialidad, dispersión"], ["KS exponencial con media estimada: aproximado"], "Validar el supuesto Poisson antes de un modelo de frecuencia.", "—")
f("ruina_jugador", "paseos", A + "int=10 sup=6 suc=6 cfg=5", ["Fórmula cerrada + comprobación por simulación"], ["Pasos ±1 de igual tamaño"], "Intuición sobre ruina y duración; enseñar martingalas.", "Reservas de una aseguradora: usa probabilidad_ruina.")
f("paseo_aleatorio", "paseos", "con=10 int=9 sup=7 esc=7 vel=8 mem=6 num=10 suc=6 rep=10 cfg=6 aju=10", ["Muestra que las martingalas mantienen su media"], ["Didáctica"], "Entender martingalas y parada opcional.", "—")
f("simular_browniano", "paseos", "con=10 int=8 sup=6 esc=7 vel=8 mem=5 num=9 suc=6 rep=10 cfg=7 aju=10", ["Aritmético y geométrico (solución exacta de Itô)", "Comprueba media, varianza y variación cuadrática"],
  ["Volatilidad constante", "Memoria n_pasos × trayectorias"], "Escenarios de precios o de activos para VaR, opciones o ALM.", "Colas pesadas o saltos: usa modelos con saltos o t de Student.")
f("generador_congruencial", "generar_aleatorios", "int=9 sup=10 esc=6 vel=5 mem=9 num=10 rep=10 cfg=8 aju=10", ["Enseña cómo funciona un generador y por qué falla RANDU"], ["Didáctico: lento y de baja calidad frente a PCG64"],
  "Entender o auditar generadores.", "Cualquier simulación real: usa np.random.default_rng.")
f("contrastes_aleatoriedad", "generar_aleatorios", "alf=8 pot=7 int=8 sup=9 esc=9 vel=9 mem=8 num=10 rep=10 cfg=5 aju=10", ["Seis contrastes, incluido el de tripletas (detecta RANDU)"], ["Pasar los contrastes no prueba la aleatoriedad"],
  "Validar una secuencia de números o un generador.", "—")
f("generar_por_inversion", "generar_aleatorios", A + "int=9 sup=9 suc=6 cfg=8", ["Exacto si hay cuantil; numérico con solo la cdf"], ["La inversión numérica en rejilla pierde precisión en colas extremas"],
  "Generar de una distribución con F⁻¹ conocida o tabulada.", "Densidades sin cdf manejable: aceptación-rechazo.")
f("generar_por_aceptacion_rechazo", "generar_aleatorios", "ins=10 con=10 int=8 sup=9 esc=8 vel=7 mem=8 num=8 rep=10 cfg=8 aju=8", ["Vale para densidades sin normalizar"], ["Desperdicia si la propuesta se parece poco (tasa 1/M)", "M estimado en rejilla"],
  "Generar de densidades raras en pocas dimensiones.", "Muchas dimensiones (la tasa se hunde): MCMC.")
f("generar_normal_multivariante", "generar_aleatorios", A + "int=9 sup=7 suc=5 cfg=7", ["Cholesky o espectral (vale para Σ singular)", "Valida que Σ sea semidefinida"], ["Solo normal: sin colas ni asimetría"],
  "Escenarios correlados para riesgo o simulación.", "Dependencia en colas: usa cópulas (simular_copula).")
f("estimar_montecarlo", "montecarlo", "ins=10 con=10 efi=8 cob=9 inc=10 int=8 sup=9 esc=10 vel=9 mem=8 num=9 rep=10 cfg=8 aju=10",
  ["Variables antitéticas y de control con reducción medida", "EE e IC siempre"], ["Integrales en el hipercubo: transforma tú la distribución"],
  "Calcular esperanzas sin fórmula (primas, probabilidades).", "Integrales de 1-2 dimensiones suaves: cuadratura es más precisa.")
f("teorema_bayes", "probabilidad", A + "int=10 sup=10 cfg=5", ["Hipótesis discretas o pruebas diagnósticas (VPP/VPN)"], ["Necesita previas y verosimilitudes fiables"],
  "Cribados, fraude, probabilidad de una causa dado un indicio.", "—")
f("analizar_distribucion_conjunta", "probabilidad", A + "int=9 sup=9 suc=6 cfg=5", ["Marginales, condicionadas, E[Y|X], covarianza, independencia"], ["Solo discretas (tabla)"],
  "Estudiar la relación entre dos variables discretas.", "Variables continuas: modelos de regresión o cópulas.")
f("distribucion_estadistico_orden", "probabilidad", A + "int=8 sup=9 cfg=5", ["Exacta por la Beta de los uniformes ordenados"], ["Muestras i.i.d."], "Máximo o mínimo de n siniestros, cuantiles de una muestra.", "—")
f("momentos_distribucion", "probabilidad", "esc=10 vel=7 mem=10 num=7 rep=10 aju=10 int=8 sup=9 cfg=4", ["Detecta FGM inexistente (lognormal, Pareto)"], ["Integración numérica: colas muy pesadas pueden avisar"],
  "Revisar momentos y FGM de un modelo de pérdidas.", "—")
f("convergencia_media_muestral", "probabilidad", "con=9 int=10 sup=10 esc=7 vel=7 mem=6 num=9 rep=10 cfg=6 aju=10", ["TCL y LGN medidos (KS, asimetría)", "Muestra cuándo NO se cumple (Cauchy)"], ["Didáctica"],
  "Saber si n es suficiente para la aproximación normal con datos asimétricos.", "—")
f("metodo_momentos", "inferencia_teoria", "ins=6 con=9 efi=5 ecm=6 int=9 rob=4 sup=6 esc=10 vel=10 mem=10 num=8 suc=5 rep=10 cfg=5 aju=10",
  ["Sencillo y con fórmula", "Compara con máxima verosimilitud"], ["Menos eficiente; inestable con colas pesadas"], "Valores iniciales y comprobaciones rápidas.", "Necesitas el estimador eficiente: máxima verosimilitud.")
f("informacion_fisher", "inferencia_teoria", "con=9 int=8 sup=6 esc=7 vel=7 mem=7 num=7 rep=10 cfg=7 aju=8", ["Observada o esperada", "Cota de Cramér-Rao directa"], ["Exige modelo regular", "Derivadas numéricas"],
  "Saber la mejor precisión posible o el EE de un MV.", "El soporte depende del parámetro (uniforme).")
f("comparar_estimadores", "inferencia_teoria", "ins=10 con=9 int=10 sup=10 esc=6 vel=5 mem=8 num=9 rep=10 cfg=10 aju=8", ["Sesgo, varianza, ECM y eficiencia frente a Cramér-Rao"], ["Simulación: depende del escenario elegido"],
  "Elegir entre estimadores o justificar uno.", "—")
f("metodo_delta", "inferencia_teoria", A + "ins=7 con=9 nas=9 cob=7 inc=9 peq=5 int=8 sup=6 cfg=8 sas=8", ["EE de cualquier función de parámetros"], ["Aproximación lineal: mala con g muy curva o EE grandes"],
  "EE de odds ratios, elasticidades, cocientes.", "Muestras pequeñas o funciones muy no lineales: bootstrap.")
f("distribucion_muestral_simulada", "inferencia_teoria", "con=10 int=10 sup=10 esc=7 vel=7 mem=6 num=9 rep=10 cfg=5 aju=10", ["Construye t, χ², F y T² desde normales", "Explica el n − 1"], ["Didáctica"],
  "Entender de dónde salen las tablas y los grados de libertad.", "—")
f("simular_regresion_a_la_media", "inferencia_teoria", A + "int=10 sup=8 cfg=5", ["Cuantifica el cambio «gratis» de los extremos"], ["Supone normal bivariante"],
  "Antes de evaluar una intervención sobre los peores casos sin grupo de control.", "—")
f("coste_de_dicotomizar", "inferencia_teoria", "int=10 sup=8 esc=7 vel=6 mem=9 num=9 rep=10 cfg=6 aju=9", ["Atenuación teórica + potencia simulada"], ["Supone normalidad"],
  "Convencer de no partir variables continuas en dos.", "—")
f("extraer_muestra", "muestreo", A + "int=9 sup=9 suc=6 cfg=8 sas=8", ["MAS, sistemática, estratificada y por conglomerados con pesos"], ["La sistemática falla con periodicidades"],
  "Diseñar la muestra de una encuesta o auditoría.", "—")
f("estimar_mas", "muestreo", A + "ins=10 con=10 efi=7 cob=9 inc=10 peq=8 int=10 sup=8 cfg=5 sas=9", ["Corrección por población finita", "Total, media y proporción"], ["Solo muestreo aleatorio simple"],
  "Estimar con una MAS de una población finita.", "Diseño estratificado o por conglomerados.")
f("tamano_muestra_encuesta", "muestreo", A + "int=10 sup=7 cfg=8 sas=8", ["Incluye efecto de diseño y no respuesta"], ["p = 0.5 es conservador"], "Planificar una encuesta.", "—")
f("asignacion_estratos", "muestreo", A + "efi=10 int=9 sup=7 cfg=8 sas=7", ["Proporcional, Neyman y óptima con costes", "Compara varianzas"], ["Necesita S_h de un piloto"],
  "Repartir la muestra entre estratos.", "—")
f("estimar_estratificado", "muestreo", A + "ins=10 con=10 efi=9 cob=8 inc=10 int=9 sup=8 cfg=5 sas=9", ["Satterthwaite y efecto de diseño"], ["≥ 2 observaciones por estrato"],
  "Estimar con muestreo estratificado.", "Conglomerados: usa estimadores de diseño completos (PROC SURVEYMEANS).")
f("estimador_razon", "muestreo", A + "ins=6 con=9 efi=9 cob=8 inc=9 int=9 sup=6 cfg=4 sas=8", ["Gana mucho si y ∝ x", "Indica si conviene"], ["Sesgo O(1/n)"],
  "Hay una variable auxiliar conocida en toda la población.", "La relación tiene ordenada lejos de 0: estimador de regresión.")
f("ajuste_no_respuesta", "muestreo", A + "ins=7 int=9 sup=5 suc=6 cfg=6 sas=8", ["Tasa de respuesta y pesos ajustados por celdas"], ["Supone MAR dentro de celda"],
  "Encuestas con respuesta desigual por grupos.", "La no respuesta depende de la propia variable (MNAR).")
# ------------------------------------------------------------------ diseño (0.10)
f("anova_factorial", "anova", "ins=10 con=10 efi=9 alf=8 pot=8 inc=9 peq=7 int=9 rob=4 sup=4 esc=8 vel=8 mem=8 num=9 suc=5 rep=10 cfg=8 aju=9 sas=10",
  ["Tipo II o III (como SAS)", "η² parcial y ω²", "Avisa de interacción en desequilibrado"], ["Normalidad y varianzas iguales"],
  "Varios factores categóricos y sus interacciones.", "Varianzas muy distintas o medidas repetidas.")
f("anova_bloques", "anova", "ins=10 con=10 efi=10 alf=8 pot=9 int=9 sup=4 esc=8 vel=8 mem=8 num=9 suc=5 rep=10 cfg=4 aju=9 sas=10",
  ["Eficiencia relativa frente a no bloquear", "Medias ajustadas; vale para bloques incompletos"], ["Supone aditividad bloque + tratamiento"],
  "Experimentos con una fuente de ruido conocida (sucursal, día, lote).", "Interacción bloque × tratamiento esperable.")
f("ancova", "anova", "ins=10 con=10 efi=9 alf=8 pot=9 inc=9 int=9 sup=4 esc=8 vel=8 mem=8 num=9 suc=5 rep=10 cfg=5 aju=9 sas=10",
  ["Contrasta pendientes homogéneas", "Medias ajustadas (LSMEANS)", "Reduce el error"], ["Pendientes paralelas y relación lineal"],
  "Comparar grupos ajustando por la medida basal.", "La covariable está afectada por el tratamiento.")
f("anova_medidas_repetidas", "anova", "ins=9 con=9 efi=8 alf=8 pot=8 inc=8 int=8 sup=4 dep=8 esc=7 vel=8 mem=8 num=8 suc=3 rep=10 cfg=5 aju=9 sas=10",
  ["Mauchly, Greenhouse-Geisser y Huynh-Feldt", "Diseño mixto (split-plot)"], ["Descarta sujetos incompletos", "Un factor intra y uno entre"],
  "Mismos sujetos medidos varias veces.", "Faltantes o tiempos irregulares: ajustar_modelo_mixto.")
f("anova_anidado", "anova", "ins=8 con=9 alf=9 int=8 sup=5 dep=8 esc=8 vel=8 mem=8 num=8 suc=4 rep=10 cfg=3 aju=9 sas=9",
  ["Evita la pseudorreplicación", "Componentes de la varianza"], ["EMS: exacto solo si equilibrado"], "Submuestreo, unidades dentro de grupos.", "Desequilibrado: REML.")
f("diagnostico_anova", "anova", A + "int=9 rob=8 sup=9 suc=6 cfg=3 sas=8", ["Recomienda remedio (Welch, Box-Cox, Kruskal)"], ["Reglas orientativas"], "Antes de interpretar cualquier ANOVA.", "—")
f("diseno_factorial_2k", "experimentos", A + "int=9 sup=10 cfg=8 sas=9", ["Completo o fraccionado con alias y resolución", "Puntos centrales y aleatorización"], ["Solo 2 niveles por factor"],
  "Cribar qué factores importan con pocos ensayos.", "Respuesta muy curva: superficie de respuesta.")
f("efectos_factorial_2k", "experimentos", "ins=10 con=9 efi=9 alf=7 pot=7 inc=7 peq=8 int=10 sup=6 esc=9 vel=9 mem=9 num=9 suc=4 rep=10 cfg=5 aju=10 sas=9",
  ["Lenth para diseños sin réplicas", "Curvatura con puntos centrales"], ["Supone pocos efectos activos (esparsidad)"], "Analizar un 2^k.", "Muchos efectos activos: Lenth pierde potencia.")
f("cuadrado_latino", "experimentos", A + "int=9 sup=10 cfg=6 sas=8", ["Controla dos fuentes de ruido con n² unidades"], ["Sin interacciones"], "Dos factores de bloqueo cruzados.", "—")
f("anova_cuadrado_latino", "experimentos", "ins=10 con=10 efi=9 alf=8 pot=7 int=9 sup=4 esc=9 vel=9 mem=9 num=9 rep=10 cfg=3 aju=10 sas=10", ["Tabla con η²"], ["Sin interacciones"],
  "Analizar un cuadrado latino.", "—")
f("diseno_central_compuesto", "experimentos", A + "int=9 sup=10 cfg=7 sas=9", ["Rotable o en caras"], ["Más ensayos que un 2^k"], "Ajustar una superficie de segundo orden.", "—")
f("superficie_respuesta", "experimentos", "ins=9 con=9 efi=8 inc=7 int=8 sup=5 esc=9 vel=9 mem=9 num=8 suc=4 rep=10 cfg=5 aju=9 sas=9",
  ["Punto estacionario y análisis canónico", "Avisa de extrapolación"], ["Aproximación cuadrática local"], "Optimizar un proceso o un precio con 2-4 factores.", "Respuesta muy irregular o muchos factores.")
f("puntuacion_propension", "causal", "ins=7 con=8 efi=6 cob=7 inc=8 peq=5 int=8 est=6 rob=5 sup=4 esc=6 vel=4 mem=8 num=7 suc=4 rep=10 cfg=8 aju=6 sas=8",
  ["IPW, emparejamiento o estratificación", "Tabla de balance (SMD) y solapamiento", "IC bootstrap"], ["Solo corrige confusores medidos", "IPW inestable con e(x) extremos"],
  "Efecto de un tratamiento o campaña sin aleatorizar.", "Hay confusores no medidos relevantes: busca un diseño (DiD, IV).")
f("diferencias_en_diferencias", "causal", "ins=8 con=9 cob=8 inc=9 int=9 sup=5 dep=9 esc=9 vel=9 mem=9 num=9 suc=5 rep=10 cfg=6 aju=10 sas=8",
  ["Errores agrupados", "Contraste de tendencias paralelas"], ["Supuesto clave: tendencias paralelas"], "Cambio de política o campaña con grupo de control y antes/después.", "Tendencias previas distintas.")
f("aleatorizar_ensayo", "ensayos", A + "int=10 sup=10 cfg=9 sas=8", ["Bloques permutados y estratos", "Asignación desigual"], ["Bloques fijos son predecibles si se conocen"], "Preparar la lista de aleatorización.", "—")
f("analisis_intencion_tratar", "ensayos", A + "ins=8 cob=8 inc=8 int=9 sup=6 cfg=3 sas=8", ["ITT, por protocolo, según recibido y CACE lado a lado"], ["CACE supone exclusión y monotonía"],
  "Ensayos con incumplimiento o cambio de brazo.", "—")
f("mediacion", "mecanismos", "ins=7 con=9 cob=8 inc=9 int=9 sup=4 esc=7 vel=5 mem=8 num=9 suc=5 rep=10 cfg=6 aju=8 sas=6",
  ["Efecto indirecto con IC bootstrap y Sobel"], ["Lineal; exige ausencia de confusión M-Y"], "Explicar por qué vía actúa un efecto.", "Mediador y resultado con confusores no medidos.")
f("moderacion", "mecanismos", "ins=9 con=9 cob=8 inc=9 int=10 sup=6 esc=9 vel=9 mem=9 num=9 suc=5 rep=10 cfg=6 aju=10 sas=8",
  ["Pendientes simples y Johnson-Neyman", "HC3"], ["Interacción lineal"], "¿El efecto de X depende de W?", "—")
f("metaanalisis", "metaanalisis", A + "ins=7 con=8 efi=8 cob=7 inc=9 peq=5 int=9 sup=5 cfg=7 sas=8",
  ["Fijos, DerSimonian-Laird y REML", "Ajuste de Hartung-Knapp para pocos estudios", "I², τ² e intervalo de predicción", "Test de Egger"], ["Con pocos estudios τ² es imprecisa y el IC de DL se queda corto (usa hartung_knapp=True)", "Egger poco potente"],
  "Combinar estudios, mercados o años.", "Estudios muy heterogéneos en diseño: no los combines sin más.")
# ------------------------------------------------------------------ regresión y modelos (0.10)
f("bandas_confianza_regresion", "regresion_inferencia", A + "ins=10 con=10 cob=10 inc=10 int=9 sup=4 cfg=6 sas=10", ["Working-Hotelling, Bonferroni o puntual", "Intervalo de predicción"], ["Regresión simple"],
  "Dibujar o usar la recta con confianza para varios x a la vez.", "Varias X: usa las predicciones del modelo con ajuste múltiple.")
f("regresion_inversa", "regresion_inferencia", A + "ins=7 con=9 cob=9 inc=10 int=9 sup=4 cfg=5 sas=8", ["IC de Fieller (detecta pendiente mal determinada)"], ["Lineal y error en y"],
  "Calibrar un instrumento o medida barata.", "Relación no lineal.")
f("regresion_por_origen", "regresion_inferencia", A + "ins=7 con=9 int=9 sup=4 cfg=3 sas=9", ["Avisa del R² no comparable y contrasta la ordenada"], ["Sesgada si la ordenada no es 0"],
  "La teoría impone y = 0 cuando x = 0.", "Datos lejos del origen.")
f("correccion_error_medida", "regresion_inferencia", "ins=8 con=8 efi=6 int=8 sup=5 esc=7 vel=6 mem=8 num=8 rep=10 cfg=7 aju=8",
  ["Calibración y SIMEX", "Corrige la atenuación"], ["Necesita la fiabilidad o la varianza del error"], "Predictores medidos con error (ingresos declarados, encuestas).", "No sabes cuánto error hay.")
f("regresion_polinomica", "regresion_no_parametrica", A + "ins=9 con=9 gen=6 par=7 int=7 est=7 sup=5 cfg=6 sas=9", ["Polinomios ortogonales", "Grado por F secuencial o CV"], ["Malos en los extremos"],
  "Curvatura suave en una variable.", "Formas complejas: splines.")
f("tamano_muestral_modelo", "regresion_inferencia", A + "int=9 sup=6 cfg=7", ["Criterios de Riley: sobreajuste, optimismo y precisión"], ["Requiere un R² esperado"], "Antes de construir un modelo de predicción.", "—")
f("regresion_local", "regresion_no_parametrica", "con=8 gen=7 int=7 rob=7 sup=9 esc=4 vel=4 mem=6 num=8 suc=6 rep=10 cfg=7 aju=6",
  ["LOESS robusto o Nadaraya-Watson con CV"], ["Sufre en los bordes", "La CV es costosa"], "Ver la forma de una relación sin suponerla.", "Muchas X.")
f("funciones_escalonadas", "regresion_no_parametrica", A + "ins=8 con=7 gen=6 par=6 int=10 sup=8 cfg=7 sas=10", ["Tramos explicables (tarifas)", "Compara AIC con lineal"], ["Discontinuo, pierde información"],
  "Tarifas o reglas por tramos.", "Necesitas una curva suave.")
f("ajustar_mars", "regresion_no_parametrica", "con=7 gen=8 dis=7 par=7 int=7 est=6 sup=8 esc=5 vel=4 mem=8 num=8 suc=6 rep=10 cfg=7 aju=5",
  ["Detecta umbrales (bisagras)", "Poda por GCV"], ["Grado 1 (sin interacciones)", "Búsqueda lenta con muchas X"], "Relaciones con codos o umbrales.", "Interacciones fuertes: boosting.")
f("regresion_inversa_cortes", "reduccion_supervisada", "con=8 int=7 sup=6 esc=9 vel=9 mem=9 num=8 rep=10 cfg=6 aju=9", ["Reducción suficiente sin suponer la forma"], ["Ciega a relaciones simétricas"],
  "Muchas X y quieres pocas direcciones relevantes.", "y depende de X de forma simétrica (x²).")
f("regresion_pls_pcr", "reduccion_supervisada", "con=8 gen=8 par=8 int=6 est=7 sup=7 esc=8 vel=7 mem=8 num=9 rep=10 cfg=6 aju=7 sas=9",
  ["PLS o PCR con CV", "Coeficientes en escala original"], ["Componentes difíciles de interpretar"], "Muchas X colineales.", "Pocas X bien separadas: OLS.")
f("modelo_loglineal", "tablas", "ins=9 con=10 efi=9 alf=8 int=8 sup=6 esc=8 vel=8 mem=8 num=8 rep=10 cfg=8 aju=9 sas=10",
  ["G² frente al saturado y comparación jerárquica"], ["Tablas grandes se vuelven dispersas"], "Asociación entre 3 o más categóricas.", "Hay una variable respuesta clara: logística.")
f("tabla_nomograma", "interpretar_modelo", A + "int=10 sup=6 cfg=5 sas=6", ["Puntos por variable y conversión a probabilidad"], ["Sin interacciones"], "Explicar o usar un modelo a mano.", "—")
f("aproximar_modelo", "interpretar_modelo", "ins=7 par=10 int=10 est=8 sup=8 esc=7 vel=6 mem=9 num=9 rep=10 cfg=5 aju=8",
  ["Simplifica sin mirar la respuesta"], ["Bucle hacia atrás: coste p²"], "Pasar de un modelo grande a uno explicable.", "—")
f("dominio_aplicabilidad", "interpretar_modelo", "int=9 rob=7 sup=7 esc=8 vel=8 mem=7 num=8 rep=10 cfg=5 aju=9", ["Cuatro criterios (apalancamiento, Mahalanobis, k-NN, rango)"], ["Solo numéricas"],
  "Antes de puntuar casos nuevos con un modelo.", "—")
f("centrar_por_grupo", "preparar", A + "int=10 sup=10 cfg=5", ["Separa efecto dentro y entre grupos"], ["—"], "Modelos multinivel.", "—")
f("diccionario_datos", "describir", "int=10 sup=10 esc=8 vel=7 mem=7 num=10 suc=8 rep=10 cfg=5 aju=10", ["Rol sugerido y alerta de fuga"], ["Heurístico"], "Documentar o recibir un dataset nuevo.", "—")
f("regresion_multivariante", "multivariante_avanzado", "ins=10 con=10 efi=9 alf=8 pot=8 int=8 sup=4 esc=8 vel=8 mem=8 num=9 rep=10 cfg=4 aju=9 sas=10",
  ["Wilks y Pillai por predictor", "Correlación de residuos"], ["Normalidad multivariante"], "Varias respuestas relacionadas.", "—")
f("analisis_perfiles", "multivariante_avanzado", "alf=8 pot=8 int=9 sup=4 esc=9 vel=9 mem=9 num=9 rep=10 cfg=3 aju=10 sas=9", ["Paralelismo, niveles y planitud"], ["Medidas conmensurables"],
  "Comparar la evolución de grupos.", "Tiempos irregulares: modelo mixto.")
f("control_t2_multivariante", "multivariante_avanzado", "alf=9 pot=8 int=8 sup=5 esc=8 vel=8 mem=9 num=8 rep=10 cfg=6 aju=9 sas=8", ["Fases I y II", "Variable que más contribuye"], ["Normalidad; sensible a fase I contaminada"],
  "Vigilar procesos con varias variables correladas.", "—")
f("analisis_conjunto", "multivariante_avanzado", "ins=9 con=9 int=10 sup=6 esc=7 vel=7 mem=8 num=9 rep=10 cfg=6 aju=9 sas=8", ["Utilidades e importancia", "Individuales para segmentar"], ["Valoraciones (no elecciones)"],
  "Diseño de producto o tarifa.", "Datos de elección discreta: logit multinomial.")
f("modelo_grafico_gaussiano", "multivariante_avanzado", "con=8 par=9 int=8 est=6 sup=5 esc=6 vel=5 mem=8 num=8 rep=10 cfg=6 aju=7", ["Independencia condicional (aristas)"], ["Normalidad; α por CV"],
  "Red de relaciones directas entre variables.", "—")
f("pca_funcional", "multivariante_avanzado", "con=8 int=7 sup=7 esc=8 vel=8 mem=7 num=8 suc=6 rep=10 cfg=7 aju=8", ["Suavizado B-spline + PCA", "Puntuaciones como variables"], ["Curvas en la misma rejilla"],
  "Perfiles horarios, curvas de tipos o de siniestralidad.", "—")
f("regresion_matriz_indicadora", "clasificacion_lineal", A + "int=9 dis=5 sup=6 cfg=2", ["Didáctico: muestra el enmascaramiento"], ["Mal clasificador con ≥ 3 clases"], "Enseñar por qué se usa LDA/logística.", "Clasificar de verdad.")
f("centroides_contraidos", "clasificacion_lineal", "gen=8 dis=7 par=9 int=8 est=7 sup=7 esc=7 vel=6 mem=8 num=9 rep=10 cfg=6 aju=7", ["p ≫ n con selección incorporada"], ["Fronteras lineales"],
  "Muchas variables y pocas observaciones.", "—")
f("pls_da", "clasificacion_lineal", "gen=7 dis=7 par=7 int=6 est=6 sup=7 esc=7 vel=6 mem=8 num=8 rep=10 cfg=6 aju=7 sas=8", ["VIP por variable", "p > n"], ["Tiende a sobreajustar si no se valida"],
  "Clasificar con variables muy colineales.", "—")
f("reglas_asociacion", "no_supervisado_ml", "int=10 sup=10 esc=5 vel=5 mem=5 num=10 suc=6 rep=10 cfg=7 aju=7", ["Soporte, confianza, lift y convicción"], ["Explosión combinatoria con soporte bajo"],
  "Venta cruzada, cesta de productos.", "—")
f("mapa_autoorganizado", "no_supervisado_ml", "con=6 int=7 est=5 sup=8 esc=6 vel=5 mem=7 num=9 rep=9 cfg=7 aju=6", ["Mapa 2D que conserva vecindades", "Matriz U"], ["Resultados dependen de semilla y tamaño"],
  "Visualizar segmentos en muchas dimensiones.", "—")
f("modelo_ngramas", "generativa", "int=9 sup=8 esc=7 vel=6 mem=6 num=9 rep=10 cfg=6 aju=8", ["Misma tarea que un LLM con conteos", "Perplejidad"], ["Juguete: contexto corto"], "Entender cómo predice un modelo de lenguaje.", "—")
f("muestrear_siguiente", "generativa", A + "int=10 sup=10 cfg=9", ["Temperatura, top-k y top-p"], ["—"], "Entender la diversidad de un LLM.", "—")
f("muestrear_texto", "generativa", "int=9 sup=10 esc=7 vel=7 mem=8 num=10 rep=10 cfg=8 aju=10", ["Autorregresivo"], ["Juguete"], "Demostración de generación.", "—")
f("ajustar_bradley_terry", "generativa", "ins=7 con=9 efi=9 inc=8 int=9 sup=7 esc=8 vel=8 mem=8 num=9 rep=10 cfg=6 aju=9", ["Modelo de recompensa (RLHF) y rankings"], ["Transitividad"],
  "Ordenar a partir de comparaciones por pares.", "—")
f("recuperar_tfidf", "generativa", "int=9 sup=9 esc=8 vel=8 mem=7 num=10 rep=10 cfg=6 aju=10", ["Recuperación simple para RAG"], ["Léxico, no semántico"], "Buscar fragmentos relevantes en documentos.", "Sinónimos y paráfrasis: embeddings.")
f("autoconsistencia_votacion", "generativa", "int=10 sup=8 esc=8 vel=8 mem=8 num=10 rep=10 cfg=6 aju=10", ["Cuándo votar mejora (Condorcet)"], ["Supone cadenas independientes"], "Decidir cuántas muestras pedir a un LLM.", "—")
f("proceso_difusion", "generativa", "int=9 sup=10 esc=9 vel=9 mem=7 num=10 rep=10 cfg=7 aju=10", ["El ruido directo de DDPM con fórmula cerrada"], ["Solo la parte directa"], "Entender los modelos de difusión.", "—")
f("arbol_black_derman_toy", "renta_fija", A + "ins=8 cal=10 int=7 sup=5 cfg=7", ["Reproduce la curva exacta", "Tipos siempre positivos (log-normal)"], ["Volatilidades del tipo corto dadas"],
  "Valorar opciones sobre bonos, caps o rescatables.", "Necesitas calibrar a precios de opciones: Hull-White u otros.")
for nombre, grupo, extra, pro in [
    ("grafico_trazas_mcmc", "graf_simulacion", "int=9 cfg=4", "Trazas por cadena + densidades + R-hat"),
    ("grafico_bandido", "graf_simulacion", "int=9 cfg=3", "Arrepentimiento acumulado por estrategia"),
    ("grafico_trayectorias", "graf_simulacion", "int=9 cfg=5", "Trayectorias con media y banda 5-95 %"),
    ("grafico_interaccion", "graf_diseno", "int=10 cfg=4 sas=9", "Medias por celda ±EE (como LSMEANS PLOT)"),
    ("grafico_efectos_2k", "graf_diseno", "int=9 cfg=3 sas=8", "Gráfico seminormal con efectos activos"),
    ("grafico_superficie_respuesta", "graf_diseno", "int=9 cfg=5 sas=8", "Contornos con punto estacionario"),
    ("grafico_balance", "graf_diseno", "int=10 cfg=3", "Love plot antes/después"),
    ("grafico_metaanalisis", "graf_diseno", "int=10 cfg=3", "Forest plot con diamante e intervalo de predicción"),
    ("grafico_hexbin", "graf_exploracion", "int=9 cfg=5", "Densidad de nubes enormes"),
    ("grafico_violin", "graf_exploracion", "int=9 cfg=3", "Distribución completa por grupo"),
    ("grafico_coordenadas_paralelas", "graf_exploracion", "int=8 cfg=4", "Perfiles multivariantes por clase"),
    ("grafico_curvas_andrews", "graf_exploracion", "int=7 cfg=3", "Curvas que preservan distancias"),
    ("grafico_caras_chernoff", "graf_exploracion", "int=6 cfg=3", "Perfiles de pocos casos como caras"),
    ("grafico_variable_anadida", "graf_modelos", "int=9 cfg=3 sas=9", "Efecto parcial limpio de una variable"),
    ("grafico_bandas_regresion", "graf_modelos", "int=10 cfg=3 sas=9", "Banda simultánea + predicción"),
    ("grafico_control_t2", "graf_exploracion", "int=9 cfg=3 sas=8", "T² con límites de fase I y II"),
]:
    f(nombre, grupo, G + extra, [pro], [], "Necesitas la figura para un notebook o un informe.", "—")


def main():
    raiz = Path(__file__).resolve().parents[1]
    props = {p["id"] for p in json.loads((raiz / "py" / "propiedades" / "propiedades.json").read_text(encoding="utf-8"))["propiedades"]}
    salida = {}
    for nombre, d in sorted(F.items()):
        notas = {}
        for par in d["notas"].split():
            k, v = par.split("=")
            notas[ABREV[k]] = int(v)
        assert set(notas) <= props, (nombre, set(notas) - props)
        salida[nombre] = dict(d, notas=notas)
    destino = raiz / "py" / "propiedades" / "fichas.json"
    destino.write_text(json.dumps({"funciones": salida}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{destino.relative_to(raiz)}: {len(salida)} fichas")


if __name__ == "__main__":
    main()
