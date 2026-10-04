"""Ejemplo de extremo a extremo con datos sintéticos: el flujo "tipo PROC LOGISTIC" del árbol.

Ejecutar:  python ejemplos/flujo_logit_binario.py   (desde la carpeta principal del árbol)
Sirve de plantilla: sustituye `df` por tu tabla y ajusta variables, referencias y unidades.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "py"))

from arbol_estadistica.contrastes import elegir_contraste
from arbol_estadistica.diagnostico import (estadisticos_asociacion, filtrar_vif_iterativo,
                                           hosmer_lemeshow, tabla_umbrales, umbral_optimo_youden)
from arbol_estadistica.modelos import (ajustar_logit, dividir_train_test, preparar_matriz_modelo,
                                       tabla_odds_ratios, tabla_parametros_wald)
from arbol_estadistica.preprocesado import corregir_probabilidades_por_balanceo, submuestreo_por_ratio
from arbol_estadistica.seleccion import cribar_variables, seleccion_forward

# 0) Datos sintéticos (sustituir por los reales)
rng = np.random.default_rng(42)
n = 6000
df = pd.DataFrame({"precio": rng.normal(30000, 6000, n), "potencia": rng.normal(150, 30, n),
                   "combustible": rng.choice(["Diesel", "Gasolina", "Electrico"], n, p=[.5, .4, .1])})
eta = -1.0 - 1.1 * (df.precio - 30000) / 6000 + 0.4 * (df.potencia - 150) / 30 \
      + df.combustible.map({"Diesel": 0, "Gasolina": 0.3, "Electrico": 0.8})
df["vendido"] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)

# 1) Cribado y contraste descriptivo
print(cribar_variables(df, "vendido")[["variable", "motivo", "p_valor", "cramer_v"]], "\n")
print(elegir_contraste(df, "precio", "vendido")["interpretacion"], "\n")

# 2) Split estratificado y balanceo SOLO del train
train, test = dividir_train_test(df, "vendido", 0.7)
train_bal = submuestreo_por_ratio(train, "vendido", ratio_neg_pos=1)

# 3) Selección de variables (BIC = más parsimonioso)
sel = seleccion_forward(train_bal, "vendido", ["precio", "potencia", "combustible"],
                        categoricas=["combustible"], criterio="bic")
print("Variables seleccionadas:", sel["seleccionadas"], "\n")

# 4) Matriz de modelo con referencia explícita y z-score; VIF
vars_, cats, nums = sel["seleccionadas"], ["combustible"], ["precio", "potencia"]
X, y, std_map = preparar_matriz_modelo(train_bal, "vendido", vars_, cats, nums, refs={"combustible": "Diesel"})
cols = [c for c in filtrar_vif_iterativo(X.assign(const=1.0), umbral=8) if c != "const"]
X = X[cols]

# 5) Ajuste y tablas "SAS"
modelo = ajustar_logit(y, X)
print(tabla_parametros_wald(modelo).round(4), "\n")
print(tabla_odds_ratios(modelo, unidades={"precio": 1000, "potencia": 10}, std_map=std_map).round(3), "\n")

# 6) Validación en test: mismas dummies y z-score con media/desv. del TRAIN (evita fuga de información)
Xt, yt, _ = preparar_matriz_modelo(test, "vendido", vars_, cats, nums, refs={"combustible": "Diesel"}, estandarizar=False)
for c in nums:
    Xt[c] = (Xt[c] - train_bal[c].mean()) / std_map[c]
Xt = Xt.reindex(columns=X.columns, fill_value=0.0)
p_bal = modelo.predict(np.column_stack([np.ones(len(Xt)), Xt.to_numpy()]))
# El modelo se entrenó con 50/50: hay que devolver las probabilidades a la prevalencia real antes de calibrar
p = corregir_probabilidades_por_balanceo(p_bal, train["vendido"].mean(), train_bal["vendido"].mean())
print("Asociación:", {k: round(v, 4) for k, v in estadisticos_asociacion(yt, p).items() if k in ("c", "Somers_D", "Gamma")})
hl = hosmer_lemeshow(yt, p)
print(f"Hosmer-Lemeshow: X2={hl['estadistico']:.2f}, gl={hl['gl']}, p={hl['p_valor']:.3f}")
print("Umbral óptimo (Youden):", round(umbral_optimo_youden(yt, p), 3))
print(tabla_umbrales(yt, p).head(5).to_string(index=False))
