"""Rama PREPROCESADO / balanceo de clases.

Origen: `balance_multiclass_df` (notebook de consultoría).
Equivale a over/under-sampling manual (en SAS, PROC SURVEYSELECT con muestreo por estratos).
IMPORTANTE: balancear SOLO el conjunto de entrenamiento, nunca el de test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils import resample


def balancear_clases(
    df: pd.DataFrame,
    objetivo: str,
    metodo: str | None = "oversample",
    semilla: int = 42,
) -> pd.DataFrame:
    """Devuelve una copia con las clases de `objetivo` equilibradas.

    metodo:
        'oversample'  -> sube todas las clases al tamaño de la mayoritaria (con reemplazo)
        'undersample' -> baja todas las clases al tamaño de la minoritaria (sin reemplazo)
        None          -> copia sin tocar
    """
    if metodo is None:
        return df.copy()
    if metodo not in ("oversample", "undersample"):
        raise ValueError("metodo debe ser None, 'oversample' o 'undersample'")

    conteos = df[objetivo].value_counts()
    if metodo == "oversample":
        n_objetivo, con_reemplazo = int(conteos.max()), True
    else:
        n_objetivo, con_reemplazo = int(conteos.min()), False

    partes = [
        resample(
            df[df[objetivo] == clase],
            replace=con_reemplazo,
            n_samples=n_objetivo,
            random_state=semilla,
        )
        for clase in conteos.index
    ]
    return (
        pd.concat(partes, axis=0)
        .sample(frac=1, random_state=semilla)
        .reset_index(drop=True)
    )


def pesos_por_clase(y) -> "pd.Series":
    """Pesos inversamente proporcionales a la frecuencia: w_c = n / (K * n_c).

    Alternativa a balancear remuestreando: se usan TODOS los datos y se pasan como `freq_weights`
    a sm.GLM (no a sm.Logit, que los ignora). Origen: modo 'class_weight' de notebook de consultoría.
    """
    y = pd.Series(y)
    conteos = y.value_counts()
    return y.map(len(y) / (len(conteos) * conteos)).astype(float)


def submuestreo_por_ratio(df: pd.DataFrame, objetivo: str, ratio_neg_pos: float = 1.0,
                          semilla: int = 42, clase_pos=1) -> pd.DataFrame:
    """Conserva todos los positivos y submuestrea los negativos a `ratio_neg_pos`:1.

    ratio 1 = 50/50 ('undersample_50' de notebook de consultoría); ratio 3 = 75/25 ('undersample_ratio').
    Si hay menos negativos de los pedidos se conservan todos. Solo sobre el train.
    """
    pos = df[df[objetivo] == clase_pos]
    neg = df[df[objetivo] != clase_pos]
    n = int(min(len(pos) * ratio_neg_pos, len(neg)))
    return pd.concat([pos, neg.sample(n=n, random_state=semilla)]).sample(frac=1, random_state=semilla)


def corregir_probabilidades_por_balanceo(p, prevalencia_real: float, prevalencia_muestra: float = 0.5):
    """Devuelve las probabilidades en la escala de la población real tras entrenar con datos balanceados.

    Entrenar con submuestreo/oversampling desplaza el intercepto: las probabilidades salen infladas
    para la clase rara. Corrección de prior (King & Zeng): se reescalan las odds,
        odds_real = odds_modelo * [prev_real/(1-prev_real)] / [prev_muestra/(1-prev_muestra)].
    Imprescindible antes de evaluar calibración (Hosmer-Lemeshow, Brier) o de fijar umbrales de negocio
    sobre datos con la prevalencia real. El AUC y el orden de las puntuaciones NO cambian.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    odds = p / (1 - p) * (prevalencia_real / (1 - prevalencia_real)) / (prevalencia_muestra / (1 - prevalencia_muestra))
    return odds / (1 + odds)
