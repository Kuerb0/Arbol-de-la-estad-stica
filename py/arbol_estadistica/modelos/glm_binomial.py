"""Rama MODELOS / GLM binomial con fórmula (patsy) + split estratificado.

Origen: `notebook de consultoría` (split 70/30 estratificado, smf.glm Binomial, AUC train/test,
AIC/BIC) y comparaciones de modelos de `notebook de consultoría`.
Equivale a PROC LOGISTIC / PROC GENMOD (dist=bin link=logit) con CLASS implícita vía C().
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

from ..diagnostico.metricas import auc_train_test


def dividir_train_test(df: pd.DataFrame, objetivo: str, train: float = 0.70,
                       semilla: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split estratificado por `objetivo` (mantiene la proporción de clases en ambos lados)."""
    # Solo train_size: pasar también test_size=1-train falla por redondeo en coma flotante (1-0.7).
    return train_test_split(df, train_size=train, random_state=semilla, stratify=df[objetivo])


def tabla_coeficientes(modelo, exponenciar: bool = True, alpha: float = 0.05) -> pd.DataFrame:
    """Coeficientes con SE, p-valor e IC; con `exponenciar` añade OR (logit) = exp(beta) y su IC."""
    ci = modelo.conf_int(alpha=alpha)
    t = pd.DataFrame({"coef": modelo.params, "se": modelo.bse, "p_valor": modelo.pvalues,
                      "ic_inf": ci[0], "ic_sup": ci[1]})
    if exponenciar:
        t["OR"] = np.exp(t["coef"])
        t["OR_ic_inf"] = np.exp(t["ic_inf"])
        t["OR_ic_sup"] = np.exp(t["ic_sup"])
    return t


def ajustar_glm_binomial(formula: str, df_train: pd.DataFrame, df_test: pd.DataFrame | None = None,
                         objetivo: str | None = None, familia=None) -> dict:
    """Ajusta smf.glm (Binomial/logit por defecto) y resume AIC, BIC y AUC train/test.

    `formula` en estilo R/patsy, p. ej. "renewed ~ C(region) + age + has_chronic".
    Si pasas df_test y objetivo se calcula también el gap de AUC y la alerta de sobreajuste.
    """
    familia = familia or sm.families.Binomial()
    modelo = smf.glm(formula, data=df_train, family=familia).fit()
    out = {"modelo": modelo, "aic": float(modelo.aic), "bic": float(modelo.bic_llf),
           "tabla": tabla_coeficientes(modelo)}
    if df_test is not None and objetivo is not None:
        out.update(auc_train_test(modelo, df_train, df_test, objetivo))
    return out
