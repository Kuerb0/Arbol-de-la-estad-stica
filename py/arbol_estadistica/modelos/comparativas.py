"""Rama MODELOS / comparativas de técnica de estimación y función de enlace (binario).

Origen: bloques C y D de la re-estimación en notebook de consultoría (Newton vs IRLS vs Firth;
logit vs probit vs cloglog). Sirven para justificar al cliente que el modelo no depende de
la técnica numérica ni del enlace elegido.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit

from ..diagnostico.metricas import metricas_binarias
from .firth import regresion_logistica_firth


def _fila(nombre, y_tr, p_tr, y_te, p_te) -> dict:
    a, b = metricas_binarias(y_tr, p_tr), metricas_binarias(y_te, p_te)
    return {"metodo": nombre,
            "AUC_train": a["AUC"], "AUC_test": b["AUC"],
            "LogLoss_train": a["LogLoss"], "LogLoss_test": b["LogLoss"],
            "Brier_train": a["Brier"], "Brier_test": b["Brier"]}


def comparar_tecnicas_estimacion(y_tr, X_tr, y_te, X_te) -> tuple[pd.DataFrame, float]:
    """Newton-Raphson vs IRLS (GLM) vs Firth. X con constante.

    Devuelve (tabla, delta_max): delta_max = máxima diferencia absoluta de coeficientes
    entre Newton y Firth (si es ~0, la penalización de Firth no cambia nada).
    """
    newton = sm.Logit(y_tr, X_tr).fit(method="newton", disp=False, maxiter=200)
    irls = sm.GLM(y_tr, X_tr, family=sm.families.Binomial()).fit()
    firth = regresion_logistica_firth(X_tr, y_tr)
    beta_f = np.asarray(firth["beta"])
    tabla = pd.DataFrame([
        _fila("Newton-Raphson", y_tr, newton.predict(X_tr), y_te, newton.predict(X_te)),
        _fila("IRLS (GLM)", y_tr, irls.predict(X_tr), y_te, irls.predict(X_te)),
        _fila("Firth", y_tr, expit(np.asarray(X_tr, float) @ beta_f), y_te,
              expit(np.asarray(X_te, float) @ beta_f)),
    ])
    return tabla, float(np.max(np.abs(newton.params.to_numpy() - beta_f)))


def comparar_enlaces(y_tr, X_tr, y_te, X_te) -> pd.DataFrame:
    """Logit vs Probit vs CLogLog (este último útil con eventos raros / asimetría)."""
    logit = sm.Logit(y_tr, X_tr).fit(method="newton", disp=False, maxiter=200)
    probit = sm.Probit(y_tr, X_tr).fit(method="newton", disp=False, maxiter=200)
    cll = sm.GLM(y_tr, X_tr, family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()
    return pd.DataFrame([
        _fila("Logit", y_tr, logit.predict(X_tr), y_te, logit.predict(X_te)),
        _fila("Probit", y_tr, probit.predict(X_tr), y_te, probit.predict(X_te)),
        _fila("CLogLog", y_tr, cll.predict(X_tr), y_te, cll.predict(X_te)),
    ])
