"""Rama MODELOS / regresión logística penalizada de Firth.

Origen: `firth_logistic` de notebook de consultoría. Equivale a MODEL ... / FIRTH de PROC LOGISTIC.
Cuándo usarla: separación (casi) perfecta, clases muy raras o muestras pequeñas, donde la
máxima verosimilitud estándar da coeficientes enormes o no converge. Con datos grandes y
equilibrados da prácticamente lo mismo que Newton (comprobado en tu notebook).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm


def regresion_logistica_firth(X, y, maxiter: int = 100, tol: float = 1e-8) -> dict:
    """Máxima verosimilitud penalizada con la penalización de Jeffreys (Firth).

    X debe incluir la columna constante si se quiere intercepto. Devuelve dict con
    `beta`, `se`, `z`, `p_valor` (Wald), `convergio` y `n_iter`; si X es DataFrame, los
    resultados son pd.Series indexadas por columna.
    """
    nombres = list(X.columns) if isinstance(X, pd.DataFrame) else None
    Xm = np.asarray(X, dtype=float)
    yv = np.asarray(y, dtype=float)
    n, p = Xm.shape
    beta = np.zeros(p)
    convergio, it = False, 0
    for it in range(1, maxiter + 1):
        pi = expit(Xm @ beta)
        w = pi * (1 - pi)
        info = Xm.T @ (w[:, None] * Xm) + np.eye(p) * 1e-10
        try:
            inv = np.linalg.inv(info)
        except np.linalg.LinAlgError:
            break
        h = np.einsum("ij,jk,ik->i", Xm, inv, Xm) * w          # diagonal de la matriz sombrero
        delta = inv @ (Xm.T @ (yv - pi + h * (0.5 - pi)))       # score modificado de Firth
        beta = beta + delta
        if np.max(np.abs(delta)) < tol:
            convergio = True
            break

    pi = expit(Xm @ beta)
    w = pi * (1 - pi)
    cov = np.linalg.inv(Xm.T @ (w[:, None] * Xm) + np.eye(p) * 1e-10)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pv = 2 * norm.sf(np.abs(z))
    if nombres:
        beta, se, z, pv = (pd.Series(a, index=nombres) for a in (beta, se, z, pv))
    return {"beta": beta, "se": se, "z": z, "p_valor": pv, "convergio": convergio, "n_iter": it}
