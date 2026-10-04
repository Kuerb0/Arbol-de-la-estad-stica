"""Evaluación de clasificadores para negocio: ganancia y lift por deciles, KS y Gini, curva precisión-exhaustividad,
descomposición del Brier (Murphy) y recalibración de probabilidades (Platt / isotónica).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .._util import _numerico


def _par(y_real, y_prob):
    y, p = np.asarray(y_real, float), np.asarray(y_prob, float)
    if y.shape != p.shape:
        raise ValueError("y_real e y_prob deben tener la misma longitud.")
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if not set(np.unique(y)) <= {0, 1} or len(np.unique(y)) < 2:
        raise ValueError("y_real debe ser 0/1 con las dos clases presentes.")
    return y.astype(int), p


def tabla_ganancia_lift(y_real, y_prob, grupos: int = 10) -> pd.DataFrame:
    """Ordena por probabilidad descendente y corta en `grupos` (deciles): por grupo, nº de casos, eventos, tasa,
    lift (tasa/tasa global), % de eventos capturados acumulado (ganancia) y lift acumulado. La tabla de campañas:
    «con el 20 % de clientes con más score capturo el 55 % de las bajas»."""
    y, p = _par(y_real, y_prob)
    orden = np.argsort(-p, kind="stable")
    y, p = y[orden], p[orden]
    corte = np.minimum((np.arange(len(y)) * grupos) // len(y), grupos - 1) + 1
    t = pd.DataFrame({"grupo": corte, "y": y, "p": p}).groupby("grupo").agg(n=("y", "size"), eventos=("y", "sum"),
                                                                          prob_media=("p", "mean"), prob_min=("p", "min"))
    base = y.mean()
    t["tasa"] = t.eventos / t.n
    t["lift"] = t.tasa / base
    t["pct_poblacion_acum"] = t.n.cumsum() / t.n.sum() * 100
    t["ganancia_acum"] = t.eventos.cumsum() / t.eventos.sum() * 100
    t["lift_acum"] = (t.eventos.cumsum() / t.n.cumsum()) / base
    return t


def estadistico_ks_gini(y_real, y_prob) -> dict:
    """KS (máxima distancia entre las distribuciones del score de eventos y no eventos, y el umbral donde se alcanza)
    y Gini = 2·AUC − 1 (el «Gini» de riesgo de crédito). Equivale a PROC NPAR1WAY EDF (KS) sobre el score."""
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    y, p = _par(y_real, y_prob)
    ks = stats.ks_2samp(p[y == 1], p[y == 0])
    auc = roc_auc_score(y, p)
    umbrales = np.unique(p)
    F1 = np.searchsorted(np.sort(p[y == 1]), umbrales, side="right") / (y == 1).sum()
    F0 = np.searchsorted(np.sort(p[y == 0]), umbrales, side="right") / (y == 0).sum()
    i = int(np.argmax(np.abs(F0 - F1)))
    return {"ks": float(ks.statistic), "p_valor_ks": float(ks.pvalue), "umbral_ks": float(umbrales[i]),
            "auc": float(auc), "gini": float(2 * auc - 1)}


def curva_precision_recall(y_real, y_prob) -> dict:
    """Precisión y exhaustividad (recall) por umbral, precisión media (AP) y el umbral de F1 máximo.
    Con eventos raros es más informativa que la ROC (la precisión depende de la prevalencia; la ROC no)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve
    y, p = _par(y_real, y_prob)
    pr, rc, th = precision_recall_curve(y, p)
    f1 = np.where(pr[:-1] + rc[:-1] > 0, 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1]), 0)
    i = int(np.argmax(f1))
    return {"tabla": pd.DataFrame({"umbral": th, "precision": pr[:-1], "recall": rc[:-1], "f1": f1}),
            "precision_media": float(average_precision_score(y, p)), "prevalencia": float(y.mean()),
            "umbral_f1": float(th[i]), "f1_max": float(f1[i])}


def descomposicion_brier(y_real, y_prob, grupos: int = 10) -> dict:
    """Brier = fiabilidad − resolución + incertidumbre (Murphy). Fiabilidad baja = bien calibrado; resolución alta =
    discrimina; incertidumbre = p̄(1−p̄) (no depende del modelo). Se calcula por deciles de probabilidad."""
    y, p = _par(y_real, y_prob)
    g = pd.qcut(p, grupos, labels=False, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "g": g}).groupby("g").agg(n=("y", "size"), o=("y", "mean"), f=("p", "mean"))
    base = y.mean(); N = len(y)
    fia = float(np.sum(t.n * (t.f - t.o) ** 2) / N)
    res = float(np.sum(t.n * (t.o - base) ** 2) / N)
    inc = float(base * (1 - base))
    return {"brier": float(np.mean((p - y) ** 2)), "fiabilidad": fia, "resolucion": res, "incertidumbre": inc,
            "brier_skill": 1 - float(np.mean((p - y) ** 2)) / inc}


def calibrar_probabilidades(y_calibracion, p_calibracion, p_nuevas=None, metodo: str = "platt"):
    """Recalibra probabilidades con un conjunto de calibración (NO el de entrenamiento): 'platt' (logística sobre el
    logit de p, 2 parámetros, estable con pocos datos) o 'isotonica' (monótona no paramétrica, necesita más datos).
    Devuelve las probabilidades recalibradas de `p_nuevas` (o de las de calibración) y la función."""
    y, p = _par(y_calibracion, p_calibracion)
    if metodo == "platt":
        from sklearn.linear_model import LogisticRegression
        z = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))).reshape(-1, 1)
        m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)

        def f(q):
            q = np.clip(np.asarray(q, float), 1e-9, 1 - 1e-9)
            return m.predict_proba(np.log(q / (1 - q)).reshape(-1, 1))[:, 1]
        params = {"a": float(m.coef_[0, 0]), "b": float(m.intercept_[0])}
    elif metodo == "isotonica":
        from sklearn.isotonic import IsotonicRegression
        m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        f = lambda q: m.predict(np.asarray(q, float))  # noqa: E731
        params = {}
    else:
        raise ValueError("metodo: 'platt' o 'isotonica'")
    nuevas = f(p if p_nuevas is None else _numerico(p_nuevas, "p_nuevas", 1))
    return {"probabilidades": nuevas, "funcion": f, "parametros": params, "metodo": metodo}
