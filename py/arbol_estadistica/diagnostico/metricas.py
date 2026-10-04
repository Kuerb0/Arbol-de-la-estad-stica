"""Rama DIAGNOSTICO / métricas de clasificación y sobreajuste.

Orígenes: bloque AUC train/test de `notebook de consultoría`, AUC multiclase de
`multinomial_logit_sas_like`, y CTABLE / Youden / `_met` de notebook de consultoría.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)


def auc_train_test(modelo, df_train: pd.DataFrame, df_test: pd.DataFrame, objetivo: str,
                   umbral_alerta: float = 0.02) -> dict:
    """AUC en train y test para un modelo de statsmodels con `.predict(df)` (fórmulas).

    Descarta filas con NaN en la predicción o en el objetivo antes de calcular.
    `alerta_sobreajuste` = gap train-test > `umbral_alerta` (0.02 en tu criterio de 3.0 GLM).
    """
    def _auc(df):
        p = modelo.predict(df)
        m = ~p.isna() & ~df[objetivo].isna()
        return float(roc_auc_score(df.loc[m, objetivo], p[m]))

    a_tr, a_te = _auc(df_train), _auc(df_test)
    return {"auc_train": a_tr, "auc_test": a_te, "gap": a_tr - a_te,
            "alerta_sobreajuste": (a_tr - a_te) > umbral_alerta}


def metricas_binarias(y_real, p_real, nombre: str = "") -> dict:
    """AUC, LogLoss y Brier de un conjunto (p_real = probabilidades predichas)."""
    return {
        "modelo": nombre,
        "AUC": round(float(roc_auc_score(y_real, p_real)), 4),
        "LogLoss": round(float(log_loss(y_real, p_real)), 4),
        "Brier": round(float(brier_score_loss(y_real, p_real)), 4),
    }


def tabla_umbrales(y_real, y_prob, umbrales=None) -> pd.DataFrame:
    """Equivalente a CTABLE de SAS: sensibilidad, especificidad, PPV, NPV y J de Youden por umbral."""
    y = np.asarray(y_real).astype(int)
    p = np.asarray(y_prob, dtype=float)
    umbrales = np.arange(0.10, 0.91, 0.05) if umbrales is None else umbrales
    filas = []
    for u in umbrales:
        pred = (p >= u).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        sens = tp / (tp + fn) if tp + fn else 0.0
        espec = tn / (tn + fp) if tn + fp else 0.0
        filas.append({
            "umbral": round(float(u), 2), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "sens": round(sens, 3), "espec": round(espec, 3),
            "PPV": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "NPV": round(tn / (tn + fn), 3) if tn + fn else 0.0,
            "Youden_J": round(sens + espec - 1, 3),
        })
    return pd.DataFrame(filas)


def umbral_optimo_youden(y_real, y_prob) -> float:
    """Umbral que maximiza J = sensibilidad + especificidad - 1 (curva ROC exacta)."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_real, y_prob)
    return float(thr[np.argmax(tpr - fpr)])


def resumen_auc_multiclase(codigos_reales, probs, categorias) -> dict:
    """AUC multiclase: macro/weighted OvR, macro OvO y OvR por clase (probs: n x K)."""
    probs = np.asarray(probs)
    return {
        "macro_ovr": roc_auc_score(codigos_reales, probs, multi_class="ovr", average="macro"),
        "weighted_ovr": roc_auc_score(codigos_reales, probs, multi_class="ovr", average="weighted"),
        "macro_ovo": roc_auc_score(codigos_reales, probs, multi_class="ovo", average="macro"),
        "por_clase_ovr": pd.DataFrame({
            "clase": list(categorias),
            "auc_ovr": roc_auc_score(codigos_reales, probs, multi_class="ovr", average=None),
        }),
    }


def matrices_confusion(y_real, y_pred, categorias) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Matriz de confusión en conteos y normalizada por filas (recall por clase)."""
    idx = [f"real_{c}" for c in categorias]
    col = [f"pred_{c}" for c in categorias]
    cm = pd.DataFrame(confusion_matrix(y_real, y_pred, labels=categorias), index=idx, columns=col)
    cm_f = pd.DataFrame(confusion_matrix(y_real, y_pred, labels=categorias, normalize="true"),
                        index=idx, columns=col)
    return cm, cm_f
