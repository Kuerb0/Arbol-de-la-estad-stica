"""Validación de modelos: validación cruzada (repetida, estratificada), optimismo por bootstrap (Harrell),
comparación de dos modelos con t corregido (Nadeau-Bengio), métricas de regresión y tabla de criterios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .._util import _columnas

_FAMILIAS = {"binomial": sm.families.Binomial, "gaussiana": sm.families.Gaussian, "poisson": sm.families.Poisson,
             "gamma": lambda: sm.families.Gamma(sm.families.links.Log())}


def metricas_regresion(y_real, y_pred) -> dict:
    """RMSE, MAE, MAPE (si no hay ceros), sesgo medio, R² y R² predictivo (sobre la media de y_real)."""
    y, p = np.asarray(y_real, float), np.asarray(y_pred, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    e = y - p
    out = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))), "sesgo": float(np.mean(p - y)),
           "r2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)), "n": int(len(y))}
    out["mape"] = float(np.mean(np.abs(e / y)) * 100) if np.all(y != 0) else np.nan
    return out


def _metricas(fam, y, p):
    if fam == "binomial":
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        return {"auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan, "logloss": log_loss(y, pc, labels=[0, 1]),
                "brier": brier_score_loss(y, p)}
    m = metricas_regresion(y, p)
    if fam == "poisson":
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = 2 * np.sum(np.where(y > 0, y * np.log(y / p), 0) - (y - p)) / len(y)
        m["deviance_media"] = float(dev)
    return {k: m[k] for k in m if k not in ("n", "mape")}


def validacion_cruzada(df: pd.DataFrame, formula: str, familia: str = "binomial", k: int = 5, repeticiones: int = 1,
                       semilla: int = 42) -> dict:
    """Validación cruzada k-fold (estratificada si es binomial), repetida `repeticiones` veces, de un GLM con fórmula.
    Devuelve las métricas por fold y un resumen (media, desviación e IC 95 % aproximado de la media).
    Binomial: AUC, log-loss y Brier. Gaussiana/Gamma: RMSE, MAE, sesgo, R². Poisson: además deviance media.
    Gauss: TODO lo que «aprende» de los datos (selección de variables, tramos, imputación) debe ir dentro de cada fold;
    si no, la estimación es optimista."""
    from sklearn.model_selection import KFold, StratifiedKFold
    y_name = formula.split("~")[0].strip()
    _columnas(df, [y_name])
    if familia not in _FAMILIAS:
        raise ValueError(f"familia: {sorted(_FAMILIAS)}")
    d = df.reset_index(drop=True)
    filas = []
    for r in range(repeticiones):
        cv = (StratifiedKFold if familia == "binomial" else KFold)(k, shuffle=True, random_state=semilla + r)
        for f, (tr, te) in enumerate(cv.split(d, d[y_name] if familia == "binomial" else None)):
            m = smf.glm(formula, d.iloc[tr], family=_FAMILIAS[familia]()).fit()
            p = np.asarray(m.predict(d.iloc[te]))
            filas.append({"repeticion": r, "fold": f, **_metricas(familia, d[y_name].to_numpy()[te], p)})
    t = pd.DataFrame(filas)
    met = [c for c in t.columns if c not in ("repeticion", "fold")]
    res = pd.DataFrame({"media": t[met].mean(), "sd": t[met].std(ddof=1)})
    res["ic_inf"] = res["media"] - 1.96 * res["sd"] / np.sqrt(len(t))
    res["ic_sup"] = res["media"] + 1.96 * res["sd"] / np.sqrt(len(t))
    return {"por_fold": t, "resumen": res, "k": k, "repeticiones": repeticiones}


def optimismo_bootstrap(df: pd.DataFrame, formula: str, familia: str = "binomial", n_boot: int = 200, semilla: int = 42) -> pd.DataFrame:
    """Validación interna de Harrell: optimismo = media de (métrica en la muestra bootstrap − métrica de ese modelo en los
    datos originales); métrica corregida = aparente − optimismo. Usa TODOS los datos (mejor que un split con n pequeño).
    Equivale a validate() de rms (R)."""
    y_name = formula.split("~")[0].strip()
    _columnas(df, [y_name])
    d = df.reset_index(drop=True)
    fam = _FAMILIAS[familia]
    y = d[y_name].to_numpy()
    aparente = _metricas(familia, y, np.asarray(smf.glm(formula, d, family=fam()).fit().predict(d)))
    rng = np.random.default_rng(semilla)
    opt = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        db = d.iloc[idx]
        try:
            m = smf.glm(formula, db, family=fam()).fit()
        except Exception:  # noqa: BLE001
            continue
        mb = _metricas(familia, db[y_name].to_numpy(), np.asarray(m.predict(db)))
        mo = _metricas(familia, y, np.asarray(m.predict(d)))
        opt.append({k: mb[k] - mo[k] for k in mb})
    o = pd.DataFrame(opt).mean()
    t = pd.DataFrame({"aparente": pd.Series(aparente), "optimismo": o})
    t["corregida"] = t["aparente"] - t["optimismo"]
    t.attrs["n_boot_validos"] = len(opt)
    return t


def comparar_modelos_cv(df: pd.DataFrame, formula_a: str, formula_b: str, familia: str = "binomial", metrica: str | None = None,
                        k: int = 10, repeticiones: int = 5, semilla: int = 42) -> dict:
    """Compara dos fórmulas con los MISMOS folds y el t corregido de Nadeau-Bengio (corrige que los folds se solapan:
    el t pareado normal es demasiado optimista). Devuelve la diferencia media (B − A) en la métrica, su IC y p."""
    ra = validacion_cruzada(df, formula_a, familia, k, repeticiones, semilla)["por_fold"]
    rb = validacion_cruzada(df, formula_b, familia, k, repeticiones, semilla)["por_fold"]
    metrica = metrica or ("auc" if familia == "binomial" else "rmse")
    dif = (rb[metrica] - ra[metrica]).to_numpy()
    J = len(dif); ratio = 1 / (k - 1)                                  # n_test / n_train
    var = (1 / J + ratio) * np.var(dif, ddof=1)
    t = dif.mean() / np.sqrt(var) if var > 0 else 0.0
    gl = J - 1
    p = float(2 * stats.t.sf(abs(t), gl))
    h = stats.t.ppf(0.975, gl) * np.sqrt(var)
    return {"metrica": metrica, "media_a": float(ra[metrica].mean()), "media_b": float(rb[metrica].mean()),
            "diferencia": float(dif.mean()), "ic": (float(dif.mean() - h), float(dif.mean() + h)), "t": float(t), "p_valor": p,
            "p_valor_sin_corregir": float(stats.ttest_1samp(dif, 0).pvalue)}


def tabla_criterios(modelos: dict, sigma2_completo: float | None = None) -> pd.DataFrame:
    """Compara modelos ajustados ({nombre: resultado de statsmodels}) por AIC, BIC, deviance, nº de parámetros y,
    si son lineales, R² ajustado y Cp de Mallows (con la σ² del modelo completo). Ordenada por BIC."""
    filas = []
    for nombre, m in modelos.items():
        m = m["modelo"] if isinstance(m, dict) else m
        k = int(m.df_model + 1)
        fila = {"modelo": nombre, "k": k, "n": int(m.nobs), "aic": float(m.aic),
                "bic": float(getattr(m, "bic_llf", None) if hasattr(m, "bic_llf") else m.bic)}
        if hasattr(m, "rsquared_adj"):
            fila["r2_ajustado"] = float(m.rsquared_adj)
            if sigma2_completo:
                fila["cp_mallows"] = float(m.ssr / sigma2_completo - m.nobs + 2 * k)
        if hasattr(m, "deviance"):
            fila["deviance"] = float(m.deviance)
        filas.append(fila)
    t = pd.DataFrame(filas).set_index("modelo").sort_values("bic")
    t["delta_aic"] = t["aic"] - t["aic"].min()
    t["peso_akaike"] = np.exp(-t["delta_aic"] / 2) / np.exp(-t["delta_aic"] / 2).sum()
    return t
