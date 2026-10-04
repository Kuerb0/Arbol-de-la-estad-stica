"""Describir y simplificar un modelo ajustado: tabla de puntos de un nomograma y aproximación del modelo completo por
uno más sencillo (Harrell)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def tabla_nomograma(modelo, datos: pd.DataFrame, variables=None, puntos_max: float = 100, n_valores: int = 6) -> dict:
    """Nomograma de un modelo lineal/GLM ajustado con fórmula (statsmodels): asigna a cada valor de cada variable unos
    PUNTOS proporcionales a su contribución al predictor lineal (la variable de mayor rango aporta de 0 a
    `puntos_max`), y una tabla de conversión puntos totales → predictor lineal → probabilidad (si es binomial).
    Así el modelo se puede usar «a mano» y se ve qué variable pesa más. Supone que las variables entran sin
    interacciones (con interacciones el nomograma clásico no vale)."""
    variables = list(variables) if variables is not None else [c for c in modelo.model.data.frame.columns if c != modelo.model.endog_names]
    d = datos.copy()
    base = d.iloc[[0]].copy()
    for c in variables:
        base[c] = d[c].mode().iloc[0] if not pd.api.types.is_numeric_dtype(d[c]) else d[c].median()
    eta0 = float(np.asarray(modelo.predict(base, which="linear") if hasattr(modelo, "family") else modelo.predict(base))[0])
    contrib = {}
    for c in variables:
        vals = sorted(d[c].dropna().unique()) if not pd.api.types.is_numeric_dtype(d[c]) else np.linspace(d[c].quantile(0.02), d[c].quantile(0.98), n_valores)
        rej = pd.concat([base] * len(vals), ignore_index=True); rej[c] = vals
        eta = np.asarray(modelo.predict(rej, which="linear") if hasattr(modelo, "family") else modelo.predict(rej)) - eta0
        contrib[c] = pd.Series(eta, index=vals)
    rango = {c: s.max() - s.min() for c, s in contrib.items()}
    escala = puntos_max / max(rango.values())
    filas = []
    for c, s in contrib.items():
        for v, e in s.items():
            filas.append({"variable": c, "valor": v, "puntos": (e - s.min()) * escala})
    puntos = pd.DataFrame(filas)
    minimo = sum(s.min() for s in contrib.values()) + eta0
    tot = np.linspace(0, sum(r * escala for r in rango.values()), 11)
    conv = pd.DataFrame({"puntos_totales": tot, "predictor_lineal": minimo + tot / escala})
    if hasattr(modelo, "family"):
        conv["prediccion"] = modelo.family.link.inverse(conv.predictor_lineal)
    importancia = pd.Series({c: r * escala for c, r in rango.items()}).sort_values(ascending=False)
    return {"puntos": puntos, "conversion": conv, "importancia_puntos": importancia}


def aproximar_modelo(X: pd.DataFrame, prediccion, r2_objetivo: float = 0.95) -> dict:
    """Simplificación por aproximación (Harrell): ajusta por MCO el PREDICTOR LINEAL del modelo completo (o cualquier
    predicción continua: de un random forest, un GLM penalizado…) con las variables de X y elimina hacia atrás la
    que menos R² aporta mientras el R² de la aproximación siga ≥ `r2_objetivo`. Da un modelo pequeño que reproduce
    casi igual las predicciones del grande, sin el sesgo de seleccionar mirando la respuesta."""
    Xd = pd.get_dummies(pd.DataFrame(X), drop_first=True, dtype=float)
    yhat = np.asarray(prediccion, float)
    if len(Xd) != len(yhat):
        raise ValueError("X y prediccion: misma longitud.")
    grupos = {c: [k for k in Xd.columns if k == c or k.startswith(f"{c}_")] for c in pd.DataFrame(X).columns}
    actuales = list(grupos)

    def r2(vs):
        cols = [k for v in vs for k in grupos[v]]
        if not cols:
            return 0.0
        return float(sm.OLS(yhat, sm.add_constant(Xd[cols])).fit().rsquared)
    hist = [{"paso": 0, "quitada": None, "r2": r2(actuales), "n_variables": len(actuales)}]
    while len(actuales) > 1:
        cand = [(r2([v for v in actuales if v != q]), q) for q in actuales]
        mejor, q = max(cand)
        if mejor < r2_objetivo:
            break
        actuales.remove(q)
        hist.append({"paso": len(hist), "quitada": q, "r2": mejor, "n_variables": len(actuales)})
    cols = [k for v in actuales for k in grupos[v]]
    m = sm.OLS(yhat, sm.add_constant(Xd[cols])).fit()
    return {"variables": actuales, "historial": pd.DataFrame(hist).set_index("paso"), "coeficientes": m.params, "r2": float(m.rsquared)}
