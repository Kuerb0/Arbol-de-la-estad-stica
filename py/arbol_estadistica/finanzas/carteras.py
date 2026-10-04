"""Teoría de carteras: covarianza robusta, frontera de Markowitz, cartera de mínima varianza y tangente, CAPM y
modelos de factores, Black-Litterman, dominancia estocástica y utilidad (equivalente cierto)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats


def matriz_covarianzas(rendimientos: pd.DataFrame, metodo: str = "ledoit_wolf") -> dict:
    """Covarianza de rendimientos: 'muestral' o 'ledoit_wolf' (contracción hacia una diana: mucho más estable cuando hay
    muchos activos y pocos datos; la muestral produce carteras «óptimas» absurdas). Devuelve covarianza, medias,
    correlaciones e intensidad de contracción."""
    R = pd.DataFrame(rendimientos).dropna()
    if metodo == "muestral":
        S, delta = R.cov(), 0.0
    elif metodo == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(R.to_numpy()); S = pd.DataFrame(lw.covariance_ * len(R) / (len(R) - 1), index=R.columns, columns=R.columns)
        delta = float(lw.shrinkage_)
    else:
        raise ValueError("metodo: 'muestral' o 'ledoit_wolf'")
    d = np.sqrt(np.diag(S))
    return {"covarianza": S, "medias": R.mean(), "correlaciones": S / np.outer(d, d), "contraccion": delta}


def _pesos_opt(mu, S, objetivo, sin_cortos):
    k = len(mu)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if objetivo is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ mu - objetivo})
    r = optimize.minimize(lambda w: w @ S @ w, np.full(k, 1 / k), constraints=cons,
                          bounds=[(0, 1)] * k if sin_cortos else None, method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
    return r.x


def frontera_eficiente(medias, covarianza, n_puntos: int = 30, sin_cortos: bool = True, tasa_libre: float | None = None) -> dict:
    """Frontera de Markowitz media-varianza: para cada rentabilidad objetivo, la cartera de mínima varianza (sumando 1;
    sin ventas en corto si `sin_cortos`). Devuelve la frontera (rentabilidad, volatilidad, pesos), la cartera de mínima
    varianza global y, si se da `tasa_libre`, la cartera tangente (máximo ratio de Sharpe).
    Gauss: los pesos óptimos son MUY sensibles a las medias estimadas (error de estimación): usa covarianza contraída,
    restricciones o Black-Litterman."""
    mu, S = np.asarray(medias, float), np.asarray(covarianza, float)
    nombres = list(medias.index) if isinstance(medias, pd.Series) else [f"a{i}" for i in range(len(mu))]
    w_min = _pesos_opt(mu, S, None, sin_cortos)
    r_min = float(w_min @ mu)
    objetivos = np.linspace(r_min, mu.max() if sin_cortos else mu.max() * 1.5, n_puntos)
    filas = []
    for o in objetivos:
        w = _pesos_opt(mu, S, o, sin_cortos)
        filas.append({"rentabilidad": float(w @ mu), "volatilidad": float(np.sqrt(w @ S @ w)), **dict(zip(nombres, w))})
    out = {"frontera": pd.DataFrame(filas), "minima_varianza": pd.Series(w_min, index=nombres),
           "minima_varianza_rent_vol": (r_min, float(np.sqrt(w_min @ S @ w_min)))}
    if tasa_libre is not None:
        k = len(mu)
        neg = lambda w: -(w @ mu - tasa_libre) / np.sqrt(w @ S @ w)       # noqa: E731
        r = optimize.minimize(neg, np.full(k, 1 / k), constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                              bounds=[(0, 1)] * k if sin_cortos else None, method="SLSQP")
        out["tangente"] = pd.Series(r.x, index=nombres)
        out["sharpe_tangente"] = float(-r.fun)
    return out


def beta_capm(rend_activo, rend_mercado, tasa_libre=0.0) -> dict:
    """CAPM por regresión de excesos de rentabilidad: r_i − r_f = α + β(r_m − r_f) + ε. β = riesgo sistemático;
    α ≠ 0 significativo = rentabilidad anormal. También R² (proporción de riesgo sistemático) y la rentabilidad
    exigida por la línea del mercado de valores (SML)."""
    import statsmodels.api as sm
    y = np.asarray(rend_activo, float) - np.asarray(tasa_libre, float)
    x = np.asarray(rend_mercado, float) - np.asarray(tasa_libre, float)
    ok = np.isfinite(y) & np.isfinite(x)
    r = sm.OLS(y[ok], sm.add_constant(x[ok])).fit(cov_type="HC3")
    a, b = r.params
    return {"alfa": float(a), "beta": float(b), "p_alfa": float(r.pvalues[0]), "ic_beta": tuple(map(float, r.conf_int()[1])),
            "r2": float(r.rsquared), "riesgo_especifico": float(np.std(r.resid, ddof=2)),
            "rentabilidad_exigida_sml": float(np.mean(np.asarray(tasa_libre, float)) + b * np.mean(x[ok]))}


def modelo_factores(rendimientos: pd.DataFrame, factores: pd.DataFrame) -> pd.DataFrame:
    """Modelo multifactor (APT / Fama-French): regresión de cada activo sobre los factores. Devuelve por activo α, las
    cargas (exposiciones) con sus p-valores y el R². Las cargas dicen a qué fuentes de riesgo está expuesto cada activo."""
    import statsmodels.api as sm
    R, F = pd.DataFrame(rendimientos), pd.DataFrame(factores)
    idx = R.dropna().index.intersection(F.dropna().index)
    filas = []
    for a in R.columns:
        r = sm.OLS(R.loc[idx, a], sm.add_constant(F.loc[idx])).fit(cov_type="HC3")
        fila = {"activo": a, "alfa": r.params["const"], "r2": r.rsquared}
        for f in F.columns:
            fila[f"beta_{f}"] = r.params[f]; fila[f"p_{f}"] = r.pvalues[f]
        filas.append(fila)
    return pd.DataFrame(filas).set_index("activo")


def black_litterman(covarianza, pesos_mercado, P, Q, confianza=None, aversion: float = 2.5, tau: float = 0.05) -> dict:
    """Black-Litterman: parte de las rentabilidades de EQUILIBRIO implícitas en los pesos de mercado (Π = δ·Σ·w) y las
    combina con las opiniones del gestor (P·μ = Q con incertidumbre Ω). Devuelve rentabilidades a posteriori y pesos
    óptimos (sin restricciones). Evita las carteras extremas de Markowitz con medias históricas.
    `P` (k×n): cada fila es una opinión; `Q` (k): rentabilidad de cada opinión; `confianza` (k, varianzas Ω) o, si es None,
    Ω = diag(τ·P·Σ·Pᵀ) (proporcional a la incertidumbre del equilibrio)."""
    S = np.asarray(covarianza, float); w = np.asarray(pesos_mercado, float)
    P, Q = np.atleast_2d(np.asarray(P, float)), np.asarray(Q, float)
    Pi = aversion * S @ w
    Om = np.diag(np.asarray(confianza, float)) if confianza is not None else np.diag(np.diag(tau * P @ S @ P.T))
    A = np.linalg.inv(np.linalg.inv(tau * S) + P.T @ np.linalg.inv(Om) @ P)
    mu = A @ (np.linalg.inv(tau * S) @ Pi + P.T @ np.linalg.inv(Om) @ Q)
    pesos = np.linalg.solve(aversion * S, mu)
    idx = covarianza.index if isinstance(covarianza, pd.DataFrame) else None
    return {"equilibrio": pd.Series(Pi, index=idx), "posterior": pd.Series(mu, index=idx), "pesos": pd.Series(pesos / pesos.sum(), index=idx)}


def dominancia_estocastica(a, b, puntos: int = 500, tolerancia: float = 1e-3) -> dict:
    """¿Domina A a B? Primer orden (FSD): F_A(x) ≤ F_B(x) para todo x (todo inversor que prefiere más a menos elige A).
    Segundo orden (SSD): ∫F_A ≤ ∫F_B (todo inversor averso al riesgo elige A). Comparación empírica en una rejilla;
    `tolerancia` (fracción del rango) absorbe el ruido de muestreo en la SSD cuando las medias son casi iguales."""
    x, y = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    g = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), puntos)
    Fa = np.searchsorted(x, g, side="right") / len(x); Fb = np.searchsorted(y, g, side="right") / len(y)
    Ia, Ib = np.cumsum(Fa) * (g[1] - g[0]), np.cumsum(Fb) * (g[1] - g[0])
    tol = 1e-12
    tol2 = tolerancia * (g[-1] - g[0])          # el área acumulada tiene ruido de muestreo del orden de la diferencia de medias
    return {"A_domina_FSD": bool(np.all(Fa <= Fb + tol)), "B_domina_FSD": bool(np.all(Fb <= Fa + tol)),
            "A_domina_SSD": bool(np.all(Ia <= Ib + tol2)), "B_domina_SSD": bool(np.all(Ib <= Ia + tol2)),
            "media_A": float(x.mean()), "media_B": float(y.mean())}


def equivalente_cierto(resultados, probabilidades=None, utilidad: str = "exponencial", aversion: float = 1.0) -> dict:
    """Equivalente cierto y prima de riesgo de una lotería (resultados con probabilidades) para una utilidad
    'exponencial' u(x) = −e^{−a·x} (aversión absoluta constante a), 'potencia' u = x^{1−γ}/(1−γ) (aversión relativa γ;
    resultados > 0) o 'logaritmica'. Prima de riesgo = E[X] − EC: lo que se pagaría por eliminar el riesgo (base del seguro)."""
    x = np.asarray(resultados, float)
    p = np.full(len(x), 1 / len(x)) if probabilidades is None else np.asarray(probabilidades, float)
    if abs(p.sum() - 1) > 1e-9:
        raise ValueError("Las probabilidades deben sumar 1.")
    if utilidad == "exponencial":
        Eu = np.sum(p * -np.exp(-aversion * x)); ec = -np.log(-Eu) / aversion
    elif utilidad == "potencia":
        if (x <= 0).any():
            raise ValueError("Utilidad potencia: resultados > 0.")
        g = aversion
        if abs(g - 1) < 1e-12:
            ec = np.exp(np.sum(p * np.log(x)))
        else:
            ec = (np.sum(p * x ** (1 - g))) ** (1 / (1 - g))
    elif utilidad == "logaritmica":
        ec = np.exp(np.sum(p * np.log(x)))
    else:
        raise ValueError("utilidad: 'exponencial', 'potencia' o 'logaritmica'")
    E = float(np.sum(p * x))
    return {"esperanza": E, "equivalente_cierto": float(ec), "prima_riesgo": float(E - ec)}
