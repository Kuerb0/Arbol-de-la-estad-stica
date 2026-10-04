"""Ajuste de distribuciones a datos, densidad, función de distribución empírica y dispersión de conteos."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._util import _numerico

CANDIDATAS = ("norm", "lognorm", "gamma", "weibull_min", "expon", "invgauss", "pareto", "loglogistic")
_NOMBRES = {"norm": "normal", "lognorm": "lognormal", "gamma": "gamma", "weibull_min": "Weibull", "expon": "exponencial",
            "invgauss": "inversa gaussiana", "pareto": "Pareto", "loglogistic": "log-logística (fisk)", "t": "t de Student"}


def ajustar_distribuciones(x, candidatas=CANDIDATAS, fijar_origen: bool = True) -> pd.DataFrame:
    """Ajusta por máxima verosimilitud varias distribuciones continuas y las ordena por AIC.

    `fijar_origen=True` fija loc = 0 en las distribuciones positivas (lo habitual con importes y tiempos).
    Columnas: distribucion, parametros, loglik, k, AIC, BIC, KS (estadístico) y p_KS.
    Equivale a PROC UNIVARIATE HISTOGRAM / (LOGNORMAL GAMMA WEIBULL …) o a fitdistrplus en R.
    OJO: el p-valor de KS con parámetros estimados de los mismos datos es OPTIMISTA (no rechaza casi nunca);
    usa el AIC para comparar y el Q-Q / P-P para validar.
    """
    v = _numerico(x, "x", 10)
    filas = []
    for nombre in candidatas:
        dist = getattr(stats, "fisk" if nombre == "loglogistic" else nombre)
        positiva = nombre not in ("norm", "t")
        if positiva and (v <= 0).any():
            continue
        try:
            params = dist.fit(v, floc=0) if (positiva and fijar_origen) else dist.fit(v)
        except Exception:
            continue
        ll = float(np.sum(dist.logpdf(v, *params)))
        if not np.isfinite(ll):
            continue
        k = len(params) - (1 if positiva and fijar_origen else 0)
        ks = stats.kstest(v, dist.cdf, args=params)
        filas.append({"distribucion": _NOMBRES.get(nombre, nombre), "scipy": nombre, "parametros": tuple(round(float(p), 6) for p in params),
                      "loglik": ll, "k": k, "AIC": 2 * k - 2 * ll, "BIC": k * np.log(len(v)) - 2 * ll,
                      "KS": float(ks.statistic), "p_KS": float(ks.pvalue)})
    if not filas:
        raise ValueError("Ninguna candidata se pudo ajustar (¿valores ≤ 0 con distribuciones positivas?).")
    t = pd.DataFrame(filas).sort_values("AIC").reset_index(drop=True)
    t["delta_AIC"] = t["AIC"] - t["AIC"].min()
    return t


def ajustar_distribucion_discreta(x) -> pd.DataFrame:
    """Ajusta Poisson y binomial negativa (y geométrica) a un conteo y las compara por AIC.

    Binomial negativa en la parametrización de seguros: media μ, varianza μ + μ²/r (r = 'tamaño').
    Incluye el test de razón de verosimilitudes Poisson vs BN (en la frontera: p-valor/2, Self-Liang).
    Útil para el número de siniestros: si la BN gana claramente, hay sobredispersión.
    """
    v = _numerico(x, "x", 5)
    if (v < 0).any() or not np.allclose(v, np.round(v)):
        raise ValueError("x debe ser un conteo (enteros ≥ 0).")
    v = v.astype(int)
    mu, var = v.mean(), v.var(ddof=1)
    ll_p = float(stats.poisson.logpmf(v, mu).sum())

    def nll(log_r):
        r = np.exp(log_r)
        return -stats.nbinom.logpmf(v, r, r / (r + mu)).sum()
    r0 = mu ** 2 / (var - mu) if var > mu else 1e3
    res = optimize.minimize_scalar(nll, bounds=(np.log(1e-4), np.log(1e7)), method="bounded")
    r = float(np.exp(res.x)); ll_nb = float(-res.fun)
    ll_g = float(stats.geom.logpmf(v + 1, 1 / (1 + mu)).sum())
    lr = max(2 * (ll_nb - ll_p), 0.0)
    filas = [{"distribucion": "Poisson", "parametros": {"media": mu}, "loglik": ll_p, "k": 1},
             {"distribucion": "binomial negativa", "parametros": {"media": mu, "r": r, "var": mu + mu ** 2 / r}, "loglik": ll_nb, "k": 2},
             {"distribucion": "geométrica", "parametros": {"media": mu}, "loglik": ll_g, "k": 1}]
    t = pd.DataFrame(filas)
    t["AIC"] = 2 * t["k"] - 2 * t["loglik"]
    t = t.sort_values("AIC").reset_index(drop=True)
    t.attrs.update({"media": float(mu), "varianza": float(var), "lr_poisson_vs_bn": lr,
                    "p_valor_lr": float(0.5 * stats.chi2.sf(lr, 1)), "r_momentos": float(r0)})
    return t


def indice_dispersion(x) -> dict:
    """Índice de dispersión D = (n−1)·s²/x̄ ~ χ²(n−1) bajo Poisson. D/(n−1) ≫ 1 = sobredispersión."""
    v = _numerico(x, "x", 5)
    n, m = len(v), v.mean()
    if m <= 0:
        raise ValueError("La media debe ser positiva.")
    d = (n - 1) * v.var(ddof=1) / m
    p_sobre = float(stats.chi2.sf(d, n - 1))
    return {"indice": float(v.var(ddof=1) / m), "estadistico": float(d), "gl": n - 1, "p_sobredispersion": p_sobre,
            "p_infradispersion": float(stats.chi2.cdf(d, n - 1)),
            "interpretacion": ("sobredispersión (varianza > media): Poisson se queda corta, usa binomial negativa"
                               if p_sobre < 0.05 else "compatible con Poisson (varianza ≈ media)")}


def funcion_distribucion_empirica(x, nivel: float = 0.95) -> pd.DataFrame:
    """F_n(t) en cada valor distinto con banda de confianza SIMULTÁNEA de Dvoretzky-Kiefer-Wolfowitz:
    ε = √(ln(2/α)/(2n)). Equivale a PROC UNIVARIATE CDFPLOT."""
    v = np.sort(_numerico(x, "x", 2))
    n = len(v)
    valores, cuentas = np.unique(v, return_counts=True)
    F = np.cumsum(cuentas) / n
    eps = np.sqrt(np.log(2 / (1 - nivel)) / (2 * n))
    return pd.DataFrame({"x": valores, "F": F, "banda_inf": np.clip(F - eps, 0, 1), "banda_sup": np.clip(F + eps, 0, 1)})


def estimar_densidad(x, ancho="scott", puntos: int = 256, limites=None) -> pd.DataFrame:
    """Densidad por núcleo gaussiano (KDE). `ancho`: 'scott', 'silverman' o un número (factor sobre la desviación).
    Devuelve columnas x, densidad. Gauss: el ancho manda más que el núcleo; con colas pesadas o soporte acotado
    (importes ≥ 0) la KDE pone masa donde no debe: considera transformar (log) antes."""
    v = _numerico(x, "x", 3)
    kde = stats.gaussian_kde(v, bw_method=ancho)
    lo, hi = limites if limites else (v.min() - 3 * v.std() * kde.factor, v.max() + 3 * v.std() * kde.factor)
    xs = np.linspace(lo, hi, puntos)
    out = pd.DataFrame({"x": xs, "densidad": kde(xs)})
    out.attrs["ancho_banda"] = float(kde.factor * v.std(ddof=1))
    return out
