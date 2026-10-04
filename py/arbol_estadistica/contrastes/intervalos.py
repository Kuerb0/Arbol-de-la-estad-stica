"""Intervalos de confianza: proporciones (Wilson, Clopper-Pearson…), bootstrap (percentil y BCa) y bondad de ajuste."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .._util import _numerico


def intervalo_proporcion(exitos: int, n: int, nivel: float = 0.95, metodo: str = "wilson") -> dict:
    """IC de una proporción. metodo: 'wilson' (recomendado), 'clopper_pearson' (exacto, conservador),
    'agresti_coull', 'jeffreys' (bayesiano con Beta(½,½)) o 'wald' (el de libro: MAL con p cerca de 0 o 1 o n pequeño).
    Equivale a PROC FREQ / BINOMIAL (WILSON CP AC JEFFREYS WALD)."""
    if not (0 <= exitos <= n and n > 0):
        raise ValueError("Necesito 0 ≤ exitos ≤ n y n > 0.")
    p, z, a = exitos / n, stats.norm.ppf(0.5 + nivel / 2), 1 - nivel
    if metodo == "wald":
        h = z * np.sqrt(p * (1 - p) / n); lo, hi = p - h, p + h
    elif metodo == "wilson":
        c = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
        h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n); lo, hi = c - h, c + h
    elif metodo == "agresti_coull":
        nt = n + z ** 2; pt = (exitos + z ** 2 / 2) / nt; h = z * np.sqrt(pt * (1 - pt) / nt); lo, hi = pt - h, pt + h
    elif metodo == "clopper_pearson":
        lo = stats.beta.ppf(a / 2, exitos, n - exitos + 1) if exitos > 0 else 0.0
        hi = stats.beta.ppf(1 - a / 2, exitos + 1, n - exitos) if exitos < n else 1.0
    elif metodo == "jeffreys":
        lo = stats.beta.ppf(a / 2, exitos + .5, n - exitos + .5) if exitos > 0 else 0.0
        hi = stats.beta.ppf(1 - a / 2, exitos + .5, n - exitos + .5) if exitos < n else 1.0
    else:
        raise ValueError("metodo: wilson, clopper_pearson, agresti_coull, jeffreys o wald")
    return {"proporcion": p, "ic_inf": float(max(lo, 0.0)), "ic_sup": float(min(hi, 1.0)), "metodo": metodo, "n": n}


def bootstrap_ic(x, estadistico=np.mean, n_boot: int = 4000, nivel: float = 0.95, metodo: str = "bca",
                 semilla: int = 42) -> dict:
    """IC bootstrap de cualquier estadístico de una muestra: 'percentil' o 'bca' (corrige sesgo y asimetría;
    recomendado). Devuelve estimación, sesgo bootstrap, error estándar e IC.

    Gauss: el bootstrap no arregla una muestra pequeña o no representativa; con n < 20 o estadísticos de
    extremos (máximo, cuantiles altos) los IC pueden quedarse cortos."""
    v = _numerico(x, "x", 5)
    rng = np.random.default_rng(semilla)
    est = float(estadistico(v))
    idx = rng.integers(0, len(v), (n_boot, len(v)))
    boot = np.array([estadistico(v[i]) for i in idx], dtype=float)
    a = 1 - nivel
    if metodo == "percentil":
        lo, hi = np.percentile(boot, [100 * a / 2, 100 * (1 - a / 2)])
    elif metodo == "bca":
        z0 = stats.norm.ppf(np.clip(np.mean(boot < est) + 0.5 * np.mean(boot == est), 1e-6, 1 - 1e-6))
        jack = np.array([estadistico(np.delete(v, i)) for i in range(len(v))]) if len(v) <= 2000 else \
            np.array([estadistico(np.delete(v, i)) for i in rng.choice(len(v), 2000, replace=False)])
        d = jack.mean() - jack
        acel = np.sum(d ** 3) / (6 * np.sum(d ** 2) ** 1.5) if np.sum(d ** 2) > 0 else 0.0
        zs = stats.norm.ppf([a / 2, 1 - a / 2])
        q = stats.norm.cdf(z0 + (z0 + zs) / (1 - acel * (z0 + zs)))
        lo, hi = np.percentile(boot, 100 * q)
    else:
        raise ValueError("metodo: 'percentil' o 'bca'")
    return {"estimacion": est, "sesgo": float(boot.mean() - est), "error_estandar": float(boot.std(ddof=1)),
            "ic_inf": float(lo), "ic_sup": float(hi), "metodo": metodo, "n_boot": n_boot, "replicas": boot}


def bondad_ajuste_multinomial(observados, probabilidades=None, nombres=None, parametros_estimados: int = 0) -> dict:
    """χ² de bondad de ajuste: ¿las frecuencias observadas siguen unas probabilidades dadas? (equiprobables si no se dan).
    `parametros_estimados` resta grados de libertad si las probabilidades salen de los mismos datos.
    Devuelve χ², gl, p y una tabla con observados, esperados y residuos de Pearson. Equivale a PROC FREQ / TESTP=."""
    o = np.asarray(observados, float)
    if (o < 0).any() or o.sum() == 0:
        raise ValueError("Frecuencias no negativas y no todas cero.")
    p = np.full(len(o), 1 / len(o)) if probabilidades is None else np.asarray(probabilidades, float)
    if len(p) != len(o) or abs(p.sum() - 1) > 1e-6:
        raise ValueError("probabilidades: misma longitud que observados y suma 1.")
    e = o.sum() * p
    chi2 = float(np.sum((o - e) ** 2 / e)); gl = len(o) - 1 - parametros_estimados
    t = pd.DataFrame({"observado": o, "esperado": e, "residuo_pearson": (o - e) / np.sqrt(e)}, index=nombres)
    return {"chi2": chi2, "gl": gl, "p_valor": float(stats.chi2.sf(chi2, gl)), "tabla": t,
            "aviso": "esperados < 5: agrupa categorías" if (e < 5).any() else ""}
