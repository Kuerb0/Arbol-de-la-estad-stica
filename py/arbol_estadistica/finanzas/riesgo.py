"""Medidas de riesgo: VaR y TVaR (histórico, normal, Cornish-Fisher), teoría de valores extremos (picos sobre umbral
con la Pareto generalizada, Hill, exceso medio), cópulas y prueba de estrés de una cartera."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._util import _numerico


def var_tvar(perdidas, niveles=(0.95, 0.99, 0.995), metodo: str = "historico") -> pd.DataFrame:
    """VaR (cuantil de la pérdida) y TVaR/ES (media de las pérdidas que superan el VaR; medida COHERENTE) por nivel.
    metodo: 'historico' (empírico), 'normal' (paramétrico: subestima colas pesadas) o 'cornish_fisher' (corrige el
    cuantil normal con asimetría y curtosis; con curtosis muy alta, p. ej. t con 3-4 gl, la expansión se dispara y no
    es fiable). Signo: valores POSITIVOS = pérdidas. Gauss: para niveles ≥ 99 % con colas pesadas usa ajustar_gpd (EVT)."""
    x = _numerico(perdidas, "perdidas", 20)
    m, s = x.mean(), x.std(ddof=1)
    sk, ku = stats.skew(x), stats.kurtosis(x)
    filas = []
    for a in niveles:
        if metodo == "historico":
            v = np.quantile(x, a); es = x[x >= v].mean()
        elif metodo == "normal":
            z = stats.norm.ppf(a); v = m + s * z; es = m + s * stats.norm.pdf(z) / (1 - a)
        elif metodo == "cornish_fisher":
            z = stats.norm.ppf(a)
            zc = z + (z ** 2 - 1) * sk / 6 + (z ** 3 - 3 * z) * ku / 24 - (2 * z ** 3 - 5 * z) * sk ** 2 / 36
            v = m + s * zc; es = x[x >= v].mean() if (x >= v).any() else v
        else:
            raise ValueError("metodo: 'historico', 'normal' o 'cornish_fisher'")
        filas.append({"nivel": a, "var": float(v), "tvar": float(es)})
    return pd.DataFrame(filas).set_index("nivel")


def ajustar_gpd(datos, umbral: float | None = None, cuantil_umbral: float = 0.9, niveles=(0.99, 0.995, 0.999)) -> dict:
    """Picos sobre umbral (POT): los excesos sobre un umbral alto siguen una Pareto generalizada (ξ forma, β escala).
    ξ > 0 = cola pesada (Pareto), ξ = 0 exponencial, ξ < 0 acotada. Devuelve parámetros (MV), VaR y TVaR extremos
    por la fórmula de McNeil y el nº de excesos. Equivale a evd / POT en R.
    Gauss: elige el umbral mirando la estabilidad de ξ y el gráfico de exceso medio; con < 50 excesos, mucha incertidumbre."""
    x = _numerico(datos, "datos", 50)
    u = float(np.quantile(x, cuantil_umbral)) if umbral is None else float(umbral)
    exc = x[x > u] - u
    if len(exc) < 10:
        raise ValueError(f"Solo {len(exc)} excesos sobre el umbral: bájalo.")
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    n, nu = len(x), len(exc)
    filas = []
    for a in niveles:
        v = u + beta / xi * (((n / nu) * (1 - a)) ** (-xi) - 1) if abs(xi) > 1e-9 else u - beta * np.log(n / nu * (1 - a))
        es = (v + beta - xi * u) / (1 - xi) if xi < 1 else np.inf
        filas.append({"nivel": a, "var": float(v), "tvar": float(es)})
    return {"xi": float(xi), "beta": float(beta), "umbral": u, "n_excesos": nu, "tabla": pd.DataFrame(filas).set_index("nivel"),
            "media_infinita": bool(xi >= 1)}


def estimador_hill(datos, k_max: int | None = None) -> pd.DataFrame:
    """Índice de cola de Hill ξ̂(k) = (1/k)·Σ log(X_(i)/X_(k+1)) para k = 5..k_max mayores observaciones (solo cola
    pesada, ξ > 0). Busca una zona estable en k ('Hill plot')."""
    x = np.sort(_numerico(datos, "datos", 20))[::-1]
    x = x[x > 0]
    k_max = k_max or min(len(x) - 1, 500)
    lx = np.log(x)
    ks = np.arange(5, k_max)
    xi = np.array([lx[:k].mean() - lx[k] for k in ks])
    se = xi / np.sqrt(ks)
    return pd.DataFrame({"k": ks, "xi": xi, "ic_inf": xi - 1.96 * se, "ic_sup": xi + 1.96 * se, "alfa": 1 / xi}).set_index("k")


def funcion_exceso_medio(datos, puntos: int = 50) -> pd.DataFrame:
    """e(u) = E[X − u | X > u] en una rejilla de umbrales: creciente y lineal = cola de Pareto (ξ > 0), plana =
    exponencial, decreciente = cola ligera. Herramienta para elegir el umbral del POT."""
    x = _numerico(datos, "datos", 20)
    us = np.quantile(x, np.linspace(0.5, 0.98, puntos))
    return pd.DataFrame({"umbral": us, "exceso_medio": [float((x[x > u] - u).mean()) for u in us], "n": [(x > u).sum() for u in us]})


def simular_copula(tipo: str = "gaussiana", parametro: float = 0.5, n: int = 10000, gl: int = 4, semilla: int = 42,
                   marginales=None) -> pd.DataFrame:
    """Simula dos variables con dependencia dada por una cópula: 'gaussiana' (ρ), 't' (ρ, gl: dependencia en colas
    simétrica), 'clayton' (θ > 0: dependencia en la cola INFERIOR) o 'gumbel' (θ ≥ 1: cola SUPERIOR, siniestros
    grandes juntos). Devuelve las uniformes (u, v) y, si se dan `marginales` (dos distribuciones scipy), las variables.
    También la τ de Kendall teórica en attrs."""
    rng = np.random.default_rng(semilla)
    if tipo in ("gaussiana", "t"):
        L = np.linalg.cholesky([[1, parametro], [parametro, 1]])
        z = rng.standard_normal((n, 2)) @ L.T
        if tipo == "gaussiana":
            uv = stats.norm.cdf(z)
        else:
            w = np.sqrt(gl / rng.chisquare(gl, n))[:, None]; uv = stats.t.cdf(z * w, gl)
        tau = 2 / np.pi * np.arcsin(parametro)
    elif tipo == "clayton":
        if parametro <= 0:
            raise ValueError("Clayton: θ > 0")
        v = rng.gamma(1 / parametro, 1, n)[:, None]; e = rng.exponential(size=(n, 2))
        uv = (1 + e / v) ** (-1 / parametro); tau = parametro / (parametro + 2)
    elif tipo == "gumbel":
        if parametro < 1:
            raise ValueError("Gumbel: θ ≥ 1")
        a = 1 / parametro
        # variable estable positiva (Chambers-Mallows-Stuck) para el método de Marshall-Olkin
        U = rng.uniform(0, np.pi, n); W = rng.exponential(size=n)
        S = (np.sin(a * U) / np.sin(U) ** (1 / a)) * (np.sin((1 - a) * U) / W) ** ((1 - a) / a) if a < 1 else np.ones(n)
        e = rng.exponential(size=(n, 2))
        uv = np.exp(-(e / S[:, None]) ** a); tau = 1 - 1 / parametro
    else:
        raise ValueError("tipo: 'gaussiana', 't', 'clayton' o 'gumbel'")
    out = pd.DataFrame(uv, columns=["u", "v"])
    if marginales is not None:
        out["x"] = marginales[0].ppf(out.u); out["y"] = marginales[1].ppf(out.v)
    out.attrs["tau_kendall"] = float(tau)
    return out


def prueba_estres(exposiciones: dict, escenarios: dict) -> pd.DataFrame:
    """Pérdida de una cartera en escenarios de estrés: `exposiciones` = {factor: importe expuesto}, `escenarios` =
    {nombre: {factor: choque relativo}} (p. ej. {'renta variable': -0.39}). Devuelve la pérdida por escenario y factor
    y el peor escenario primero. Es la lógica de los choques estándar de Solvencia II o de un test de la EBA."""
    filas = []
    for nombre, choques in escenarios.items():
        fila = {"escenario": nombre}
        for f, e in exposiciones.items():
            fila[f] = -e * choques.get(f, 0.0)
        fila["perdida_total"] = sum(fila[f] for f in exposiciones)
        filas.append(fila)
    return pd.DataFrame(filas).set_index("escenario").sort_values("perdida_total", ascending=False)
