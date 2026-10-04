"""Fundamentos de inferencia por simulación y cálculo: método de los momentos, información de Fisher y cota de
Cramér-Rao, comparación de estimadores (sesgo, varianza, ECM, eficiencia), método delta, distribuciones muestrales
(t, χ², F, T² de Hotelling y el porqué de n−1), regresión a la media y coste de dicotomizar una variable continua."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._util import _numerico

_MM = {
    # nombre: (función momentos -> parámetros, constructor scipy, nombres)
    "normal": (lambda m, v, x: (m, np.sqrt(v)), lambda a, b: stats.norm(a, b), ("media", "desviacion")),
    "exponencial": (lambda m, v, x: (1 / m,), lambda l: stats.expon(scale=1 / l), ("tasa",)),
    "gamma": (lambda m, v, x: (m ** 2 / v, m / v), lambda a, r: stats.gamma(a, scale=1 / r), ("forma", "tasa")),
    "lognormal": (lambda m, v, x: (np.log(m ** 2 / np.sqrt(v + m ** 2)), np.sqrt(np.log(1 + v / m ** 2))),
                  lambda mu, s: stats.lognorm(s, scale=np.exp(mu)), ("mu_log", "sigma_log")),
    "beta": (lambda m, v, x: (m * (m * (1 - m) / v - 1), (1 - m) * (m * (1 - m) / v - 1)), lambda a, b: stats.beta(a, b), ("a", "b")),
    "uniforme": (lambda m, v, x: (m - np.sqrt(3 * v), m + np.sqrt(3 * v)), lambda a, b: stats.uniform(a, b - a), ("min", "max")),
    "poisson": (lambda m, v, x: (m,), lambda l: stats.poisson(l), ("lambda",)),
    "binomial_negativa": (lambda m, v, x: (m ** 2 / (v - m), m / v), lambda r, p: stats.nbinom(r, p), ("r", "p")),
    "pareto": (lambda m, v, x: ((1 + np.sqrt(1 + m ** 2 / v)), None), None, ("alfa", "escala")),
}


def metodo_momentos(datos, distribucion: str, comparar_mv: bool = True) -> pd.DataFrame:
    """Estimadores por el método de los momentos (igualar media y varianza muestrales a las teóricas) para normal,
    exponencial, gamma, lognormal, beta, uniforme, Poisson, binomial negativa y Pareto, y (opcional) los de máxima
    verosimilitud para comparar. Gauss: los de momentos son consistentes y sencillos (buenos valores iniciales), pero
    suelen ser menos eficientes que los MV, sobre todo con colas pesadas; en la uniforme incluso pueden dejar
    observaciones fuera del soporte."""
    x = _numerico(datos, "datos", 3)
    if distribucion not in _MM:
        raise ValueError(f"distribucion: una de {sorted(_MM)}")
    m, v = x.mean(), x.var(ddof=0)
    f, cons, nombres = _MM[distribucion]
    if distribucion == "binomial_negativa" and v <= m:
        raise ValueError("Varianza ≤ media: no hay sobredispersión, la binomial negativa por momentos no existe (usa Poisson).")
    if distribucion == "pareto":
        a = 1 + np.sqrt(1 + m ** 2 / v); est = (a, m * (a - 1) / a)
    else:
        est = f(m, v, x)
    t = pd.DataFrame({"momentos": est}, index=list(nombres))
    if comparar_mv:
        mv = {"normal": lambda: (x.mean(), x.std(ddof=0)), "exponencial": lambda: (1 / x.mean(),),
              "poisson": lambda: (x.mean(),),
              "gamma": lambda: (lambda a, l, s: (a, 1 / s))(*stats.gamma.fit(x, floc=0)),
              "lognormal": lambda: (np.log(x).mean(), np.log(x).std(ddof=0)),
              "beta": lambda: stats.beta.fit(x, floc=0, fscale=1)[:2],
              "uniforme": lambda: (x.min(), x.max()),
              "pareto": lambda: (len(x) / np.sum(np.log(x / x.min())), x.min()),
              "binomial_negativa": lambda: _nb_mv(x)}[distribucion]
        t["max_verosimilitud"] = list(mv())
    return t


def _nb_mv(x):
    def nll(th):
        r, p = np.exp(th[0]), 1 / (1 + np.exp(-th[1]))
        return -stats.nbinom.logpmf(x, r, p).sum()
    m, v = x.mean(), x.var()
    r0 = max(m ** 2 / max(v - m, 1e-3), 0.1)
    s = optimize.minimize(nll, [np.log(r0), np.log(m / v / (1 - m / v + 1e-9) + 1e-9)], method="Nelder-Mead")
    return np.exp(s.x[0]), 1 / (1 + np.exp(-s.x[1]))


def informacion_fisher(log_densidad: Callable, theta, datos=None, distribucion=None, n: int = 1, n_mc: int = 200_000,
                       semilla: int = 42) -> dict:
    """Información de Fisher I(θ) y cota de Cramér-Rao (varianza mínima de un estimador insesgado: I(θ)⁻¹/n).
    `log_densidad(x, theta)` vectorizada en x. Con `datos` calcula la información OBSERVADA (−hessiano de la
    log-verosimilitud, por diferencias finitas); con `distribucion` (scipy, la verdadera) la ESPERADA por Monte Carlo
    E[score·scoreᵀ]. Gauss: la cota exige regularidad (el soporte no depende de θ; falla en la uniforme(0, θ))."""
    th = np.atleast_1d(np.asarray(theta, float)); p = len(th)
    h = 1e-4 * np.maximum(1, np.abs(th))

    def ll(t, x):
        return np.asarray(log_densidad(x, t if p > 1 else t[0]), float)
    if datos is not None:
        x = np.asarray(datos, float)
        H = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                ei, ej = np.eye(p)[i] * h[i], np.eye(p)[j] * h[j]
                H[i, j] = (ll(th + ei + ej, x).sum() - ll(th + ei - ej, x).sum() - ll(th - ei + ej, x).sum() + ll(th - ei - ej, x).sum()) / (4 * h[i] * h[j])
        I = -H
        tipo = "observada (total de la muestra)"
        cota = np.linalg.inv(I)
    elif distribucion is not None:
        x = distribucion.rvs(size=n_mc, random_state=np.random.default_rng(semilla))
        S = np.column_stack([(ll(th + np.eye(p)[i] * h[i], x) - ll(th - np.eye(p)[i] * h[i], x)) / (2 * h[i]) for i in range(p)])
        I = S.T @ S / n_mc
        tipo = "esperada por observación (Monte Carlo)"
        cota = np.linalg.inv(I) / n
    else:
        raise ValueError("Pasa `datos` (información observada) o `distribucion` (esperada).")
    return {"informacion": I if p > 1 else float(I[0, 0]), "tipo": tipo,
            "cota_cramer_rao": cota if p > 1 else float(cota[0, 0]),
            "error_estandar_minimo": np.sqrt(np.diag(cota)) if p > 1 else float(np.sqrt(cota[0, 0]))}


def comparar_estimadores(estimadores: dict, generar: Callable, verdadero: float, n: int = 30, n_sim: int = 5000,
                         cota_cr: float | None = None, semilla: int = 42) -> pd.DataFrame:
    """Compara estimadores por Monte Carlo: `generar(n, rng)` simula una muestra, cada estimador es una función de la
    muestra. Devuelve media, sesgo, varianza, ECM = sesgo² + varianza, eficiencia relativa al de menor ECM y, si das la
    cota de Cramér-Rao, la eficiencia frente a ella (1 = eficiente). Sirve para ver suficiencia/Rao-Blackwell (el
    estimador que usa el estadístico suficiente gana) o insesgadez vs ECM (el sesgado puede ganar)."""
    rng = np.random.default_rng(semilla)
    res = {k: np.empty(n_sim) for k in estimadores}
    for s in range(n_sim):
        x = generar(n, rng)
        for k, f in estimadores.items():
            res[k][s] = f(x)
    filas = []
    for k, r in res.items():
        filas.append({"estimador": k, "media": r.mean(), "sesgo": r.mean() - verdadero, "varianza": r.var(ddof=1),
                      "ecm": np.mean((r - verdadero) ** 2)})
    t = pd.DataFrame(filas).set_index("estimador")
    t["eficiencia_relativa"] = t.ecm.min() / t.ecm
    if cota_cr is not None:
        t["eficiencia_cramer_rao"] = cota_cr / t.varianza
    return t


def metodo_delta(funcion: Callable, estimacion, covarianza, nivel: float = 0.95, nombres=None) -> dict:
    """Método delta: error estándar de g(θ̂) ≈ √(∇gᵀ Σ ∇g) con gradiente numérico. Para ratios, elasticidades,
    odds ratio a partir de coeficientes, diferencias de probabilidades… Devuelve valor, EE, IC normal y el gradiente.
    Gauss: aproximación de primer orden; con g muy no lineal o EE grandes compara con bootstrap (`bootstrap_ic`)."""
    th = np.atleast_1d(np.asarray(estimacion, float)); S = np.atleast_2d(np.asarray(covarianza, float))
    if S.shape != (len(th), len(th)):
        raise ValueError("covarianza debe ser p×p con p = len(estimacion).")
    h = 1e-6 * np.maximum(1, np.abs(th))
    def g(t):
        return float(funcion(t if len(th) > 1 else t[0]))
    E = np.eye(len(th))
    grad = np.array([(g(th + E[i] * h[i]) - g(th - E[i] * h[i])) / (2 * h[i]) for i in range(len(th))])
    g0 = g(th)
    se = float(np.sqrt(grad @ S @ grad))
    z = stats.norm.ppf(0.5 + nivel / 2)
    return {"valor": g0, "error_estandar": se, "ic": (g0 - z * se, g0 + z * se),
            "gradiente": pd.Series(grad, index=nombres) if nombres is not None else grad}


def distribucion_muestral_simulada(tipo: str, gl=10, n_sim: int = 20_000, semilla: int = 42) -> dict:
    """Construye por simulación desde normales las distribuciones muestrales clásicas y las compara con la teórica (KS):
    - ``'chi2'``: (n−1)S²/σ² con n = gl+1 → χ²(gl);  ``'t'``: (X̄−μ)/(S/√n) → t(gl);
    - ``'f'``: (S₁²/σ₁²)/(S₂²/σ₂²) con gl = (gl1, gl2) → F; ``'hotelling'``: T² de una normal p-variante con gl = (p, n)
      → (n−p)/(p(n−1))·T² ~ F(p, n−p) (y la Wishart detrás de S);
    - ``'varianza'``: sesgo de dividir entre n frente a n−1 (los grados de libertad que «consume» estimar la media)."""
    rng = np.random.default_rng(semilla)
    if tipo == "chi2":
        n = int(gl) + 1; x = rng.normal(size=(n_sim, n)); s = (n - 1) * x.var(axis=1, ddof=1); teo = stats.chi2(gl)
    elif tipo == "t":
        n = int(gl) + 1; x = rng.normal(size=(n_sim, n)); s = x.mean(axis=1) / (x.std(axis=1, ddof=1) / np.sqrt(n)); teo = stats.t(gl)
    elif tipo == "f":
        a, b = gl
        s = rng.normal(size=(n_sim, a + 1)).var(axis=1, ddof=1) / rng.normal(size=(n_sim, b + 1)).var(axis=1, ddof=1); teo = stats.f(a, b)
    elif tipo == "hotelling":
        p, n = gl
        if n <= p:
            raise ValueError("hotelling: n > p.")
        x = rng.normal(size=(n_sim, n, p)); m = x.mean(axis=1); xc = x - m[:, None, :]
        S = np.einsum("kni,knj->kij", xc, xc) / (n - 1)
        T2 = n * np.einsum("ki,kij,kj->k", m, np.linalg.inv(S), m)
        s = (n - p) / (p * (n - 1)) * T2; teo = stats.f(p, n - p)
    elif tipo == "varianza":
        n = int(gl); x = rng.normal(0, 1, size=(n_sim, n))
        return {"tabla": pd.DataFrame({"media_estimador": [x.var(axis=1, ddof=0).mean(), x.var(axis=1, ddof=1).mean()],
                                       "esperanza_teorica": [(n - 1) / n, 1.0]}, index=["divide entre n", "divide entre n−1"]),
                "verdadera": 1.0}
    else:
        raise ValueError("tipo: 'chi2', 't', 'f', 'hotelling' o 'varianza'")
    ks = stats.kstest(s, teo.cdf)
    qs = [0.05, 0.5, 0.95, 0.99]
    return {"muestra": s, "teorica": teo, "ks": float(ks.statistic), "p_valor_ks": float(ks.pvalue),
            "cuantiles": pd.DataFrame({"simulado": np.quantile(s, qs), "teorico": teo.ppf(qs)}, index=qs)}


def simular_regresion_a_la_media(correlacion: float = 0.5, n: int = 10_000, cuantil_seleccion: float = 0.9, media: float = 100,
                                 desviacion: float = 15, semilla: int = 42) -> dict:
    """Regresión a la media: dos mediciones con correlación ρ (test-retest, siniestralidad de un año y del siguiente).
    Si seleccionas a los extremos en la 1ª medición, en la 2ª se acercan a la media SIN ninguna intervención:
    E[X₂ | X₁] = μ + ρ(X₁ − μ). Devuelve la media de los seleccionados en ambas mediciones y la predicción teórica.
    Gauss: por eso un «antes/después» sobre los peores casos sin grupo de control exagera cualquier efecto."""
    if not -1 < correlacion < 1:
        raise ValueError("correlacion en (−1, 1).")
    S = desviacion ** 2 * np.array([[1, correlacion], [correlacion, 1]])
    x = np.random.default_rng(semilla).multivariate_normal([media, media], S, n)
    sel = x[:, 0] >= np.quantile(x[:, 0], cuantil_seleccion)
    m1, m2 = x[sel, 0].mean(), x[sel, 1].mean()
    return {"media_1a_seleccionados": float(m1), "media_2a_seleccionados": float(m2), "prediccion_teorica_2a": float(media + correlacion * (m1 - media)),
            "cambio_aparente": float(m2 - m1), "fraccion_que_regresa": float(1 - correlacion)}


def coste_de_dicotomizar(correlacion: float = 0.5, punto_corte: float = 0.5, n: int = 200, n_sim: int = 2000,
                         alfa: float = 0.05, semilla: int = 42) -> dict:
    """Cuánto se pierde al partir una variable continua en dos (alto/bajo) — también aplica a discretizar la respuesta.
    Teoría (normal): la correlación se atenúa por φ(z_c)/√(p(1−p)) (0.798 en la mediana), lo que equivale a tirar
    ≈ 1 − 0.798² = 36 % de la muestra. Simula la potencia del contraste de correlación con la X continua y dicotomizada."""
    p = punto_corte
    z = stats.norm.ppf(1 - p)
    aten = stats.norm.pdf(z) / np.sqrt(p * (1 - p))
    rng = np.random.default_rng(semilla)
    rc = rd = 0
    S = np.array([[1, correlacion], [correlacion, 1]])
    for _ in range(n_sim):
        xy = rng.multivariate_normal([0, 0], S, n)
        rc += stats.pearsonr(xy[:, 0], xy[:, 1])[1] < alfa
        xd = (xy[:, 0] > z).astype(float)
        if 0 < xd.sum() < n:
            rd += stats.pearsonr(xd, xy[:, 1])[1] < alfa
    return {"factor_atenuacion": float(aten), "correlacion_dicotomizada_teorica": float(correlacion * aten),
            "eficiencia_relativa": float(aten ** 2), "muestra_equivalente_perdida": float(1 - aten ** 2),
            "potencia_continua": rc / n_sim, "potencia_dicotomizada": rd / n_sim}
