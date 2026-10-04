"""Estadística bayesiana: familias conjugadas, Bayes empírico (contracción hacia la media), Metropolis con diagnósticos
(R-hat, tamaño efectivo), chequeos predictivos, decisión con pérdida (acción de Bayes), test A/B bayesiano y bandidos
multibrazo (Thompson, UCB, ε-greedy)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .._util import _numerico


# ------------------------------------------------------------------ conjugadas
def posterior_conjugado(modelo: str, datos, previa: tuple, sigma: float | None = None, exposicion=None,
                        nivel: float = 0.95) -> dict:
    """Actualización conjugada previa → posterior, con media, IC creíble de colas iguales, predictiva y el peso de los
    datos (factor de credibilidad Z: media posterior = Z·media muestral + (1−Z)·media previa).

    - ``'beta_binomial'``: datos = (éxitos, ensayos) o vector 0/1; previa = (a, b).
    - ``'gamma_poisson'``: datos = conteos; previa = (a, b) con b la TASA (media previa a/b); `exposicion` opcional.
    - ``'normal_normal'``: datos = observaciones con σ conocida (`sigma`); previa = (μ0, τ0) (media y desviación previas).

    Equivale a: PROC MCMC/GENMOD BAYES con previas conjugadas (aquí sin simulación: es exacto).
    Gauss: la previa pesa como (a+b) ensayos en la beta o b unidades de exposición en la gamma; con previa
    «plana» (1, 1) el IC creíble se parece al de Wilson, pero su interpretación es probabilística sobre el parámetro."""
    a_ = (1 - nivel) / 2
    if modelo == "beta_binomial":
        if isinstance(datos, tuple) and len(datos) == 2:
            x, n = map(float, datos)
        else:
            v = np.asarray(datos, float)
            if not set(np.unique(v)) <= {0.0, 1.0}:
                raise ValueError("beta_binomial: pasa (éxitos, ensayos) como tupla o un vector de 0/1.")
            x, n = v.sum(), len(v)
        if not 0 <= x <= n:
            raise ValueError("éxitos debe estar entre 0 y ensayos.")
        a, b = previa
        A, B = a + x, b + n - x
        post = stats.beta(A, B)
        pred = stats.betabinom(1, A, B)
        Z = n / (n + a + b)
        return {"modelo": modelo, "posterior": {"a": A, "b": B}, "media": post.mean(), "mediana": post.median(),
                "moda": (A - 1) / (A + B - 2) if A > 1 and B > 1 else np.nan, "ic": tuple(post.ppf([a_, 1 - a_])),
                "media_previa": a / (a + b), "media_muestral": x / n if n else np.nan, "credibilidad_Z": Z,
                "prob_predictiva_exito": float(pred.pmf(1)), "distribucion": post}
    if modelo == "gamma_poisson":
        y = _numerico(datos, "datos", 1)
        e = np.ones_like(y) if exposicion is None else _numerico(exposicion, "exposicion", 1)
        if len(e) != len(y):
            raise ValueError("exposicion debe tener la misma longitud que datos.")
        a, b = previa
        A, B = a + y.sum(), b + e.sum()
        post = stats.gamma(A, scale=1 / B)
        Z = e.sum() / (e.sum() + b)
        return {"modelo": modelo, "posterior": {"a": A, "b": B}, "media": post.mean(), "mediana": post.median(),
                "ic": tuple(post.ppf([a_, 1 - a_])), "media_previa": a / b, "media_muestral": y.sum() / e.sum(),
                "credibilidad_Z": Z, "predictiva": stats.nbinom(A, B / (B + 1)), "distribucion": post}
    if modelo == "normal_normal":
        if sigma is None or sigma <= 0:
            raise ValueError("normal_normal necesita sigma (desviación conocida de los datos) > 0.")
        y = _numerico(datos, "datos", 1)
        mu0, tau0 = previa
        prec = 1 / tau0 ** 2 + len(y) / sigma ** 2
        m = (mu0 / tau0 ** 2 + y.sum() / sigma ** 2) / prec
        post = stats.norm(m, np.sqrt(1 / prec))
        Z = (len(y) / sigma ** 2) / prec
        return {"modelo": modelo, "posterior": {"media": m, "desviacion": float(np.sqrt(1 / prec))}, "media": m, "mediana": m,
                "ic": tuple(post.ppf([a_, 1 - a_])), "media_previa": mu0, "media_muestral": float(y.mean()), "credibilidad_Z": Z,
                "predictiva": stats.norm(m, np.sqrt(1 / prec + sigma ** 2)), "distribucion": post}
    raise ValueError("modelo: 'beta_binomial', 'gamma_poisson' o 'normal_normal'")


def bayes_empirico_beta(exitos, ensayos, nombres=None) -> dict:
    """Bayes empírico beta-binomial: estima la previa Beta(a, b) maximizando la verosimilitud marginal de TODOS los
    grupos y contrae la tasa de cada grupo hacia la media común (más cuanto menos ensayos tiene). Es el «préstamo de
    información» de los modelos jerárquicos (ensayos de cesta, tasas por oficina o por agente, credibilidad).

    Gauss: la previa estimada no recoge su propia incertidumbre (los IC son algo estrechos con pocos grupos);
    con < 5 grupos usa un modelo jerárquico completo."""
    x = _numerico(exitos, "exitos", 2); n = _numerico(ensayos, "ensayos", 2)
    if len(x) != len(n) or np.any(x > n) or np.any(x < 0):
        raise ValueError("exitos y ensayos: misma longitud y 0 ≤ éxitos ≤ ensayos.")

    def nll(th):
        a, b = np.exp(th)
        return -np.sum(special.betaln(x + a, n - x + b) - special.betaln(a, b))
    m0 = np.clip(x.sum() / n.sum(), 1e-3, 1 - 1e-3)
    r = optimize.minimize(nll, np.log([m0 * 10, (1 - m0) * 10]), method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 4000})
    a, b = np.exp(r.x)
    A, B = a + x, b + n - x
    idx = nombres if nombres is not None else range(len(x))
    tabla = pd.DataFrame({"exitos": x, "ensayos": n, "tasa_bruta": x / n, "tasa_contraida": A / (A + B),
                          "ic_inf": stats.beta.ppf(0.025, A, B), "ic_sup": stats.beta.ppf(0.975, A, B),
                          "peso_datos": n / (n + a + b)}, index=pd.Index(idx, name="grupo"))
    return {"previa": {"a": float(a), "b": float(b), "media": float(a / (a + b)), "tamano_previo": float(a + b)}, "tabla": tabla}


# ------------------------------------------------------------------ MCMC
def diagnostico_mcmc(muestras) -> pd.DataFrame:
    """Diagnósticos de convergencia para un array (cadenas, iteraciones, parámetros): R-hat dividido (Gelman-Rubin,
    versión split; < 1.01 ideal, > 1.1 mal), tamaño muestral efectivo (ESS, autocorrelación con la regla de Geyer)
    y error de Monte Carlo de la media. Equivale a: el resumen de PROC MCMC / ArviZ."""
    M = np.asarray(muestras, float)
    if M.ndim == 2:
        M = M[:, :, None]
    c, n, p = M.shape
    if n < 8:
        raise ValueError("Hacen falta al menos 8 iteraciones por cadena.")
    h = n // 2
    S = np.concatenate([M[:, :h], M[:, h:2 * h]], axis=0)              # split: 2c cadenas de longitud h
    filas = []
    for j in range(p):
        x = S[:, :, j]
        m, k = x.shape
        W = x.var(axis=1, ddof=1).mean(); Bm = k * x.mean(axis=1).var(ddof=1)
        var_mas = (k - 1) / k * W + Bm / k
        rhat = float(np.sqrt(var_mas / W)) if W > 0 else np.nan
        # ESS combinado (Vehtari et al. 2021, sin rangos)
        xc = x - x.mean(axis=1, keepdims=True)
        f = np.fft.rfft(xc, n=2 * k, axis=1)
        acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :k] / k
        rho = 1 - (W - acov.mean(axis=0)) / var_mas if var_mas > 0 else np.zeros(k)
        rho[0] = 1.0
        suma, t = 0.0, 1
        while t + 1 < k:
            par = rho[t] + rho[t + 1]
            if par < 0:
                break
            suma += par; t += 2
        ess = m * k / (1 + 2 * suma) if (1 + 2 * suma) > 0 else float(m * k)
        todo = M[:, :, j].ravel()
        filas.append({"media": todo.mean(), "desviacion": todo.std(ddof=1), "q2.5": np.quantile(todo, .025),
                      "q97.5": np.quantile(todo, .975), "r_hat": rhat, "ess": min(ess, m * k * np.log10(m * k)),
                      "error_mc": todo.std(ddof=1) / np.sqrt(max(ess, 1))})
    return pd.DataFrame(filas)


def metropolis(log_posterior: Callable, inicial, n_iter: int = 5000, cadenas: int = 4, quemado: int | None = None,
               escala=None, nombres=None, semilla: int = 42) -> dict:
    """Metropolis de paseo aleatorio (propuesta normal) con varias cadenas desde puntos dispersos y adaptación de la
    escala durante el calentamiento (objetivo: aceptación ≈ 0.23-0.44). `log_posterior(theta)` devuelve el log de la
    posterior SIN normalizar (−inf fuera del soporte). Devuelve las muestras tras el calentamiento, la tasa de
    aceptación y el diagnóstico (R-hat, ESS) de cada parámetro.

    Equivale a: PROC MCMC (algoritmo de paseo aleatorio). Gauss: mira R-hat < 1.01 y ESS > 400 antes de usar las
    medias; parámetros muy correlados mezclan mal (reparametriza o usa HMC/Stan)."""
    th0 = np.atleast_1d(np.asarray(inicial, float))
    p = len(th0)
    if not np.isfinite(log_posterior(th0)):
        raise ValueError("log_posterior(inicial) no es finito: elige un punto inicial dentro del soporte.")
    rng = np.random.default_rng(semilla)
    quemado = n_iter // 2 if quemado is None else quemado
    esc0 = np.full(p, 0.5) if escala is None else np.broadcast_to(np.asarray(escala, float), (p,)).copy()
    total = quemado + n_iter
    out = np.empty((cadenas, n_iter, p)); acept = np.zeros(cadenas)
    for c in range(cadenas):
        th = th0 + (rng.normal(0, 1, p) * esc0 * 2 if c else 0)
        lp = log_posterior(th)
        if not np.isfinite(lp):
            th, lp = th0.copy(), log_posterior(th0)
        esc = esc0.copy(); cov = np.diag(esc ** 2); hist = []
        lam, ac_vent = 2.38 ** 2 / p, 0
        for i in range(total):
            if i < quemado and i >= 200 and i % 50 == 0 and len(hist) > 50:
                cov = np.cov(np.asarray(hist[-1000:]).T).reshape(p, p) + 1e-10 * np.eye(p)
            if (i < quemado and i % 50 == 0) or i == quemado:
                L = np.linalg.cholesky(lam * cov)
            prop = th + L @ rng.standard_normal(p)
            lpp = log_posterior(prop)
            if np.log(rng.random()) < lpp - lp:
                th, lp = prop, lpp; ac_vent += 1
                if i >= quemado:
                    acept[c] += 1
            if i < quemado:
                hist.append(th.copy())
                if (i + 1) % 100 == 0:
                    tasa = ac_vent / 100
                    lam *= np.exp((tasa - 0.3) * 2)
                    ac_vent = 0
            else:
                out[c, i - quemado] = th
    nombres = list(nombres) if nombres is not None else [f"theta{j}" for j in range(p)]
    diag = diagnostico_mcmc(out); diag.index = nombres
    muestras = pd.DataFrame(out.reshape(-1, p), columns=nombres)
    muestras.insert(0, "cadena", np.repeat(np.arange(cadenas), n_iter))
    avisos = []
    if (diag.r_hat > 1.01).any():
        avisos.append("R-hat > 1.01: las cadenas no han convergido; alarga el calentamiento o reparametriza.")
    if (diag.ess < 400).any():
        avisos.append("ESS < 400 en algún parámetro: las medias y sobre todo los cuantiles tienen error de Monte Carlo apreciable.")
    return {"muestras": muestras, "resumen": diag, "aceptacion": float(acept.mean() / n_iter), "cadenas": out, "avisos": avisos}


def chequeo_predictivo(observado, muestras_parametros, simulador: Callable, estadisticos: dict | None = None,
                       semilla: int = 42) -> pd.DataFrame:
    """Chequeo predictivo (posterior o previo): para cada muestra de parámetros simula un conjunto de datos réplica
    con `simulador(theta, n, rng)` y compara estadísticos de las réplicas con los observados. El p-valor bayesiano
    P(T(réplica) ≥ T(observado)) cerca de 0 o 1 indica que el modelo no reproduce ese aspecto de los datos.
    Pasa muestras de la PREVIA para un chequeo previo (¿la previa genera datos plausibles?)."""
    y = _numerico(observado, "observado", 2)
    rng = np.random.default_rng(semilla)
    est = estadisticos or {"media": np.mean, "desviacion": np.std, "minimo": np.min, "maximo": np.max,
                           "prop_ceros": lambda v: np.mean(np.asarray(v) == 0)}
    th = np.asarray(muestras_parametros, float)
    th = th[:, None] if th.ndim == 1 else th
    rep = {k: [] for k in est}
    for t in th:
        yr = np.asarray(simulador(t if len(t) > 1 else t[0], len(y), rng))
        for k, f in est.items():
            rep[k].append(f(yr))
    filas = []
    for k, f in est.items():
        r = np.asarray(rep[k]); o = f(y)
        p = np.mean(r >= o) if np.ptp(r) > 0 else np.nan
        filas.append({"estadistico": k, "observado": o, "replicas_media": r.mean(), "replicas_q2.5": np.quantile(r, .025),
                      "replicas_q97.5": np.quantile(r, .975), "p_bayesiano": p, "alarma": bool(np.isfinite(p) and min(p, 1 - p) < 0.025)})
    return pd.DataFrame(filas).set_index("estadistico")


def accion_bayes(muestras, perdida="cuadratica", acciones=None, estados=None) -> dict:
    """Teoría de la decisión: la acción que minimiza la pérdida esperada a posteriori.
    - ``'cuadratica'`` → media posterior; ``'absoluta'`` → mediana; ``'0-1'`` → moda (aprox. por KDE);
      ``('asimetrica', k)`` → cuantil k/(1+k) (sobrestimar cuesta 1, infraestimar cuesta k: provisiones prudentes).
    - Matriz de pérdidas (DataFrame acciones × estados) con `muestras` = probabilidades de cada estado → acción con menor
      pérdida esperada (decisiones discretas: aceptar/rechazar, tarifa A/B…)."""
    if isinstance(perdida, pd.DataFrame):
        L = perdida
        p = pd.Series(np.asarray(muestras, float), index=L.columns)
        if not np.isclose(p.sum(), 1):
            raise ValueError("Con matriz de pérdidas, `muestras` son las probabilidades de cada estado (suman 1).")
        esperada = L @ p
        return {"accion": esperada.idxmin(), "perdida_esperada": esperada.sort_values()}
    x = _numerico(muestras, "muestras", 2)
    if perdida == "cuadratica":
        a = x.mean()
    elif perdida == "absoluta":
        a = np.median(x)
    elif perdida == "0-1":
        kde = stats.gaussian_kde(x); g = np.linspace(x.min(), x.max(), 2000); a = g[np.argmax(kde(g))]
    elif isinstance(perdida, tuple) and perdida[0] == "asimetrica":
        k = float(perdida[1]); a = np.quantile(x, k / (1 + k))
    else:
        raise ValueError("perdida: 'cuadratica', 'absoluta', '0-1', ('asimetrica', k) o una matriz DataFrame")
    rejilla = np.quantile(x, np.linspace(0.01, 0.99, 99))
    return {"accion": float(a), "perdida_esperada_cuadratica": float(np.mean((x - a) ** 2)),
            "perdida_esperada_absoluta": float(np.mean(np.abs(x - a))), "rejilla": rejilla}


# ------------------------------------------------------------------ A/B y bandidos
def ab_bayesiano(exitos_a: int, n_a: int, exitos_b: int, n_b: int, previa=(1, 1), n_sim: int = 200_000,
                 umbral_perdida: float = 0.001, semilla: int = 42) -> dict:
    """Test A/B bayesiano (beta-binomial): P(B > A), uplift relativo con IC creíble y PÉRDIDA ESPERADA de elegir cada
    variante (lo que perderías si te equivocas). Regla de parada adaptativa: parar cuando la pérdida esperada de la
    mejor sea < `umbral_perdida`. Gauss: mirar los datos a cada rato no invalida la posterior, pero sí las garantías
    frecuentistas de error tipo I; si las necesitas, usa un diseño secuencial."""
    rng = np.random.default_rng(semilla)
    a, b = previa
    pa = rng.beta(a + exitos_a, b + n_a - exitos_a, n_sim); pb = rng.beta(a + exitos_b, b + n_b - exitos_b, n_sim)
    perd_a, perd_b = np.mean(np.maximum(pb - pa, 0)), np.mean(np.maximum(pa - pb, 0))
    up = pb / pa - 1
    mejor = "B" if perd_b < perd_a else "A"
    return {"prob_b_mejor": float(np.mean(pb > pa)), "uplift_medio": float(up.mean()), "uplift_ic": tuple(np.quantile(up, [.025, .975])),
            "perdida_esperada_a": float(perd_a), "perdida_esperada_b": float(perd_b), "mejor": mejor,
            "parar": bool(min(perd_a, perd_b) < umbral_perdida)}


def simular_bandido(probabilidades, n_rondas: int = 2000, estrategia: str = "thompson", epsilon: float = 0.1,
                    repeticiones: int = 1, semilla: int = 42) -> dict:
    """Bandido multibrazo Bernoulli: en cada ronda se elige un brazo (variante, oferta, creatividad) y se observa éxito
    o fracaso. Estrategias: ``'thompson'`` (muestreo de la posterior beta), ``'ucb'`` (UCB1), ``'epsilon'`` (ε-greedy),
    ``'uniforme'`` (reparto fijo, como un A/B clásico). Devuelve el arrepentimiento acumulado medio (lo perdido frente a
    jugar siempre el mejor brazo) y cuántas veces se jugó cada brazo."""
    P = _numerico(probabilidades, "probabilidades", 2)
    if np.any((P < 0) | (P > 1)):
        raise ValueError("probabilidades en [0, 1].")
    if estrategia not in ("thompson", "ucb", "epsilon", "uniforme"):
        raise ValueError("estrategia: 'thompson', 'ucb', 'epsilon' o 'uniforme'")
    rng = np.random.default_rng(semilla)
    K = len(P); arrep = np.zeros(n_rondas); jugadas = np.zeros(K)
    for _ in range(repeticiones):
        s, f = np.zeros(K), np.zeros(K); reg = np.empty(n_rondas)
        for t in range(n_rondas):
            if estrategia == "thompson":
                k = int(np.argmax(rng.beta(s + 1, f + 1)))
            elif estrategia == "ucb":
                nk = s + f
                k = t if t < K else int(np.argmax(s / nk + np.sqrt(2 * np.log(t + 1) / nk)))
            elif estrategia == "epsilon":
                nk = s + f
                k = int(rng.integers(K)) if (rng.random() < epsilon or t < K) else int(np.argmax(np.where(nk > 0, s / np.maximum(nk, 1), 0)))
            else:
                k = t % K
            r = rng.random() < P[k]
            s[k] += r; f[k] += 1 - r
            reg[t] = P.max() - P[k]
        arrep += np.cumsum(reg); jugadas += s + f
    arrep /= repeticiones; jugadas /= repeticiones
    return {"arrepentimiento": pd.Series(arrep, index=np.arange(1, n_rondas + 1), name=estrategia),
            "jugadas": pd.Series(jugadas, index=[f"brazo_{i}" for i in range(K)]),
            "prop_mejor_brazo": float(jugadas[np.argmax(P)] / n_rondas)}
