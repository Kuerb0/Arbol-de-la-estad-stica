"""Procesos estocásticos: cadenas de Markov discretas (clasificación de estados, distribución estacionaria, absorción),
bonus-malus, cadenas en tiempo continuo, nacimiento y muerte / colas M/M/c, proceso de Poisson (homogéneo y no
homogéneo), ruina del jugador, paseo aleatorio y martingalas, movimiento browniano aritmético y geométrico."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import linalg, stats


def _matriz_transicion(P, nombres=None) -> pd.DataFrame:
    M = pd.DataFrame(P).astype(float)
    if M.shape[0] != M.shape[1]:
        raise ValueError("La matriz de transición debe ser cuadrada.")
    if (M.to_numpy() < -1e-12).any() or not np.allclose(M.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Cada fila de P debe ser una distribución de probabilidad (≥ 0 y suma 1).")
    if nombres is not None:
        M.index = M.columns = list(nombres)
    elif not isinstance(P, pd.DataFrame):
        M.index = M.columns = list(range(M.shape[0]))
    return M


def _estacionaria(P: np.ndarray) -> np.ndarray:
    n = P.shape[0]
    A = np.vstack([P.T - np.eye(n), np.ones(n)])
    b = np.r_[np.zeros(n), 1]
    return np.linalg.lstsq(A, b, rcond=None)[0]


def cadena_markov(P, inicial=None, pasos: int = 10, nombres=None) -> dict:
    """Análisis de una cadena de Markov en tiempo discreto con matriz de transición P (filas = estado actual).

    Devuelve: P^n, distribución tras `pasos` desde `inicial`, clases comunicantes con su tipo (recurrente/transitoria)
    y periodo, si es regular (alguna potencia con todo > 0 → converge a la estacionaria desde cualquier inicio), la
    distribución estacionaria π (π P = π) con tiempos medios de retorno 1/π, y, si hay estados absorbentes, la matriz
    fundamental N = (I − Q)⁻¹, las probabilidades de absorción y el tiempo esperado hasta absorción."""
    M = _matriz_transicion(P, nombres)
    est = list(M.index); n = len(est); A = M.to_numpy()
    alcanza = (A > 0) | np.eye(n, dtype=bool)
    for k in range(n):                                    # cierre transitivo (Warshall)
        alcanza = alcanza | (alcanza[:, [k]] & alcanza[[k], :])
    comunica = alcanza & alcanza.T
    clases, visto = [], set()
    for i in range(n):
        if i in visto:
            continue
        c = [j for j in range(n) if comunica[i, j]]
        visto.update(c)
        cerrada = not any(alcanza[i, j] and not comunica[i, j] for j in range(n))
        # periodo: mcd de las longitudes de ciclos que vuelven a i (hasta 2n pasos)
        Pk, mcd = np.eye(n), 0
        for k in range(1, 2 * n + 1):
            Pk = Pk @ A
            if Pk[i, i] > 1e-14:
                mcd = np.gcd(mcd, k)
        clases.append({"estados": [est[j] for j in c], "tipo": "recurrente" if cerrada else "transitoria", "periodo": int(mcd) if mcd else np.nan})
    regular = bool((np.linalg.matrix_power(A, (n - 1) ** 2 + 1) > 1e-14).all())
    out = {"matriz": M, "clases": pd.DataFrame(clases), "regular": regular,
           "P_n": pd.DataFrame(np.linalg.matrix_power(A, pasos), index=est, columns=est)}
    if inicial is not None:
        if np.ndim(inicial) == 0:
            if inicial not in est:
                raise ValueError(f"Estado inicial {inicial!r} no está entre {est}.")
            v = np.eye(n)[est.index(inicial)]
        else:
            v = np.asarray(inicial, float)
            if len(v) != n or not np.isclose(v.sum(), 1):
                raise ValueError("inicial: un estado o una distribución de probabilidad sobre los estados.")
        tray = [v]
        for _ in range(pasos):
            tray.append(tray[-1] @ A)
        out["trayectoria"] = pd.DataFrame(tray, columns=est).rename_axis("paso")
    recurrentes = [c for c in clases if c["tipo"] == "recurrente"]
    if len(recurrentes) == 1:
        pi = _estacionaria(A)
        out["estacionaria"] = pd.Series(pi, index=est)
        out["tiempo_medio_retorno"] = pd.Series(np.where(pi > 1e-12, 1 / np.maximum(pi, 1e-300), np.inf), index=est)
    absorb = [i for i in range(n) if np.isclose(A[i, i], 1)]
    trans = [i for i in range(n) if not any(est[i] in c["estados"] for c in recurrentes)]
    if absorb and trans:
        Q = A[np.ix_(trans, trans)]; R = A[np.ix_(trans, absorb)]
        N = np.linalg.inv(np.eye(len(trans)) - Q)
        out["matriz_fundamental"] = pd.DataFrame(N, index=[est[i] for i in trans], columns=[est[i] for i in trans])
        out["prob_absorcion"] = pd.DataFrame(N @ R, index=[est[i] for i in trans], columns=[est[i] for i in absorb])
        out["tiempo_hasta_absorcion"] = pd.Series(N.sum(axis=1), index=[est[i] for i in trans])
    return out


def simular_cadena_markov(P, n_pasos: int, inicial=0, n_trayectorias: int = 1, semilla: int = 42) -> pd.DataFrame:
    """Simula trayectorias de una cadena de Markov discreta (filas = paso, columnas = trayectoria) con los NOMBRES de
    los estados. La frecuencia de visitas a largo plazo converge a la estacionaria (teorema ergódico)."""
    M = _matriz_transicion(P)
    est = np.array(M.index); A = M.to_numpy(); C = np.cumsum(A, axis=1)
    rng = np.random.default_rng(semilla)
    i0 = list(M.index).index(inicial) if inicial in list(M.index) else int(inicial)
    X = np.empty((n_pasos + 1, n_trayectorias), int); X[0] = i0
    for t in range(n_pasos):
        u = rng.random(n_trayectorias)
        X[t + 1] = np.minimum((u[:, None] > C[X[t]]).sum(axis=1), len(est) - 1)
    return pd.DataFrame(est[X]).rename_axis("paso")


def bonus_malus(coeficientes, reglas, frecuencia: float, max_siniestros: int = 4, anios: int = 30, nivel_inicial=None) -> dict:
    """Sistema bonus-malus como cadena de Markov: niveles con su coeficiente de prima y `reglas[nivel][k]` = nivel al
    que se pasa tras k siniestros en el año (el último valor de la lista se usa para k ≥ len−1). Con siniestros
    Poisson(`frecuencia`) construye la matriz de transición, la distribución estacionaria, el coeficiente medio a largo
    plazo, la evolución desde el nivel inicial y la eficiencia de Loimaranta η = d ln(prima media)/d ln(λ)
    (1 = la prima sigue proporcionalmente al riesgo; los sistemas reales suelen quedarse en 0.2-0.5)."""
    coef = np.asarray(coeficientes, float); K = len(coef)
    if len(reglas) != K:
        raise ValueError("Hace falta una regla por nivel.")

    def matriz(lam):
        P = np.zeros((K, K))
        pk = stats.poisson.pmf(np.arange(max_siniestros), lam); pk = np.r_[pk, 1 - pk.sum()]
        for i, r in enumerate(reglas):
            for k, p in enumerate(pk):
                P[i, int(r[min(k, len(r) - 1)])] += p
        return P

    P = matriz(frecuencia)
    pi = _estacionaria(P)
    media = float(pi @ coef)
    h = 1e-4
    eta = (np.log(_estacionaria(matriz(frecuencia * (1 + h))) @ coef) - np.log(_estacionaria(matriz(frecuencia * (1 - h))) @ coef)) / (np.log(1 + h) - np.log(1 - h))
    i0 = int(np.argmin(np.abs(coef - 1))) if nivel_inicial is None else int(nivel_inicial)
    v = np.eye(K)[i0]; ev = []
    for t in range(anios + 1):
        ev.append({"anio": t, "coeficiente_medio": v @ coef, "dist_estacionaria_tv": 0.5 * np.abs(v - pi).sum()})
        v = v @ P
    return {"matriz": pd.DataFrame(P), "estacionaria": pd.Series(pi, name="prob"), "coeficiente_medio": media,
            "eficiencia_loimaranta": float(eta), "evolucion": pd.DataFrame(ev).set_index("anio")}


def cadena_markov_continua(Q, t=1.0, nombres=None) -> dict:
    """Cadena de Markov en tiempo continuo con generador Q (filas suman 0, fuera de la diagonal tasas ≥ 0).
    P(t) = exp(Q t), distribución estacionaria (π Q = 0), tiempo medio de permanencia en cada estado (1/q_i) y cadena
    de saltos inmersa. Útil para modelos multiestado (sano-inválido-fallecido), fiabilidad y colas."""
    G = pd.DataFrame(Q).astype(float)
    A = G.to_numpy()
    if A.shape[0] != A.shape[1] or not np.allclose(A.sum(axis=1), 0, atol=1e-8) or (A - np.diag(np.diag(A)) < -1e-12).any():
        raise ValueError("Q: cuadrada, tasas fuera de la diagonal ≥ 0 y filas que suman 0.")
    est = list(nombres) if nombres is not None else (list(G.index) if isinstance(Q, pd.DataFrame) else list(range(len(A))))
    q = -np.diag(A)
    salto = np.where(q[:, None] > 0, (A + np.diag(q)) / np.where(q > 0, q, 1)[:, None], np.eye(len(A)))
    n = len(A)
    pi = np.linalg.lstsq(np.vstack([A.T, np.ones(n)]), np.r_[np.zeros(n), 1], rcond=None)[0]
    ts = np.atleast_1d(t)
    Pt = {float(s): pd.DataFrame(linalg.expm(A * s), index=est, columns=est) for s in ts}
    return {"P_t": Pt[float(ts[0])] if len(ts) == 1 else Pt, "estacionaria": pd.Series(pi, index=est),
            "permanencia_media": pd.Series(np.where(q > 0, 1 / np.where(q > 0, q, 1), np.inf), index=est),
            "cadena_saltos": pd.DataFrame(salto, index=est, columns=est)}


def nacimiento_muerte(nacimiento, muerte, capacidad: int | None = None, servidores: int | None = None) -> dict:
    """Proceso de nacimiento y muerte en equilibrio. Dos usos:
    - tasas por estado: `nacimiento[i]` (λ_i, de i a i+1) y `muerte[i]` (μ_{i+1}, de i+1 a i) como listas → π_n ∝ Π λ/μ;
    - cola M/M/c(/K): `nacimiento` y `muerte` escalares (llegadas λ, servicio μ por servidor) con `servidores` = c y
      `capacidad` opcional K → L, Lq, W, Wq (Little), probabilidad de esperar (Erlang C) y de rechazo.
    Gauss: supone llegadas Poisson y servicio exponencial; con servicio poco variable la cola real es menor (M/D/c)."""
    if np.ndim(nacimiento) == 0:
        lam, mu = float(nacimiento), float(muerte)
        c = int(servidores or 1)
        rho = lam / (c * mu)
        if capacidad is None and rho >= 1:
            raise ValueError(f"ρ = λ/(cμ) = {rho:.3f} ≥ 1: la cola crece sin límite (añade servidores o una capacidad K).")
        K = int(capacidad) if capacidad is not None else None
        nmax = K if K is not None else int(max(200, c + 50 / max(1e-9, -np.log(rho))))
        nn = np.arange(nmax + 1)
        lam_i = np.full(nmax, lam); mu_i = mu * np.minimum(nn[1:], c)
    else:
        lam_i, mu_i = np.asarray(nacimiento, float), np.asarray(muerte, float)
        if len(lam_i) != len(mu_i):
            raise ValueError("nacimiento y muerte: misma longitud (λ_0..λ_{K−1} y μ_1..μ_K).")
        nn = np.arange(len(lam_i) + 1); c = K = None
    w = np.r_[1.0, np.cumprod(lam_i / mu_i)]
    pi = w / w.sum()
    out = {"distribucion": pd.Series(pi, index=nn, name="prob"), "media_estado": float(pi @ nn)}
    if c is not None:
        L = float(pi @ nn); Lq = float(pi @ np.maximum(nn - c, 0))
        lam_ef = lam * (1 - (pi[-1] if K is not None else 0))
        out.update({"utilizacion": lam_ef / (c * mu), "L": L, "Lq": Lq, "W": L / lam_ef, "Wq": Lq / lam_ef,
                    "prob_esperar": float(pi[c:].sum()) if K is None else float(pi[c:-1].sum()),
                    "prob_rechazo": float(pi[-1]) if K is not None else 0.0})
    return out


def simular_proceso_poisson(tasa, horizonte: float, n_trayectorias: int = 1, tasa_maxima: float | None = None,
                            semilla: int = 42) -> list:
    """Tiempos de llegada de un proceso de Poisson en [0, horizonte]. `tasa` escalar (homogéneo: interllegadas
    exponenciales) o función λ(t) (no homogéneo, por adelgazamiento de Lewis-Shedler con `tasa_maxima` ≥ λ(t)).
    Devuelve una lista de arrays (uno por trayectoria). N(T) ~ Poisson(∫λ)."""
    rng = np.random.default_rng(semilla)
    fun = callable(tasa)
    if fun:
        if tasa_maxima is None:
            g = np.linspace(0, horizonte, 2001); tasa_maxima = float(np.max(tasa(g))) * 1.05
        lam = tasa_maxima
    else:
        lam = float(tasa)
    if lam <= 0:
        raise ValueError("La tasa debe ser > 0.")
    out = []
    for _ in range(n_trayectorias):
        n = rng.poisson(lam * horizonte)
        t = np.sort(rng.uniform(0, horizonte, n))       # dado N(T)=n, los tiempos son uniformes ordenados
        if fun:
            t = t[rng.random(n) < tasa(t) / lam]
        out.append(t)
    return out


def contraste_proceso_poisson(tiempos, horizonte: float) -> pd.DataFrame:
    """¿Son estos tiempos de llegada un proceso de Poisson homogéneo? Contrasta (1) que, condicionado a N(T), los tiempos
    son uniformes en [0, T] (KS) y (2) que las interllegadas son exponenciales (Lilliefors-exponencial vía KS con media
    estimada, aproximado) y la dispersión de los conteos por subintervalos (índice varianza/media ≈ 1)."""
    t = np.sort(np.asarray(tiempos, float))
    if len(t) < 10:
        raise ValueError("Hacen falta al menos 10 llegadas.")
    ks_u = stats.kstest(t / horizonte, "uniform")
    d = np.diff(np.r_[0, t])
    ks_e = stats.kstest(d, "expon", args=(0, d.mean()))
    k = max(5, int(np.sqrt(len(t))))
    cnt = np.histogram(t, bins=k, range=(0, horizonte))[0]
    disp = cnt.var(ddof=1) / cnt.mean()
    chi = (k - 1) * disp
    return pd.DataFrame([
        {"contraste": "tiempos uniformes (KS)", "estadistico": ks_u.statistic, "p_valor": ks_u.pvalue},
        {"contraste": "interllegadas exponenciales (KS, aprox.)", "estadistico": ks_e.statistic, "p_valor": ks_e.pvalue},
        {"contraste": f"dispersión de conteos ({k} tramos)", "estadistico": disp, "p_valor": 2 * min(stats.chi2.sf(chi, k - 1), stats.chi2.cdf(chi, k - 1))},
    ]).set_index("contraste")


def ruina_jugador(capital: int, objetivo: int, p: float = 0.5, simular: int = 0, semilla: int = 42) -> dict:
    """Ruina del jugador: empieza con `capital`, gana 1 con prob. p y pierde 1 con q = 1−p, hasta llegar a 0 (ruina) u
    `objetivo`. Probabilidad de ruina (fórmula cerrada), duración esperada del juego y, con `simular` > 0, la comprobación
    por Monte Carlo. Con p < 1/2 y objetivo infinito la ruina es segura (casino); con p > 1/2 vale (q/p)^capital."""
    i, N = int(capital), int(objetivo)
    if not 0 < i < N or not 0 < p < 1:
        raise ValueError("0 < capital < objetivo y 0 < p < 1.")
    q = 1 - p
    if np.isclose(p, 0.5):
        ruina, dur = 1 - i / N, i * (N - i)
    else:
        r = q / p
        ruina = (r ** i - r ** N) / (1 - r ** N)
        dur = i / (q - p) - N / (q - p) * (1 - r ** i) / (1 - r ** N)
    out = {"prob_ruina": float(ruina), "prob_exito": float(1 - ruina), "duracion_esperada": float(dur),
           "ruina_objetivo_infinito": float(min(1.0, (q / p) ** i)) if p > 0.5 else 1.0}
    if simular:
        rng = np.random.default_rng(semilla)
        x = np.full(simular, i); vivo = np.ones(simular, bool); pasos = np.zeros(simular)
        while vivo.any():
            x[vivo] += np.where(rng.random(vivo.sum()) < p, 1, -1); pasos[vivo] += 1
            vivo &= (x > 0) & (x < N)
        out["prob_ruina_simulada"] = float(np.mean(x == 0)); out["duracion_simulada"] = float(pasos.mean())
    return out


def paseo_aleatorio(n_pasos: int, p: float = 0.5, n_trayectorias: int = 2000, inicio: float = 0.0, semilla: int = 42) -> dict:
    """Paseo aleatorio simple (+1 con prob. p, −1 con q) y sus martingalas: M_n = S_n − n(p − q) y, si p ≠ q,
    (q/p)^{S_n}; con p = 1/2 también S_n² − n. Devuelve las trayectorias y la media de cada martingala por paso,
    que debe mantenerse constante (E[M_n] = M_0): la base del teorema de parada opcional y de la ruina del jugador."""
    rng = np.random.default_rng(semilla)
    pasos = np.where(rng.random((n_pasos, n_trayectorias)) < p, 1, -1)
    S = inicio + np.vstack([np.zeros(n_trayectorias), np.cumsum(pasos, axis=0)])
    n = np.arange(n_pasos + 1)[:, None]
    q = 1 - p
    mart = {"S_n - n(p-q)": (S - n * (p - q)).mean(axis=1)}
    if np.isclose(p, 0.5):
        mart["S_n^2 - n"] = (S ** 2 - n).mean(axis=1)
    else:
        mart["(q/p)^S_n"] = ((q / p) ** S).mean(axis=1)
    return {"trayectorias": pd.DataFrame(S).rename_axis("paso"), "martingalas": pd.DataFrame(mart).rename_axis("paso"),
            "media_teorica_final": float(inicio + n_pasos * (p - q)), "var_teorica_final": float(4 * n_pasos * p * q)}


def simular_browniano(horizonte: float = 1.0, n_pasos: int = 250, n_trayectorias: int = 1000, mu: float = 0.0,
                      sigma: float = 1.0, inicio: float = 0.0, geometrico: bool = False, semilla: int = 42) -> dict:
    """Movimiento browniano con deriva (X_t = x0 + μt + σW_t) o geométrico (dS = μS dt + σS dW, solución exacta de Itô:
    S_t = S0·exp((μ − σ²/2)t + σW_t)). Devuelve las trayectorias (filas = tiempo), y la comprobación de propiedades:
    media y varianza finales frente a las teóricas y la variación cuadrática (≈ σ²T, la base del lema de Itô)."""
    if n_pasos < 1 or horizonte <= 0 or sigma < 0:
        raise ValueError("n_pasos ≥ 1, horizonte > 0 y sigma ≥ 0.")
    rng = np.random.default_rng(semilla)
    dt = horizonte / n_pasos
    dW = rng.normal(0, np.sqrt(dt), (n_pasos, n_trayectorias))
    W = np.vstack([np.zeros(n_trayectorias), np.cumsum(dW, axis=0)])
    t = np.linspace(0, horizonte, n_pasos + 1)[:, None]
    if geometrico:
        if inicio <= 0:
            raise ValueError("El browniano geométrico necesita inicio > 0.")
        X = inicio * np.exp((mu - sigma ** 2 / 2) * t + sigma * W)
        media_t, var_t = inicio * np.exp(mu * horizonte), inicio ** 2 * np.exp(2 * mu * horizonte) * (np.exp(sigma ** 2 * horizonte) - 1)
        vq = np.sum(np.diff(np.log(X), axis=0) ** 2, axis=0).mean()
    else:
        X = inicio + mu * t + sigma * W
        media_t, var_t = inicio + mu * horizonte, sigma ** 2 * horizonte
        vq = np.sum(np.diff(X, axis=0) ** 2, axis=0).mean()
    fin = X[-1]
    return {"trayectorias": pd.DataFrame(X, index=pd.Index(t.ravel(), name="t")),
            "comprobacion": pd.DataFrame({"simulado": [fin.mean(), fin.var(ddof=1), vq], "teorico": [media_t, var_t, sigma ** 2 * horizonte]},
                                         index=["media final", "varianza final", "variación cuadrática (log si geométrico)"])}
