"""Simulación y Monte Carlo: generadores de números aleatorios y contrastes de aleatoriedad, generación de variables
(inversión, aceptación-rechazo, normal multivariante), integración Monte Carlo con reducción de varianza y
herramientas de probabilidad (Bayes, distribuciones conjuntas, estadísticos de orden, momentos, TCL y LGN)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import integrate, interpolate, optimize, stats

from .._util import _numerico


# ------------------------------------------------------------------ números aleatorios
def generador_congruencial(n: int, semilla: int = 12345, a: int = 1664525, c: int = 1013904223, m: int = 2 ** 32) -> np.ndarray:
    """Generador lineal congruencial x_{k+1} = (a·x_k + c) mod m → u = x/m. Por defecto los parámetros de Numerical
    Recipes (periodo completo 2³²). Didáctico: para trabajo real usa `np.random.default_rng` (PCG64).
    Con a = 65539, c = 0, m = 2³¹ obtienes RANDU, el ejemplo clásico de mal generador (falla el test de tripletas)."""
    if n < 1 or m <= 1:
        raise ValueError("n ≥ 1 y m > 1.")
    x = np.empty(n, dtype=np.uint64); v = int(semilla) % m
    for k in range(n):
        v = (a * v + c) % m
        x[k] = v
    return x.astype(float) / m


def contrastes_aleatoriedad(u, bins: int = 20) -> pd.DataFrame:
    """Batería de contrastes para una secuencia que debería ser U(0,1) independiente: KS de uniformidad, χ² de
    frecuencias, rachas por encima/debajo de la mediana, autocorrelación de orden 1, χ² de pares consecutivos
    (uniformidad en 2D) y de tripletas (3D, detecta RANDU). p-valores pequeños = el generador falla."""
    x = _numerico(u, "u", 100)
    n = len(x)
    filas = []
    ks = stats.kstest(x, "uniform"); filas.append(("uniformidad (KS)", ks.statistic, ks.pvalue))
    f = np.histogram(x, bins=bins, range=(0, 1))[0]; ch = stats.chisquare(f); filas.append((f"frecuencias (χ², {bins} celdas)", ch.statistic, ch.pvalue))
    s = x > np.median(x); r = 1 + np.sum(s[1:] != s[:-1]); n1, n2 = s.sum(), (~s).sum()
    mr = 2 * n1 * n2 / n + 1; vr = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    z = (r - mr) / np.sqrt(vr); filas.append(("rachas (mediana)", z, 2 * stats.norm.sf(abs(z))))
    rho = np.corrcoef(x[:-1], x[1:])[0, 1]; z = rho * np.sqrt(n); filas.append(("autocorrelación lag 1", rho, 2 * stats.norm.sf(abs(z))))
    k = max(2, int(np.sqrt(n / 2 / 10)))                              # ≥ 10 esperados por celda
    m = n // 2; cel = (np.floor(x[:2 * m:2] * k) * k + np.floor(x[1:2 * m:2] * k)).astype(int)
    ch = stats.chisquare(np.bincount(cel, minlength=k * k)); filas.append((f"pares 2D (χ², {k}×{k})", ch.statistic, ch.pvalue))
    k3 = max(2, int((n / 3 / 3) ** (1 / 3) + 1e-9))                  # ≥ 3 esperados por celda (muchas celdas: χ² aún fiable)
    m3 = n // 3; tri = (np.floor(x[:3 * m3:3] * k3) * k3 ** 2 + np.floor(x[1:3 * m3:3] * k3) * k3 + np.floor(x[2:3 * m3:3] * k3)).astype(int)
    ch = stats.chisquare(np.bincount(tri, minlength=k3 ** 3)); filas.append((f"tripletas 3D (χ², {k3}³)", ch.statistic, ch.pvalue))
    return pd.DataFrame(filas, columns=["contraste", "estadistico", "p_valor"]).set_index("contraste")


# ------------------------------------------------------------------ generación de variables
def generar_por_inversion(n: int, cuantil: Callable | None = None, distribucion=None, cdf: Callable | None = None,
                          soporte=None, semilla: int = 42) -> np.ndarray:
    """Método de la transformada inversa: X = F⁻¹(U) con U ~ U(0,1). Pasa la función cuantil (`cuantil`), una
    distribución de scipy (`distribucion`) o solo la `cdf` y su `soporte` (a, b) (se invierte numéricamente sobre una
    rejilla). Funciona también para discretas si `cuantil` devuelve el menor x con F(x) ≥ u."""
    rng = np.random.default_rng(semilla)
    u = rng.random(n)
    if distribucion is not None:
        return distribucion.ppf(u)
    if cuantil is not None:
        return np.asarray(cuantil(u), float)
    if cdf is None or soporte is None:
        raise ValueError("Pasa `cuantil`, `distribucion` o (`cdf` y `soporte`).")
    g = np.linspace(soporte[0], soporte[1], 20001)
    F = np.maximum.accumulate(np.asarray(cdf(g), float))
    F, idx = np.unique(F, return_index=True)
    return np.interp(u, F, g[idx])


def generar_por_aceptacion_rechazo(n: int, densidad: Callable, propuesta, M: float | None = None, semilla: int = 42) -> dict:
    """Aceptación-rechazo: genera de una densidad f (sin normalizar vale) a partir de una propuesta g de scipy con
    f ≤ M·g. Si no das M, se estima con un margen del 10 % sobre una rejilla de cuantiles de g. Devuelve la muestra y
    la tasa de aceptación (= 1/M si f está normalizada): cuanto más se parece g a f, menos se desperdicia."""
    rng = np.random.default_rng(semilla)
    if M is None:
        g = propuesta.ppf(np.linspace(1e-4, 1 - 1e-4, 4001))
        M = 1.1 * float(np.max(densidad(g) / propuesta.pdf(g)))
    out, intentos, aceptados = [], 0, 0
    while aceptados < n:
        k = int(1.2 * (n - aceptados) * M) + 10
        y = propuesta.rvs(size=k, random_state=rng); u = rng.random(k)
        intentos += k
        out.append(y[u * M * propuesta.pdf(y) <= densidad(y)]); aceptados += len(out[-1])
    x = np.concatenate(out)[:n]
    return {"muestra": x, "tasa_aceptacion": aceptados / intentos, "M": M}


def generar_normal_multivariante(medias, covarianza, n: int, metodo: str = "cholesky", semilla: int = 42) -> pd.DataFrame:
    """Normal multivariante X = μ + A·Z con A·Aᵀ = Σ: Cholesky (rápido; exige Σ definida positiva) o descomposición
    espectral ('eigen', vale para Σ semidefinida, p. ej. variables combinación lineal de otras). Comprueba que Σ es
    simétrica y semidefinida positiva y devuelve las columnas con los nombres de `medias` si es una Series."""
    mu = np.asarray(medias, float); S = np.asarray(covarianza, float)
    if S.shape != (len(mu), len(mu)) or not np.allclose(S, S.T):
        raise ValueError("covarianza debe ser simétrica p×p con p = len(medias).")
    w, V = np.linalg.eigh(S)
    if w.min() < -1e-10 * max(1, w.max()):
        raise ValueError(f"covarianza no es semidefinida positiva (autovalor mínimo {w.min():.3g}).")
    if metodo == "cholesky":
        A = np.linalg.cholesky(S)
    elif metodo == "eigen":
        A = V * np.sqrt(np.clip(w, 0, None))
    else:
        raise ValueError("metodo: 'cholesky' o 'eigen'")
    Z = np.random.default_rng(semilla).standard_normal((n, len(mu)))
    cols = list(medias.index) if isinstance(medias, pd.Series) else [f"x{i + 1}" for i in range(len(mu))]
    return pd.DataFrame(mu + Z @ A.T, columns=cols)


# ------------------------------------------------------------------ integración Monte Carlo
def estimar_montecarlo(f: Callable, n: int = 10_000, dim: int = 1, metodo: str = "simple", control: Callable | None = None,
                       media_control: float | None = None, nivel: float = 0.95, semilla: int = 42) -> dict:
    """Estima θ = E[f(U)], U ~ U(0,1)^dim (cualquier integral en el hipercubo; para otras distribuciones transforma
    dentro de f). Métodos: ``'simple'``, ``'antiteticas'`` (promedia f(U) y f(1−U): reduce varianza si f es monótona),
    ``'control'`` (variable de control c(U) con media conocida, coeficiente óptimo estimado). Devuelve estimación, error
    estándar, IC y la reducción de varianza frente al método simple con el mismo nº de evaluaciones de f."""
    rng = np.random.default_rng(semilla)
    U = rng.random((n, dim)) if dim > 1 else rng.random(n)
    fu = np.asarray(f(U), float)
    var_simple = fu.var(ddof=1)
    if metodo == "simple":
        y = fu; ne = n
    elif metodo == "antiteticas":
        h = n // 2
        y = 0.5 * (fu[:h] + np.asarray(f(1 - U[:h]), float)); ne = h
    elif metodo == "control":
        if control is None or media_control is None:
            raise ValueError("metodo='control' necesita `control` y `media_control`.")
        cu = np.asarray(control(U), float)
        b = np.cov(fu, cu)[0, 1] / cu.var(ddof=1)
        y = fu - b * (cu - media_control); ne = n
    else:
        raise ValueError("metodo: 'simple', 'antiteticas' o 'control'")
    est, se = y.mean(), y.std(ddof=1) / np.sqrt(ne)
    z = stats.norm.ppf(0.5 + nivel / 2)
    se_simple = np.sqrt(var_simple / n)
    return {"estimacion": float(est), "error_estandar": float(se), "ic": (float(est - z * se), float(est + z * se)),
            "reduccion_varianza": float(1 - se ** 2 / se_simple ** 2) if metodo != "simple" else 0.0, "evaluaciones_f": n}


# ------------------------------------------------------------------ probabilidad
def teorema_bayes(previas, verosimilitudes, nombres=None) -> pd.DataFrame:
    """Teorema de Bayes sobre hipótesis excluyentes: P(H_i | E) = P(E | H_i) P(H_i) / Σ_j P(E | H_j) P(H_j).
    Atajo para pruebas diagnósticas: `previas` = prevalencia (escalar) y `verosimilitudes` = (sensibilidad,
    especificidad) → valor predictivo positivo y negativo (la falacia de la tasa base en acción)."""
    if np.ndim(previas) == 0:
        prev = float(previas); se, es = verosimilitudes
        vpp = se * prev / (se * prev + (1 - es) * (1 - prev)); vpn = es * (1 - prev) / (es * (1 - prev) + (1 - se) * prev)
        return pd.DataFrame({"valor": [prev, se, es, se * prev + (1 - es) * (1 - prev), vpp, vpn, se / (1 - es), (1 - se) / es]},
                            index=["prevalencia", "sensibilidad", "especificidad", "P(test +)", "VPP = P(enfermo | +)",
                                   "VPN = P(sano | −)", "razón de verosimilitud +", "razón de verosimilitud −"])
    p = np.asarray(previas, float); L = np.asarray(verosimilitudes, float)
    if len(p) != len(L) or not np.isclose(p.sum(), 1) or (p < 0).any():
        raise ValueError("previas: probabilidades que suman 1, una por verosimilitud.")
    conj = p * L; ev = conj.sum()
    return pd.DataFrame({"previa": p, "verosimilitud": L, "conjunta": conj, "posterior": conj / ev},
                        index=nombres if nombres is not None else range(len(p)))


def analizar_distribucion_conjunta(tabla, valores_x=None, valores_y=None) -> dict:
    """Distribución conjunta discreta de (X, Y) a partir de una tabla de probabilidades o de frecuencias (filas = X,
    columnas = Y; índices numéricos para medias). Devuelve marginales, condicionadas P(Y|X) y P(X|Y), esperanza
    condicionada E[Y|X=x] (y la comprobación E[E[Y|X]] = E[Y]), covarianza, correlación y si hay independencia
    (máxima diferencia |P(x,y) − P(x)P(y)|; con frecuencias además el χ² de independencia)."""
    T = pd.DataFrame(tabla).astype(float)
    if valores_x is not None:
        T.index = valores_x
    if valores_y is not None:
        T.columns = valores_y
    frec = T.to_numpy().sum() > 1 + 1e-9
    P = T / T.to_numpy().sum()
    px, py = P.sum(axis=1), P.sum(axis=0)
    out = {"conjunta": P, "marginal_x": px, "marginal_y": py, "y_dado_x": P.div(px, axis=0), "x_dado_y": P.div(py, axis=1)}
    ind = float(np.abs(P.to_numpy() - np.outer(px, py)).max())
    out["max_desviacion_independencia"] = ind
    try:
        x = np.asarray(P.index, float); y = np.asarray(P.columns, float)
        ey_x = out["y_dado_x"].to_numpy() @ y
        ex, ey = px.to_numpy() @ x, py.to_numpy() @ y
        exy = x @ P.to_numpy() @ y
        vx, vy = px.to_numpy() @ (x - ex) ** 2, py.to_numpy() @ (y - ey) ** 2
        out.update({"esperanza_y_dado_x": pd.Series(ey_x, index=P.index), "E_y": float(ey), "E_E_y_dado_x": float(px.to_numpy() @ ey_x),
                    "covarianza": float(exy - ex * ey), "correlacion": float((exy - ex * ey) / np.sqrt(vx * vy))})
    except (TypeError, ValueError):
        pass
    if frec:
        chi = stats.chi2_contingency(T.to_numpy(), correction=False)
        out["chi2_independencia"] = {"chi2": float(chi[0]), "gl": int(chi[2]), "p_valor": float(chi[1])}
    return out


def distribucion_estadistico_orden(distribucion, n: int, k: int, n_sim: int = 0, semilla: int = 42) -> dict:
    """Distribución del k-ésimo estadístico de orden X_(k) de una muestra de tamaño n de `distribucion` (scipy):
    F_(k)(x) = P(Bin(n, F(x)) ≥ k), densidad n!/((k−1)!(n−k)!) F^{k−1}(1−F)^{n−k} f, media y cuantiles (por
    transformación: U_(k) ~ Beta(k, n−k+1)). k = 1 es el mínimo y k = n el máximo (valores extremos)."""
    if not 1 <= k <= n:
        raise ValueError("1 ≤ k ≤ n.")
    B = stats.beta(k, n - k + 1)
    q = distribucion.ppf(B.ppf([0.025, 0.5, 0.975]))
    media = integrate.quad(lambda u: distribucion.ppf(u) * B.pdf(u), 0, 1, limit=200)[0]
    out = {"media": media, "mediana": q[1], "ic95": (q[0], q[2]),
           "cdf": lambda x: stats.binom.sf(k - 1, n, distribucion.cdf(x)),
           "pdf": lambda x: B.pdf(distribucion.cdf(x)) * distribucion.pdf(x)}
    if n_sim:
        rng = np.random.default_rng(semilla)
        s = np.sort(distribucion.rvs(size=(n_sim, n), random_state=rng), axis=1)[:, k - 1]
        out["media_simulada"] = float(s.mean()); out["muestra_simulada"] = s
    return out


def momentos_distribucion(distribucion, orden: int = 4, t=(0.1, 0.5)) -> dict:
    """Momentos de una distribución de scipy: crudos E[X^k], centrales, asimetría y curtosis en exceso, y la función
    generadora de momentos M(t) = E[e^{tX}] (numérica; ∞ si no existe, como en la lognormal o la Pareto) con la
    comprobación de que M'(0) ≈ E[X] y M''(0) ≈ E[X²] por diferencias finitas."""
    crudos = {k: float(distribucion.moment(k)) for k in range(1, orden + 1)}
    mu = crudos[1]
    centrales = {k: float(distribucion.expect(lambda x, k=k: (x - mu) ** k)) for k in range(2, orden + 1)}
    m, v, s, c = distribucion.stats(moments="mvsk")

    a_, b_ = distribucion.support()

    def fgm(tt):
        with np.errstate(all="ignore"):
            discreta = not hasattr(distribucion, "pdf")
            logf = distribucion.logpmf if discreta else distribucion.logpdf
            g = lambda x: tt * x + logf(np.floor(x) if discreta else x)          # noqa: E731
            for lim, lejos, cerca in ((b_, 1e6, 1e3), (a_, -1e6, -1e3)):
                if not np.isfinite(lim) and (g(lejos) > g(cerca) or g(lejos) > -50):
                    return np.inf
            if discreta:
                k = np.arange(max(a_, -10 ** 6), distribucion.ppf(1 - 1e-15) + 1)
                return float(np.sum(np.exp(tt * k) * distribucion.pmf(k)))
            v = integrate.quad(lambda x: np.exp(np.clip(g(x), -745, 700)), a_, b_, limit=200)[0]
        return float(v) if np.isfinite(v) and v < 1e300 else np.inf
    h = 1e-3
    M = {float(tt): fgm(tt) for tt in t}
    d1 = (fgm(h) - fgm(-h)) / (2 * h); d2 = (fgm(h) - 2 + fgm(-h)) / h ** 2
    return {"crudos": crudos, "centrales": centrales, "media": float(m), "varianza": float(v), "asimetria": float(s),
            "curtosis_exceso": float(c), "fgm": M, "fgm_derivada1_en_0": d1, "fgm_derivada2_en_0": d2}


def convergencia_media_muestral(distribucion, tamanos=(1, 2, 5, 10, 30, 100), n_rep: int = 5000, semilla: int = 42) -> dict:
    """Ley de los grandes números y teorema central del límite por simulación: para cada n, distribución de la media
    muestral de `distribucion` (scipy) en `n_rep` repeticiones. Devuelve su media y desviación frente a μ y σ/√n,
    asimetría, curtosis y distancia de Kolmogorov a la normal (cuánto le falta al TCL), y una trayectoria de la media
    acumulada (LGN). Con colas pesadas (Cauchy, Pareto α<2) el TCL no se cumple y lo verás aquí."""
    rng = np.random.default_rng(semilla)
    m, v = distribucion.stats(moments="mv")
    filas, medias = [], {}
    for n in tamanos:
        x = distribucion.rvs(size=(n_rep, n), random_state=rng).mean(axis=1)
        z = (x - x.mean()) / x.std(ddof=1)
        filas.append({"n": n, "media": x.mean(), "desviacion": x.std(ddof=1), "teorica_sigma_raiz_n": np.sqrt(v / n) if np.isfinite(v) else np.nan,
                      "asimetria": stats.skew(x), "curtosis_exceso": stats.kurtosis(x), "ks_normal": stats.kstest(z, "norm").statistic})
        medias[n] = x
    tray = np.cumsum(distribucion.rvs(size=max(tamanos) * 100, random_state=rng)) / np.arange(1, max(tamanos) * 100 + 1)
    return {"tabla": pd.DataFrame(filas).set_index("n"), "medias": medias, "media_teorica": float(m),
            "media_acumulada": pd.Series(tray, index=np.arange(1, len(tray) + 1), name="media acumulada")}
