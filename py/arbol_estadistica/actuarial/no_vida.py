"""Matemática actuarial de no vida: reservas (chain ladder con error de Mack y bootstrap ODP), siniestralidad agregada
(recursión de Panjer y simulación), teoría de la ruina, principios de cálculo de primas y credibilidad de Bühlmann-Straub.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats


def _triangulo(tri) -> pd.DataFrame:
    T = pd.DataFrame(tri).astype(float)
    if T.shape[0] < 2 or T.shape[1] < 2:
        raise ValueError("El triángulo necesita al menos 2 orígenes y 2 desarrollos.")
    return T


def chain_ladder(triangulo, acumulado: bool = True) -> dict:
    """Chain ladder (método de los factores de desarrollo) sobre un triángulo origen × desarrollo (NaN en el futuro).
    Devuelve factores de desarrollo ponderados, último estimado, reserva IBNR por año de origen, total y el error
    estándar de Mack (proceso + parámetro) por año y total. `acumulado=False` si el triángulo es de pagos incrementales.
    Gauss: supone que el patrón de desarrollo pasado se repite; revisa factores con pocos datos (últimas columnas)."""
    T = _triangulo(triangulo)
    C = T.cumsum(axis=1) if not acumulado else T.copy()
    C = C.where(T.notna())
    n, m = C.shape
    f, sigma2 = np.ones(m - 1), np.zeros(m - 1)
    for j in range(m - 1):
        ok = C.iloc[:, j].notna() & C.iloc[:, j + 1].notna()
        a, b = C.iloc[:, j][ok].to_numpy(), C.iloc[:, j + 1][ok].to_numpy()
        f[j] = b.sum() / a.sum()
        if ok.sum() > 1:
            sigma2[j] = np.sum(a * (b / a - f[j]) ** 2) / (ok.sum() - 1)
    for j in range(m - 1):                                     # extrapolación de Mack para la última σ²
        if sigma2[j] == 0 and j > 1:
            sigma2[j] = min(sigma2[j - 1] ** 2 / max(sigma2[j - 2], 1e-12), min(sigma2[j - 2], sigma2[j - 1]))
    proy = C.copy()
    for j in range(m - 1):
        falta = proy.iloc[:, j + 1].isna() & proy.iloc[:, j].notna()
        proy.loc[falta, proy.columns[j + 1]] = proy.loc[falta, proy.columns[j]] * f[j]
    ultimo = proy.iloc[:, -1]
    diag = C.apply(lambda r: r.dropna().iloc[-1], axis=1)
    reserva = ultimo - diag
    # error de Mack
    se = np.zeros(n)
    sumC = [C.iloc[:, j][C.iloc[:, j + 1].notna()].sum() for j in range(m - 1)]
    for i in range(n):
        ult_j = int(C.iloc[i].notna().sum()) - 1
        acum = 0.0
        for j in range(ult_j, m - 1):
            Cij = proy.iloc[i, j]
            acum += sigma2[j] / f[j] ** 2 * (1 / Cij + 1 / sumC[j]) if Cij > 0 else 0
        se[i] = ultimo.iloc[i] * np.sqrt(acum)
    # total (Mack 1993): suma de varianzas + covarianzas por parámetro
    var_tot = np.sum(se ** 2)
    for i in range(n):
        for k in range(i + 1, n):
            ult_j = int(C.iloc[i].notna().sum()) - 1
            ult_k = int(C.iloc[k].notna().sum()) - 1
            ini = max(ult_j, ult_k)
            var_tot += 2 * ultimo.iloc[i] * ultimo.iloc[k] * sum(sigma2[j] / f[j] ** 2 / sumC[j] for j in range(ini, m - 1))
    tabla = pd.DataFrame({"pagado_actual": diag, "ultimo": ultimo, "reserva": reserva, "se_mack": se,
                          "cv": np.where(reserva > 0, se / reserva.replace(0, np.nan), np.nan)})
    return {"tabla": tabla, "factores": pd.Series(f, index=[f"{C.columns[j]}→{C.columns[j + 1]}" for j in range(m - 1)]),
            "reserva_total": float(reserva.sum()), "se_total": float(np.sqrt(var_tot)), "triangulo_completo": proy}


def _proyectar(Cb: np.ndarray, mask: np.ndarray):
    """Chain ladder vectorizado (solo factores y proyección) para el bootstrap."""
    n, m = Cb.shape
    P = Cb.copy()
    for j in range(m - 1):
        ok = mask[:, j] & mask[:, j + 1]
        den = Cb[ok, j].sum()
        if den <= 0:
            return None
        f = Cb[ok, j + 1].sum() / den
        falta = ~mask[:, j + 1]
        P[falta, j + 1] = P[falta, j] * f
    return P


def bootstrap_chain_ladder(triangulo, n_sim: int = 2000, acumulado: bool = True, semilla: int = 42) -> dict:
    """Bootstrap ODP (England-Verrall): remuestrea los residuos de Pearson del triángulo incremental, rehace el chain
    ladder y añade el error de proceso (gamma con la dispersión del ODP). Da la DISTRIBUCIÓN de la reserva total:
    media, desviación, percentiles 75/95/99.5 (el VaR de Solvencia II) y las simulaciones."""
    T = _triangulo(triangulo)
    C = T if acumulado else T.cumsum(axis=1).where(T.notna())
    I = C.diff(axis=1); I.iloc[:, 0] = C.iloc[:, 0]
    base = chain_ladder(C)
    f = base["factores"].to_numpy()
    n, m = C.shape
    # incrementales ajustados (hacia atrás desde la diagonal)
    ajust = pd.DataFrame(np.nan, index=C.index, columns=C.columns)
    for i in range(n):
        ult = int(C.iloc[i].notna().sum()) - 1
        acu = C.iloc[i, ult]
        for j in range(ult, -1, -1):
            ajust.iloc[i, j] = acu
            if j > 0:
                acu = acu / f[j - 1]
    inc_aj = ajust.diff(axis=1); inc_aj.iloc[:, 0] = ajust.iloc[:, 0]
    mask = I.notna().to_numpy()
    r = ((I - inc_aj) / np.sqrt(inc_aj.abs())).to_numpy()[mask]
    npar = n + m - 1
    phi = float(np.sum(r ** 2) / max(mask.sum() - npar, 1))
    r = r * np.sqrt(mask.sum() / max(mask.sum() - npar, 1))
    rng = np.random.default_rng(semilla)
    reservas = np.empty(n_sim)
    for s in range(n_sim):
        Ib = inc_aj.to_numpy().copy()
        Ib[mask] = inc_aj.to_numpy()[mask] + rng.choice(r, mask.sum()) * np.sqrt(np.abs(inc_aj.to_numpy()[mask]))
        Cb = np.where(mask, np.cumsum(np.where(mask, Ib, 0.0), axis=1), np.nan)
        proy = _proyectar(Cb, mask)
        if proy is None:
            reservas[s] = np.nan; continue
        fut = np.diff(proy, axis=1)[~mask[:, 1:]]
        fut = np.clip(fut, 1e-9, None)
        reservas[s] = np.sum(rng.gamma(fut / phi, phi))           # error de proceso ODP
    reservas = reservas[np.isfinite(reservas)]
    pct = {p: float(np.percentile(reservas, p)) for p in (50, 75, 95, 99.5)}
    return {"media": float(reservas.mean()), "desviacion": float(reservas.std(ddof=1)), "percentiles": pct,
            "reserva_chain_ladder": base["reserva_total"], "phi": phi, "simulaciones": reservas}


def recursion_panjer(frecuencia: str, parametros: dict, severidad, max_s: int | None = None) -> pd.DataFrame:
    """Distribución EXACTA de la siniestralidad agregada S = X1 + … + XN con severidad discreta (vector de probabilidades
    de 0, 1, 2… unidades monetarias) y N de la clase (a, b, 0): 'poisson' {lambda}, 'binomial_negativa' {r, p} o
    'binomial' {n, p}. Devuelve P(S = s), la función de distribución y los momentos. Discretiza antes la severidad."""
    f = np.asarray(severidad, float)
    if abs(f.sum() - 1) > 1e-6:
        raise ValueError("La severidad debe ser un vector de probabilidades que sume 1.")
    if frecuencia == "poisson":
        lam = parametros["lambda"]; a, b = 0.0, lam; g0 = np.exp(lam * (f[0] - 1)); media_n = lam
    elif frecuencia == "binomial_negativa":
        r, p = parametros["r"], parametros["p"]; a, b = 1 - p, (r - 1) * (1 - p); g0 = (p / (1 - (1 - p) * f[0])) ** r; media_n = r * (1 - p) / p
    elif frecuencia == "binomial":
        nn, p = parametros["n"], parametros["p"]; a, b = -p / (1 - p), (nn + 1) * p / (1 - p); g0 = (1 - p + p * f[0]) ** nn; media_n = nn * p
    else:
        raise ValueError("frecuencia: 'poisson', 'binomial_negativa' o 'binomial'")
    media_x = float(np.sum(np.arange(len(f)) * f))
    max_s = max_s or int(media_n * media_x * 8 + 10 * len(f))
    g = np.zeros(max_s + 1); g[0] = g0
    fx = np.r_[f, np.zeros(max_s + 1 - len(f))] if len(f) <= max_s else f[:max_s + 1]
    den = 1 - a * fx[0]
    for s in range(1, max_s + 1):
        j = np.arange(1, s + 1)
        g[s] = np.sum((a + b * j / s) * fx[j] * g[s - j]) / den
    out = pd.DataFrame({"s": np.arange(max_s + 1), "probabilidad": g, "acumulada": np.cumsum(g)}).set_index("s")
    out.attrs.update({"media": float(np.sum(out.index * g)), "masa_cubierta": float(g.sum())})
    return out


def simular_siniestralidad_agregada(frecuencia, severidad, n_sim: int = 100000, semilla: int = 42, niveles=(0.95, 0.99, 0.995)) -> dict:
    """Monte Carlo de S = Σ X_i: `frecuencia` y `severidad` son distribuciones congeladas de scipy (p. ej.
    stats.poisson(2), stats.lognorm(1.2, scale=1000)) o funciones (rng, tamaño) -> muestra. Devuelve media, desviación,
    asimetría, VaR y TVaR a los niveles pedidos y la muestra."""
    rng = np.random.default_rng(semilla)
    N = frecuencia.rvs(size=n_sim, random_state=rng) if hasattr(frecuencia, "rvs") else frecuencia(rng, n_sim)
    N = np.asarray(N, int)
    tot = int(N.sum())
    X = severidad.rvs(size=tot, random_state=rng) if hasattr(severidad, "rvs") else severidad(rng, tot)
    S = np.zeros(n_sim)
    np.add.at(S, np.repeat(np.arange(n_sim), N), X)
    var = {q: float(np.quantile(S, q)) for q in niveles}
    tvar = {q: float(S[S >= var[q]].mean()) for q in niveles}
    return {"media": float(S.mean()), "desviacion": float(S.std(ddof=1)), "asimetria": float(stats.skew(S)),
            "prob_cero": float((S == 0).mean()), "var": var, "tvar": tvar, "muestra": S}


def probabilidad_ruina(capital_inicial: float, recargo: float, media_siniestro: float, lambda_: float = 1.0,
                       severidad=None, horizonte: float | None = None, n_sim: int = 5000, semilla: int = 42) -> dict:
    """Modelo clásico de Cramér-Lundberg: U(t) = u + c·t − S(t), c = (1 + θ)·λ·μ.
    - Severidad exponencial: ψ(u) exacta = exp(−θu/((1+θ)μ))/(1+θ).
    - Cota de Lundberg ψ(u) ≤ exp(−R·u) con el coeficiente de ajuste R (resuelto numéricamente si se da la severidad).
    - Con `horizonte`, probabilidad de ruina en tiempo finito por simulación (severidad: scipy congelada o exponencial).
    Gauss: con colas pesadas (Pareto, lognormal) no existe R y la cota exponencial no vale."""
    u, th, mu = capital_inicial, recargo, media_siniestro
    out = {"capital_inicial": u, "recargo": th}
    out["psi_exponencial"] = float(np.exp(-th * u / ((1 + th) * mu)) / (1 + th))
    dist = severidad if severidad is not None else stats.expon(scale=mu)
    nombre = getattr(getattr(dist, "dist", None), "name", "")
    forma = getattr(dist, "args", ()) or tuple(getattr(dist, "kwds", {}).values())
    cola_pesada = nombre in {"lognorm", "pareto", "lomax", "burr", "burr12", "fisk", "genpareto", "cauchy", "t", "loglaplace",
                             "invweibull", "levy"} or (nombre == "weibull_min" and forma and forma[0] < 1)
    if severidad is None:
        R = th / ((1 + th) * mu)                                    # exacto con severidad exponencial
    elif cola_pesada:
        R = np.nan                                                  # sin función generatriz: no hay coeficiente de ajuste
    else:
        try:
            mgf = lambda r: dist.expect(lambda x: np.exp(r * x))   # noqa: E731
            hi = 1.0 / mu
            while np.isfinite(mgf(hi)) and 1 + (1 + th) * mu * hi - mgf(hi) > 0 and hi < 100 / mu:
                hi *= 2
            g = lambda r: 1 + (1 + th) * mu * r - mgf(r)            # noqa: E731
            rejilla = np.linspace(hi / 400, hi, 400)
            valores = np.array([g(r) for r in rejilla])
            cambio = np.where((valores[:-1] > 0) & (valores[1:] <= 0))[0]
            R = optimize.brentq(g, rejilla[cambio[0]], rejilla[cambio[0] + 1]) if len(cambio) else np.nan
        except Exception:  # noqa: BLE001  (colas pesadas: no hay coeficiente de ajuste)
            R = np.nan
    out["coef_ajuste"] = float(R); out["cota_lundberg"] = float(np.exp(-R * u)) if np.isfinite(R) else np.nan
    if horizonte:
        rng = np.random.default_rng(semilla); c = (1 + th) * lambda_ * mu; ruinas = 0
        for _ in range(n_sim):
            t, U = 0.0, u
            while True:
                w = rng.exponential(1 / lambda_); t += w
                if t > horizonte:
                    break
                U += c * w - float(dist.rvs(random_state=rng))
                if U < 0:
                    ruinas += 1; break
        out["psi_horizonte"] = ruinas / n_sim; out["horizonte"] = horizonte
    return out


def prima_por_principios(x=None, distribucion=None, parametros: dict | None = None, n_sim: int = 200000, semilla: int = 42) -> pd.DataFrame:
    """Primas según los principios clásicos sobre una pérdida X (muestra `x` o distribución de scipy):
    valor esperado (1+θ)E, varianza E + α·Var, desviación típica E + β·σ, exponencial (1/a)·log E[e^{aX}],
    Esscher E[X e^{hX}]/E[e^{hX}] y percentil (VaR). `parametros` = {theta, alfa, beta, a, h, p}; por defecto
    α = a = h = 0.1/E[X] (en unidades de la pérdida). OJO: con colas pesadas (lognormal, Pareto) la función generatriz
    no existe: los principios exponencial y de Esscher dependen de la muestra y no son fiables."""
    v = np.asarray(x, float) if x is not None else distribucion.rvs(size=n_sim, random_state=np.random.default_rng(semilla))
    E, Var = v.mean(), v.var(ddof=1)
    p = {"theta": 0.1, "alfa": 0.1 / E, "beta": 0.2, "a": 0.1 / E, "h": 0.1 / E, "p": 0.95, **(parametros or {})}
    with np.errstate(over="ignore"):
        ea = np.mean(np.exp(p["a"] * v)); eh = np.exp(p["h"] * v)
    filas = [("prima pura E[X]", E), ("valor esperado (θ)", (1 + p["theta"]) * E), ("varianza (α)", E + p["alfa"] * Var),
             ("desviación típica (β)", E + p["beta"] * np.sqrt(Var)), ("exponencial (a)", np.log(ea) / p["a"] if np.isfinite(ea) else np.inf),
             ("Esscher (h)", np.mean(v * eh) / np.mean(eh) if np.isfinite(eh).all() else np.inf), (f"percentil {p['p']:.0%}", np.quantile(v, p["p"]))]
    t = pd.DataFrame(filas, columns=["principio", "prima"]).set_index("principio")
    t["recargo_pct"] = (t.prima / E - 1) * 100
    return t


def credibilidad_buhlmann(df: pd.DataFrame, riesgo: str, valor: str, exposicion: str | None = None) -> dict:
    """Credibilidad de Bühlmann-Straub: prima de cada riesgo = Z·(media propia) + (1−Z)·(media colectiva),
    Z = w/(w + k), k = EPV/VHM. Estimadores no paramétricos de la media colectiva μ, la esperanza de la varianza del
    proceso (EPV) y la varianza de las medias hipotéticas (VHM). `exposicion` = pesos (pólizas-año); sin ella, todos 1.
    Si VHM ≤ 0 (no hay heterogeneidad detectable) todos los Z = 0. Es el «encogimiento» de los modelos mixtos."""
    d = df[[riesgo, valor] + ([exposicion] if exposicion else [])].dropna().copy()
    d["_w"] = d[exposicion] if exposicion else 1.0
    g = d.groupby(riesgo)
    w_i = g["_w"].sum(); n_i = g.size()
    xbar_i = g.apply(lambda s: np.sum(s["_w"] * s[valor]) / s["_w"].sum(), include_groups=False)
    w = w_i.sum(); xbar = float(np.sum(w_i * xbar_i) / w)
    epv = float(sum(np.sum(s["_w"] * (s[valor] - xbar_i[k]) ** 2) for k, s in g) / (n_i - 1).clip(lower=0).sum())
    r = len(w_i)
    vhm = float((np.sum(w_i * (xbar_i - xbar) ** 2) - (r - 1) * epv) / (w - np.sum(w_i ** 2) / w))
    if vhm <= 0:
        Z = pd.Series(0.0, index=w_i.index); k = np.inf
    else:
        k = epv / vhm; Z = w_i / (w_i + k)
    mu = float(np.sum(Z * xbar_i) / Z.sum()) if Z.sum() > 0 else xbar
    prima = Z * xbar_i + (1 - Z) * mu
    t = pd.DataFrame({"exposicion": w_i, "media_propia": xbar_i, "credibilidad_Z": Z, "prima_credibilidad": prima})
    return {"tabla": t, "media_colectiva": mu, "epv": epv, "vhm": vhm, "k": k}
