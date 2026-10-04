"""Regresión: inferencias simultáneas (Working-Hotelling, Bonferroni), regresión inversa (calibración), regresión por
el origen, errores de medida en X (calibración de la regresión y SIMEX), polinómica con grado elegido, tamaño muestral
mínimo para un modelo (Riley), regresión local (LOESS y Nadaraya-Watson), funciones escalonadas, MARS, regresión
inversa por cortes (SIR), PCR y PLS, y modelos loglineales para tablas de contingencia."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize, stats

from .._util import _columnas, _numerico


def _xy(x, y):
    X, Y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(X) & np.isfinite(Y)
    if ok.sum() < 4:
        raise ValueError("Hacen falta al menos 4 pares (x, y) válidos.")
    return X[ok], Y[ok]


def bandas_confianza_regresion(x, y, x_nuevos=None, nivel: float = 0.95, metodo: str = "working_hotelling") -> pd.DataFrame:
    """Bandas de confianza para la recta de regresión simple E[y|x] en varios puntos a la vez: ``'puntual'`` (cada IC
    por separado: la confianza conjunta es menor), ``'bonferroni'`` (t con α/g, para g puntos fijados de antemano) o
    ``'working_hotelling'`` (W = √(2F₂,ₙ₋₂): válida para TODA la recta a la vez). Incluye el intervalo de predicción de
    una observación nueva (más ancho: suma la varianza del error). Equivale a: PROC REG con CLM/CLI."""
    X, Y = _xy(x, y)
    n = len(X)
    m = sm.OLS(Y, sm.add_constant(X)).fit()
    xn = np.linspace(X.min(), X.max(), 50) if x_nuevos is None else np.atleast_1d(np.asarray(x_nuevos, float))
    g = len(xn)
    yh = m.params[0] + m.params[1] * xn
    s2 = m.mse_resid; sxx = np.sum((X - X.mean()) ** 2)
    se = np.sqrt(s2 * (1 / n + (xn - X.mean()) ** 2 / sxx))
    a = 1 - nivel
    if metodo == "puntual":
        c = stats.t.ppf(1 - a / 2, n - 2)
    elif metodo == "bonferroni":
        c = stats.t.ppf(1 - a / (2 * g), n - 2)
    elif metodo == "working_hotelling":
        c = np.sqrt(2 * stats.f.ppf(nivel, 2, n - 2))
    else:
        raise ValueError("metodo: 'puntual', 'bonferroni' o 'working_hotelling'")
    sp = np.sqrt(s2 + se ** 2); tp = stats.t.ppf(1 - a / 2, n - 2)
    return pd.DataFrame({"x": xn, "ajuste": yh, "banda_inf": yh - c * se, "banda_sup": yh + c * se,
                         "prediccion_inf": yh - tp * sp, "prediccion_sup": yh + tp * sp, "multiplicador": c})


def regresion_inversa(x, y, y0, m_replicas: int = 1, nivel: float = 0.95) -> dict:
    """Calibración (regresión inversa): ajustas y = a + b·x con patrones de x conocido y, para una nueva lectura y0
    (media de m réplicas), estimas x0 = (y0 − a)/b con el IC de Fieller (puede ser no acotado si la pendiente no es
    claramente distinta de 0: g ≥ 1). Típico: curvas patrón de laboratorio, convertir una medida barata en la cara."""
    X, Y = _xy(x, y)
    n = len(X)
    m = sm.OLS(Y, sm.add_constant(X)).fit()
    a, b = m.params
    s2 = m.mse_resid; sxx = np.sum((X - X.mean()) ** 2)
    t = stats.t.ppf(0.5 + nivel / 2, n - 2)
    x0 = (y0 - a) / b
    g = t ** 2 * s2 / (b ** 2 * sxx)
    out = {"x0": float(x0), "pendiente": float(b), "ordenada": float(a), "g": float(g)}
    if g >= 1:
        out["ic"] = (-np.inf, np.inf); out["aviso"] = "g ≥ 1: la pendiente no está bien determinada; el IC de Fieller no es acotado."
        return out
    d = x0 - X.mean()
    c = t * np.sqrt(s2) / b * np.sqrt((1 - g) * (1 / m_replicas + 1 / n) + d ** 2 / sxx)
    centro = X.mean() + d / (1 - g)
    lo, hi = sorted([centro - c / (1 - g), centro + c / (1 - g)])
    out["ic"] = (float(lo), float(hi))
    out["ee_aprox_delta"] = float(np.sqrt(s2) / abs(b) * np.sqrt(1 / m_replicas + 1 / n + d ** 2 / sxx))
    return out


def regresion_por_origen(x, y) -> dict:
    """Regresión sin ordenada (y = b·x): pendiente, EE, IC y contraste de que la ordenada sea 0 en el modelo completo.
    OJO: el R² sin constante se calcula respecto a 0 (Σy²) y NO es comparable con el R² habitual (sale inflado);
    úsala solo si la teoría impone que x = 0 ⇒ y = 0 (y los datos llegan cerca del origen)."""
    X, Y = _xy(x, y)
    m0 = sm.OLS(Y, X).fit(); m1 = sm.OLS(Y, sm.add_constant(X)).fit()
    ci = m0.conf_int()[0]
    return {"pendiente": float(m0.params[0]), "ee": float(m0.bse[0]), "ic": (float(ci[0]), float(ci[1])),
            "r2_no_centrado": float(m0.rsquared), "r2_con_ordenada": float(m1.rsquared),
            "ordenada_modelo_completo": float(m1.params[0]), "p_ordenada": float(m1.pvalues[0]),
            "recomendacion": "por el origen" if m1.pvalues[0] > 0.05 else "mantén la ordenada (p < 0.05)"}


def correccion_error_medida(x_observada, y, fiabilidad: float | None = None, var_error: float | None = None,
                            metodo: str = "simex", lambdas=(0.5, 1.0, 1.5, 2.0), n_sim: int = 100, semilla: int = 42) -> dict:
    """Errores de medida clásicos en el predictor (X_obs = X + U): la pendiente MCO se atenúa por la fiabilidad
    λ = var(X)/var(X_obs). Corrige con ``'calibracion'`` (pendiente / fiabilidad) o ``'simex'`` (añade más error
    simulado, ve cómo empeora la pendiente y extrapola cuadráticamente a «error 0»). Necesita la fiabilidad o la
    varianza del error (de réplicas o de un estudio de validación)."""
    X, Y = _xy(x_observada, y)
    vx = X.var(ddof=1)
    if var_error is None:
        if fiabilidad is None or not 0 < fiabilidad <= 1:
            raise ValueError("Da la fiabilidad (0, 1] o la varianza del error de medida.")
        var_error = (1 - fiabilidad) * vx
    fiab = 1 - var_error / vx
    ingenua = float(np.polyfit(X, Y, 1)[0])
    if metodo == "calibracion":
        return {"pendiente_ingenua": ingenua, "pendiente_corregida": ingenua / fiab, "fiabilidad": float(fiab)}
    if metodo != "simex":
        raise ValueError("metodo: 'calibracion' o 'simex'")
    rng = np.random.default_rng(semilla)
    lam = np.r_[0.0, np.asarray(lambdas, float)]
    pend = [ingenua] + [np.mean([np.polyfit(X + rng.normal(0, np.sqrt(l * var_error), len(X)), Y, 1)[0] for _ in range(n_sim)]) for l in lam[1:]]
    c = np.polyfit(lam, pend, 2)
    return {"pendiente_ingenua": ingenua, "pendiente_corregida": float(np.polyval(c, -1)), "fiabilidad": float(fiab),
            "curva_simex": pd.Series(pend, index=lam, name="pendiente"), "pendiente_calibracion": ingenua / fiab}


def regresion_polinomica(x, y, grado_max: int = 6, criterio: str = "f_secuencial", alfa: float = 0.05, cv: int = 10,
                         semilla: int = 42) -> dict:
    """Regresión polinómica con polinomios ORTOGONALES (sin colinealidad entre potencias) eligiendo el grado por
    contrastes F secuenciales (se para en el primer término no significativo) o por validación cruzada (``'cv'``).
    Devuelve la tabla por grado (R², R² ajustado, AIC, p del término añadido, ECM de CV) y el grado elegido.
    Gauss: los polinomios de grado alto se disparan en los extremos; si hace falta grado > 3 usa splines."""
    X, Y = _xy(x, y)
    n = len(X)
    grado_max = min(grado_max, n - 2)
    Q = np.linalg.qr(np.vander((X - X.mean()) / X.std(), grado_max + 1, increasing=True))[0]
    rng = np.random.default_rng(semilla); folds = rng.permutation(n) % cv
    filas, prev = [], None
    for g in range(1, grado_max + 1):
        m = sm.OLS(Y, Q[:, :g + 1]).fit()
        p = float(m.pvalues[g])
        ecm = np.mean([np.mean((Y[folds == f] - Q[folds == f, :g + 1] @ np.linalg.lstsq(Q[folds != f, :g + 1], Y[folds != f], rcond=None)[0]) ** 2) for f in range(cv)])
        filas.append({"grado": g, "r2": m.rsquared, "r2_ajustado": m.rsquared_adj, "aic": m.aic, "p_termino": p, "ecm_cv": ecm})
    t = pd.DataFrame(filas).set_index("grado")
    if criterio == "cv":
        g = int(t.ecm_cv.idxmin())
    elif criterio == "f_secuencial":
        no = t.index[t.p_termino > alfa]
        g = int(no[0] - 1) if len(no) else grado_max
        g = max(g, 1)
    else:
        raise ValueError("criterio: 'f_secuencial' o 'cv'")
    coef = np.polyfit(X, Y, g)
    return {"tabla": t, "grado": g, "coeficientes_crudos": coef, "predecir": lambda v: np.polyval(coef, np.asarray(v, float))}


def tamano_muestral_modelo(n_parametros: int, tipo: str = "binario", prevalencia: float | None = None, r2: float | None = None,
                           contraccion: float = 0.9, delta_r2: float = 0.05, sd_y: float | None = None, margen_media: float | None = None) -> pd.DataFrame:
    """Tamaño muestral mínimo para DESARROLLAR un modelo de predicción (Riley et al. 2019-2020) según los grados de
    libertad que se gastan (`n_parametros`, contando dummies, splines e interacciones):
    1) contracción esperada ≥ `contraccion` (sobreajuste pequeño), 2) optimismo del R² ≤ `delta_r2`, 3) estimar con
    precisión el riesgo medio (binario: ±0.05) o la media (continuo). `r2` = R² de Cox-Snell esperado (binario) o R²
    (continuo). Devuelve n por criterio, el máximo y los eventos por parámetro (EPP) resultantes — compárese con la
    vieja regla de 10 EPP, que a menudo se queda corta."""
    p = int(n_parametros)
    if r2 is None or not 0 < r2 < 1:
        raise ValueError("Indica r2 esperado en (0, 1) (Cox-Snell si binario; de estudios previos o conservador ≈ 0.15·máx).")
    filas = []
    if tipo == "binario":
        if prevalencia is None:
            raise ValueError("tipo binario necesita la prevalencia.")
        n1 = p / ((contraccion - 1) * np.log(1 - r2 / contraccion))
        lnLnull = prevalencia * np.log(prevalencia) + (1 - prevalencia) * np.log(1 - prevalencia)
        r2max = 1 - np.exp(2 * lnLnull)
        s2 = r2 / (r2 + delta_r2 * r2max)
        n2 = p / ((s2 - 1) * np.log(1 - r2 / s2))
        n3 = (1.96 / 0.05) ** 2 * prevalencia * (1 - prevalencia)
        filas = [("1) contracción ≥ %.2f" % contraccion, n1), ("2) optimismo R² ≤ %.2f" % delta_r2, n2), ("3) precisión del riesgo medio ±0.05", n3)]
    elif tipo == "continuo":
        n1 = p / ((contraccion - 1) * np.log(1 - r2 / contraccion))       # misma fórmula con R² de la regresión
        n2 = p + 1 + p * (1 - r2) / delta_r2                         # R² aparente − ajustado ≤ δ
        filas = [("1) contracción ≥ %.2f" % contraccion, n1), ("2) R² ajustado ≈ R² (diferencia ≤ %.2f)" % delta_r2, n2)]
        if sd_y and margen_media:
            filas.append(("3) precisión de la media", (1.96 * sd_y / margen_media) ** 2))
    else:
        raise ValueError("tipo: 'binario' o 'continuo'")
    t = pd.DataFrame(filas, columns=["criterio", "n"]).set_index("criterio")
    t["n"] = np.ceil(t.n).astype(int)
    nmax = int(t.n.max())
    t.attrs.update({"n_minimo": nmax, "eventos_por_parametro": (nmax * min(prevalencia, 1 - prevalencia) / p) if tipo == "binario" else nmax / p})
    return t


def regresion_local(x, y, metodo: str = "loess", fraccion: float | None = None, ancho: float | None = None, x_nuevos=None) -> dict:
    """Regresión no paramétrica en una variable: ``'loess'`` (regresión lineal local ponderada con vecindario =
    `fraccion` de los datos, robusta a atípicos) o ``'nadaraya_watson'`` (media local con núcleo gaussiano de ancho
    `ancho`). Si no das el parámetro de suavizado se elige por validación cruzada dejando uno fuera.
    Gauss: muestra la forma de la relación sin suponerla; no extrapola y sufre en los bordes y con muchas X."""
    X, Y = _xy(x, y)
    o = np.argsort(X); X, Y = X[o], Y[o]
    xn = np.linspace(X.min(), X.max(), 200) if x_nuevos is None else np.asarray(x_nuevos, float)
    if metodo == "loess":
        from statsmodels.nonparametric.smoothers_lowess import lowess

        delta = 0.005 * np.ptp(X) if len(X) > 2000 else 0.0            # interpolación lineal: O(n) con n grande
        sub = np.sort(np.random.default_rng(0).choice(len(X), 2000, replace=False)) if len(X) > 2000 else np.arange(len(X))
        Xs, Ys = X[sub], Y[sub]

        def cv(fr):
            r = []
            for i in np.arange(len(Xs))[:: max(1, len(Xs) // 150)]:
                m = np.ones(len(Xs), bool); m[i] = False
                r.append(Ys[i] - lowess(Ys[m], Xs[m], frac=fr, it=1, xvals=np.array([Xs[i]]))[0])
            return np.nanmean(np.square(r))
        if fraccion is None:
            grid = np.array([0.15, 0.25, 0.35, 0.5, 0.65, 0.8])
            errs = [cv(f) for f in grid]; fraccion = float(grid[int(np.nanargmin(errs))])
        yh = lowess(Y, X, frac=fraccion, it=3, delta=delta, xvals=xn) if delta == 0 else np.interp(xn, *lowess(Y, X, frac=fraccion, it=3, delta=delta).T)
        return {"x": xn, "ajuste": yh, "parametro": fraccion, "metodo": metodo}
    if metodo == "nadaraya_watson":
        def nw(xe, h, excl=False):
            K = np.exp(-0.5 * ((xe[:, None] - X[None, :]) / h) ** 2)
            if excl:
                np.fill_diagonal(K, 0)
            return (K @ Y) / K.sum(axis=1)
        if ancho is None:                                             # LOO-CV (en una submuestra de 2000 si n es grande)
            k = min(len(X), 2000)
            sub = np.sort(np.random.default_rng(0).choice(len(X), k, replace=False))
            Xs, Ys = X[sub], Y[sub]
            h0 = 1.06 * Xs.std() * k ** -0.2
            grid = h0 * np.array([0.2, 0.35, 0.5, 0.75, 1, 1.5, 2])

            def loo(h):
                K = np.exp(-0.5 * ((Xs[:, None] - Xs[None, :]) / h) ** 2); np.fill_diagonal(K, 0)
                return np.mean((Ys - (K @ Ys) / K.sum(axis=1)) ** 2)
            ancho = float(grid[int(np.argmin([loo(h) for h in grid]))] * (k / len(X)) ** 0.2)
        ajuste = np.concatenate([nw(xn[i:i + 50], ancho) for i in range(0, len(xn), 50)])
        return {"x": xn, "ajuste": ajuste, "parametro": ancho, "metodo": metodo}
    raise ValueError("metodo: 'loess' o 'nadaraya_watson'")


def funciones_escalonadas(df: pd.DataFrame, y: str, x: str, cortes=5, familia: str = "gaussiana") -> dict:
    """Regresión con funciones escalonadas: parte x en tramos (nº de cuantiles o cortes explícitos) y ajusta una
    constante por tramo (GLM con la variable tramificada). Fácil de explicar (tarifas por tramos), pero discontinua y
    pierde información dentro de cada tramo; compara su AIC con el lineal y con splines."""
    _columnas(df, [y, x])
    d = df[[y, x]].dropna().copy()
    d["_tramo"] = pd.qcut(d[x], cortes, duplicates="drop") if np.ndim(cortes) == 0 else pd.cut(d[x], cortes, include_lowest=True)
    fam = {"gaussiana": sm.families.Gaussian(), "binomial": sm.families.Binomial(), "poisson": sm.families.Poisson()}[familia]
    m = smf.glm(f'Q("{y}") ~ C(_tramo)', d, family=fam).fit()
    ml = smf.glm(f'Q("{y}") ~ Q("{x}")', d, family=fam).fit()
    tab = d.groupby("_tramo", observed=True)[y].agg(["count", "mean"]).rename(columns={"count": "n", "mean": "media"})
    return {"tabla": tab, "aic_escalonado": float(m.aic), "aic_lineal": float(ml.aic), "modelo": m}


def ajustar_mars(X, y, max_terminos: int = 21, penalizacion: float = 3.0, max_nudos: int = 30) -> dict:
    """MARS aditivo (Friedman 1991, grado 1): construye funciones bisagra max(0, x − t) y max(0, t − x) por pares en
    una pasada hacia delante (el par que más reduce el error) y poda hacia atrás con el GCV
    (penalización `penalizacion` por nudo). Devuelve los términos con sus coeficientes, el GCV, R² y una función para
    predecir. Útil para relaciones con codos (umbrales) y como alternativa interpretable a los árboles."""
    Xd = pd.DataFrame(X).astype(float); Y = np.asarray(y, float)
    n, p = Xd.shape
    if n < 20:
        raise ValueError("MARS necesita al menos 20 observaciones.")
    A = Xd.to_numpy()
    base = [np.ones(n)]; terms = [("(intercepto)", None, None, 0)]

    def sse(B):
        b = np.linalg.lstsq(np.column_stack(B), Y, rcond=None)[0]
        r = Y - np.column_stack(B) @ b
        return r @ r
    while len(base) + 2 <= max_terminos:
        mejor = (sse(base), None)
        for j in range(p):
            cand = np.unique(np.quantile(A[:, j], np.linspace(0.05, 0.95, max_nudos)))
            for t in cand:
                h1, h2 = np.maximum(0, A[:, j] - t), np.maximum(0, t - A[:, j])
                s = sse(base + [h1, h2])
                if s < mejor[0] - 1e-12:
                    mejor = (s, (j, t, h1, h2))
        if mejor[1] is None:
            break
        j, t, h1, h2 = mejor[1]
        base += [h1, h2]; terms += [(f"max(0, {Xd.columns[j]} - {t:.4g})", j, t, 1), (f"max(0, {t:.4g} - {Xd.columns[j]})", j, t, -1)]

    def gcv(idx):
        B = [base[i] for i in idx]; M = len(idx) + penalizacion * (len(idx) - 1) / 2
        return sse(B) / n / (1 - M / n) ** 2 if M < n else np.inf
    actual = list(range(len(base))); mejor_idx, mejor_g = actual[:], gcv(actual)
    while len(actual) > 1:
        opciones = [(gcv([i for i in actual if i != k]), k) for k in actual if k != 0]
        g, k = min(opciones)
        actual = [i for i in actual if i != k]
        if g <= mejor_g:
            mejor_g, mejor_idx = g, actual[:]
    B = np.column_stack([base[i] for i in mejor_idx])
    b = np.linalg.lstsq(B, Y, rcond=None)[0]
    sel = [terms[i] for i in mejor_idx]

    def predecir(Xn):
        An = pd.DataFrame(Xn).astype(float).to_numpy()
        cols = [np.ones(len(An)) if tr[1] is None else (np.maximum(0, An[:, tr[1]] - tr[2]) if tr[3] == 1 else np.maximum(0, tr[2] - An[:, tr[1]])) for tr in sel]
        return np.column_stack(cols) @ b
    r2 = 1 - sse([base[i] for i in mejor_idx]) / np.sum((Y - Y.mean()) ** 2)
    return {"terminos": pd.DataFrame({"termino": [t[0] for t in sel], "coeficiente": b}), "gcv": float(mejor_g), "r2": float(r2),
            "predecir": predecir}


def regresion_inversa_cortes(X, y, n_cortes: int = 10, n_direcciones: int = 2) -> dict:
    """Regresión inversa por cortes (SIR, Li 1991): reducción de dimensión SUFICIENTE para y = f(β₁ᵀx, …, β_kᵀx, ε)
    sin suponer f. Estandariza X, corta y en `n_cortes` tramos, hace PCA de las medias de X por tramo; los primeros
    autovectores (retransformados) son las direcciones. Autovalores grandes = direcciones relevantes.
    Gauss: no detecta relaciones simétricas (y = x²) — para eso, SAVE o projection pursuit."""
    Xd = pd.DataFrame(X).astype(float); Y = np.asarray(y, float)
    A = Xd.to_numpy(); n, p = A.shape
    mu = A.mean(axis=0); S = np.cov(A, rowvar=False)
    w, V = np.linalg.eigh(S)
    if w.min() <= 1e-12:
        raise ValueError("La covarianza de X es singular: quita variables colineales.")
    Sm12 = V @ np.diag(w ** -0.5) @ V.T
    Z = (A - mu) @ Sm12
    corte = pd.qcut(pd.Series(Y).rank(method="first"), n_cortes, labels=False).to_numpy()
    M = np.zeros((p, p))
    for h in range(n_cortes):
        m = corte == h
        zh = Z[m].mean(axis=0); M += m.mean() * np.outer(zh, zh)
    lam, E = np.linalg.eigh(M)
    o = np.argsort(lam)[::-1]; lam, E = lam[o], E[:, o]
    B = Sm12 @ E[:, :n_direcciones]
    B /= np.linalg.norm(B, axis=0)
    dirs = pd.DataFrame(B, index=Xd.columns, columns=[f"dir{i + 1}" for i in range(n_direcciones)])
    return {"direcciones": dirs, "autovalores": pd.Series(lam, index=[f"dir{i + 1}" for i in range(p)]),
            "proyecciones": pd.DataFrame((A - mu) @ B, columns=dirs.columns)}


def regresion_pls_pcr(X, y, metodo: str = "pls", max_componentes: int | None = None, cv: int = 10, semilla: int = 42) -> dict:
    """Regresión por componentes principales (PCR: componentes de X ignorando y) o mínimos cuadrados parciales (PLS:
    componentes que maximizan la covarianza con y; suele necesitar menos). Elige el nº de componentes por validación
    cruzada (RMSE) y devuelve la curva, los coeficientes en la escala original y la varianza de X explicada.
    Para muchas X colineales (espectros, ratios financieros). Equivale a: PROC PLS (METHOD=PLS / PCR)."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xd = pd.DataFrame(X).astype(float); Y = np.asarray(y, float)
    kmax = min(max_componentes or Xd.shape[1], Xd.shape[1], len(Y) - 2)

    def modelo(k):
        if metodo == "pls":
            return make_pipeline(StandardScaler(), PLSRegression(n_components=k, scale=False))
        if metodo == "pcr":
            return make_pipeline(StandardScaler(), PCA(n_components=k), LinearRegression())
        raise ValueError("metodo: 'pls' o 'pcr'")
    kf = KFold(cv, shuffle=True, random_state=semilla)
    curva = pd.Series({k: float(np.sqrt(-cross_val_score(modelo(k), Xd, Y, cv=kf, scoring="neg_mean_squared_error").mean())) for k in range(1, kmax + 1)},
                      name="rmse_cv")
    k = int(curva.idxmin())
    m = modelo(k).fit(Xd, Y)
    sc = m[0]
    if metodo == "pls":
        coef = np.ravel(m[-1].coef_) / sc.scale_
        var_x = float(np.sum(np.var(m[-1].x_scores_, axis=0)) / Xd.shape[1])
    else:
        coef = (m[1].components_.T @ m[2].coef_) / sc.scale_
        var_x = float(m[1].explained_variance_ratio_.sum())
    return {"n_componentes": k, "curva_cv": curva, "coeficientes": pd.Series(coef, index=Xd.columns), "var_x_explicada": var_x,
            "modelo": m}


def modelo_loglineal(datos, factores=None, frecuencia: str | None = None, modelo: str = "independencia") -> dict:
    """Modelo loglineal (Poisson) para una tabla de contingencia: log μ = términos de los factores y sus interacciones.
    ``modelo``: 'independencia' (solo efectos principales), 'saturado', 'dobles' (todas las interacciones de orden 2)
    o una fórmula con la parte derecha ('A*B + C'). Devuelve G² (desviación) frente al saturado con su p (¿basta este
    modelo?), AIC, frecuencias ajustadas y la comparación jerárquica con el de independencia.
    Equivale a: PROC CATMOD LOGLIN / PROC GENMOD DIST=POISSON. `datos`: DataFrame de casos (una fila por individuo) o
    de frecuencias (`frecuencia`)."""
    d = pd.DataFrame(datos)
    if frecuencia is None:
        if factores is None:
            raise ValueError("Con datos de casos indica los factores.")
        tab = d.groupby(list(factores), observed=False).size().rename("_n").reset_index()
        frecuencia = "_n"
    else:
        tab = d.copy()
    f = list(factores) if factores is not None else [c for c in tab.columns if c != frecuencia]
    tab[f] = tab[f].astype(str)
    q = [f'Q("{c}")' for c in f]
    if modelo == "independencia":
        rhs = " + ".join(q)
    elif modelo == "saturado":
        rhs = " * ".join(q)
    elif modelo == "dobles":
        rhs = " + ".join(q + [f"{a}:{b}" for a, b in itertools.combinations(q, 2)])
    else:
        rhs = modelo                                                  # fórmula con los nombres de las columnas
    import warnings
    with warnings.catch_warnings():                                  # el saturado ajusta exacto (0 gl): avisos esperables
        warnings.simplefilter("ignore")
        m = smf.glm(f'Q("{frecuencia}") ~ {rhs}', tab, family=sm.families.Poisson()).fit()
    ind = smf.glm(f'Q("{frecuencia}") ~ ' + " + ".join(q), tab, family=sm.families.Poisson()).fit()
    G2, gl = float(m.deviance), int(m.df_resid)
    out = {"G2": G2, "gl": gl, "p_valor": float(stats.chi2.sf(G2, gl)) if gl > 0 else np.nan, "aic": float(m.aic),
           "ajustadas": tab.assign(ajustada=m.fittedvalues), "modelo": m}
    if modelo != "independencia":
        dG = float(ind.deviance - m.deviance); dgl = int(ind.df_resid - m.df_resid)
        out["frente_a_independencia"] = {"delta_G2": dG, "gl": dgl, "p_valor": float(stats.chi2.sf(dG, dgl)) if dgl > 0 else np.nan}
    return out
