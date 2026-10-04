"""Inferencia causal y ensayos: puntuación de propensión (IPW, emparejamiento, estratificación con diagnóstico de
balance), diferencias en diferencias, aleatorización por bloques, análisis por intención de tratar frente a por
protocolo y efecto en cumplidores (switchers), mediación y moderación, y metaanálisis de efectos fijos y aleatorios."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from .._util import _columnas


def _smd(x, t, w=None):
    w = np.ones(len(x)) if w is None else w
    m1 = np.average(x[t == 1], weights=w[t == 1]); m0 = np.average(x[t == 0], weights=w[t == 0])
    s = np.sqrt((x[t == 1].var() + x[t == 0].var()) / 2)
    return (m1 - m0) / s if s > 0 else 0.0


def puntuacion_propension(df: pd.DataFrame, tratamiento: str, resultado: str, covariables, metodo: str = "ipw",
                          estimando: str = "ate", recorte: float = 0.01, n_boot: int = 200, semilla: int = 42) -> dict:
    """Efecto causal de un tratamiento binario en datos OBSERVACIONALES ajustando por confusores mediante la puntuación
    de propensión e(x) = P(T=1 | x) (logística). Métodos:
    - ``'ipw'``: ponderación por el inverso de la probabilidad (estabilizada; ATE o ATT), con recorte de e(x);
    - ``'emparejamiento'``: vecino más cercano 1:1 con reemplazo sobre logit(e) con caliper 0.2·sd (estima ATT);
    - ``'estratificacion'``: quintiles de e(x) (elimina ~90 % del sesgo por confusores observados).
    Devuelve el efecto con IC bootstrap, el efecto ingenuo (diferencia bruta) y la tabla de BALANCE (diferencia de
    medias estandarizada antes/después: |SMD| < 0.1 = equilibrado) y el solapamiento.
    Gauss: solo corrige confusores MEDIDOS; sin solapamiento (e ≈ 0 o 1) no hay comparación posible."""
    cov = [covariables] if isinstance(covariables, str) else list(covariables)
    _columnas(df, [tratamiento, resultado] + cov)
    d = df[[tratamiento, resultado] + cov].dropna().reset_index(drop=True)
    X = pd.get_dummies(d[cov], drop_first=True, dtype=float)
    t = d[tratamiento].astype(int).to_numpy(); y = d[resultado].to_numpy(float)
    if set(np.unique(t)) != {0, 1}:
        raise ValueError("tratamiento debe ser 0/1 con ambos grupos presentes.")
    if metodo not in ("ipw", "emparejamiento", "estratificacion"):
        raise ValueError("metodo: 'ipw', 'emparejamiento' o 'estratificacion'")

    def estimar(idx):
        Xi, ti, yi = X.iloc[idx], t[idx], y[idx]
        e = sm.GLM(ti, sm.add_constant(Xi, has_constant="add"), family=sm.families.Binomial()).fit().predict()
        e = np.clip(e, recorte, 1 - recorte)
        if metodo == "ipw":
            if estimando == "att":
                w = np.where(ti == 1, 1, e / (1 - e))
            else:
                w = np.where(ti == 1, ti.mean() / e, (1 - ti.mean()) / (1 - e))
            ef = np.average(yi[ti == 1], weights=w[ti == 1]) - np.average(yi[ti == 0], weights=w[ti == 0])
        elif metodo == "emparejamiento":
            lg = np.log(e / (1 - e)); cal = 0.2 * lg.std()
            tr, co = np.where(ti == 1)[0], np.where(ti == 0)[0]
            orden = np.argsort(lg[co]); lc = lg[co][orden]
            pos = np.clip(np.searchsorted(lc, lg[tr]), 1, len(lc) - 1)
            izq, der = lc[pos - 1], lc[pos]
            j = np.where(np.abs(lg[tr] - izq) <= np.abs(der - lg[tr]), pos - 1, pos)
            ok = np.abs(lc[j] - lg[tr]) <= cal
            pareja = co[orden][j]
            ef = np.mean(yi[tr[ok]] - yi[pareja[ok]])
            w = np.zeros(len(ti)); w[tr[ok]] = 1; np.add.at(w, pareja[ok], 1)
        else:
            q = pd.qcut(e, 5, labels=False, duplicates="drop")
            efs, pesos = [], []
            for s in np.unique(q):
                m = q == s
                if 0 < ti[m].sum() < m.sum():
                    efs.append(yi[m & (ti == 1)].mean() - yi[m & (ti == 0)].mean())
                    pesos.append(m.sum() if estimando == "ate" else (m & (ti == 1)).sum())
            ef = np.average(efs, weights=pesos)
            w = np.ones(len(ti))
            for s in np.unique(q):
                m = q == s
                for g in (0, 1):
                    mm = m & (ti == g)
                    if mm.any():
                        w[mm] = m.sum() / mm.sum() / 2
        return ef, e, w
    ef, e, w = estimar(np.arange(len(d)))
    rng = np.random.default_rng(semilla)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        if 0 < t[idx].sum() < len(idx):
            boot.append(estimar(idx)[0])
    bal = pd.DataFrame({"smd_antes": [_smd(X[c].to_numpy(), t) for c in X], "smd_despues": [_smd(X[c].to_numpy(), t, w) for c in X]},
                       index=X.columns)
    avisos = []
    if (bal.smd_despues.abs() > 0.1).any():
        avisos.append("Hay covariables con |SMD| > 0.1 tras el ajuste: el balance no es suficiente (añade interacciones o términos no lineales).")
    sol = (e <= recorte + 1e-12).mean() + (e >= 1 - recorte - 1e-12).mean()
    if sol > 0.05:
        avisos.append(f"{sol:.0%} de las observaciones tiene e(x) en el recorte: solapamiento pobre.")
    return {"efecto": float(ef), "ic": tuple(np.quantile(boot, [0.025, 0.975])) if boot else (np.nan, np.nan),
            "ee_bootstrap": float(np.std(boot, ddof=1)) if len(boot) > 1 else np.nan,
            "efecto_ingenuo": float(y[t == 1].mean() - y[t == 0].mean()), "balance": bal, "propension": e,
            "metodo": metodo, "estimando": "att" if metodo == "emparejamiento" else estimando, "avisos": avisos}


def diferencias_en_diferencias(df: pd.DataFrame, resultado: str, tratado: str, post: str, cluster: str | None = None,
                               covariables=None, periodo: str | None = None) -> dict:
    """Diferencias en diferencias: efecto = (tratados después − antes) − (control después − antes), estimado como la
    interacción tratado×post en una regresión con errores robustos (agrupados por `cluster` si hay medidas repetidas
    de la misma unidad). Con `periodo` (varios periodos previos) contrasta las TENDENCIAS PARALELAS previas
    (tratado × tendencia antes del tratamiento): el supuesto clave. Equivale a: PROC GLM/SURVEYREG con la interacción."""
    cov = list(covariables or [])
    cols = [resultado, tratado, post] + cov + ([cluster] if cluster else []) + ([periodo] if periodo else [])
    _columnas(df, cols)
    d = df[cols].dropna().copy()
    q = lambda c: f'Q("{c}")'  # noqa: E731
    f = f"{q(resultado)} ~ {q(tratado)} * {q(post)}" + "".join(f" + {q(c)}" for c in cov)
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": pd.factorize(d[cluster])[0]}} if cluster else {"cov_type": "HC3"}
    m = smf.ols(f, d).fit(**kw)
    nom = f"{q(tratado)}:{q(post)}"
    tab = d.groupby([tratado, post])[resultado].mean().unstack()
    out = {"efecto": float(m.params[nom]), "ee": float(m.bse[nom]), "ic": tuple(m.conf_int().loc[nom]), "p_valor": float(m.pvalues[nom]),
           "medias_2x2": tab, "efecto_bruto_2x2": float((tab.loc[1, 1] - tab.loc[1, 0]) - (tab.loc[0, 1] - tab.loc[0, 0])), "modelo": m}
    if periodo:
        pre = d[d[post] == 0]
        if pre[periodo].nunique() >= 2:
            mp = smf.ols(f"{q(resultado)} ~ {q(tratado)} * {q(periodo)}", pre).fit(**({"cov_type": "cluster", "cov_kwds": {"groups": pd.factorize(pre[cluster])[0]}} if cluster else {"cov_type": "HC3"}))
            k = f"{q(tratado)}:{q(periodo)}"
            out["tendencias_previas"] = {"diferencia_pendiente": float(mp.params[k]), "p_valor": float(mp.pvalues[k])}
            if mp.pvalues[k] < 0.05:
                out["aviso"] = "Las tendencias previas NO son paralelas: el DiD está sesgado."
    return out


def aleatorizar_ensayo(n: int, brazos=("control", "tratamiento"), proporciones=None, tamano_bloque: int | None = None,
                       estratos: pd.Series | None = None, semilla: int = 42) -> pd.DataFrame:
    """Lista de aleatorización para un ensayo: por bloques permutados (garantiza equilibrio cada `tamano_bloque`
    pacientes; por defecto 2×nº de brazos) y, opcionalmente, estratificada (una lista independiente por estrato).
    `proporciones` permite asignación desigual (p. ej. 2:1 → (2, 1))."""
    rng = np.random.default_rng(semilla)
    brazos = list(brazos)
    prop = list(proporciones) if proporciones is not None else [1] * len(brazos)
    unidad = sum(prop)
    tb = tamano_bloque or 2 * unidad
    if tb % unidad:
        raise ValueError(f"tamano_bloque debe ser múltiplo de {unidad}.")
    plantilla = np.repeat(brazos, [p * tb // unidad for p in prop])

    def lista(m):
        out = []
        while len(out) < m:
            out.extend(rng.permutation(plantilla))
        return out[:m]
    if estratos is None:
        return pd.DataFrame({"id": np.arange(1, n + 1), "brazo": lista(n)})
    est = pd.Series(estratos).reset_index(drop=True)
    if len(est) != n:
        raise ValueError("estratos debe tener n valores.")
    asig = pd.Series(index=est.index, dtype=object)
    for s, idx in est.groupby(est).groups.items():
        asig[idx] = lista(len(idx))
    return pd.DataFrame({"id": np.arange(1, n + 1), "estrato": est, "brazo": asig})


def analisis_intencion_tratar(df: pd.DataFrame, asignado: str, recibido: str, resultado: str) -> pd.DataFrame:
    """Compara los análisis de un ensayo con incumplimiento o cambio de tratamiento (switchers):
    - ITT (por intención de tratar): por brazo ASIGNADO; protege la aleatorización (efecto de ofrecer el tratamiento);
    - por protocolo: solo quienes cumplieron; - según tratamiento recibido (as-treated): por lo recibido;
    - CACE (efecto en cumplidores, variable instrumental): ITT / (diferencia de cumplimiento entre brazos).
    Los dos intermedios rompen la aleatorización y pueden estar sesgados. Variables 0/1."""
    _columnas(df, [asignado, recibido, resultado])
    d = df[[asignado, recibido, resultado]].dropna()
    z, t, y = d[asignado].astype(int), d[recibido].astype(int), d[resultado].astype(float)

    def dif(mask1, mask0):
        a, b = y[mask1], y[mask0]
        ee = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        return a.mean() - b.mean(), ee
    filas = []
    for nom, (m1, m0) in {"ITT": (z == 1, z == 0), "por protocolo": ((z == 1) & (t == 1), (z == 0) & (t == 0)),
                          "según recibido": (t == 1, t == 0)}.items():
        e, s = dif(m1, m0)
        filas.append({"analisis": nom, "efecto": e, "ee": s, "n": int(m1.sum() + m0.sum())})
    itt, ee_itt = filas[0]["efecto"], filas[0]["ee"]
    cumpl = t[z == 1].mean() - t[z == 0].mean()
    if cumpl <= 0.05:
        raise ValueError("Casi no hay diferencia de tratamiento recibido entre brazos: el instrumento es débil.")
    filas.append({"analisis": "CACE (VI)", "efecto": itt / cumpl, "ee": ee_itt / cumpl, "n": len(d)})
    r = pd.DataFrame(filas).set_index("analisis")
    r["ic_inf"] = r.efecto - 1.96 * r.ee; r["ic_sup"] = r.efecto + 1.96 * r.ee
    r.attrs["cumplimiento"] = float(cumpl)
    return r


def mediacion(df: pd.DataFrame, x: str, mediador: str, y: str, covariables=None, n_boot: int = 1000, semilla: int = 42) -> dict:
    """Análisis de mediación (lineal, Baron-Kenny / producto de coeficientes): a = efecto de X en M, b = efecto de M en
    Y controlando X, efecto indirecto a·b con IC bootstrap percentil (preferible al test de Sobel, que también se da),
    efecto directo c' y total c = c' + a·b, y proporción mediada. Gauss: la interpretación causal exige que no haya
    confusores M-Y no medidos (ni siquiera aleatorizando X)."""
    cov = list(covariables or [])
    _columnas(df, [x, mediador, y] + cov)
    d = df[[x, mediador, y] + cov].dropna().reset_index(drop=True)
    q = lambda c: f'Q("{c}")'  # noqa: E731
    extra = "".join(f" + {q(c)}" for c in cov)

    def ajustar(dd):
        ma = smf.ols(f"{q(mediador)} ~ {q(x)}{extra}", dd).fit()
        mb = smf.ols(f"{q(y)} ~ {q(x)} + {q(mediador)}{extra}", dd).fit()
        mc = smf.ols(f"{q(y)} ~ {q(x)}{extra}", dd).fit()
        return ma, mb, mc
    ma, mb, mc = ajustar(d)
    a, b = ma.params[q(x)], mb.params[q(mediador)]
    c_, c = mb.params[q(x)], mc.params[q(x)]
    rng = np.random.default_rng(semilla)
    Xa = np.column_stack([np.ones(len(d)), d[[x] + cov].to_numpy(float)])
    Xb = np.column_stack([np.ones(len(d)), d[[x, mediador] + cov].to_numpy(float)])
    M, Y = d[mediador].to_numpy(float), d[y].to_numpy(float)
    ind = []
    for _ in range(n_boot):
        i = rng.integers(0, len(d), len(d))
        ba = np.linalg.lstsq(Xa[i], M[i], rcond=None)[0][1]; bb = np.linalg.lstsq(Xb[i], Y[i], rcond=None)[0][2]
        ind.append(ba * bb)
    sobel_se = np.sqrt(b ** 2 * ma.bse[q(x)] ** 2 + a ** 2 * mb.bse[q(mediador)] ** 2)
    return {"a": float(a), "b": float(b), "indirecto": float(a * b), "ic_indirecto": tuple(np.quantile(ind, [0.025, 0.975])),
            "directo": float(c_), "total": float(c), "proporcion_mediada": float(a * b / c) if c != 0 else np.nan,
            "sobel_z": float(a * b / sobel_se), "sobel_p": float(2 * stats.norm.sf(abs(a * b / sobel_se)))}


def moderacion(df: pd.DataFrame, y: str, x: str, moderador: str, covariables=None, centrar: bool = True) -> dict:
    """Moderación (interacción X×W): ¿el efecto de X sobre Y depende de W? Ajusta y = b0 + b1·X + b2·W + b3·X·W (con X y
    W centradas para que b1 y b2 sean efectos en la media), da las PENDIENTES SIMPLES de X en W = media ± 1 DE con su
    IC y la región de Johnson-Neyman (valores de W donde el efecto de X es significativo). Errores HC3."""
    cov = list(covariables or [])
    _columnas(df, [y, x, moderador] + cov)
    d = df[[y, x, moderador] + cov].dropna().copy()
    mx, mw = (d[x].mean(), d[moderador].mean()) if centrar else (0.0, 0.0)
    d["_x"], d["_w"] = d[x] - mx, d[moderador] - mw
    m = smf.ols(f'Q("{y}") ~ _x * _w' + "".join(f' + Q("{c}")' for c in cov), d).fit(cov_type="HC3")
    V = m.cov_params()
    b1, b3 = m.params["_x"], m.params["_x:_w"]
    v11, v33, v13 = V.loc["_x", "_x"], V.loc["_x:_w", "_x:_w"], V.loc["_x", "_x:_w"]
    sd = d["_w"].std()
    filas = []
    for et, w in [("media − 1 DE", -sd), ("media", 0.0), ("media + 1 DE", sd)]:
        pen, ee = b1 + b3 * w, np.sqrt(v11 + w ** 2 * v33 + 2 * w * v13)
        filas.append({"moderador": et, "valor_moderador": w + mw, "pendiente_x": pen, "ee": ee, "p_valor": 2 * stats.norm.sf(abs(pen / ee))})
    z = 1.96
    A, B, C = b3 ** 2 - z ** 2 * v33, 2 * (b1 * b3 - z ** 2 * v13), b1 ** 2 - z ** 2 * v11
    disc = B ** 2 - 4 * A * C
    jn = sorted(((-B + s * np.sqrt(disc)) / (2 * A) + mw) for s in (-1, 1)) if disc >= 0 and A != 0 else []
    return {"interaccion": float(b3), "p_interaccion": float(m.pvalues["_x:_w"]), "pendientes_simples": pd.DataFrame(filas).set_index("moderador"),
            "johnson_neyman": jn, "modelo": m}


def metaanalisis(efectos, errores, metodo: str = "dl", nombres=None, nivel: float = 0.95, hartung_knapp: bool = False) -> dict:
    """Metaanálisis de estudios con su efecto y error estándar: efectos fijos (inverso de la varianza) o aleatorios
    (DerSimonian-Laird ``'dl'`` o REML ``'reml'``). Devuelve el efecto combinado con IC, heterogeneidad (Q de Cochran,
    I², τ²), intervalo de PREDICCIÓN (dónde caería el efecto de un estudio nuevo), pesos por estudio y el test de Egger
    de asimetría del embudo (sesgo de publicación; poco potente con < 10 estudios).
    `hartung_knapp=True` usa el ajuste de Hartung-Knapp-Sidik-Jonkman (t con k−1 gl y varianza reescalada): con pocos
    estudios el IC de DerSimonian-Laird se queda corto (≈ 88 % en vez de 95 % con 8 estudios; ver la ficha) y HKSJ lo
    corrige. Equivale a: PROC MIXED con varianzas fijas / metafor::rma en R (test="knha")."""
    y = np.asarray(efectos, float); se = np.asarray(errores, float)
    if len(y) != len(se) or len(y) < 2 or (se <= 0).any():
        raise ValueError("efectos y errores: misma longitud (≥ 2) y errores > 0.")
    k = len(y); v = se ** 2; w = 1 / v
    mf = np.sum(w * y) / w.sum()
    Q = float(np.sum(w * (y - mf) ** 2))
    if metodo == "fijo":
        tau2 = 0.0
    elif metodo == "dl":
        tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    elif metodo == "reml":
        def nreml(t2):
            ww = 1 / (v + t2); mu = np.sum(ww * y) / ww.sum()
            return 0.5 * (np.sum(np.log(v + t2)) + np.log(ww.sum()) + np.sum(ww * (y - mu) ** 2))
        from scipy.optimize import minimize_scalar
        tau2 = float(minimize_scalar(nreml, bounds=(0, max(10 * np.var(y), 1e-8)), method="bounded").x)
    else:
        raise ValueError("metodo: 'fijo', 'dl' o 'reml'")
    ws = 1 / (v + tau2)
    mu = float(np.sum(ws * y) / ws.sum()); se_mu = float(np.sqrt(1 / ws.sum()))
    z = stats.norm.ppf(0.5 + nivel / 2)
    p_mu = float(2 * stats.norm.sf(abs(mu / se_mu)))
    if hartung_knapp:
        q = float(np.sum(ws * (y - mu) ** 2) / (k - 1))
        se_mu = float(np.sqrt(max(q, 1.0) / ws.sum()))                # versión «modificada»: nunca más estrecha que DL
        z = stats.t.ppf(0.5 + nivel / 2, k - 1)
        p_mu = float(2 * stats.t.sf(abs(mu / se_mu), k - 1))
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    tq = stats.t.ppf(0.5 + nivel / 2, k - 2) if k > 2 else np.nan
    pred = (mu - tq * np.sqrt(tau2 + se_mu ** 2), mu + tq * np.sqrt(tau2 + se_mu ** 2)) if k > 2 else (np.nan, np.nan)
    egger = None
    if k >= 3:
        me = sm.OLS(y / se, sm.add_constant(1 / se)).fit()
        egger = {"sesgo": float(me.params[0]), "p_valor": float(me.pvalues[0])}
    tabla = pd.DataFrame({"efecto": y, "ee": se, "ic_inf": y - z * se, "ic_sup": y + z * se, "peso": 100 * ws / ws.sum()},
                         index=nombres if nombres is not None else [f"estudio {i + 1}" for i in range(k)])
    return {"efecto": mu, "ee": se_mu, "ic": (mu - z * se_mu, mu + z * se_mu), "p_valor": p_mu,
            "Q": Q, "p_heterogeneidad": float(stats.chi2.sf(Q, k - 1)), "I2": float(I2), "tau2": float(tau2),
            "intervalo_prediccion": pred, "egger": egger, "tabla": tabla, "metodo": metodo + ("+HK" if hartung_knapp else "")}
