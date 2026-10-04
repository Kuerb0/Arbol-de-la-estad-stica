"""Multivariante (2): regresión multivariante, análisis de perfiles, gráfico de control T² de Hotelling, análisis
conjunto (conjoint), modelos gráficos gaussianos (lasso gráfico), datos funcionales (PCA funcional) y clasificadores
lineales clásicos (regresión de la matriz indicadora, centroides contraídos, PLS-DA)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .._util import _columnas


def regresion_multivariante(df: pd.DataFrame, respuestas, predictores) -> dict:
    """Regresión lineal con VARIAS respuestas a la vez (Y = XB + E, errores correlados entre respuestas). Los
    coeficientes coinciden con regresiones separadas, pero los contrastes multivariantes (Wilks, Pillai,
    Hotelling-Lawley, Roy) de cada predictor sobre TODAS las respuestas controlan el error global y aprovechan la
    correlación. Equivale a: PROC GLM con MANOVA / PROC REG con MTEST."""
    from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
    respuestas, predictores = list(respuestas), list(predictores)
    _columnas(df, respuestas + predictores)
    d = df[respuestas + predictores].dropna()
    X = sm.add_constant(pd.get_dummies(d[predictores], drop_first=True, dtype=float))
    Y = d[respuestas].astype(float)
    m = _MultivariateOLS(Y, X).fit()
    tests = m.mv_test()
    filas = []
    for i, efecto in enumerate(X.columns):
        if efecto == "const":
            continue
        r = tests.results[efecto if efecto in tests.results else f"x{i}"]["stat"]
        filas.append({"efecto": efecto, "wilks": r.loc["Wilks' lambda", "Value"], "p_wilks": r.loc["Wilks' lambda", "Pr > F"],
                      "pillai": r.loc["Pillai's trace", "Value"], "p_pillai": r.loc["Pillai's trace", "Pr > F"]})
    B = pd.DataFrame(np.linalg.lstsq(X.to_numpy(), Y.to_numpy(), rcond=None)[0], index=X.columns, columns=respuestas)
    E = Y.to_numpy() - X.to_numpy() @ B.to_numpy()
    Sig = E.T @ E / (len(d) - X.shape[1])
    return {"coeficientes": B, "contrastes": pd.DataFrame(filas).set_index("efecto"),
            "correlacion_residuos": pd.DataFrame(np.corrcoef(E, rowvar=False), index=respuestas, columns=respuestas),
            "covarianza_residuos": pd.DataFrame(Sig, index=respuestas, columns=respuestas)}


def analisis_perfiles(df: pd.DataFrame, medidas, grupo: str) -> pd.DataFrame:
    """Análisis de perfiles de varios grupos medidos en p variables conmensurables (mismas unidades: tiempos,
    ítems de una escala). Tres preguntas con contrastes multivariantes sobre diferencias consecutivas:
    1) PARALELISMO (¿misma forma? = sin interacción grupo × medida), 2) NIVELES (¿misma altura media?, si paralelos),
    3) PLANITUD (¿perfil horizontal? = sin efecto de la medida). Base de las curvas de crecimiento."""
    medidas = list(medidas)
    _columnas(df, medidas + [grupo])
    d = df[medidas + [grupo]].dropna()
    p = len(medidas)
    if p < 2:
        raise ValueError("Hacen falta al menos 2 medidas.")
    Y = d[medidas].to_numpy(float)
    contr = np.diff(np.eye(p), axis=0)                           # (no llamar C: patsy usa C() del entorno)
    D = Y @ contr.T
    from statsmodels.multivariate.manova import MANOVA
    dd = pd.DataFrame(D, columns=[f"d{i}" for i in range(p - 1)]).assign(_g=d[grupo].to_numpy())
    par = MANOVA.from_formula(" + ".join(dd.columns[:-1]) + " ~ C(_g)", dd).mv_test().results["C(_g)"]["stat"]
    media = Y.mean(axis=1)
    niv = stats.f_oneway(*[media[(d[grupo] == k).to_numpy()] for k in d[grupo].unique()])
    # planitud: T² de las diferencias medias (con covarianza intra-grupos)
    gr = d[grupo].to_numpy()
    resid = np.vstack([D[gr == k] - D[gr == k].mean(axis=0) for k in np.unique(gr)])
    S = resid.T @ resid / (len(D) - len(np.unique(gr)))
    m = D.mean(axis=0); n = len(D); q = p - 1; gle = len(D) - len(np.unique(gr))
    T2 = n * m @ np.linalg.solve(S, m)
    F = (gle - q + 1) / (gle * q) * T2
    return pd.DataFrame([
        {"hipotesis": "paralelismo (forma)", "estadistico": par.loc["Wilks' lambda", "Value"], "p_valor": par.loc["Wilks' lambda", "Pr > F"]},
        {"hipotesis": "niveles (altura)", "estadistico": niv.statistic, "p_valor": niv.pvalue},
        {"hipotesis": "planitud (cambio entre medidas)", "estadistico": T2, "p_valor": stats.f.sf(F, q, gle - q + 1)},
    ]).set_index("hipotesis")


def control_t2_multivariante(X_fase1, X_fase2=None, alfa: float = 0.0027) -> dict:
    """Gráfico de control T² de Hotelling para observaciones individuales con p variables correladas (un gráfico
    univariante por variable ignora la correlación y multiplica las falsas alarmas). Fase I: límite con la beta
    ((n−1)²/n · Beta(p/2, (n−p−1)/2)) sobre los datos de referencia; Fase II (vigilancia): límite F
    p(n+1)(n−1)/(n(n−p)) F(p, n−p). Devuelve T² y si cada punto está fuera de control, con la variable que más
    contribuye (descomposición por eliminación)."""
    A = pd.DataFrame(X_fase1).astype(float)
    n, p = A.shape
    if n <= p + 1:
        raise ValueError("Fase I necesita n > p + 1 observaciones.")
    mu = A.mean().to_numpy(); S = np.cov(A.to_numpy(), rowvar=False); Si = np.linalg.inv(S)

    def t2(Z):
        Zc = Z - mu
        return np.einsum("ij,jk,ik->i", Zc, Si, Zc)

    def contrib(Z):
        out = []
        for z in Z:
            tot = (z - mu) @ Si @ (z - mu); mejor, cual = -np.inf, None
            for j in range(p):
                k = [i for i in range(p) if i != j]
                r = (z[k] - mu[k]) @ np.linalg.solve(S[np.ix_(k, k)], z[k] - mu[k]) if k else 0.0
                if tot - r > mejor:
                    mejor, cual = tot - r, A.columns[j]
            out.append(cual)
        return out
    lim1 = (n - 1) ** 2 / n * stats.beta.ppf(1 - alfa, p / 2, (n - p - 1) / 2)
    f1 = pd.DataFrame({"T2": t2(A.to_numpy()), "limite": lim1})
    f1["fuera"] = f1.T2 > lim1
    f1["variable_principal"] = None
    if f1.fuera.any():
        f1.loc[f1.fuera, "variable_principal"] = contrib(A.to_numpy()[f1.fuera.to_numpy()])
    out = {"fase1": f1, "media": pd.Series(mu, index=A.columns), "covarianza": pd.DataFrame(S, index=A.columns, columns=A.columns)}
    if X_fase2 is not None:
        B = pd.DataFrame(X_fase2)[A.columns].astype(float)
        lim2 = p * (n + 1) * (n - 1) / (n * (n - p)) * stats.f.ppf(1 - alfa, p, n - p)
        f2 = pd.DataFrame({"T2": t2(B.to_numpy()), "limite": lim2}, index=B.index)
        f2["fuera"] = f2.T2 > lim2
        f2["variable_principal"] = None
        if f2.fuera.any():
            f2.loc[f2.fuera, "variable_principal"] = contrib(B.to_numpy()[f2.fuera.to_numpy()])
        out["fase2"] = f2
    return out


def analisis_conjunto(df: pd.DataFrame, valoracion: str, atributos, encuestado: str | None = None) -> dict:
    """Análisis conjunto (conjoint) de valoraciones de perfiles: utilidades parciales (part-worths) de cada nivel con
    codificación de efectos (suman 0 dentro de cada atributo) por MCO, IMPORTANCIA relativa de cada atributo (rango de
    sus utilidades / suma de rangos) y, con `encuestado`, las utilidades individuales para segmentar.
    Útil para diseñar productos/tarifas: cuánto vale cada característica para el cliente."""
    atributos = list(atributos)
    _columnas(df, [valoracion] + atributos + ([encuestado] if encuestado else []))
    d = df.dropna(subset=[valoracion] + atributos)

    def ajustar(dd):
        cols, nombres = [], []
        for a in atributos:
            niveles = sorted(dd[a].astype(str).unique())
            for nv in niveles[:-1]:
                cols.append(np.where(dd[a].astype(str) == nv, 1.0, np.where(dd[a].astype(str) == niveles[-1], -1.0, 0.0)))
                nombres.append((a, nv))
        X = sm.add_constant(np.column_stack(cols))
        b = np.linalg.lstsq(X, dd[valoracion].to_numpy(float), rcond=None)[0]
        util = {}
        k = 1
        for a in atributos:
            niveles = sorted(dd[a].astype(str).unique())
            v = b[k:k + len(niveles) - 1]; k += len(niveles) - 1
            util[a] = pd.Series(np.r_[v, -v.sum()], index=niveles)
        return util, b[0]
    util, cte = ajustar(d)
    tabla = pd.concat(util, names=["atributo", "nivel"]).rename("utilidad").reset_index()
    rangos = pd.Series({a: u.max() - u.min() for a, u in util.items()})
    out = {"utilidades": tabla, "importancia": (100 * rangos / rangos.sum()).sort_values(ascending=False), "constante": float(cte)}
    if encuestado:
        ind = {}
        for e, g in d.groupby(encuestado):
            try:
                u, _ = ajustar(g)
                ind[e] = pd.concat(u)
            except np.linalg.LinAlgError:
                continue
        out["individuales"] = pd.DataFrame(ind).T
    return out


def modelo_grafico_gaussiano(X, alfa: float | None = None, cv: int = 5) -> dict:
    """Modelo gráfico no dirigido gaussiano: estima la matriz de PRECISIÓN (inversa de la covarianza) con lasso gráfico
    (alfa por validación cruzada si no se da). Un cero en la precisión = independencia condicional dado el resto →
    no hay arista. Devuelve las correlaciones parciales y la lista de aristas ordenadas por fuerza.
    Gauss: supone normalidad multivariante; con variables muy asimétricas transforma antes (rangos normales)."""
    from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
    Xd = pd.DataFrame(X).astype(float).dropna()
    Z = (Xd - Xd.mean()) / Xd.std(ddof=1)
    m = GraphicalLassoCV(cv=cv).fit(Z) if alfa is None else GraphicalLasso(alpha=alfa).fit(Z)
    P = m.precision_
    d = np.sqrt(np.diag(P))
    pc = -P / np.outer(d, d); np.fill_diagonal(pc, 1)
    cols = list(Xd.columns)
    aristas = [{"de": cols[i], "a": cols[j], "correlacion_parcial": pc[i, j]} for i in range(len(cols)) for j in range(i + 1, len(cols)) if abs(pc[i, j]) > 1e-6]
    ar = pd.DataFrame(aristas, columns=["de", "a", "correlacion_parcial"])
    ar = ar.reindex(ar.correlacion_parcial.abs().sort_values(ascending=False).index).reset_index(drop=True)
    return {"correlaciones_parciales": pd.DataFrame(pc, index=cols, columns=cols), "aristas": ar,
            "alfa": float(getattr(m, "alpha_", alfa)), "densidad": len(ar) / (len(cols) * (len(cols) - 1) / 2)}


def pca_funcional(curvas, t=None, n_bases: int = 15, n_componentes: int = 3, suavizado: float = 1e-4) -> dict:
    """Análisis de datos funcionales: cada fila de `curvas` es una curva observada en los puntos `t` (p. ej. un perfil
    de consumo horario, una curva de tipos, la siniestralidad por mes de antigüedad). Suaviza cada curva con B-splines
    cúbicos penalizados (segunda derivada) y hace PCA funcional: curva media, funciones propias (modos de variación),
    varianza explicada y puntuaciones de cada curva. Las puntuaciones se usan como variables en otros modelos."""
    from scipy.interpolate import BSpline
    Y = np.asarray(curvas, float)
    if Y.ndim != 2 or Y.shape[1] < 6:
        raise ValueError("curvas: matriz (n curvas × ≥ 6 puntos).")
    tt = np.linspace(0, 1, Y.shape[1]) if t is None else np.asarray(t, float)
    k = 3
    nb = min(n_bases, Y.shape[1])
    nudos = np.r_[[tt[0]] * k, np.linspace(tt[0], tt[-1], nb - k + 1), [tt[-1]] * k]
    B = BSpline.design_matrix(tt, nudos, k).toarray()
    D2 = np.diff(np.eye(nb), 2, axis=0)
    lam = suavizado * np.trace(B.T @ B) / max(np.trace(D2.T @ D2), 1e-12)
    H = np.linalg.solve(B.T @ B + lam * D2.T @ D2, B.T)
    C = (H @ np.nan_to_num(Y.T, nan=np.nanmean(Y))).T
    malla = np.linspace(tt[0], tt[-1], 200)
    Bm = BSpline.design_matrix(malla, nudos, k).toarray()
    F = C @ Bm.T
    mu = F.mean(axis=0); Fc = F - mu
    dt = (malla[-1] - malla[0]) / (len(malla) - 1)
    U, s, Vt = np.linalg.svd(Fc * np.sqrt(dt), full_matrices=False)
    var = s ** 2 / (len(F) - 1)
    q = min(n_componentes, len(s))
    phi = Vt[:q] / np.sqrt(dt)
    scores = Fc @ phi.T * dt
    return {"t": malla, "media": mu, "funciones_propias": pd.DataFrame(phi.T, index=malla, columns=[f"fpc{i + 1}" for i in range(q)]),
            "varianza_explicada": pd.Series(var[:q] / var.sum(), index=[f"fpc{i + 1}" for i in range(q)]),
            "puntuaciones": pd.DataFrame(scores, columns=[f"fpc{i + 1}" for i in range(q)]), "curvas_suavizadas": F}


def regresion_matriz_indicadora(X, y) -> dict:
    """Clasificación por regresión lineal de la matriz indicadora (una columna 0/1 por clase, MCO, se asigna la clase
    con mayor ajuste). Didáctica: con ≥ 3 clases sufre ENMASCARAMIENTO (una clase intermedia nunca gana) — compárala
    con el discriminante lineal. Devuelve las predicciones, la exactitud y qué clases no se predicen nunca."""
    Xd = sm.add_constant(pd.get_dummies(pd.DataFrame(X), drop_first=True, dtype=float))
    yy = pd.Series(np.asarray(y)).astype(str)
    Yi = pd.get_dummies(yy, dtype=float)
    B = np.linalg.lstsq(Xd.to_numpy(), Yi.to_numpy(), rcond=None)[0]
    F = Xd.to_numpy() @ B
    pred = Yi.columns[F.argmax(axis=1)]
    return {"prediccion": pd.Series(pred), "exactitud": float((pred == yy.to_numpy()).mean()), "ajustes": pd.DataFrame(F, columns=Yi.columns),
            "clases_enmascaradas": sorted(set(Yi.columns) - set(pred)), "coeficientes": pd.DataFrame(B, index=Xd.columns, columns=Yi.columns)}


def centroides_contraidos(X, y, umbrales=None, cv: int = 5, semilla: int = 42) -> dict:
    """Centroides contraídos más cercanos (PAM, Tibshirani 2002): clasifica por el centroide más cercano (variables
    estandarizadas) contrayendo cada centroide hacia el global con un umbral Δ (soft-thresholding) que se elige por
    validación cruzada; las variables cuyo centroide queda igual en todas las clases se eliminan. Para p ≫ n
    (genes, muchos indicadores). Devuelve Δ, exactitud CV y las variables que sobreviven."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import NearestCentroid
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xd = pd.DataFrame(X).astype(float); yy = np.asarray(y)
    grid = np.r_[0, np.linspace(0.1, 3, 15)] if umbrales is None else np.asarray(umbrales, float)
    kf = StratifiedKFold(cv, shuffle=True, random_state=semilla)
    res = {}
    for d in grid:
        mdl = make_pipeline(StandardScaler(), NearestCentroid(shrink_threshold=d if d > 0 else None))
        res[float(d)] = cross_val_score(mdl, Xd, yy, cv=kf).mean()
    curva = pd.Series(res, name="exactitud_cv")
    mejor = float(curva[curva >= curva.max() - 0.01].index.max())         # el más contraído casi igual de bueno
    m = make_pipeline(StandardScaler(), NearestCentroid(shrink_threshold=mejor if mejor > 0 else None)).fit(Xd, yy)
    cen = m[-1].centroids_
    activas = list(Xd.columns[np.ptp(cen, axis=0) > 1e-10])
    return {"umbral": mejor, "curva_cv": curva, "exactitud_cv": float(curva[mejor]), "variables_activas": activas, "modelo": m}


def pls_da(X, y, max_componentes: int = 10, cv: int = 5, semilla: int = 42) -> dict:
    """PLS-DA: análisis discriminante por mínimos cuadrados parciales (PLS sobre la matriz indicadora de clases y
    asignación a la clase de mayor predicción). Funciona con muchas variables colineales (p > n), donde el LDA no
    puede invertir la covarianza. Elige el nº de componentes por validación cruzada y da la importancia VIP de cada
    variable (VIP > 1 = relevante). Equivale a: PROC PLS con respuesta indicadora."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import StratifiedKFold
    Xd = pd.DataFrame(X).astype(float); yy = pd.Series(np.asarray(y)).astype(str)
    Y = pd.get_dummies(yy, dtype=float); clases = Y.columns
    mu, sd = Xd.mean(), Xd.std(ddof=1).replace(0, 1)
    Z = ((Xd - mu) / sd).to_numpy()
    kf = StratifiedKFold(cv, shuffle=True, random_state=semilla)
    kmax = min(max_componentes, Z.shape[1], len(Z) - 2)
    curva = {}
    for k in range(1, kmax + 1):
        ac = []
        for tr, te in kf.split(Z, yy):
            m = PLSRegression(k, scale=False).fit(Z[tr], Y.to_numpy()[tr])
            ac.append(np.mean(clases[m.predict(Z[te]).argmax(axis=1)] == yy.to_numpy()[te]))
        curva[k] = float(np.mean(ac))
    curva = pd.Series(curva, name="exactitud_cv")
    k = int(curva.idxmax())
    m = PLSRegression(k, scale=False).fit(Z, Y.to_numpy())
    T, W, Q = m.x_scores_, m.x_weights_, m.y_loadings_
    ss = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    vip = np.sqrt(Z.shape[1] * (W ** 2 / np.sum(W ** 2, axis=0)) @ ss / ss.sum())
    return {"n_componentes": k, "curva_cv": curva, "exactitud_cv": float(curva[k]), "vip": pd.Series(vip, index=Xd.columns).sort_values(ascending=False),
            "puntuaciones": pd.DataFrame(T[:, :2], columns=[f"t{i + 1}" for i in range(min(2, k))]).assign(clase=yy.to_numpy()) if k >= 1 else None,
            "modelo": m}
