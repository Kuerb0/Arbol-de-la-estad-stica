"""Diseño de experimentos: factoriales 2^k completos y fraccionados (estructura de alias y resolución), estimación de
efectos con el método de Lenth para diseños sin réplicas, cuadrados latinos, diseño central compuesto y metodología de
superficie de respuesta (ajuste de segundo orden, punto estacionario y análisis canónico)."""
from __future__ import annotations

import itertools
import string

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from .._util import _columnas


def _palabra(cols):
    return "".join(sorted(cols))


def diseno_factorial_2k(k: int, generadores: dict | None = None, replicas: int = 1, centros: int = 0, aleatorizar: bool = True,
                        nombres=None, semilla: int = 42) -> dict:
    """Matriz de un diseño factorial 2^k (niveles −1/+1, orden estándar de Yates) o fraccionado 2^(k−p) con
    `generadores` del tipo {"D": "ABC"} (D = A·B·C). Devuelve el diseño (con orden de ejecución aleatorio si
    `aleatorizar`), la relación de definición, la RESOLUCIÓN (III: efectos principales confundidos con interacciones
    dobles; IV: principales limpios, dobles confundidas entre sí; V: dobles limpias) y la tabla de alias de los efectos
    principales y dobles. `centros` añade puntos centrales (0) para estimar error puro y contrastar curvatura."""
    letras = list(nombres) if nombres is not None else list(string.ascii_uppercase[:k])
    if len(letras) != k:
        raise ValueError("nombres debe tener k elementos.")
    gen = dict(generadores or {})
    base = [l for l in letras if l not in gen]
    X = pd.DataFrame(list(itertools.product([-1, 1], repeat=len(base)))[::1], columns=base[::-1])[base]
    for f, g in gen.items():
        if any(c not in base for c in g):
            raise ValueError(f"El generador de {f} solo puede usar factores básicos {base}.")
        X[f] = np.prod([X[c] for c in g], axis=0)
    X = X[letras]
    palabras = {_palabra(set(g) | {f}) for f, g in gen.items()}
    defin = set()
    for r in range(1, len(palabras) + 1):
        for comb in itertools.combinations(palabras, r):
            s = set()
            for w in comb:
                s ^= set(w)
            defin.add(_palabra(s))
    resolucion = min((len(w) for w in defin), default=np.inf)

    def alias(efecto):
        e = set(efecto)
        return sorted({_palabra(e ^ set(w)) for w in defin}, key=lambda w: (len(w), w))
    efectos = letras + ["".join(c) for c in itertools.combinations(letras, 2)]
    tabla_alias = pd.Series({e: " = ".join(a for a in alias(e) if len(a) <= 3) for e in efectos}, name="alias") if defin else pd.Series(dtype=str)
    D = pd.concat([X] * replicas, ignore_index=True)
    if centros:
        D = pd.concat([D, pd.DataFrame(0, index=range(centros), columns=letras)], ignore_index=True)
    D.insert(0, "orden_estandar", np.arange(1, len(D) + 1))
    if aleatorizar:
        D = D.sample(frac=1, random_state=semilla).reset_index(drop=True)
    D.insert(0, "orden_ejecucion", np.arange(1, len(D) + 1))
    return {"diseno": D, "relacion_definicion": sorted(defin, key=len), "resolucion": resolucion, "alias": tabla_alias,
            "n_ensayos": len(D)}


def efectos_factorial_2k(df: pd.DataFrame, respuesta: str, factores, orden_max: int | None = None, alfa: float = 0.05) -> dict:
    """Estima los efectos de un factorial 2^k (efecto = media en +1 − media en −1 = 2·coeficiente) hasta el orden
    `orden_max`. Si hay réplicas o puntos centrales, error estándar y p-valores clásicos; SIEMPRE el pseudo error estándar
    de Lenth (PSE) para diseños SIN réplicas: un efecto es activo si |efecto| > ME (margen individual) o > SME
    (simultáneo). Incluye la curvatura (media de centros − media factorial) si hay puntos centrales.
    Equivale a: el análisis de PROC FACTEX/ADX o del half-normal plot de Daniel."""
    factores = list(factores)
    _columnas(df, [respuesta] + factores)
    d = df[[respuesta] + factores].dropna()
    cen = (d[factores] == 0).all(axis=1)
    f = d[~cen]
    if not set(np.unique(f[factores].to_numpy())) <= {-1, 1}:
        raise ValueError("Los factores deben estar codificados −1/+1 (y 0 solo en puntos centrales).")
    om = orden_max or len(factores)
    filas, vistos = [], set()
    for r in range(1, om + 1):
        for c in itertools.combinations(factores, r):
            x = np.prod(f[list(c)].to_numpy(), axis=1)
            firma = tuple(x * x[0])
            if len(np.unique(x)) < 2 or firma in vistos:          # constante o alias de un efecto ya listado
                continue
            vistos.add(firma)
            filas.append({"efecto": ":".join(c), "orden": r, "estimacion": f[respuesta][x > 0].mean() - f[respuesta][x < 0].mean()})
    t = pd.DataFrame(filas).set_index("efecto")
    a = t.estimacion.abs()
    s0 = 1.5 * a.median(); pse = 1.5 * a[a < 2.5 * s0].median()
    m = len(t); gl = m / 3
    me = stats.t.ppf(1 - alfa / 2, gl) * pse
    sme = stats.t.ppf((1 + (1 - alfa) ** (1 / m)) / 2, gl) * pse
    t["lenth_t"] = t.estimacion / pse
    t["activo_me"] = a > me; t["activo_sme"] = a > sme
    rep = len(f) - len(f.drop_duplicates(subset=factores))
    out = {"tabla": t.sort_values("estimacion", key=np.abs, ascending=False), "pse_lenth": float(pse), "me": float(me), "sme": float(sme)}
    if rep > 0 or cen.sum() > 1:
        piezas = [g[respuesta].to_numpy() - g[respuesta].mean() for _, g in d.groupby(factores) if len(g) > 1]
        ss = sum((p ** 2).sum() for p in piezas); gle = sum(len(p) - 1 for p in piezas)
        if gle > 0:
            s2 = ss / gle; N = len(f)
            ee = np.sqrt(4 * s2 / N)
            out["tabla"]["ee"] = ee
            out["tabla"]["p_valor"] = 2 * stats.t.sf(np.abs(out["tabla"].estimacion / ee), gle)
            out["error_puro"] = {"varianza": float(s2), "gl": int(gle)}
            if cen.sum() > 0:
                curv = d.loc[cen, respuesta].mean() - f[respuesta].mean()
                se_c = np.sqrt(s2 * (1 / len(f) + 1 / cen.sum()))
                out["curvatura"] = {"estimacion": float(curv), "p_valor": float(2 * stats.t.sf(abs(curv / se_c), gle))}
    return out


def cuadrado_latino(n: int, tratamientos=None, aleatorizar: bool = True, semilla: int = 42) -> pd.DataFrame:
    """Genera un cuadrado latino n×n (cada tratamiento una vez por fila y por columna: controla DOS fuentes de
    variación, p. ej. día y operario), aleatorizando filas, columnas y etiquetas. Devuelve formato largo
    (fila, columna, tratamiento) listo para añadir la respuesta y analizar con `anova_cuadrado_latino`."""
    if n < 3:
        raise ValueError("n ≥ 3.")
    trat = list(tratamientos) if tratamientos is not None else (list(string.ascii_uppercase[:n]) if n <= 26 else [f"T{i + 1}" for i in range(n)])
    if len(trat) != n:
        raise ValueError("Hacen falta n tratamientos.")
    L = (np.arange(n)[:, None] + np.arange(n)[None, :]) % n
    if aleatorizar:
        rng = np.random.default_rng(semilla)
        L = L[rng.permutation(n)][:, rng.permutation(n)]
        trat = list(np.array(trat)[rng.permutation(n)])
    return pd.DataFrame([{"fila": i + 1, "columna": j + 1, "tratamiento": trat[L[i, j]]} for i in range(n) for j in range(n)])


def anova_cuadrado_latino(df: pd.DataFrame, respuesta: str, fila: str = "fila", columna: str = "columna",
                          tratamiento: str = "tratamiento") -> dict:
    """ANOVA de un cuadrado latino: y = μ + fila + columna + tratamiento + ε (sin interacciones: es lo que se paga por
    usar solo n² unidades). Devuelve la tabla con η² parcial y las medias de tratamiento ajustadas."""
    _columnas(df, [respuesta, fila, columna, tratamiento])
    from .anova import _tabla_anova
    q = lambda c: f'C(Q("{c}"))'  # noqa: E731
    m = smf.ols(f'Q("{respuesta}") ~ {q(fila)} + {q(columna)} + {q(tratamiento)}', df).fit()
    t = _tabla_anova(m, 2)
    return {"tabla": t, "medias": df.groupby(tratamiento)[respuesta].mean(), "modelo": m}


def diseno_central_compuesto(k: int, alfa: str | float = "rotable", centros: int = 4, nombres=None) -> pd.DataFrame:
    """Diseño central compuesto para superficies de respuesta: factorial 2^k (±1) + 2k puntos axiales (±α) + centros.
    α = (2^k)^(1/4) lo hace ROTABLE (varianza de predicción igual a igual distancia del centro); α = 1 da el diseño
    centrado en las caras (solo 3 niveles por factor)."""
    letras = list(nombres) if nombres is not None else [f"x{i + 1}" for i in range(k)]
    a = (2 ** k) ** 0.25 if alfa == "rotable" else (1.0 if alfa == "caras" else float(alfa))
    fac = np.array(list(itertools.product([-1, 1], repeat=k)), float)
    ax = np.vstack([s * a * np.eye(k)[i] for i in range(k) for s in (-1, 1)])
    D = np.vstack([fac, ax, np.zeros((centros, k))])
    out = pd.DataFrame(D, columns=letras)
    out.insert(0, "tipo", ["factorial"] * len(fac) + ["axial"] * len(ax) + ["centro"] * centros)
    return out


def superficie_respuesta(df: pd.DataFrame, respuesta: str, factores, objetivo: str = "max") -> dict:
    """Ajusta un modelo de segundo orden y = β0 + Σβi xi + Σβii xi² + Σβij xi xj, localiza el punto estacionario
    x* = −½ B⁻¹ b y lo clasifica con el análisis canónico (autovalores de B: todos < 0 máximo, > 0 mínimo, mezcla
    punto de silla; autovalores ≈ 0 = cresta). Incluye la falta de ajuste del primer orden (¿hay curvatura?) y avisa
    si el punto estacionario cae fuera de la región experimental (extrapolación)."""
    factores = list(factores)
    _columnas(df, [respuesta] + factores)
    d = df[[respuesta] + factores].dropna()
    q = [f'Q("{f}")' for f in factores]
    cuad = [f"I({x}**2)" for x in q]
    inter = [f"{a}:{b}" for a, b in itertools.combinations(q, 2)]
    m = smf.ols(f'Q("{respuesta}") ~ ' + " + ".join(q + cuad + inter), d).fit()
    m1 = smf.ols(f'Q("{respuesta}") ~ ' + " + ".join(q), d).fit()
    k = len(factores)
    p = m.params
    b = np.array([p[x] for x in q])
    B = np.zeros((k, k))
    for i, x in enumerate(q):
        B[i, i] = p[f"I({x} ** 2)"] if f"I({x} ** 2)" in p else p[f"I({x}**2)"]
    for (i, a), (j, c) in itertools.combinations(enumerate(q), 2):
        B[i, j] = B[j, i] = p[f"{a}:{c}"] / 2
    xs = -0.5 * np.linalg.solve(B, b)
    ys = float(m.params.iloc[0] + 0.5 * b @ xs)
    lam, V = np.linalg.eigh(B)
    tol = 1e-3 * max(1, np.abs(lam).max())
    tipo = "máximo" if (lam < -tol).all() else ("mínimo" if (lam > tol).all() else ("punto de silla" if (lam < -tol).any() and (lam > tol).any() else "cresta"))
    dentro = bool(((xs >= d[factores].min().to_numpy() - 1e-9) & (xs <= d[factores].max().to_numpy() + 1e-9)).all())
    from statsmodels.stats.anova import anova_lm
    curv = anova_lm(m1, m)
    avisos = []
    if not dentro:
        avisos.append("El punto estacionario está fuera de la región experimental: es una extrapolación; explora en la dirección del máximo ascenso.")
    if (objetivo == "max" and tipo != "máximo") or (objetivo == "min" and tipo != "mínimo"):
        avisos.append(f"La superficie tiene un {tipo}, no un {objetivo}imo: el óptimo está en el borde de la región.")
    return {"modelo": m, "punto_estacionario": pd.Series(xs, index=factores), "respuesta_estacionaria": ys,
            "autovalores": pd.Series(lam), "autovectores": pd.DataFrame(V, index=factores), "tipo": tipo,
            "dentro_region": dentro, "r2": float(m.rsquared), "p_curvatura": float(curv["Pr(>F)"].iloc[1]), "avisos": avisos}
