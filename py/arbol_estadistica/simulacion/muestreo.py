"""Muestreo de poblaciones finitas y encuestas: extracción de muestras (aleatoria simple, sistemática, estratificada,
por conglomerados), estimación con corrección por población finita, muestreo estratificado (afijación proporcional,
de Neyman y óptima), estimador de razón, tamaño de muestra y tasa de respuesta con ajuste por no respuesta."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .._util import _columnas, _numerico


def extraer_muestra(df: pd.DataFrame, n: int, metodo: str = "mas", estrato: str | None = None, conglomerado: str | None = None,
                    afijacion: str = "proporcional", semilla: int = 42) -> pd.DataFrame:
    """Extrae una muestra de un marco (DataFrame) y añade la columna `peso` (inverso de la probabilidad de inclusión,
    el factor de elevación). Métodos: ``'mas'`` (aleatoria simple sin reposición), ``'sistematica'`` (arranque aleatorio
    y salto N/n: cuidado con periodicidades en el orden del marco), ``'estratificada'`` (por `estrato`, con afijación
    proporcional o igual) y ``'conglomerados'`` (se eligen n conglomerados enteros de la columna `conglomerado`)."""
    rng = np.random.default_rng(semilla)
    N = len(df)
    if metodo == "mas":
        if not 0 < n <= N:
            raise ValueError("0 < n ≤ N.")
        idx = rng.choice(N, n, replace=False)
        return df.iloc[np.sort(idx)].assign(peso=N / n)
    if metodo == "sistematica":
        k = N / n; inicio = rng.uniform(0, k)
        idx = np.floor(inicio + k * np.arange(n)).astype(int)
        return df.iloc[idx].assign(peso=N / n)
    if metodo == "estratificada":
        _columnas(df, [estrato])
        Nh = df[estrato].value_counts()
        nh = (Nh / N * n).round().astype(int).clip(lower=1) if afijacion == "proporcional" else pd.Series(max(1, n // len(Nh)), index=Nh.index)
        partes = []
        for h, g in df.groupby(estrato, sort=False):
            k = int(min(nh[h], len(g)))
            partes.append(g.iloc[np.sort(rng.choice(len(g), k, replace=False))].assign(peso=len(g) / k))
        return pd.concat(partes)
    if metodo == "conglomerados":
        _columnas(df, [conglomerado])
        u = df[conglomerado].unique()
        if not 0 < n <= len(u):
            raise ValueError("n = nº de conglomerados a elegir (≤ nº de conglomerados del marco).")
        elegidos = rng.choice(u, n, replace=False)
        return df[df[conglomerado].isin(elegidos)].assign(peso=len(u) / n)
    raise ValueError("metodo: 'mas', 'sistematica', 'estratificada' o 'conglomerados'")


def estimar_mas(muestra, N: float | None = None, nivel: float = 0.95, proporcion: bool | None = None) -> pd.DataFrame:
    """Estimación con muestreo aleatorio simple: media, total y (si los datos son 0/1) proporción, con error estándar
    que incluye la corrección por población finita √(1 − n/N), IC y coeficiente de variación.
    Equivale a: PROC SURVEYMEANS (sin estratos ni conglomerados). Gauss: si n/N < 5 % la corrección es despreciable."""
    y = _numerico(muestra, "muestra", 2)
    n = len(y)
    fpc = 1 - n / N if N else 1.0
    if N is not None and n > N:
        raise ValueError("n > N.")
    z = stats.t.ppf(0.5 + nivel / 2, n - 1)
    m = y.mean(); se = np.sqrt(fpc * y.var(ddof=1) / n)
    filas = [{"parametro": "media", "estimacion": m, "error_estandar": se}]
    if N:
        filas.append({"parametro": "total", "estimacion": N * m, "error_estandar": N * se})
    if proporcion or (proporcion is None and set(np.unique(y)) <= {0.0, 1.0}):
        p = m; sp = np.sqrt(fpc * p * (1 - p) / (n - 1))
        filas.append({"parametro": "proporcion", "estimacion": p, "error_estandar": sp})
    t = pd.DataFrame(filas).set_index("parametro")
    t["ic_inf"] = t.estimacion - z * t.error_estandar; t["ic_sup"] = t.estimacion + z * t.error_estandar
    t["cv"] = t.error_estandar / t.estimacion.abs()
    t.attrs.update({"n": n, "N": N, "fraccion_muestreo": n / N if N else 0.0})
    return t


def tamano_muestra_encuesta(margen: float, N: float | None = None, proporcion: float = 0.5, desviacion: float | None = None,
                            nivel: float = 0.95, tasa_respuesta: float = 1.0, efecto_diseno: float = 1.0) -> dict:
    """Tamaño de muestra para estimar una proporción (por defecto p = 0.5, el caso más desfavorable) o una media
    (`desviacion`) con un margen de error dado: n0 = z²·S²/e², corrección por población finita n = n0/(1 + (n0−1)/N),
    multiplicado por el efecto de diseño (conglomerados) y dividido por la tasa de respuesta esperada (cuántos hay
    que CONTACTAR). Gauss: la no respuesta no solo reduce n, también puede sesgar (si quien no responde es distinto)."""
    if not 0 < tasa_respuesta <= 1 or margen <= 0:
        raise ValueError("margen > 0 y tasa_respuesta en (0, 1].")
    z = stats.norm.ppf(0.5 + nivel / 2)
    S2 = desviacion ** 2 if desviacion is not None else proporcion * (1 - proporcion)
    n0 = z ** 2 * S2 / margen ** 2
    n = n0 / (1 + (n0 - 1) / N) if N else n0
    n *= efecto_diseno
    return {"n_sin_corregir": float(n0), "n_necesario": int(np.ceil(n)), "n_a_contactar": int(np.ceil(n / tasa_respuesta))}


def asignacion_estratos(N_h, S_h, n: int, metodo: str = "neyman", costes=None) -> pd.DataFrame:
    """Afijación de una muestra de tamaño n entre estratos: ``'proporcional'`` (n_h ∝ N_h), ``'neyman'``
    (n_h ∝ N_h·S_h: más muestra donde hay más variabilidad) u ``'optima'`` (n_h ∝ N_h·S_h/√c_h con costes por unidad).
    Devuelve n_h y la varianza de la media estratificada resultante frente a la proporcional y a la MAS.
    `S_h` son desviaciones típicas por estrato (de un piloto o de la encuesta anterior)."""
    Nh = pd.Series(N_h, dtype=float)

    def alinear(v):
        s_ = pd.Series(v, dtype=float)
        if set(s_.index) == set(Nh.index):
            return s_.reindex(Nh.index)
        if len(s_) != len(Nh):
            raise ValueError("N_h, S_h y costes deben tener un valor por estrato.")
        return pd.Series(s_.to_numpy(), index=Nh.index)
    Sh = alinear(S_h)
    ch = pd.Series(1.0, index=Nh.index) if costes is None else alinear(costes)
    if metodo == "proporcional":
        w = Nh
    elif metodo == "neyman":
        w = Nh * Sh
    elif metodo == "optima":
        w = Nh * Sh / np.sqrt(ch)
    else:
        raise ValueError("metodo: 'proporcional', 'neyman' u 'optima'")
    nh = n * w / w.sum()
    nh = np.minimum(nh, Nh)
    N = Nh.sum(); Wh = Nh / N

    def var(nn):
        return float(np.sum(Wh ** 2 * Sh ** 2 / nn * (1 - nn / Nh)))
    S2_total = float(np.sum(Wh * Sh ** 2))        # sin variación entre medias (cota inferior de la MAS)
    t = pd.DataFrame({"N_h": Nh, "S_h": Sh, "n_h": nh, "n_h_entero": np.maximum(1, nh.round()).astype(int)})
    t.attrs.update({"var_media": var(nh), "var_proporcional": var(n * Nh / N), "var_mas_aprox": S2_total / n * (1 - n / N)})
    return t


def estimar_estratificado(muestra: pd.DataFrame, variable: str, estrato: str, N_h, nivel: float = 0.95) -> dict:
    """Estimación con muestreo estratificado: media ȳ_st = Σ W_h ȳ_h y total, varianza Σ W_h² (1−f_h) s_h²/n_h,
    grados de libertad de Satterthwaite, y el efecto de diseño (varianza estratificada / varianza MAS equivalente;
    < 1 = la estratificación ha ganado precisión). `N_h`: tamaños poblacionales por estrato (dict o Series)."""
    _columnas(muestra, [variable, estrato])
    Nh = pd.Series(N_h, dtype=float)
    g = muestra.groupby(estrato)[variable].agg(["mean", "var", "count"])
    falta = set(g.index) - set(Nh.index)
    if falta:
        raise ValueError(f"Faltan N_h para los estratos {falta}.")
    Nh = Nh.loc[g.index]
    if (g["count"] < 2).any():
        raise ValueError("Cada estrato necesita al menos 2 observaciones para estimar su varianza.")
    N = Nh.sum(); W = Nh / N
    m = float(np.sum(W * g["mean"]))
    a = W ** 2 * (1 - g["count"] / Nh) * g["var"] / g["count"]
    v = float(a.sum())
    gl = v ** 2 / float(np.sum(a ** 2 / (g["count"] - 1)))
    z = stats.t.ppf(0.5 + nivel / 2, gl)
    n = g["count"].sum()
    v_mas = muestra[variable].var(ddof=1) / n * (1 - n / N)
    return {"media": m, "error_estandar": float(np.sqrt(v)), "ic": (m - z * np.sqrt(v), m + z * np.sqrt(v)), "total": float(N * m),
            "error_total": float(N * np.sqrt(v)), "gl": float(gl), "efecto_diseno": float(v / v_mas),
            "por_estrato": g.rename(columns={"mean": "media", "var": "varianza", "count": "n"}).assign(N_h=Nh, peso=W)}


def estimador_razon(y, x, media_x_poblacional: float, N: float | None = None, nivel: float = 0.95) -> dict:
    """Estimador de razón: R̂ = ȳ/x̄ y ȳ_R = R̂·X̄ (usa una variable auxiliar x conocida en toda la población, p. ej.
    la prima del año pasado para estimar la de este). Varianza linealizada (1−f)/n · s²(y − R̂x), comparada con la MAS.
    Gauss: es sesgado (sesgo O(1/n)) pero gana mucho si y es casi proporcional a x (recta por el origen) y
    corr(x, y) > CV(x)/(2·CV(y)); si la recta tiene ordenada lejos de 0, usa el estimador de regresión."""
    yv, xv = np.asarray(y, float), np.asarray(x, float)
    if len(yv) != len(xv) or len(yv) < 3:
        raise ValueError("y y x: misma longitud (≥ 3).")
    n = len(yv); f = n / N if N else 0
    R = yv.mean() / xv.mean()
    est = R * media_x_poblacional
    e = yv - R * xv
    v = (1 - f) / n * e.var(ddof=1) * (media_x_poblacional / xv.mean()) ** 2
    v_mas = (1 - f) / n * yv.var(ddof=1)
    z = stats.t.ppf(0.5 + nivel / 2, n - 1)
    r = np.corrcoef(xv, yv)[0, 1]
    cvx, cvy = xv.std(ddof=1) / xv.mean(), yv.std(ddof=1) / yv.mean()
    return {"razon": float(R), "media_estimada": float(est), "error_estandar": float(np.sqrt(v)),
            "ic": (float(est - z * np.sqrt(v)), float(est + z * np.sqrt(v))), "total_estimado": float(N * est) if N else np.nan,
            "media_mas": float(yv.mean()), "error_mas": float(np.sqrt(v_mas)), "ganancia_varianza": float(1 - v / v_mas),
            "conviene": bool(r > cvx / (2 * cvy))}


def ajuste_no_respuesta(df: pd.DataFrame, respondio: str, celdas, peso: str | None = None) -> dict:
    """Tasa de respuesta (global y por celda, ponderada si hay pesos de diseño) y ajuste por no respuesta por celdas de
    ponderación: dentro de cada celda (combinación de variables conocidas para TODOS los seleccionados: zona, edad,
    canal…) el peso de los que responden se multiplica por 1/tasa de respuesta de la celda. Supone que, dentro de la
    celda, la no respuesta es al azar (MAR). Devuelve los respondentes con `peso_ajustado` y la tabla de tasas."""
    celdas = [celdas] if isinstance(celdas, str) else list(celdas)
    _columnas(df, [respondio] + celdas + ([peso] if peso else []))
    d = df.copy()
    d["_w"] = d[peso] if peso else 1.0
    d["_r"] = d[respondio].astype(int)
    tab = d.groupby(celdas).apply(lambda g: pd.Series({"seleccionados": len(g), "respondentes": g._r.sum(),
                                                      "tasa_respuesta": (g._w * g._r).sum() / g._w.sum()}), include_groups=False)
    if (tab.respondentes == 0).any():
        raise ValueError("Hay celdas sin ningún respondente: agrúpalas antes de ajustar.")
    d = d.join(tab["tasa_respuesta"], on=celdas)
    resp = d[d._r == 1].copy()
    resp["peso_ajustado"] = resp._w / resp.tasa_respuesta
    tasa = float((d._w * d._r).sum() / d._w.sum())
    avisos = []
    if tab.tasa_respuesta.max() - tab.tasa_respuesta.min() > 0.2:
        avisos.append("La tasa de respuesta varía mucho entre celdas: sin ajuste la estimación estaría sesgada hacia las celdas que más responden.")
    return {"tasa_respuesta": tasa, "por_celda": tab, "respondentes": resp.drop(columns=["_w", "_r"]), "avisos": avisos}
