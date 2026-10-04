"""Demografía: tasas específicas con IC, estandarización directa e indirecta (SMR), exposición al riesgo por edad
(diagrama de Lexis), indicadores de fecundidad y proyección de población con la matriz de Leslie."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _ic_poisson(d, nivel=0.95):
    a = 1 - nivel
    lo = np.where(d > 0, stats.chi2.ppf(a / 2, 2 * d) / 2, 0.0)
    hi = stats.chi2.ppf(1 - a / 2, 2 * (d + 1)) / 2
    return lo, hi


def tasas_especificas(defunciones, poblacion, grupos=None, por: float = 1000.0, nivel: float = 0.95) -> pd.DataFrame:
    """Tasas específicas (por edad o grupo) m = D/P con IC exacto de Poisson, y la tasa bruta global en `attrs`.
    `poblacion` = población media o exposición en personas-año. Expresadas por `por` (1000 por defecto)."""
    D, P = np.asarray(defunciones, float), np.asarray(poblacion, float)
    if D.shape != P.shape or (P <= 0).any():
        raise ValueError("defunciones y poblacion: misma longitud y población > 0.")
    lo, hi = _ic_poisson(D, nivel)
    t = pd.DataFrame({"defunciones": D, "poblacion": P, "tasa": D / P * por, "ic_inf": lo / P * por, "ic_sup": hi / P * por},
                     index=grupos if grupos is not None else None)
    t.attrs["tasa_bruta"] = float(D.sum() / P.sum() * por)
    return t


def estandarizar_tasas(defunciones, poblacion, referencia, metodo: str = "directa", por: float = 1000.0) -> dict:
    """Compara poblaciones con estructuras de edad distintas.
    'directa': aplica las tasas por edad del estudio a una población ESTÁNDAR (`referencia` = población estándar por edad)
      → tasa estandarizada con IC (aproximación gamma de Fay-Feuer).
    'indirecta': aplica tasas ESTÁNDAR (`referencia` = tasas por edad de referencia, por unidad) a la población del estudio
      → esperadas E, razón estandarizada de mortalidad SMR = O/E con IC exacto de Poisson (SMR > 1 = exceso de mortalidad)."""
    D, P, R = (np.asarray(v, float) for v in (defunciones, poblacion, referencia))
    if metodo == "directa":
        w = R / R.sum()
        m = D / P
        tasa = float(np.sum(w * m))
        var = float(np.sum(w ** 2 * D / P ** 2))
        wm = float(np.max(w / P))
        lo = stats.gamma.ppf(0.025, tasa ** 2 / var, scale=var / tasa) if var > 0 else 0.0
        hi = stats.gamma.ppf(0.975, (tasa + wm) ** 2 / (var + wm ** 2), scale=(var + wm ** 2) / (tasa + wm))
        return {"tasa_estandarizada": tasa * por, "ic_inf": float(lo) * por, "ic_sup": float(hi) * por,
                "tasa_bruta": float(D.sum() / P.sum()) * por, "metodo": "directa"}
    if metodo == "indirecta":
        E = float(np.sum(R * P)); O = float(D.sum())
        lo, hi = _ic_poisson(np.array([O]))
        return {"observadas": O, "esperadas": E, "smr": O / E, "ic_inf": float(lo[0] / E), "ic_sup": float(hi[0] / E),
                "p_valor": float(2 * min(stats.poisson.cdf(O, E), stats.poisson.sf(O - 1, E))), "metodo": "indirecta"}
    raise ValueError("metodo: 'directa' o 'indirecta'")


def exposicion_por_edad(df: pd.DataFrame, fecha_nacimiento: str, fecha_entrada: str, fecha_salida: str, evento: str,
                        edad_min: int = 0, edad_max: int = 110) -> pd.DataFrame:
    """Reparte el tiempo observado de cada persona entre las edades cumplidas (diagrama de Lexis: personas-año en cada
    edad) y asigna el evento a la edad en que ocurre. Devuelve por edad: exposición central, eventos y tasa bruta m_x.
    Es la base para estimar qx de una cartera propia (experiencia) y compararla con una tabla (SMR)."""
    d = df[[fecha_nacimiento, fecha_entrada, fecha_salida, evento]].dropna().copy()
    nac = pd.to_datetime(d[fecha_nacimiento]); ini = pd.to_datetime(d[fecha_entrada]); fin = pd.to_datetime(d[fecha_salida])
    a0 = ((ini - nac).dt.days / 365.25).to_numpy(); a1 = ((fin - nac).dt.days / 365.25).to_numpy()
    if (a1 < a0).any():
        raise ValueError("Hay salidas anteriores a la entrada.")
    edades = np.arange(edad_min, edad_max + 1)
    expo = np.zeros(len(edades)); ev = np.zeros(len(edades))
    for k, x in enumerate(edades):
        expo[k] = np.clip(np.minimum(a1, x + 1) - np.maximum(a0, x), 0, None).sum()
    edad_ev = np.floor(a1[d[evento].to_numpy().astype(bool)]).astype(int)
    for x in edad_ev:
        if edad_min <= x <= edad_max:
            ev[x - edad_min] += 1
    t = pd.DataFrame({"exposicion": expo, "eventos": ev}, index=pd.Index(edades, name="edad"))
    t = t[t.exposicion > 0]
    t["m_x"] = t.eventos / t.exposicion
    t["q_x_aprox"] = 1 - np.exp(-t.m_x)
    return t


def indicadores_fecundidad(nacimientos, mujeres, edades, proporcion_ninas: float = 0.4878, amplitud: int = 1) -> dict:
    """Tasas específicas de fecundidad por edad (o grupo de `amplitud` años), índice sintético de fecundidad
    ISF = amplitud·Σ f_x (hijos por mujer; reemplazo ≈ 2.1), tasa bruta de reproducción (hijas por mujer) y
    edad media a la maternidad."""
    B, W, x = np.asarray(nacimientos, float), np.asarray(mujeres, float), np.asarray(edades, float)
    f = B / W
    isf = float(amplitud * f.sum())
    return {"tasas": pd.Series(f, index=x, name="f_x"), "isf": isf, "tbr": isf * proporcion_ninas,
            "edad_media_maternidad": float(np.sum((x + amplitud / 2) * f) / f.sum())}


def proyeccion_leslie(poblacion_inicial, supervivencia, fecundidad, pasos: int = 10) -> dict:
    """Proyección por cohortes con la matriz de Leslie L: primera fila = fecundidad (hijas por mujer de cada grupo que
    sobreviven al primer grupo), subdiagonal = probabilidades de pasar al grupo siguiente. n(t+1) = L·n(t).
    Devuelve las poblaciones por paso, la tasa de crecimiento a largo plazo λ (autovalor dominante: >1 crece) y la
    estructura estable (autovector)."""
    n0 = np.asarray(poblacion_inicial, float)
    s, F = np.asarray(supervivencia, float), np.asarray(fecundidad, float)
    k = len(n0)
    if len(F) != k or len(s) != k - 1:
        raise ValueError("fecundidad: k valores; supervivencia: k − 1 valores.")
    L = np.zeros((k, k)); L[0] = F; L[np.arange(1, k), np.arange(k - 1)] = s
    pobl = [n0]
    for _ in range(pasos):
        pobl.append(L @ pobl[-1])
    w, V = np.linalg.eig(L)
    i = int(np.argmax(np.abs(w)))
    estable = np.abs(np.real(V[:, i])); estable /= estable.sum()
    return {"poblaciones": pd.DataFrame(pobl, columns=[f"g{j}" for j in range(k)]).rename_axis("paso"),
            "lambda": float(np.real(w[i])), "estructura_estable": estable, "matriz": L}
