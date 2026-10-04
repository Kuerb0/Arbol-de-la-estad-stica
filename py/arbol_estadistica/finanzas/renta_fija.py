"""Renta fija y derivados: convenciones de días, bonos (precio, TIR, duración, convexidad), estructura temporal
(bootstrapping), inmunización, FRA y swaps, Black-Scholes con griegas y árbol binomial (valoración neutral al riesgo)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from scipy import optimize, stats


def fraccion_anio(inicio, fin, convencion: str = "act/365") -> float:
    """Fracción de año entre dos fechas: 'act/365', 'act/360' (monetario), '30/360' (bono europeo, 30E/360) o
    'act/act' (ISDA: reparte por años naturales)."""
    a, b = pd.Timestamp(inicio).date(), pd.Timestamp(fin).date()
    if convencion == "act/365":
        return (b - a).days / 365
    if convencion == "act/360":
        return (b - a).days / 360
    if convencion == "30/360":
        d1, d2 = min(a.day, 30), min(b.day, 30)
        return ((b.year - a.year) * 360 + (b.month - a.month) * 30 + (d2 - d1)) / 360
    if convencion == "act/act":
        total, cur = 0.0, a
        while cur < b:
            fin_anio = dt.date(cur.year + 1, 1, 1)
            tramo_fin = min(b, fin_anio)
            dias_anio = 366 if (cur.year % 4 == 0 and (cur.year % 100 != 0 or cur.year % 400 == 0)) else 365
            total += (tramo_fin - cur).days / dias_anio
            cur = tramo_fin
        return total
    raise ValueError("convencion: 'act/365', 'act/360', '30/360' o 'act/act'")


def _flujos(nominal, cupon, anios, frecuencia):
    n = int(round(anios * frecuencia))
    t = np.arange(1, n + 1) / frecuencia
    c = np.full(n, nominal * cupon / frecuencia); c[-1] += nominal
    return t, c


def precio_bono(nominal: float, cupon: float, anios: float, tir: float, frecuencia: int = 1) -> dict:
    """Precio de un bono con cupón fijo descontando a la TIR (compuesta con la frecuencia de cupón), duración de
    Macaulay, duración modificada (sensibilidad: ΔP/P ≈ −D_mod·Δy) y convexidad."""
    t, c = _flujos(nominal, cupon, anios, frecuencia)
    v = (1 + tir / frecuencia) ** (-t * frecuencia)
    P = float(np.sum(c * v))
    D = float(np.sum(t * c * v) / P)
    conv = float(np.sum(c * v * t * (t + 1 / frecuencia)) / (P * (1 + tir / frecuencia) ** 2))
    return {"precio": P, "duracion_macaulay": D, "duracion_modificada": D / (1 + tir / frecuencia), "convexidad": conv}


def tir_bono(precio: float, nominal: float, cupon: float, anios: float, frecuencia: int = 1) -> float:
    """TIR (rentabilidad al vencimiento) que iguala el precio al valor actual de los flujos (Brent)."""
    f = lambda y: precio_bono(nominal, cupon, anios, y, frecuencia)["precio"] - precio      # noqa: E731
    return float(optimize.brentq(f, -0.99 * frecuencia + 1e-6, 5.0))


def bootstrapping_etti(plazos, cupones, precios, nominal: float = 100.0) -> pd.DataFrame:
    """Estructura temporal de tipos (ETTI) por bootstrapping a partir de bonos con cupón anual y vencimientos 1, 2, …:
    despeja cada factor de descuento con los anteriores. Devuelve factor de descuento, tipo cupón cero y tipo forward
    a un año. Base para valorar cualquier flujo cierto."""
    T = np.asarray(plazos, int); C = np.asarray(cupones, float); P = np.asarray(precios, float)
    orden = np.argsort(T); T, C, P = T[orden], C[orden], P[orden]
    if not np.array_equal(T, np.arange(1, len(T) + 1)):
        raise ValueError("Los plazos deben ser 1, 2, …, n años (uno por año).")
    df = []
    for k in range(len(T)):
        cup = C[k] * nominal
        df.append((P[k] - cup * sum(df)) / (cup + nominal))
    df = np.array(df)
    cero = df ** (-1 / T) - 1
    fwd = np.r_[cero[0], df[:-1] / df[1:] - 1]
    return pd.DataFrame({"plazo": T, "factor_descuento": df, "tipo_cero": cero, "forward_1a": fwd}).set_index("plazo")


def inmunizacion(duracion_pasivo: float, valor_pasivo: float, bonos: pd.DataFrame) -> dict:
    """Inmunización de Redington con dos bonos: pesos que igualan valor actual y duración del activo a los del pasivo
    (`bonos` con columnas precio y duracion, dos filas). Comprueba la convexidad si hay columna 'convexidad'."""
    b = pd.DataFrame(bonos)
    if len(b) != 2:
        raise ValueError("Pasa exactamente dos bonos.")
    d1, d2 = b["duracion"].to_numpy(float)
    if not min(d1, d2) <= duracion_pasivo <= max(d1, d2):
        raise ValueError("La duración del pasivo debe estar entre las de los dos bonos.")
    w1 = (duracion_pasivo - d2) / (d1 - d2)
    w = np.array([w1, 1 - w1])
    importe = w * valor_pasivo
    out = {"pesos": pd.Series(w, index=b.index), "importe": pd.Series(importe, index=b.index),
           "titulos": pd.Series(importe / b["precio"].to_numpy(float), index=b.index)}
    if "convexidad" in b:
        out["convexidad_activo"] = float(w @ b["convexidad"].to_numpy(float))
    return out


def valorar_swap(tipo_fijo: float, factores_descuento, nominal: float = 1.0, frecuencia: int = 1) -> dict:
    """Swap de tipo de interés (pagas fijo, recibes variable) con la curva de factores de descuento de las fechas de
    pago: valor de la pata variable = N·(1 − DF_n); fija = N·c·Σ DF/frecuencia. Devuelve valor para el pagador del fijo
    y el tipo swap de equilibrio (valor 0). Un FRA es el caso de un solo periodo."""
    DF = np.asarray(factores_descuento, float)
    anualidad = DF.sum() / frecuencia
    flotante = nominal * (1 - DF[-1]); fija = nominal * tipo_fijo * anualidad
    return {"valor_pagador_fijo": float(flotante - fija), "tipo_swap_par": float((1 - DF[-1]) / anualidad), "pata_variable": float(flotante),
            "pata_fija": float(fija)}


def black_scholes(S: float, K: float, T: float, r: float, sigma: float, tipo: str = "call", q: float = 0.0) -> dict:
    """Precio de una opción europea de Black-Scholes-Merton (dividendo continuo q) y sus griegas: delta, gamma, vega
    (por 1 punto de volatilidad), theta (por año) y rho. Supone volatilidad constante y log-normalidad."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T y sigma deben ser > 0.")
    d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T)); d2 = d1 - sigma * np.sqrt(T)
    N, n = stats.norm.cdf, stats.norm.pdf
    if tipo == "call":
        p = S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2); delta = np.exp(-q * T) * N(d1)
        theta = -S * np.exp(-q * T) * n(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N(d2) + q * S * np.exp(-q * T) * N(d1)
        rho = K * T * np.exp(-r * T) * N(d2)
    elif tipo == "put":
        p = K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1); delta = -np.exp(-q * T) * N(-d1)
        theta = -S * np.exp(-q * T) * n(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * N(-d2) - q * S * np.exp(-q * T) * N(-d1)
        rho = -K * T * np.exp(-r * T) * N(-d2)
    else:
        raise ValueError("tipo: 'call' o 'put'")
    return {"precio": float(p), "delta": float(delta), "gamma": float(np.exp(-q * T) * n(d1) / (S * sigma * np.sqrt(T))),
            "vega": float(S * np.exp(-q * T) * n(d1) * np.sqrt(T) / 100), "theta": float(theta), "rho": float(rho), "d1": float(d1), "d2": float(d2)}


def arbol_binomial(S: float, K: float, T: float, r: float, sigma: float, pasos: int = 200, tipo: str = "call",
                   americana: bool = False) -> dict:
    """Árbol binomial de Cox-Ross-Rubinstein: u = e^{σ√Δt}, d = 1/u, probabilidad NEUTRAL AL RIESGO
    p = (e^{rΔt} − d)/(u − d); valor = descuento de la esperanza bajo p. Con `americana=True` permite ejercicio
    anticipado (la put americana vale más que la europea). Converge a Black-Scholes al aumentar los pasos."""
    dtt = T / pasos; u = np.exp(sigma * np.sqrt(dtt)); d = 1 / u
    p = (np.exp(r * dtt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValueError("Probabilidad neutral al riesgo fuera de (0, 1): aumenta los pasos.")
    j = np.arange(pasos + 1)
    ST = S * u ** j * d ** (pasos - j)
    V = np.maximum(ST - K, 0) if tipo == "call" else np.maximum(K - ST, 0)
    for i in range(pasos - 1, -1, -1):
        V = np.exp(-r * dtt) * (p * V[1:] + (1 - p) * V[:-1])
        if americana:
            Si = S * u ** np.arange(i + 1) * d ** (i - np.arange(i + 1))
            V = np.maximum(V, (Si - K) if tipo == "call" else (K - Si))
    return {"precio": float(V[0]), "p_neutral_riesgo": float(p), "u": float(u), "d": float(d)}


def arbol_black_derman_toy(tipos_cero, volatilidades, dt: float = 1.0) -> dict:
    """Modelo de tipos de Black-Derman-Toy calibrado por inducción hacia delante con precios de Arrow-Debreu: árbol
    binomial recombinante del tipo corto r_{i,j} = a_i·exp(2σ_i√dt·j) (log-normal, probabilidades 1/2) que reproduce
    EXACTAMENTE la curva cupón cero dada (`tipos_cero`, compuestos por periodo, plazos dt, 2dt, …) con las
    volatilidades del tipo corto de cada periodo. Sirve para valorar flujos dependientes de la trayectoria de tipos
    (opciones sobre bonos, caps, bonos rescatables). Devuelve el árbol de tipos y los factores de descuento."""
    z = np.asarray(tipos_cero, float); s = np.broadcast_to(np.asarray(volatilidades, float), z.shape).astype(float)
    n = len(z)
    P = (1 + z) ** (-dt * np.arange(1, n + 1))                      # tipos cero con capitalización anual
    arbol = []
    Q = np.array([1.0])                                            # precios de Arrow-Debreu en el nivel i
    for i in range(n):
        j = np.arange(i + 1)
        forma = np.exp(2 * s[i] * np.sqrt(dt) * j)
        f = lambda a: np.sum(Q / (1 + a * forma * dt)) - P[i]     # noqa: E731
        a = optimize.brentq(f, 1e-10, 10.0)
        r = a * forma
        arbol.append(r)
        desc = Q / (1 + r * dt)
        Qn = np.zeros(i + 2)
        Qn[:-1] += 0.5 * desc; Qn[1:] += 0.5 * desc
        Q = Qn
    T = pd.DataFrame([np.r_[r, [np.nan] * (n - len(r))] for r in arbol]).T
    T.index.name = "nodo (subidas)"; T.columns.name = "periodo"
    return {"arbol_tipos": T, "factores_descuento": pd.Series(P, index=np.arange(1, n + 1) * dt), "volatilidades": s}
