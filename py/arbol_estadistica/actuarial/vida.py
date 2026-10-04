"""Matemática actuarial de vida: conmutados, seguros, rentas (también fraccionadas), primas por equivalencia
(neta y de tarifa con gastos), provisión matemática y riesgo de longevidad. Tiempo discreto anual, interés técnico i.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tablas_vida import tabla_mortalidad


def conmutados(tabla: pd.DataFrame, interes: float) -> pd.DataFrame:
    """Valores conmutados: Dx = v^x·lx, Nx = Σ D, Sx = Σ N, Cx = v^(x+1)·dx, Mx = Σ C, Rx = Σ M (v = 1/(1+i)).
    Las fórmulas clásicas: äx = Nx/Dx, Ax = Mx/Dx, nEx = D(x+n)/Dx."""
    v = 1 / (1 + interes)
    x = tabla.index.to_numpy(float)
    D = v ** x * tabla.lx.to_numpy(); C = v ** (x + 1) * tabla.dx.to_numpy()
    N = D[::-1].cumsum()[::-1]; M = C[::-1].cumsum()[::-1]
    return pd.DataFrame({"Dx": D, "Nx": N, "Sx": N[::-1].cumsum()[::-1], "Cx": C, "Mx": M, "Rx": M[::-1].cumsum()[::-1]}, index=tabla.index)


def _probs(tabla, x):
    if x not in tabla.index:
        raise KeyError(f"La edad {x} no está en la tabla.")
    lx = tabla.loc[x:, "lx"].to_numpy(); dx = tabla.loc[x:, "dx"].to_numpy()
    return lx / lx[0], dx / lx[0]          # k p x, k|q x


def seguro_vida(tabla: pd.DataFrame, x: int, interes: float, tipo: str = "vida_entera", n: int | None = None,
                diferido: int = 0, capital: float = 1.0, momento: str = "final_del_año") -> dict:
    """Valor actual actuarial de un seguro de capital `capital` sobre (x).
    tipo: 'vida_entera', 'temporal' (n años), 'dotal_puro' (paga si vive a x+n: nEx), 'mixto' (temporal + dotal puro).
    `diferido` años sin cobertura al principio. momento: 'final_del_año' (A) o 'momento_muerte' (Ā ≈ i/δ·A, UDD).
    Devuelve el valor (prima única pura), la varianza de la pérdida (con el doble de fuerza: ²A − A²) y la desviación."""
    if tipo in ("temporal", "dotal_puro", "mixto") and n is None:
        raise ValueError(f"El seguro {tipo} necesita el plazo n.")
    kp, kq = _probs(tabla, x)
    v = 1 / (1 + interes)

    def valor(vv):
        m, d = diferido, diferido + (n if n is not None else len(kq))
        fallec = 0.0 if tipo == "dotal_puro" else float(np.sum(vv ** np.arange(m + 1, d + 1) * kq[m:d]))
        vivo = float(vv ** d * kp[d]) if tipo in ("dotal_puro", "mixto") and d < len(kp) else 0.0
        if tipo == "vida_entera":
            fallec = float(np.sum(vv ** np.arange(m + 1, len(kq) + 1) * kq[m:]))
        return fallec, vivo
    f1, s1 = valor(v)
    f2, s2 = valor(v ** 2)
    if momento == "momento_muerte":
        delta = np.log(1 + interes); f1 *= interes / delta; f2 *= ((1 + interes) ** 2 - 1) / (2 * delta)
    elif momento != "final_del_año":
        raise ValueError("momento: 'final_del_año' o 'momento_muerte'")
    A, A2 = f1 + s1, f2 + s2
    return {"valor": capital * A, "valor_unitario": A, "varianza": capital ** 2 * (A2 - A ** 2), "desviacion": capital * np.sqrt(max(A2 - A ** 2, 0))}


def renta_actuarial(tabla: pd.DataFrame, x: int, interes: float, n: int | None = None, anticipada: bool = True,
                    diferido: int = 0, fraccionamiento: int = 1, creciente: bool = False, cuantia: float = 1.0) -> float:
    """Valor actual de una renta vitalicia (o temporal de n pagos) de cuantía anual `cuantia` sobre (x):
    anticipada (ä) o vencida (a), diferida `diferido` años, creciente (Iä: 1, 2, 3…) y fraccionada en m pagos al año
    (aproximación UDD/Woolhouse de 2 términos: ä^(m) ≈ ä − (m−1)/(2m)·(1 − nEx)). Ej.: pensión mensual → fraccionamiento=12."""
    kp, _ = _probs(tabla, x)
    v = 1 / (1 + interes)
    ini = diferido + (0 if anticipada else 1)
    fin = len(kp) if n is None else min(diferido + n + (0 if anticipada else 1), len(kp))
    k = np.arange(ini, fin)
    pesos = (k - ini + 1) if creciente else np.ones(len(k))
    a = float(np.sum(pesos * v ** k * kp[k]))
    if fraccionamiento > 1:
        m = fraccionamiento
        E_ini = v ** diferido * kp[diferido] if diferido < len(kp) else 0.0
        E_fin = v ** (diferido + n) * kp[diferido + n] if n is not None and diferido + n < len(kp) else 0.0
        a += (-1 if anticipada else 1) * (m - 1) / (2 * m) * (E_ini - E_fin)
    return cuantia * a


def prima_neta(tabla: pd.DataFrame, x: int, interes: float, tipo: str = "vida_entera", n: int | None = None,
               pagos: int | None = None, capital: float = 1.0) -> dict:
    """Prima pura anual nivelada por el principio de equivalencia: P·ä(x:pagos) = capital·A. `pagos` = años de pago
    (por defecto, la duración del seguro; vida entera = de por vida). Devuelve prima única, anual y la renta de pagos."""
    pu = seguro_vida(tabla, x, interes, tipo, n, capital=capital)["valor"]
    plazo = pagos if pagos is not None else n
    a = renta_actuarial(tabla, x, interes, n=plazo, anticipada=True)
    return {"prima_unica": pu, "prima_anual": pu / a, "renta_pagos": a}


def provision_matematica(tabla: pd.DataFrame, x: int, interes: float, tipo: str = "vida_entera", n: int | None = None,
                         pagos: int | None = None, capital: float = 1.0) -> pd.DataFrame:
    """Provisión matemática prospectiva al inicio de cada año t: tV = capital·A(x+t) − P·ä(x+t: pagos restantes),
    con P la prima neta anual. Para el temporal/mixto, t = 0..n (al vencimiento del mixto, tV = capital)."""
    P = prima_neta(tabla, x, interes, tipo, n, pagos, capital)["prima_anual"]
    hasta = n if n is not None else int(tabla.index.max() - x)
    plazo_p = pagos if pagos is not None else n
    filas = []
    for t in range(0, hasta + 1):
        xt = x + t
        if xt > tabla.index.max():
            break
        resto = None if n is None else n - t
        if resto == 0:
            filas.append({"t": t, "edad": xt, "provision": capital if tipo in ("mixto", "dotal_puro") else 0.0}); continue
        A = seguro_vida(tabla, xt, interes, tipo, resto, capital=capital)["valor"]
        pp = None if plazo_p is None else max(plazo_p - t, 0)
        a = renta_actuarial(tabla, xt, interes, n=pp) if (pp is None or pp > 0) else 0.0
        filas.append({"t": t, "edad": xt, "provision": A - P * a})
    out = pd.DataFrame(filas).set_index("t")
    out.attrs["prima_anual"] = P
    return out


def prima_tarifa(tabla: pd.DataFrame, x: int, interes: float, tipo: str, n: int | None = None, pagos: int | None = None,
                 capital: float = 1.0, alfa: float = 0.03, beta: float = 0.05, gamma: float = 0.002) -> dict:
    """Prima comercial (de tarifa) anual con gastos por equivalencia: G·ä = capital·A + alfa·capital (adquisición, al
    inicio) + beta·G·ä (gastos de cobro, sobre cada prima) + gamma·capital·ä(duración) (gestión, cada año).
    También devuelve la prima de inventario (neta + gestión)."""
    base = prima_neta(tabla, x, interes, tipo, n, pagos, capital)
    a_pag = base["renta_pagos"]
    a_dur = renta_actuarial(tabla, x, interes, n=n, anticipada=True)
    G = (base["prima_unica"] + alfa * capital + gamma * capital * a_dur) / ((1 - beta) * a_pag)
    inventario = (base["prima_unica"] + gamma * capital * a_dur) / a_pag
    return {"prima_tarifa": G, "prima_inventario": inventario, "prima_neta": base["prima_anual"],
            "recargo_total_pct": (G / base["prima_anual"] - 1) * 100}


def sensibilidad_longevidad(tabla: pd.DataFrame, x: int, interes: float, choque: float = -0.20, n: int | None = None) -> dict:
    """Riesgo de longevidad: cuánto sube el valor de una renta vitalicia si la mortalidad baja un `choque` (Solvencia II:
    −20 % permanente en todas las edades). Devuelve valor base, valor con choque y el impacto en %."""
    base = renta_actuarial(tabla, x, interes, n=n)
    q2 = (tabla.qx * (1 + choque)).clip(0, 1)
    t2 = tabla_mortalidad(q2, l0=tabla.lx.iloc[0])
    choc = renta_actuarial(t2, x, interes, n=n)
    return {"renta_base": base, "renta_choque": choc, "impacto_pct": (choc / base - 1) * 100,
            "e_base": float(tabla.loc[x, "e_completa"]), "e_choque": float(t2.loc[x, "e_completa"])}
