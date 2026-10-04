"""Biometría: tablas de mortalidad, leyes de Gompertz-Makeham, fracciones de año, vida futura, varias cabezas y
múltiples decrementos. Notación actuarial internacional (lx, dx, qx, px, ex, tpx).

Origen: temario de Biometría y Matemática Actuarial de Vida (máster UAH).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from .._util import _numerico


def qx_ley(edades, ley: str = "makeham", A: float = 0.0005, B: float = 0.00007, c: float = 1.10) -> pd.Series:
    """Probabilidades anuales de fallecimiento qx de una ley de mortalidad con fuerza μ(x) = A + B·c^x
    (Gompertz: A = 0). qx = 1 − exp(−A − B·c^x·(c − 1)/ln c). Útil para tener una tabla de ejemplo o suavizar."""
    x = np.asarray(edades, float)
    if ley == "gompertz":
        A = 0.0
    elif ley != "makeham":
        raise ValueError("ley: 'gompertz' o 'makeham'")
    return pd.Series(1 - np.exp(-A - B * c ** x * (c - 1) / np.log(c)), index=np.asarray(edades), name="qx")


def tabla_mortalidad(qx, edad_inicial: int = 0, l0: float = 100000.0) -> pd.DataFrame:
    """Tabla completa a partir de qx (Serie indexada por edad o vector desde `edad_inicial`): qx, px, lx, dx,
    Lx (UDD: lx − dx/2), Tx, esperanza de vida completa e̊x = Tx/lx y abreviada ex = Σ_{k≥1} l_{x+k}/lx.
    La última edad se cierra con q = 1. Equivale a una tabla de PERM/F o PASEM construida a mano."""
    q = pd.Series(qx).astype(float) if isinstance(qx, pd.Series) else pd.Series(np.asarray(qx, float), index=np.arange(edad_inicial, edad_inicial + len(qx)))
    if ((q < 0) | (q > 1)).any():
        raise ValueError("qx debe estar en [0, 1].")
    q = q.copy(); q.iloc[-1] = 1.0
    lx = l0 * np.r_[1.0, np.cumprod(1 - q.to_numpy())[:-1]]
    dx = lx * q.to_numpy()
    Lx = lx - dx / 2
    Tx = Lx[::-1].cumsum()[::-1]
    lsig = np.r_[lx[1:], 0.0]
    ex = np.where(lx > 0, (lsig[::-1].cumsum()[::-1]) / np.where(lx > 0, lx, 1), 0.0)
    return pd.DataFrame({"qx": q.to_numpy(), "px": 1 - q.to_numpy(), "lx": lx, "dx": dx, "Lx": Lx, "Tx": Tx,
                         "e_completa": np.where(lx > 0, Tx / np.where(lx > 0, lx, 1), 0.0), "e_abreviada": ex}, index=pd.Index(q.index, name="edad"))


def ajustar_ley_mortalidad(edades, defunciones=None, expuestos=None, qx=None, ley: str = "makeham") -> dict:
    """Ajusta Gompertz (μ = B·c^x) o Makeham (μ = A + B·c^x).
    Con `defunciones` y `expuestos` (exposición central): máxima verosimilitud de Poisson, D_x ~ Poisson(E_x·μ(x+½)).
    Con `qx` brutas: mínimos cuadrados sobre log qx. Devuelve parámetros, qx ajustadas y la desviación (χ² de Pearson
    si hay defunciones). Gauss: con pocas muertes en edades extremas, el ajuste manda; compara con los datos brutos."""
    x = np.asarray(edades, float)

    def mu(p, xx):
        A, lB, lc = (0.0, *p) if ley == "gompertz" else p
        return A + np.exp(lB) * np.exp(lc) ** xx
    p0 = [-10.0, np.log(1.1)] if ley == "gompertz" else [0.0005, -10.0, np.log(1.1)]
    if defunciones is not None and expuestos is not None:
        D, E = np.asarray(defunciones, float), np.asarray(expuestos, float)

        def nll(p):
            m = mu(p, x + 0.5)
            return np.inf if (m <= 0).any() else -np.sum(D * np.log(m) - E * m)
        r = optimize.minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10})
        m = mu(r.x, x + 0.5)
        chi2 = float(np.sum((D - E * m) ** 2 / (E * m)))
        metodo = "Poisson (MV)"
    elif qx is not None:
        qo = np.asarray(qx, float)

        def ss(p):
            m = mu(p, x + 0.5)
            return np.inf if (m <= 0).any() else np.sum((np.log(1 - np.exp(-m)) - np.log(qo)) ** 2)
        r = optimize.minimize(ss, p0, method="Nelder-Mead", options={"maxiter": 20000})
        chi2, metodo = np.nan, "mínimos cuadrados (log qx)"
    else:
        raise ValueError("Pasa defunciones + expuestos o qx.")
    A, lB, lc = (0.0, *r.x) if ley == "gompertz" else r.x
    par = {"A": float(A), "B": float(np.exp(lB)), "c": float(np.exp(lc))}
    return {"parametros": par, "qx_ajustadas": qx_ley(edades, ley, **par), "chi2": chi2, "metodo": metodo, "convergio": bool(r.success)}


def probabilidad_supervivencia(tabla: pd.DataFrame, x: float, t: float, hipotesis: str = "udd") -> float:
    """t p x para edades y plazos NO enteros con la hipótesis de fracciones de año: 'udd' (distribución uniforme de las
    muertes), 'constante' (fuerza de mortalidad constante en el año) o 'balducci'. Para edades enteras coincide con lx+t/lx."""
    def frac(xe, s):          # s p xe, xe entero y 0 ≤ s ≤ 1
        q = float(tabla.loc[xe, "qx"])
        if hipotesis == "udd":
            return 1 - s * q
        if hipotesis == "constante":
            return (1 - q) ** s
        if hipotesis == "balducci":
            return (1 - q) / (1 - (1 - s) * q)
        raise ValueError("hipotesis: 'udd', 'constante' o 'balducci'")
    if t < 0:
        raise ValueError("t ≥ 0")
    x0, f0 = int(np.floor(x)), x - np.floor(x)
    fin = x + t
    x1, f1 = int(np.floor(fin)), fin - np.floor(fin)
    if x1 > tabla.index.max():
        return 0.0
    # l(x) relativo con la hipótesis dentro de cada año
    l_ini = tabla.loc[x0, "lx"] * frac(x0, f0)
    l_fin = tabla.loc[x1, "lx"] * frac(x1, f1) if x1 in tabla.index else 0.0
    return float(l_fin / l_ini) if l_ini > 0 else 0.0


def vida_futura(tabla: pd.DataFrame, x: int) -> dict:
    """Vida futura abreviada K_x (años completos que vivirá alguien de edad x) como variable aleatoria: función de
    probabilidad P(K = k) = k|qx, esperanza (= e_x abreviada), varianza y cuantiles (mediana de vida restante)."""
    if x not in tabla.index:
        raise KeyError(f"La edad {x} no está en la tabla.")
    t = tabla.loc[x:]
    pk = (t["dx"] / t.loc[x, "lx"]).to_numpy()
    k = np.arange(len(pk))
    media = float(np.sum(k * pk)); var = float(np.sum(k ** 2 * pk) - media ** 2)
    F = np.cumsum(pk)
    cuant = {q: int(k[np.searchsorted(F, q)]) for q in (0.25, 0.5, 0.75, 0.95)}
    return {"funcion_probabilidad": pd.Series(pk, index=k, name="P(K=k)"), "esperanza": media, "varianza": var,
            "esperanza_completa_aprox": media + 0.5, "cuantiles": cuant}


def tabla_conjunta(tabla_x: pd.DataFrame, x: int, tabla_y: pd.DataFrame, y: int) -> pd.DataFrame:
    """Dos cabezas independientes de edades x e y: t p x, t p y, vida conjunta t p xy (viven ambos) y último superviviente
    t p x̄ȳ = tpx + tpy − tpxy, con sus esperanzas abreviadas en `attrs`. Base de rentas de viudedad y seguros sobre dos vidas."""
    n = min(len(tabla_x.loc[x:]), len(tabla_y.loc[y:]))
    px = tabla_x.loc[x:, "lx"].to_numpy()[:n] / tabla_x.loc[x, "lx"]
    py = tabla_y.loc[y:, "lx"].to_numpy()[:n] / tabla_y.loc[y, "lx"]
    out = pd.DataFrame({"t": np.arange(n), "tpx": px, "tpy": py, "tpxy": px * py, "tp_ultimo": px + py - px * py}).set_index("t")
    out.attrs.update({"e_xy": float(out.tpxy.iloc[1:].sum()), "e_ultimo": float(out.tp_ultimo.iloc[1:].sum())})
    return out


def decrementos_multiples(q_dependientes: pd.DataFrame | None = None, q_independientes: pd.DataFrame | None = None) -> dict:
    """Múltiples decrementos (muerte, invalidez, rescate…) con UDD en cada decremento:
    de tasas DEPENDIENTES q^(j) (las observadas, compitiendo) a INDEPENDIENTES q'^(j) = 1 − (p^(τ))^(q^(j)/q^(τ)) y al revés
    (aproximación de producto: p^(τ) = Π (1 − q'^(j))). Pasa uno de los dos DataFrames (filas = edades, columnas = causas)."""
    if (q_dependientes is None) == (q_independientes is None):
        raise ValueError("Pasa q_dependientes O q_independientes.")
    if q_dependientes is not None:
        qd = pd.DataFrame(q_dependientes).astype(float)
        qt = qd.sum(axis=1)
        if (qt > 1).any():
            raise ValueError("La suma de tasas dependientes supera 1 en alguna edad.")
        pt = 1 - qt
        qi = qd.apply(lambda col: 1 - pt ** (col / qt.replace(0, np.nan))).fillna(0.0)
    else:
        qi = pd.DataFrame(q_independientes).astype(float)
        pt = (1 - qi).prod(axis=1); qt = 1 - pt
        # UDD: q^(j) = q'^(j)·(1 − ½ Σ_{k≠j} q'^(k) + ⅓ Σ_{k<l, ≠j} q'^(k)q'^(l)) (exacta para 2 y 3 causas)
        cols = list(qi.columns)
        qd = pd.DataFrame(index=qi.index)
        for j in cols:
            otras = [c for c in cols if c != j]
            s1 = qi[otras].sum(axis=1)
            s2 = sum(qi[a] * qi[b] for i, a in enumerate(otras) for b in otras[i + 1:]) if len(otras) > 1 else 0
            qd[j] = qi[j] * (1 - s1 / 2 + s2 / 3)
    return {"dependientes": qd, "independientes": qi, "q_total": qt, "p_total": pt}
