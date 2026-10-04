"""Aprendizaje no supervisado (2): reglas de asociación (cesta de la compra, Apriori) y mapas autoorganizados (SOM)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def reglas_asociacion(transacciones, soporte_min: float = 0.01, confianza_min: float = 0.3, max_longitud: int = 3) -> dict:
    """Apriori: conjuntos frecuentes (soporte ≥ `soporte_min`) y reglas A → B con soporte, confianza P(B|A), LIFT
    (confianza / P(B): > 1 asociación positiva) y convicción. `transacciones`: lista de listas (cestas) o DataFrame
    0/1 (filas = cestas, columnas = productos). Gauss: con miles de reglas aparecen asociaciones por azar; prioriza
    lift alto con soporte razonable y valida en otro periodo."""
    if isinstance(transacciones, pd.DataFrame):
        M = transacciones.astype(bool)
    else:
        items = sorted({i for t in transacciones for i in t})
        M = pd.DataFrame([[i in set(t) for i in items] for t in transacciones], columns=items)
    A = M.to_numpy(); n = len(A); cols = list(M.columns)
    sop1 = A.mean(axis=0)
    frecuentes = {(j,): s for j, s in enumerate(sop1) if s >= soporte_min}
    nivel = list(frecuentes)
    for k in range(2, max_longitud + 1):
        cand = set()
        for a, b in itertools.combinations(nivel, 2):
            u = tuple(sorted(set(a) | set(b)))
            if len(u) == k and all(tuple(sorted(s)) in frecuentes for s in itertools.combinations(u, k - 1)):
                cand.add(u)
        nivel = []
        for c in cand:
            s = A[:, list(c)].all(axis=1).mean()
            if s >= soporte_min:
                frecuentes[c] = s; nivel.append(c)
        if not nivel:
            break
    reglas = []
    for c, s in frecuentes.items():
        if len(c) < 2:
            continue
        for r in range(1, len(c)):
            for ant in itertools.combinations(c, r):
                con = tuple(j for j in c if j not in ant)
                sa = frecuentes.get(tuple(sorted(ant)), A[:, list(ant)].all(axis=1).mean())
                sc = frecuentes.get(tuple(sorted(con)), A[:, list(con)].all(axis=1).mean())
                conf = s / sa
                if conf >= confianza_min:
                    reglas.append({"antecedente": ", ".join(cols[j] for j in ant), "consecuente": ", ".join(cols[j] for j in con),
                                   "soporte": s, "confianza": conf, "lift": conf / sc, "conviccion": (1 - sc) / (1 - conf) if conf < 1 else np.inf})
    R = pd.DataFrame(reglas, columns=["antecedente", "consecuente", "soporte", "confianza", "lift", "conviccion"]).sort_values("lift", ascending=False, ignore_index=True)
    F = pd.DataFrame([{"conjunto": ", ".join(cols[j] for j in c), "longitud": len(c), "soporte": s} for c, s in frecuentes.items()]).sort_values("soporte", ascending=False, ignore_index=True)
    return {"frecuentes": F, "reglas": R, "n_transacciones": n}


def mapa_autoorganizado(X, filas: int = 6, columnas: int = 6, iteraciones: int = 5000, tasa: float = 0.5,
                        radio: float | None = None, semilla: int = 42) -> dict:
    """Mapa autoorganizado de Kohonen (SOM) en una rejilla rectangular: cada neurona tiene un prototipo; en cada paso
    la más parecida a un dato (BMU) y sus vecinas se acercan a él (tasa y radio decrecen). Resultado: una
    proyección 2D que conserva la topología (clientes parecidos caen en celdas cercanas). Devuelve los prototipos (en
    escala original), la celda de cada observación, el error de cuantización y la matriz U (distancia media a las
    vecinas: valores altos = fronteras entre grupos)."""
    Xd = pd.DataFrame(X).astype(float)
    mu, sd = Xd.mean(), Xd.std(ddof=1).replace(0, 1)
    Z = ((Xd - mu) / sd).to_numpy()
    rng = np.random.default_rng(semilla)
    K = filas * columnas
    pos = np.array([(i, j) for i in range(filas) for j in range(columnas)], float)
    W = Z[rng.choice(len(Z), K, replace=len(Z) < K)] + rng.normal(0, 0.01, (K, Z.shape[1]))
    r0 = radio or max(filas, columnas) / 2
    D2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    for t in range(iteraciones):
        fr = t / iteraciones
        a, r = tasa * (1 - fr) + 0.01, r0 * (1 - fr) + 0.5
        x = Z[rng.integers(len(Z))]
        b = int(np.argmin(((W - x) ** 2).sum(1)))
        h = np.exp(-D2[b] / (2 * r ** 2))
        W += a * h[:, None] * (x - W)
    dist = ((Z[:, None, :] - W[None, :, :]) ** 2).sum(-1)
    bmu = dist.argmin(1)
    U = np.zeros(K)
    for k in range(K):
        vec = np.where(D2[k] == 1)[0]
        U[k] = np.mean(np.sqrt(((W[vec] - W[k]) ** 2).sum(1)))
    return {"prototipos": pd.DataFrame(W * sd.to_numpy() + mu.to_numpy(), columns=Xd.columns).assign(fila=pos[:, 0].astype(int), columna=pos[:, 1].astype(int)),
            "celda": pd.DataFrame({"fila": pos[bmu, 0].astype(int), "columna": pos[bmu, 1].astype(int)}, index=Xd.index),
            "error_cuantizacion": float(np.sqrt(dist.min(1)).mean()), "matriz_u": U.reshape(filas, columnas),
            "conteos": np.bincount(bmu, minlength=K).reshape(filas, columnas)}
