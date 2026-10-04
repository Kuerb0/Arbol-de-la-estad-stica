"""IA generativa vista desde la estadística: modelo de lenguaje de n-gramas con muestreo por temperatura/top-k/top-p,
modelo de recompensa de Bradley-Terry (RLHF), recuperación TF-IDF (la «R» de RAG), autoconsistencia por votación
(razonamiento en cadena muestreado varias veces) y el proceso de difusión directo (ruido gaussiano)."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


def _tokens(texto):
    return re.findall(r"\w+|[^\w\s]", texto.lower())


def modelo_ngramas(textos, n: int = 2, suavizado: float = 0.1) -> dict:
    """Modelo de lenguaje de n-gramas: P(siguiente palabra | n−1 anteriores) por conteo con suavizado aditivo. Es la
    misma tarea que un LLM (predecir el siguiente token) con un modelo trivial. Devuelve el vocabulario, una función
    `distribucion(contexto)` y la PERPLEJIDAD sobre los propios textos (exp de la entropía cruzada: cuanto menor,
    mejor predice). Úsalo con `muestrear_texto` para generar."""
    textos = [textos] if isinstance(textos, str) else list(textos)
    toks = [["<s>"] * (n - 1) + _tokens(t) + ["</s>"] for t in textos]
    vocab = sorted({w for t in toks for w in t})
    idx = {w: i for i, w in enumerate(vocab)}
    cuenta = defaultdict(Counter)
    for t in toks:
        for i in range(n - 1, len(t)):
            cuenta[tuple(t[i - n + 1:i])][t[i]] += 1
    V = len(vocab)

    def distribucion(contexto):
        ctx = tuple((["<s>"] * (n - 1) + _tokens(contexto) if isinstance(contexto, str) else list(contexto))[-(n - 1):]) if n > 1 else ()
        c = cuenta.get(ctx, Counter())
        v = np.full(V, suavizado, float)
        for w, k in c.items():
            v[idx[w]] += k
        return pd.Series(v / v.sum(), index=vocab)
    totales = {c: sum(k.values()) for c, k in cuenta.items()}
    ll, m = 0.0, 0
    for t in toks:                                                  # O(nº de tokens): sin construir la distribución entera
        for i in range(n - 1, len(t)):
            c = tuple(t[i - n + 1:i])
            ll += np.log((cuenta[c][t[i]] + suavizado) / (totales[c] + suavizado * V)); m += 1
    return {"vocabulario": vocab, "distribucion": distribucion, "perplejidad": float(np.exp(-ll / m)), "n": n}


def muestrear_siguiente(probabilidades, temperatura: float = 1.0, top_k: int | None = None, top_p: float | None = None,
                        n: int = 1, semilla: int = 42) -> dict:
    """Cómo elige un LLM el siguiente token: reescala las log-probabilidades por la TEMPERATURA (T → 0: siempre el más
    probable; T > 1: más diversidad), recorta a los `top_k` más probables y/o al núcleo `top_p` (los más probables
    hasta acumular p) y muestrea. Devuelve la distribución final, su entropía y las muestras."""
    p = pd.Series(probabilidades, dtype=float)
    if (p < 0).any() or p.sum() <= 0:
        raise ValueError("probabilidades ≥ 0 con suma > 0.")
    lp = np.log(np.clip(p / p.sum(), 1e-300, None))
    if temperatura <= 0:
        q = (lp == lp.max()).astype(float)
    else:
        q = np.exp((lp - lp.max()) / temperatura)
    q = q / q.sum()
    orden = q.sort_values(ascending=False)
    keep = pd.Series(True, index=q.index)
    if top_k is not None:
        keep &= q.index.isin(orden.index[:top_k])
    if top_p is not None:
        cum = orden.cumsum()
        nucleo = orden.index[: int(np.searchsorted(cum.to_numpy(), top_p) + 1)]
        keep &= q.index.isin(nucleo)
    q = q.where(keep, 0); q = q / q.sum()
    rng = np.random.default_rng(semilla)
    m = rng.choice(q.index.to_numpy(), size=n, p=q.to_numpy())
    return {"distribucion": q, "entropia_bits": float(-(q[q > 0] * np.log2(q[q > 0])).sum()), "muestras": list(m)}


def muestrear_texto(modelo: dict, inicio: str = "", max_palabras: int = 20, temperatura: float = 1.0, top_k=None, top_p=None,
                    semilla: int = 42) -> str:
    """Genera texto con un `modelo_ngramas` token a token (autorregresivo) usando `muestrear_siguiente`."""
    n = modelo["n"]
    ctx = ["<s>"] * (n - 1) + _tokens(inicio)
    out = _tokens(inicio)
    for i in range(max_palabras):
        d = modelo["distribucion"](ctx[-(n - 1):] if n > 1 else [])
        w = muestrear_siguiente(d, temperatura, top_k, top_p, semilla=semilla + i)["muestras"][0]
        if w == "</s>":
            break
        out.append(w); ctx.append(w)
    return " ".join(out)


def ajustar_bradley_terry(comparaciones: pd.DataFrame, ganador: str = "ganador", perdedor: str = "perdedor",
                          regularizacion: float = 0.01) -> pd.DataFrame:
    """Modelo de Bradley-Terry: P(i gana a j) = σ(r_i − r_j). Es el modelo de RECOMPENSA de RLHF (a partir de
    preferencias humanas entre pares de respuestas se estima una puntuación latente de cada una) y el de los
    rankings tipo Elo/arena. Ajuste por máxima verosimilitud con una pequeña penalización L2 (identifica la escala
    y evita infinitos con ganadores invictos). Devuelve la puntuación, su EE y la probabilidad de ganar a la media."""
    d = comparaciones[[ganador, perdedor]].dropna()
    items = sorted(set(d[ganador]) | set(d[perdedor]))
    ix = {k: i for i, k in enumerate(items)}
    g, p = d[ganador].map(ix).to_numpy(), d[perdedor].map(ix).to_numpy()
    K = len(items)

    def nll(r):
        z = r[g] - r[p]
        return -np.sum(-np.logaddexp(0, -z)) + regularizacion * np.sum(r ** 2)

    def grad(r):
        z = r[g] - r[p]; s = special.expit(-z)
        gr = np.zeros(K); np.add.at(gr, g, -s); np.add.at(gr, p, s)
        return gr + 2 * regularizacion * r
    r = optimize.minimize(nll, np.zeros(K), jac=grad, method="L-BFGS-B").x
    r -= r.mean()
    z = r[g] - r[p]; w = special.expit(z) * special.expit(-z)
    H = np.zeros((K, K)); np.add.at(H, (g, g), w); np.add.at(H, (p, p), w); np.add.at(H, (g, p), -w); np.add.at(H, (p, g), -w)
    H += 2 * regularizacion * np.eye(K)
    ee = np.sqrt(np.clip(np.diag(np.linalg.pinv(H)), 0, None))
    out = pd.DataFrame({"puntuacion": r, "ee": ee, "prob_ganar_a_la_media": special.expit(r),
                        "victorias": np.bincount(g, minlength=K), "derrotas": np.bincount(p, minlength=K)}, index=items)
    return out.sort_values("puntuacion", ascending=False)


def recuperar_tfidf(documentos, consulta: str, k: int = 3) -> pd.DataFrame:
    """Recuperación por similitud coseno sobre TF-IDF: el paso de búsqueda de un RAG (se recuperan los k fragmentos
    más parecidos a la pregunta y se pasan al modelo como contexto). Los RAG reales usan embeddings densos, pero la
    lógica (representar, comparar, quedarse con los k mejores) es la misma."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = list(documentos)
    if not docs:
        raise ValueError("No hay documentos.")
    v = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    M = v.fit_transform(docs + [consulta])
    sim = (M[:-1] @ M[-1].T).toarray().ravel()
    o = np.argsort(sim)[::-1][:k]
    return pd.DataFrame({"documento": [docs[i] for i in o], "similitud": sim[o]}, index=o)


def autoconsistencia_votacion(prob_acierto: float, n_muestras=(1, 3, 5, 11, 21, 41), n_respuestas_erroneas: int = 1,
                              n_sim: int = 20000, semilla: int = 42) -> pd.DataFrame:
    """Autoconsistencia (self-consistency): muestrear varias cadenas de razonamiento y quedarse con la respuesta más
    votada. Si cada cadena acierta con prob. p y los errores se reparten entre `n_respuestas_erroneas` respuestas
    distintas, la mayoría acierta con más probabilidad que una cadena sola cuando p supera a la respuesta errónea más
    frecuente (teorema del jurado de Condorcet). Devuelve la precisión simulada por nº de muestras."""
    rng = np.random.default_rng(semilla)
    q = (1 - prob_acierto) / n_respuestas_erroneas
    probs = np.r_[prob_acierto, np.full(n_respuestas_erroneas, q)]
    filas = []
    for m in n_muestras:
        votos = rng.multinomial(m, probs, size=n_sim)
        mejor = votos.max(axis=1)
        gana = (votos[:, 0] == mejor) & ((votos == mejor[:, None]).sum(axis=1) == 1)
        empate = (votos[:, 0] == mejor) & ((votos == mejor[:, None]).sum(axis=1) > 1)
        filas.append({"muestras": m, "precision": float(np.mean(gana + empate / (votos == mejor[:, None]).sum(axis=1)))})
    return pd.DataFrame(filas).set_index("muestras")


def proceso_difusion(x0, pasos: int = 1000, beta_ini: float = 1e-4, beta_fin: float = 0.02, instantes=(0, 50, 200, 500, 999),
                     semilla: int = 42) -> dict:
    """Proceso de difusión DIRECTO (DDPM): x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·ε con ᾱ_t = Π(1−β_s). Los datos se convierten en ruido
    N(0, 1) de forma controlada; un modelo de difusión aprende el camino inverso (quitar el ruido paso a paso).
    Devuelve las muestras en varios instantes y la comprobación de que al final la distribución es normal estándar."""
    x = np.asarray(x0, float)
    b = np.linspace(beta_ini, beta_fin, pasos)
    ab = np.cumprod(1 - b)
    rng = np.random.default_rng(semilla)
    out = {}
    for t in instantes:
        t = min(int(t), pasos - 1)
        out[t] = np.sqrt(ab[t]) * x + np.sqrt(1 - ab[t]) * rng.normal(size=x.shape)
    final = out[max(out)]
    return {"muestras": out, "alfa_barra": pd.Series(ab, name="alfa_barra"), "media_final": float(final.mean()), "sd_final": float(final.std()),
            "p_normal_final": float(stats.kstest(final.ravel(), "norm").pvalue)}
