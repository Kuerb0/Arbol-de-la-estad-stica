"""Rama CONTRASTES / potencia y tamaño muestral (analítico y por simulación).

Origen: hueco del catálogo (Very Normal: «Explaining Power», «An easier way to do sample size
calculations», «2-Minute Power Analysis in R», «Why does the number 30 appear so much…»).
Equivale a PROC POWER (TWOSAMPLEMEANS, TWOSAMPLEFREQ) de SAS / al paquete `pwr` de R.

Recordatorio: potencia = P(rechazar H0 | H0 es falsa) = 1 - beta. Depende de cuatro cosas que se
fijan entre sí: tamaño del efecto, n, alpha y potencia. Dadas tres, se despeja la cuarta.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

_ALT = {"two-sided": "two-sided", "dos_colas": "two-sided", "larger": "larger", "mayor": "larger",
        "smaller": "smaller", "menor": "smaller"}


def _alternativa(a: str) -> str:
    if a not in _ALT:
        raise ValueError(f"alternativa debe ser una de {sorted(_ALT)}")
    return _ALT[a]


def _efecto_d(efecto, diferencia, desviacion) -> float:
    if efecto is not None:
        return float(efecto)
    if diferencia is None or desviacion is None or desviacion <= 0:
        raise ValueError("Indica `efecto` (d de Cohen) o `diferencia` y `desviacion` (> 0).")
    return float(diferencia) / float(desviacion)


def tamano_muestral_medias(efecto: float | None = None, diferencia: float | None = None,
                           desviacion: float | None = None, alpha: float = 0.05, potencia: float = 0.80,
                           ratio: float = 1.0, alternativa: str = "two-sided") -> dict:
    """n necesario para comparar dos medias con un t-test (grupos independientes).

    efecto  : d de Cohen = diferencia / desviación típica común (0.2 pequeño, 0.5 medio, 0.8 grande).
              Alternativamente pasa `diferencia` y `desviacion` en unidades reales.
    ratio   : n2 / n1 (1 = grupos iguales).
    Devuelve dict con n_grupo1, n_grupo2, n_total (redondeados hacia arriba), d y la potencia real
    que se obtiene con esos n.  Ejemplo clásico: d=0.5, alpha=0.05, potencia=0.8 -> 64 por grupo.
    OJO: el «n = 30» no es una regla de tamaño muestral; el n lo marca el efecto que quieres detectar.
    """
    d = _efecto_d(efecto, diferencia, desviacion)
    if d == 0:
        raise ValueError("Con efecto 0 no existe n que dé potencia > alpha.")
    alt = _alternativa(alternativa)
    calc = TTestIndPower()
    n1 = calc.solve_power(effect_size=abs(d), alpha=alpha, power=potencia, ratio=ratio, alternative=alt)
    n1 = math.ceil(float(n1))
    n2 = math.ceil(n1 * ratio)
    real = float(calc.power(effect_size=abs(d), nobs1=n1, alpha=alpha, ratio=n2 / n1, alternative=alt))
    return {"n_grupo1": n1, "n_grupo2": n2, "n_total": n1 + n2, "d": d, "alpha": alpha,
            "potencia_objetivo": potencia, "potencia_real": real}


def tamano_muestral_proporciones(p1: float, p2: float, alpha: float = 0.05, potencia: float = 0.80,
                                 ratio: float = 1.0, alternativa: str = "two-sided") -> dict:
    """n por grupo para detectar la diferencia entre dos proporciones (p. ej. tasas de conversión).

    Usa el tamaño de efecto h de Cohen (transformación arcoseno), como `pwr.2p.test` de R.
    Ejemplo: 50 % frente a 60 % con potencia 0.8 -> unos 388 por grupo.
    Gauss: detectar diferencias pequeñas en proporciones cuesta MUCHA muestra; con prevalencias
    raras (p < 0.05) la aproximación normal empeora: comprobar con potencia_por_simulacion.
    """
    for p in (p1, p2):
        if not 0 < p < 1:
            raise ValueError("p1 y p2 deben estar en (0, 1).")
    if p1 == p2:
        raise ValueError("p1 y p2 son iguales: no hay diferencia que detectar.")
    h = float(proportion_effectsize(p2, p1))
    alt = _alternativa(alternativa)
    calc = NormalIndPower()
    n1 = math.ceil(float(calc.solve_power(effect_size=abs(h), alpha=alpha, power=potencia, ratio=ratio,
                                          alternative=alt)))
    n2 = math.ceil(n1 * ratio)
    real = float(calc.power(effect_size=abs(h), nobs1=n1, alpha=alpha, ratio=n2 / n1, alternative=alt))
    return {"n_grupo1": n1, "n_grupo2": n2, "n_total": n1 + n2, "h_cohen": h, "alpha": alpha,
            "potencia_objetivo": potencia, "potencia_real": real}


def potencia_contraste_medias(n_grupo1: int, efecto: float, alpha: float = 0.05, ratio: float = 1.0,
                              alternativa: str = "two-sided") -> float:
    """Potencia de un t-test de dos grupos con n_grupo1 (y n_grupo1 * ratio) observaciones y d = efecto."""
    if n_grupo1 < 2:
        raise ValueError("n_grupo1 debe ser >= 2.")
    return float(TTestIndPower().power(effect_size=abs(efecto), nobs1=n_grupo1, alpha=alpha, ratio=ratio,
                                       alternative=_alternativa(alternativa)))


def curva_potencia(efecto: float, n_valores=None, alpha: float = 0.05, ratio: float = 1.0,
                   alternativa: str = "two-sided") -> pd.DataFrame:
    """Tabla n -> potencia para un efecto dado (para dibujar o elegir n). Por defecto n = 5..300."""
    n_valores = np.unique(np.linspace(5, 300, 60).astype(int)) if n_valores is None else np.asarray(n_valores)
    pot = [potencia_contraste_medias(int(n), efecto, alpha, ratio, alternativa) for n in n_valores]
    return pd.DataFrame({"n_grupo1": n_valores.astype(int), "potencia": pot})


def potencia_por_simulacion(generador: Callable, contraste: Callable, n: int, n_sim: int = 1000,
                            alpha: float = 0.05, semilla: int = 42) -> dict:
    """Potencia por Monte Carlo: simula `n_sim` estudios y cuenta en cuántos p < alpha.

    generador(rng, n) -> tupla de muestras (p. ej. dos arrays); contraste(*muestras) -> p-valor.
    Sirve para cualquier test/diseño sin fórmula cerrada (no paramétricos, datos sesgados,
    clústeres...). Si el generador NO tiene efecto, la «potencia» estimada es el error tipo I real.
    Devuelve potencia, su error estándar Monte Carlo y el nº de simulaciones.
    Linus: coste = n_sim llamadas al contraste; 1000 da ±1.5 pp de precisión alrededor de 0.8.
    """
    if n_sim < 10:
        raise ValueError("n_sim debe ser >= 10.")
    rng = np.random.default_rng(semilla)
    rechazos = 0
    for _ in range(n_sim):
        muestras = generador(rng, n)
        if not isinstance(muestras, tuple):
            muestras = (muestras,)
        rechazos += float(contraste(*muestras)) < alpha
    pot = rechazos / n_sim
    return {"potencia": pot, "error_mc": math.sqrt(pot * (1 - pot) / n_sim), "n_sim": n_sim, "n": n, "alpha": alpha}
