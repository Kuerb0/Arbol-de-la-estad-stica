"""Rama CONTRASTES / elegir y ejecutar el contraste de igualdad adecuado.

Implementa en código el árbol de decisión de tu nota `Contrastes de Igualdad.txt`
(ver teoria/contrastes.md): tipo de variable -> normalidad -> igualdad de varianzas -> nº de grupos.
Equivale a PROC TTEST / PROC NPAR1WAY / PROC ANOVA / PROC FREQ CHISQ, eligiendo el procedimiento.
Solo grupos INDEPENDIENTES (para muestras apareadas: pendiente, ver INDEX.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def elegir_contraste(df: pd.DataFrame, valor: str, grupo: str, alpha: float = 0.05,
                     tipo_valor: str = "auto", n_tcl: int = 30) -> dict:
    """Compara `valor` entre los niveles de `grupo`, eligiendo y ejecutando el contraste correcto.

    tipo_valor : 'auto' | 'numerico' | 'categorico'. En 'auto', numérica con > 10 valores
                 distintos = numérica; si no, categórica (un 0/1 va a chi-cuadrado).
    n_tcl      : con >= n_tcl observaciones por grupo se acepta normalidad aproximada (TCL), como
                 en tu nota (n > 30). Con menos se exige Shapiro-Wilk p > alpha.
    Devuelve dict: contraste, estadistico, p_valor, significativo, efecto_nombre, efecto, supuestos,
    avisos, interpretacion, n, n_grupos.
    """
    d = df[[valor, grupo]].dropna()
    niveles = list(d[grupo].unique())
    k = len(niveles)
    if k < 2:
        raise ValueError("`grupo` necesita al menos 2 niveles.")

    if tipo_valor == "auto":
        es_num = pd.api.types.is_numeric_dtype(d[valor]) and d[valor].nunique() > 10
        tipo_valor = "numerico" if es_num else "categorico"

    avisos: list[str] = []
    supuestos: dict = {}
    n = len(d)

    if tipo_valor == "categorico":
        tabla = pd.crosstab(d[grupo], d[valor])
        esperado = stats.chi2_contingency(tabla)[3]
        supuestos["frecuencia_esperada_minima"] = float(esperado.min())
        if tabla.shape == (2, 2) and esperado.min() < 5:
            est, p = stats.fisher_exact(tabla)
            nombre, efecto_nombre = "Fisher exacto", "odds_ratio"
            efecto = float(est)
        else:
            chi2, p, _, _ = stats.chi2_contingency(tabla)
            est, nombre, efecto_nombre = float(chi2), "Chi-cuadrado de independencia", "V_Cramer"
            efecto = float(np.sqrt(chi2 / (n * (min(tabla.shape) - 1))))
            if (esperado < 5).mean() > 0.20:
                avisos.append("Más del 20 % de celdas con frecuencia esperada < 5: agrupar niveles o usar test exacto.")
    else:
        muestras = [d.loc[d[grupo] == g, valor].to_numpy(dtype=float) for g in niveles]
        tamanos = [len(m) for m in muestras]
        if min(tamanos) < 2:
            raise ValueError("Cada grupo necesita al menos 2 observaciones.")

        normal = True
        for g, m in zip(niveles, muestras):
            if len(m) >= n_tcl:
                supuestos[f"normalidad[{g}]"] = "aprox. normal por TCL (n >= %d)" % n_tcl
            else:
                p_sw = float(stats.shapiro(m).pvalue) if len(m) >= 3 else 1.0
                supuestos[f"normalidad[{g}]"] = f"Shapiro p={p_sw:.4f}"
                normal &= p_sw > alpha
        p_lev = float(stats.levene(*muestras, center="median").pvalue)
        supuestos["levene_p"] = p_lev
        var_iguales = p_lev > alpha

        if k == 2:
            a, b = muestras
            if normal:
                nombre = "t de Student" if var_iguales else "t de Welch"
                r = stats.ttest_ind(a, b, equal_var=var_iguales)
                est, p = float(r.statistic), float(r.pvalue)
                efecto_nombre, efecto = "d_Cohen", _cohen_d(a, b)
            else:
                r = stats.mannwhitneyu(a, b, alternative="two-sided")
                est, p, nombre = float(r.statistic), float(r.pvalue), "U de Mann-Whitney"
                efecto_nombre, efecto = "r_biserial_rangos", float(1 - 2 * r.statistic / (len(a) * len(b)))
        else:
            if normal:
                if var_iguales:
                    r = stats.f_oneway(*muestras)
                    est, p, nombre = float(r.statistic), float(r.pvalue), "ANOVA de un factor"
                else:
                    r = anova_oneway(muestras, use_var="unequal", welch_correction=True)
                    est, p, nombre = float(r.statistic), float(r.pvalue), "ANOVA de Welch"
                gran = np.concatenate(muestras).mean()
                ssb = sum(len(m) * (m.mean() - gran) ** 2 for m in muestras)
                sst = sum(((m - gran) ** 2).sum() for m in muestras)
                efecto_nombre, efecto = "eta_cuadrado", float(ssb / sst) if sst > 0 else 0.0
            else:
                r = stats.kruskal(*muestras)
                est, p, nombre = float(r.statistic), float(r.pvalue), "Kruskal-Wallis"
                efecto_nombre, efecto = "eta_cuadrado_H", float(max((est - k + 1) / (n - k), 0.0))
            if p < alpha:
                avisos.append("Significativo con >2 grupos: ejecutar post-hoc (tukey_entre_grupos si es paramétrico).")

    if n > 5000:
        avisos.append(f"n={n:,}: con muestras grandes el p-valor casi siempre sale significativo; valora el tamaño del efecto.")

    sig = bool(p < alpha)
    return {
        "contraste": nombre, "estadistico": float(est), "p_valor": float(p), "significativo": sig,
        "efecto_nombre": efecto_nombre, "efecto": float(efecto),
        "supuestos": supuestos, "avisos": avisos, "n": n, "n_grupos": k,
        "interpretacion": (f"{nombre}: p={p:.4g} -> {'se rechaza' if sig else 'no se rechaza'} H0 "
                           f"(igualdad entre grupos) a alpha={alpha}. {efecto_nombre}={efecto:.3f}."),
    }
