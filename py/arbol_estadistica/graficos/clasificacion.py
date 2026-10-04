"""Rama GRAFICOS / modelos de clasificación (logística): ROC, calibración, odds ratios, umbrales.

Origen: nuevo; acompaña a `diagnostico` (tabla_umbrales, hosmer_lemeshow) y a
`modelos.tabla_odds_ratios` (notebook de consultoría).
Equivale a PROC LOGISTIC con ODS GRAPHICS: PLOTS=(ROC ODDSRATIO CALIBRATION) y CTABLE.
"""
from __future__ import annotations

import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from ..diagnostico.bondad_ajuste import hosmer_lemeshow
from ..diagnostico.metricas import tabla_umbrales, umbral_optimo_youden
from ._base import APAGADO, AZUL, AQUA, NARANJA, TINTA2, _color, _ejes, _figura, _leyenda


def grafico_curva_roc(y_real, probabilidades, ax=None, marcar_youden: bool = True):
    """Curva ROC con AUC en la leyenda. `probabilidades` puede ser un array o un dict
    {nombre_modelo: array} para comparar hasta 8 modelos sobre las mismas `y_real`.

    Con un solo modelo se marca el punto de Youden (umbral que maximiza sens + espec - 1).
    Gauss: AUC mide ORDENACIÓN, no calibración; un modelo con buen AUC puede dar probabilidades
    mal escaladas (mirar grafico_calibracion).
    """
    y = np.asarray(y_real).astype(int)
    series = probabilidades if isinstance(probabilidades, dict) else {"modelo": probabilidades}
    fig, ax = _figura(ax, (5.2, 5.0))
    ax.plot([0, 1], [0, 1], color=APAGADO, lw=1, label="azar (AUC = 0.5)")
    for i, (nombre, p) in enumerate(series.items()):
        p = np.asarray(p, float)
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=_color(i), lw=2, label=f"{nombre} (AUC = {roc_auc_score(y, p):.3f})")
        if marcar_youden and len(series) == 1:
            u = umbral_optimo_youden(y, p)
            pred = p >= u
            sens, fpr_u = pred[y == 1].mean(), pred[y == 0].mean()
            ax.scatter([fpr_u], [sens], s=64, color=_color(i), edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(f"Youden: umbral {u:.2f}\nsens {sens:.2f} · espec {1 - fpr_u:.2f}", (fpr_u, sens),
                        xytext=(12, -28), textcoords="offset points", fontsize=8.5, color=TINTA2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
    _ejes(ax, "Curva ROC", "1 − especificidad (falsos positivos)", "sensibilidad (verdaderos positivos)", rejilla="both")
    _leyenda(ax, loc="lower right")
    return fig


def grafico_calibracion(y_real, y_prob, grupos: int = 10, ax=None):
    """Calibración por deciles: tasa observada (con IC 95 % de Wilson) frente a probabilidad media
    predicha. Bien calibrado = puntos sobre la diagonal. El título incluye Hosmer-Lemeshow.

    Gauss: con n grande el Hosmer-Lemeshow rechaza casi siempre; este gráfico dice SI la
    descalibración importa (cuánto se separan los puntos y en qué zona de probabilidad).
    Si entrenaste con datos balanceados, corrige antes las probabilidades
    (preprocesado.corregir_probabilidades_por_balanceo).
    """
    d = pd.DataFrame({"y": np.asarray(y_real).astype(int), "p": np.asarray(y_prob, float)})
    d["g"] = pd.qcut(d["p"], grupos, duplicates="drop")
    t = d.groupby("g", observed=True).agg(p_media=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    z = 1.96
    centro = (t["obs"] + z * z / (2 * t["n"])) / (1 + z * z / t["n"])
    medio = z * np.sqrt(t["obs"] * (1 - t["obs"]) / t["n"] + z * z / (4 * t["n"] ** 2)) / (1 + z * z / t["n"])
    hl = hosmer_lemeshow(d["y"], d["p"], g=grupos)
    fig, ax = _figura(ax, (5.2, 5.0))
    tope = float(max(t["p_media"].max(), (centro + medio).max())) * 1.05
    ax.plot([0, tope], [0, tope], color=APAGADO, lw=1, label="calibración perfecta")
    ax.errorbar(t["p_media"], t["obs"], yerr=[t["obs"] - (centro - medio).clip(lower=0), (centro + medio) - t["obs"]],
                fmt="o", color=AZUL, ms=6, mec="white", mew=1.5, elinewidth=1.5, capsize=0, label="decil (IC 95 %)")
    ax.set_xlim(0, tope); ax.set_ylim(0, tope); ax.set_aspect("equal")
    _ejes(ax, f"Calibración · Hosmer-Lemeshow p = {hl['p_valor']:.3g}", "probabilidad media predicha",
          "tasa observada", rejilla="both")
    _leyenda(ax, loc="upper left")
    return fig


def grafico_odds_ratios(tabla: pd.DataFrame, ax=None, alpha: float = 0.05):
    """Forest plot de odds ratios (salida de modelos.tabla_odds_ratios: columnas OR, IC_2.5%,
    IC_97.5%, p_valor). Escala logarítmica, línea en OR = 1; en azul las significativas.

    Leer: el IC que cruza la línea vertical no es significativo; OR 2 y OR 0.5 son efectos
    igual de grandes (por eso el eje es logarítmico).
    """
    falta = {"OR", "IC_2.5%", "IC_97.5%"} - set(tabla.columns)
    if falta:
        raise KeyError(f"Faltan columnas {falta}: usa la salida de tabla_odds_ratios.")
    t = tabla.sort_values("OR")
    pos = np.arange(len(t))
    sig = (t["p_valor"] < alpha).to_numpy() if "p_valor" in t else np.ones(len(t), bool)
    fig, ax = _figura(ax, (6.4, max(2.4, 0.38 * len(t) + 1.2)))
    ax.axvline(1, color=APAGADO, lw=1)
    for i, (_, f) in enumerate(t.iterrows()):
        c = AZUL if sig[i] else APAGADO
        ax.plot([f["IC_2.5%"], f["IC_97.5%"]], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.scatter([f["OR"]], [i], s=48, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(f"{f['OR']:.2f}", (f["IC_97.5%"], i), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=TINTA2)
    ax.set_xscale("log")
    lo, hi = float(t["IC_2.5%"].min()), float(t["IC_97.5%"].max())
    marcas = [v for v in (0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 10, 20) if lo * 0.8 <= v <= hi * 1.25] or [1]
    ax.set_xticks(marcas, [f"{v:g}" for v in marcas])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(pos, [str(v) for v in t.index])
    _ejes(ax, f"Odds ratios (IC 95 %) · azul: p < {alpha}", "odds ratio (escala log)", "", rejilla="x")
    return fig


def grafico_umbrales(y_real, y_prob, ax=None):
    """Sensibilidad, especificidad y J de Youden según el umbral de decisión (CTABLE de SAS en gráfico).

    El umbral 0.5 no tiene nada de especial: el corte se elige por el coste de cada error.
    La línea vertical marca el umbral de Youden.
    """
    t = tabla_umbrales(y_real, y_prob, np.round(np.arange(0.02, 0.981, 0.02), 2))
    u = umbral_optimo_youden(y_real, y_prob)
    fig, ax = _figura(ax)
    for i, (col, nombre) in enumerate([("sens", "sensibilidad"), ("espec", "especificidad"), ("Youden_J", "J de Youden")]):
        ax.plot(t["umbral"], t[col], color=[AZUL, NARANJA, AQUA][i], lw=2, label=nombre)
    ax.axvline(u, color=APAGADO, lw=1)
    ax.annotate(f"umbral Youden {u:.2f}", (u, 0.04), xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=TINTA2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _ejes(ax, "Métricas según el umbral de decisión", "umbral de probabilidad", "valor")
    _leyenda(ax, loc="center right")
    return fig


def grafico_residuos_agrupados(y_real, y_prob, grupos: int | None = None, ax=None):
    """Residuos agrupados (Gelman y Hill) para modelos binarios: se ordenan las observaciones por
    probabilidad predicha, se agrupan y se dibuja el residuo medio (observado - predicho) con
    bandas ±2 errores estándar. Bien especificado: ~95 % de los puntos dentro de las bandas.

    Es el sustituto del gráfico de residuos de OLS cuando la respuesta es 0/1 (los residuos
    crudos de una logística forman dos líneas y no dicen nada).
    """
    y = np.asarray(y_real).astype(float)
    p = np.asarray(y_prob, float)
    n = len(y)
    grupos = grupos or int(np.clip(np.sqrt(n), 10, 40))
    d = pd.DataFrame({"y": y, "p": p}).sort_values("p")
    d["g"] = np.arange(n) * grupos // n
    t = d.groupby("g").agg(p=("p", "mean"), r=("y", "mean"), n=("y", "size"))
    t["r"] = t["r"] - t["p"]
    t["se2"] = 2 * np.sqrt(t["p"] * (1 - t["p"]) / t["n"])
    dentro = (t["r"].abs() <= t["se2"]).to_numpy()
    fig, ax = _figura(ax)
    orden = np.argsort(t["p"].to_numpy())
    ax.fill_between(t["p"].to_numpy()[orden], -t["se2"].to_numpy()[orden], t["se2"].to_numpy()[orden],
                    color=AZUL, alpha=0.10, label="±2 EE")
    ax.axhline(0, color=APAGADO, lw=1)
    ax.scatter(t["p"][dentro], t["r"][dentro], s=30, color=AZUL, edgecolor="white", linewidth=1.2, label="dentro")
    ax.scatter(t["p"][~dentro], t["r"][~dentro], s=36, color=NARANJA, edgecolor="white", linewidth=1.2, label="fuera")
    _ejes(ax, f"Residuos agrupados · {dentro.mean():.0%} dentro de ±2 EE (esperado ≈ 95 %)",
          "probabilidad predicha media del grupo", "residuo medio (obs − pred)")
    _leyenda(ax, loc="best")
    return fig


def grafico_ganancia_lift(y_real, y_prob, grupos: int = 10):
    """Dos paneles: curva de ganancia acumulada (% de eventos capturados frente a % de población, con la diagonal
    del azar y el modelo perfecto) y lift por grupo (salida de diagnostico.tabla_ganancia_lift)."""
    import matplotlib.pyplot as plt
    from ..diagnostico.negocio import tabla_ganancia_lift
    from ._base import FONDO
    t = tabla_ganancia_lift(y_real, y_prob, grupos)
    base = float(np.mean(np.asarray(y_real, float)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    fig.patch.set_facecolor(FONDO)
    a1.plot(np.r_[0, t.pct_poblacion_acum], np.r_[0, t.ganancia_acum], "o-", color=AZUL, lw=2, label="modelo")
    a1.plot([0, 100], [0, 100], color=APAGADO, ls=":", label="azar")
    a1.plot([0, base * 100, 100], [0, 100, 100], color=NARANJA, lw=1, ls="--", label="perfecto")
    _ejes(a1, "Ganancia acumulada", "% de población (por score)", "% de eventos capturados", rejilla="both"); _leyenda(a1)
    a2.bar(t.index, t.lift, color=AZUL, alpha=0.85)
    a2.axhline(1, color=APAGADO, ls=":")
    _ejes(a2, "Lift por grupo", "grupo (1 = mayor score)", "lift")
    return fig


def grafico_precision_recall(y_real, y_prob, ax=None):
    """Curva precisión-exhaustividad con la precisión media (AP), la línea base (prevalencia) y el punto de F1 máximo."""
    from ..diagnostico.negocio import curva_precision_recall
    r = curva_precision_recall(y_real, y_prob)
    t = r["tabla"]
    fig, ax = _figura(ax)
    ax.plot(t.recall, t.precision, color=AZUL, lw=2, label=f"AP = {r['precision_media']:.3f}")
    ax.axhline(r["prevalencia"], color=APAGADO, ls=":", label=f"azar = {r['prevalencia']:.3f}")
    i = int(t.f1.idxmax())
    ax.plot(t.recall[i], t.precision[i], "o", color=NARANJA, ms=8, label=f"F1 máx {r['f1_max']:.2f} (umbral {r['umbral_f1']:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _ejes(ax, "Precisión - exhaustividad", "recall (exhaustividad)", "precisión", rejilla="both"); _leyenda(ax)
    return fig
