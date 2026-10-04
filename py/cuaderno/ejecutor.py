"""Ejecuta celdas de código Python como un notebook mínimo (lo usa la ventana de la app).

Cada «sesión» (una por función del visor) guarda sus variables entre celdas. Devuelve lo impreso,
el valor de la última expresión (tablas en HTML) y los gráficos de matplotlib como PNG en base64.
Solo se usa en local, desde la app; no abre puertos ni ejecuta nada que no escribas tú.
"""
from __future__ import annotations

import ast
import base64
import io
import pprint
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MAX_TEXTO = 20000


def _mostrar(valor) -> tuple[str, str]:
    """(html, texto) para el valor de la última expresión."""
    if valor is None:
        return "", ""
    try:
        import pandas as pd
        if isinstance(valor, pd.DataFrame):
            return valor.to_html(max_rows=40, max_cols=20, border=0, classes="nb-tabla", float_format=lambda x: f"{x:.4g}"), ""
        if isinstance(valor, pd.Series):
            return valor.to_frame().to_html(max_rows=40, border=0, classes="nb-tabla", float_format=lambda x: f"{x:.4g}"), ""
    except ImportError:
        pass
    modulo = type(valor).__module__ or ""
    if modulo.startswith("matplotlib"):
        return "", ""
    if isinstance(valor, (list, tuple)) and valor and all(type(v).__module__.startswith("matplotlib") for v in valor):
        return "", ""
    return "", pprint.pformat(valor, width=100, compact=True)[:MAX_TEXTO]


def _figuras() -> list[str]:
    imgs = []
    for num in plt.get_fignums():
        buf = io.BytesIO()
        plt.figure(num).savefig(buf, format="png", dpi=96, bbox_inches="tight")
        imgs.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    plt.close("all")
    return imgs


class Cuaderno:
    def __init__(self) -> None:
        self._sesiones: dict[str, dict] = {}

    def reiniciar(self, sesion: str) -> bool:
        self._sesiones.pop(sesion, None)
        plt.close("all")
        return True

    def ejecutar(self, sesion: str, codigo: str) -> dict:
        ns = self._sesiones.setdefault(sesion, {"__name__": "__cuaderno__"})
        salida, valor, error = io.StringIO(), None, ""
        plt.close("all")
        try:
            arbol = ast.parse(codigo, filename="<celda>", mode="exec")
            ultima = None
            if arbol.body and isinstance(arbol.body[-1], ast.Expr):
                ultima = ast.Expression(arbol.body.pop().value)
            with redirect_stdout(salida), redirect_stderr(salida), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(compile(arbol, "<celda>", "exec"), ns)
                if ultima is not None:
                    valor = eval(compile(ultima, "<celda>", "eval"), ns)
        except Exception as e:  # el error se enseña en la celda, no rompe la app
            pila = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "<celda>"]
            linea = f" (línea {pila[-1].lineno})" if pila else ""
            error = f"{type(e).__name__}{linea}: {e}"
        html, texto = _mostrar(valor) if not error else ("", "")
        return {"salida": salida.getvalue()[-MAX_TEXTO:], "html": html, "texto": texto,
                "imagenes": _figuras(), "error": error}
