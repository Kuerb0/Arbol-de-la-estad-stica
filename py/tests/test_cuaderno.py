"""Cada función del árbol tiene un ejemplo ejecutable y todos los ejemplos funcionan."""
import ast
import sys
from pathlib import Path

import pytest

CODIGO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODIGO))
from cuaderno import ejemplos  # noqa: E402
from cuaderno.ejecutor import Cuaderno  # noqa: E402


def _publicas():
    out = set()
    for py in (CODIGO / "arbol_estadistica").rglob("*.py"):
        if py.name == "__init__.py" or py.name.startswith("_"):
            continue
        out |= {n.name for n in ast.parse(py.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}
    return out


def test_todas_las_funciones_tienen_ejemplo():
    faltan = _publicas() - set(ejemplos.E)
    sobran = set(ejemplos.E) - _publicas()
    assert not faltan and not sobran, f"sin ejemplo: {faltan}; ejemplo de algo inexistente: {sobran}"


@pytest.mark.parametrize("funcion", sorted(ejemplos.E))
def test_el_ejemplo_se_ejecuta_sin_errores(funcion):
    c = Cuaderno()
    celdas = ejemplos.celdas_de(funcion)
    imagenes = 0
    for i, codigo in enumerate(celdas):
        r = c.ejecutar(funcion, codigo)
        assert not r["error"], f"{funcion}, celda {i}: {r['error']}"
        imagenes += len(r["imagenes"])
    if any("grafico_" in x or ".plot" in x or "plt." in x for x in celdas[1:]):
        assert imagenes >= 1, f"{funcion}: el ejemplo dibuja pero no salió ninguna imagen"


def test_ejecutor_muestra_tablas_errores_y_conserva_variables():
    c = Cuaderno()
    assert c.ejecutar("s", "import pandas as pd\nx = 2\npd.DataFrame({'a': [1, x]})")["html"].startswith("<table")
    assert c.ejecutar("s", "x * 10")["texto"] == "20"
    r = c.ejecutar("s", "1 / 0")
    assert r["error"].startswith("ZeroDivisionError")
    c.reiniciar("s")
    assert "NameError" in c.ejecutar("s", "x")["error"]


def test_ejemplos_de_conceptos_existen_en_el_catalogo():
    import json
    ids = {k["id"] for k in json.loads((CODIGO.parent / "conceptos" / "catalogo.json").read_text(encoding="utf-8"))["conceptos"]}
    assert set(ejemplos.CONCEPTOS) <= ids, set(ejemplos.CONCEPTOS) - ids


@pytest.mark.parametrize("concepto", sorted(ejemplos.CONCEPTOS))
def test_el_ejemplo_del_concepto_se_ejecuta_y_dibuja(concepto):
    c = Cuaderno()
    imagenes = 0
    for i, codigo in enumerate(ejemplos.celdas_concepto(concepto, [])):
        r = c.ejecutar(concepto, codigo)
        assert not r["error"], f"{concepto}, celda {i}: {r['error']}"
        imagenes += len(r["imagenes"])
    assert imagenes >= 1
