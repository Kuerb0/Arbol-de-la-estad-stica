---
name: anadir-funcion-arbol
description: Añade una función estadística nueva al Árbol de la estadística siguiendo los 11 pasos obligatorios (código, export, test, ejemplo, ficha, bench, INDEX, catálogo, visor). Úsala cuando Mario pida "añade <función> al árbol" o falte algo en una rama.
---

# Añadir una función al árbol

Antes: busca en `INDEX.md` si ya existe algo parecido. No dupliques. Hay tests que fallan si falta cualquier paso.

Checklist (marca cada uno; no cierres sin completarlos):

1. **Código** en `py/arbol_estadistica/<rama>/<modulo>.py`: nombre en español `snake_case` sin tildes/eñes, no mutar entradas, semilla 42, devuelve `dict`/`DataFrame` con claves en español, valida con `ValueError` claro. Docstring: qué hace, *Equivale a:* (SAS), *Gauss:* (supuestos/trampas). Nada de nombres de negocio.
2. **Export** en `<rama>/__init__.py`: import + `__all__` como **lista literal** (nunca `sorted([...])`).
3. **Test** en `py/tests/`: caso donde la teoría dé la respuesta, rng con semilla, y un `pytest.raises(ValueError)`.
4. **Ejemplo** en `py/cuaderno/ejemplos.py` dentro de `E`: `"funcion": ("escenario", ["celda1", ...])`. Si dibuja (`grafico_`, `.plot`, `plt.`) debe salir imagen. Rama nueva → añadir import en `IMPORTS`.
5. **Ficha** en `herramientas/fichas_fuente.py` antes de `def main()`: `f(nombre, grupo, notas, pros, contras, usar_si, evitar_si)`. El grupo debe existir en `py/propiedades/propiedades.json` (gráficos: `graf_*`). Luego `python herramientas/fichas_fuente.py`.
6. **Bench** en `py/medir_propiedades.py`: `@bench("nombre")` con `_(n)` (o `fijo=True`; gráficos con `_cerrar(...)`). O(n²) en memoria → `ValueError` por encima de un límite. Si hay propiedad medible, añade/amplía `sim_*` en `SIMULACIONES`. Luego `cd py && python medir_propiedades.py nombre`.
7. **INDEX.md**: fila `| \`firma(args)\` | Para qué | SAS | Origen |` en su rama.
8. **Catálogo**: añade el nombre a `funciones` de al menos un concepto en `conceptos/catalogo.json` **y** `catalogo_base.json` (respeta el sangrado).
9. **Rama nueva**: `RAMAS` en `construir_visor.py` (colores), `packages` en `py/pyproject.toml`, import y `__all__` en `arbol_estadistica/__init__.py`, árbol de texto en `INDEX.md`.
10. **Cierre**: `python py/construir_visor.py`; `cd py && python -m pytest -q` (~4 min); si la tabla «qué usar» de `CLAUDE.md` gana fila, añadirla.
11. **Si cambias `py/`, `assets/` o `.bat`**: subir versión en `py/VERSION.txt`, `py/pyproject.toml` y `__version__`, y `python herramientas/generar_instaladores.py`, y publica con `python herramientas/publicar.py "qué cambió"` (commit + push); entrega a Mario las rutas de ambos `.bat` de `instaladores/` para que los pruebe.

Trampas: variables locales `C`/`Q`/`I` rompen patsy; columnas raras → `Q("col")`; pandas 3 (`axis=`, `to_numpy(copy=True)`); sklearn 1.9 (`LassoCV(alphas=)`, helper `_logistica`); nada O(n²) sin submuestrear; Windows: rutas entre comillas, OneDrive puede bloquear escrituras (reintentar). Respeta ediciones del usuario y sus datos (`catalogo.json`, `conceptos/fichas_mias.json`).
