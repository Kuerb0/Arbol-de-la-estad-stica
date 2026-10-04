"""Fichas de propiedades (barras 0-10): completas, válidas, medidas y visibles en el visor."""
import ast
import importlib
import importlib.util
import json
import sys
from pathlib import Path

CODIGO = Path(__file__).resolve().parents[1]
RAIZ = CODIGO.parent
sys.path.insert(0, str(CODIGO))
construir_visor = importlib.import_module("construir_visor")
PROP = CODIGO / "propiedades"


def _publicas():
    out = set()
    for py in (CODIGO / "arbol_estadistica").rglob("*.py"):
        if py.name.startswith("_"):
            continue
        out |= {n.name for n in ast.parse(py.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}
    return out


def _json(nombre):
    return json.loads((PROP / nombre).read_text(encoding="utf-8"))


def test_definiciones_perfiles_y_grupos_son_coherentes():
    meta = _json("propiedades.json")
    bloques = {b["id"] for b in meta["bloques"]}
    ids = [p["id"] for p in meta["propiedades"]]
    assert len(ids) == len(set(ids)) and len(ids) >= 25
    assert all(p["bloque"] in bloques and p["nombre"] and p["def"] for p in meta["propiedades"])
    # los nombres estadísticos clásicos se mantienen
    assert {"insesgadez", "consistencia", "eficiencia", "normalidad_asintotica", "potencia", "robustez"} <= set(ids)
    assert {"escalabilidad", "velocidad", "memoria", "reproducibilidad"} <= set(ids)
    perfiles = [p["id"] for p in meta["perfiles"]]
    assert perfiles[0] == "equilibrado" and len(perfiles) == len(set(perfiles)) and "mio" not in perfiles
    for p in meta["perfiles"]:
        assert set(p["pesos"]) <= set(ids), p["id"]
        assert all(0 <= v <= 10 for v in p["pesos"].values())
    grupos = [g["id"] for g in meta["grupos"]]
    assert len(grupos) == len(set(grupos))


def test_toda_funcion_publica_tiene_ficha_valida():
    meta, fichas = _json("propiedades.json"), _json("fichas.json")["funciones"]
    ids, grupos = {p["id"] for p in meta["propiedades"]}, {g["id"] for g in meta["grupos"]}
    assert set(fichas) == _publicas(), f"faltan: {_publicas() - set(fichas)} · sobran: {set(fichas) - _publicas()}"
    for nombre, f in fichas.items():
        assert f["grupo"] in grupos, nombre
        assert set(f["notas"]) <= ids and len(f["notas"]) >= 5, nombre
        assert all(isinstance(v, int) and 0 <= v <= 10 for v in f["notas"].values()), nombre
        assert f["pros"] and isinstance(f["contras"], list), nombre
    usados = {f["grupo"] for f in fichas.values()}
    assert usados == grupos, f"grupos sin funciones: {grupos - usados}"


def test_fichas_json_esta_al_dia_con_su_fuente(tmp_path):
    spec = importlib.util.spec_from_file_location("fichas_fuente", RAIZ / "herramientas" / "fichas_fuente.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    generadas = {k: v for k, v in mod.F.items()}
    actuales = _json("fichas.json")["funciones"]
    assert set(generadas) == set(actuales), "ejecuta python herramientas/fichas_fuente.py"
    for k, v in generadas.items():
        notas = {mod.ABREV[x.split("=")[0]]: int(x.split("=")[1]) for x in v["notas"].split()}
        assert actuales[k]["notas"] == notas, f"{k}: fichas.json desactualizado (ejecuta herramientas/fichas_fuente.py)"


def test_medidas_validas_y_cubren_todas_las_funciones():
    ids = {p["id"] for p in _json("propiedades.json")["propiedades"]}
    med = _json("medidas.json")
    assert med["generado"] and med["funciones"]
    assert _publicas() <= set(med["funciones"]), f"sin medir: {_publicas() - set(med['funciones'])}"
    for f, props in med["funciones"].items():
        assert set(props) <= ids, f
        for p, v in props.items():
            assert 0 <= v["nota"] <= 10 and v["detalle"], (f, p)
        assert {"velocidad", "escalabilidad", "memoria"} <= set(props), f


def test_todas_las_funciones_tienen_benchmark():
    medir = importlib.import_module("medir_propiedades")
    assert set(medir.BENCH) == _publicas(), f"sin benchmark: {_publicas() - set(medir.BENCH)}"


def test_reglas_de_nota_dan_lo_esperado():
    m = importlib.import_module("medir_propiedades")
    assert m.nota_velocidad(0.01) == 10 and m.nota_velocidad(1) == 6 and m.nota_velocidad(100) == 2
    assert m.nota_escalabilidad(1.0) == 10 and m.nota_escalabilidad(2.0) == 3 and m.nota_escalabilidad(0.5) == 10
    assert m.nota_memoria(1) == 10 and m.nota_memoria(100) == 6
    assert m.nota_desviacion(0.0, 1000) == 10 and m.nota_desviacion(0.05, 1000) < 3 and m.nota_desviacion(0.10, 1000) == 0
    assert m.nota_sesgo(0) == 10 and m.nota_sesgo(0.25) == 0 and m.nota_sesgo(-0.05) == 8


def test_medir_una_funcion_rapida():
    m = importlib.import_module("medir_propiedades")
    r = m.medir_codigo("perfil_respuesta", rapido=True)
    assert set(r) == {"velocidad", "escalabilidad", "memoria"} and all(0 <= v["nota"] <= 10 for v in r.values())
    r = m.medir_codigo("potencia_contraste_medias")
    assert r["escalabilidad"]["nota"] == 10                       # no depende de n


def test_el_visor_lleva_las_fichas_y_las_notas_medidas_mandan():
    datos = construir_visor.construir()
    items = {i["nombre"]: i for r in datos["ramas"] for m in r["modulos"] for i in m["items"] if i["tipo"] == "funcion"}
    assert all(i["ficha"]["notas"] for i in items.values())
    firth = items["regresion_logistica_firth"]["ficha"]["notas"]
    assert firth["velocidad"]["origen"] == "medido" and firth["consistencia"]["origen"] == "estimado"
    assert items["ajustar_logit"]["ficha"]["notas"]["cobertura_tests"]["origen"] == "medido"
    meta = datos["propiedades"]
    assert meta["perfiles"] and meta["bloques"] and meta["medido"]
    html = construir_visor.ensamblar(datos)
    assert 'id="perfil"' in html and "htmlPropiedades" in html and "detalleGrupoFunciones" in html


def test_tus_notas_en_fichas_mias_mandan(monkeypatch, tmp_path):
    mias = tmp_path / "fichas_mias.json"
    mias.write_text(json.dumps({"funciones": {"ajustar_logit": {"notas": {"robustez": 9, "velocidad": None},
                                                                "usar_si": "mi criterio"}}}), encoding="utf-8")
    monkeypatch.setattr(construir_visor, "FICHAS_MIAS", mias)
    _, fichas = construir_visor.cargar_propiedades({"ajustar_logit"})
    f = fichas["ajustar_logit"]
    assert f["notas"]["robustez"] == {"nota": 9, "origen": "tuyo"}
    assert "velocidad" not in f["notas"]                        # None = quitar la nota
    assert f["usar_si"] == "mi criterio" and f["pros"]
