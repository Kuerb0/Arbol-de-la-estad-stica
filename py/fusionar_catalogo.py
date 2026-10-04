"""Añade al catálogo del usuario los conceptos y enlaces nuevos que trae una actualización.

conceptos/catalogo_base.json  -> el que viene con cada versión (se sobrescribe al actualizar)
conceptos/catalogo.json       -> el tuyo (lo que edites a mano se conserva)

Regla: nunca se borra nada tuyo. Se añaden áreas y conceptos nuevos (por id) y, en los conceptos que
ya tienes, las funciones, sinónimos y fuentes que falten. La organización (temas, nombre/tema/ámbito de las
áreas y el área de cada concepto) la manda la base, salvo en los conceptos que marques con "area_fija": true. Lo ejecuta el actualizador; a mano:
    python py/fusionar_catalogo.py
"""
from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def _union(mia: list, nueva: list, clave=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False)) -> tuple[list, int]:
    vistas = {clave(x) for x in mia}
    extra = [x for x in nueva if clave(x) not in vistas]
    return mia + extra, len(extra)


def fusionar(raiz: Path = RAIZ) -> dict:
    base_p, user_p = raiz / "conceptos" / "catalogo_base.json", raiz / "conceptos" / "catalogo.json"
    if not base_p.exists():
        return {"estado": "sin catalogo_base.json: nada que fusionar"}
    base = json.loads(base_p.read_text(encoding="utf-8"))
    if not user_p.exists():
        user_p.write_text(json.dumps(base, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return {"estado": "catálogo creado desde la base", "conceptos_nuevos": len(base["conceptos"])}
    mio = json.loads(user_p.read_text(encoding="utf-8"))
    mio["areas"], n_areas = _union(mio.get("areas", []), base.get("areas", []), clave=lambda a: a["id"])
    n_org = 0
    # Organización: la base manda sobre temas y datos de las áreas (nombre, descripción, tema, ámbito).
    if base.get("temas") and mio.get("temas") != base["temas"]:
        mio["temas"] = base["temas"]; n_org += 1
    areas_mias = {a["id"]: a for a in mio["areas"]}
    for a in base.get("areas", []):
        for k, v in a.items():
            if areas_mias[a["id"]].get(k) != v:
                areas_mias[a["id"]][k] = v; n_org += 1
    por_id = {c["id"]: c for c in mio["conceptos"]}
    n_conc = n_enl = 0
    for c in base["conceptos"]:
        if c["id"] not in por_id:
            mio["conceptos"].append(c); por_id[c["id"]] = c; n_conc += 1
            continue
        m = por_id[c["id"]]
        if "area" in c and not m.get("area_fija"):
            if m.get("area") != c["area"]:
                m["area"] = c["area"]; n_org += 1
        if c.get("prioridad") and not m.get("prioridad"):
            m["prioridad"] = c["prioridad"]; n_org += 1
        m["funciones"], a = _union(m.get("funciones", []), c.get("funciones", []), clave=str)
        m["sinonimos"], _ = _union(m.get("sinonimos", []), c.get("sinonimos", []), clave=str)
        m["fuentes"], _ = _union(m.get("fuentes", []), c.get("fuentes", []),
                                 clave=lambda f: (f.get("tipo"), f.get("url") or f.get("ref")))
        n_enl += a
    # Áreas antiguas que la base ya no tiene y que se han quedado sin conceptos: se retiran (no pierdes nada).
    usadas = {c.get("area") for c in mio["conceptos"]}
    en_base = {a["id"] for a in base.get("areas", [])}
    huerfanas = [a for a in mio["areas"] if a["id"] not in en_base and a["id"] not in usadas] if base.get("temas") else []
    if huerfanas:
        mio["areas"] = [a for a in mio["areas"] if a not in huerfanas]; n_org += len(huerfanas)
    if n_areas or n_conc or n_enl or n_org:
        user_p.write_text(json.dumps(mio, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"estado": "fusionado", "areas_nuevas": n_areas, "conceptos_nuevos": n_conc, "enlaces_nuevos": n_enl, "reorganizados": n_org}


if __name__ == "__main__":
    r = fusionar()
    print("Catálogo:", ", ".join(f"{k}={v}" for k, v in r.items()))
