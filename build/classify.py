# -*- coding: utf-8 -*-
"""
Fase 2 — Clasificación de los títulos que las reglas base no alcanzaron.

El vocabulario de acá salió de leer las 1.698 firmas distintas de los 3.204
títulos sin clasificar, no de adivinar sinónimos. Por eso incluye cosas que
ningún diccionario genérico traería: "espuma de afeitar", "moonsand",
"guarde motricidad fina", "abanderados", "Rancho de Popy".

Dos categorías nuevas salen de ese material, porque aparecen mucho y no
entraban en ninguna existente:
  Rutinas     — período de inicio, adaptación, siesta, almuerzo, guardería.
                Es la mitad del archivo de Kinder.
  Cumpleaños  — cumples de sala y el "Cumple de Teddy".

Uso:
    python classify.py --in ../data/folders.json --out ../data/tags.json
"""
import argparse
import json
import re
import sys
import unicodedata
from collections import Counter


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def norm(s):
    return strip_accents(str(s or "").lower())


# Un título que sólo nombra el nivel es una jornada de aula sin descripción.
# Admite el año pegado y el número suelto, que es como aparecen de verdad:
# "Year 4", "Kinder 2022", "Year 6 - 2024", "Kinder 3", "K3S".
ONLY_LEVEL = re.compile(
    r"^\s*(pre[\s-]?k(inder)?[gs]?|k(inder)?\s*[123]?[gs]?|y(ear)?\s*[1-6][gs]?|"
    r"[1-6][gs]|s[1-6]|primary|secondary|sala)"
    r"(\s*[-–]?\s*(\d{1,4}|[gs]|[123]))*\s*[-–]?\s*$"
)

# El orden importa: la primera categoría que matchea gana el ai_event_type,
# pero todas las que matchean quedan en ai_tags.
RULES = [
    ("Retratos", (
        "mugshot", "mugshots", "mug shots", "portrait", "portraits", "retratos",
        "groups", "grupales", "group photos", "photo shoot", "photoshoot",
        "missing photos", "informal", "fotos carnet", "photomovie",
        "nenas", "nenes", "group photos with names",
    )),
    ("Ceremonia", (
        "acto", "ceremony", "ceremonia", "graduation", "graduacion", "valete",
        "prizegiving", "assembly", "founders", "speech day", "colacion",
        "diplomas", "diploma", "bandera", "25 de mayo", "9 de julio",
        "san martin", "tradicion", "dia del maestro", "dia de la maestra",
        "abanderados", "abandorados", "teachers day", "teachers' day",
        "independencia", "himno", "izamiento",
    )),
    ("Escenario", (
        "play", "drama", "concert", "concierto", "music", "musica", "coro",
        "choir", "songs", "gala", "recital", "artes", "exhibition", "muestra",
        "puppet", "titeres", "production", "cast", "obra", "cancionero",
        "pelicula", "movie", "movies", "video", "videos", "footage", "drone",
        "cortes", "filmacion",
        "disfraces", "payasos", "bailando", "baile", "danza", "murga",
        "storyteller", "show and tell", "desfile",
    )),
    ("Deporte", (
        "rugby", "hockey", "sports", "football", "futbol", "cricket",
        "athletics", "atletismo", "swimming", "natacion", "cross country",
        "interhouse", "interhouses", "tournament", "torneo", "match",
        "bowling", "yoga", "gym", "gimnasio", "houses", "house sorting",
        "bicicleteada", "bicicletas", "wheels", "house captains", "captains",
        "fun run", "tug", "golf", "pumas", "paracaidas", "carrera",
        "sports day", "field day", "olimpiadas", "basquet", "voley",
    )),
    ("Viaje", (
        "camp", "campamento", "camping", "trip", "excursion", "salida",
        "viaje", "cordoba", "tandil", "angostura", "africa", "new zealand",
        "bariloche", "mar del plata", "colonia suiza", "peninsula valdes",
        "salta", "lujan", "la boca", "rosario", "uruguay", "casa rosada",
        "entre rios", "museo", "granja", "zoo", "planetario", "teatro colon",
        "visita", "visit", "picnic", "malba", "tigre", "iguazu", "mendoza",
        "rancho", "chacra", "estancia", "quinta", "outing", "paseo",
    )),
    ("Feria", (
        "fair", "feria", "expo", "open day", "openday", "science fair",
        "book week", "kermes", "kermesse", "pyp week", "exhibicion",
        "mercado", "market",
    )),
    ("Comunidad", (
        "solidario", "solidaria", "solidarios", "community", "comunidad",
        "service", "donacion", "campana solidaria", "techo por mi pais",
        "techo", "cas", "abuelos", "family day", "dia de la familia",
        "solidaridad", "despedida", "farewell",
        "mothers", "fathers", "madres", "padres", "dia de la madre",
        "dia del padre", "voluntariado", "caritas",
    )),
    ("Cumpleanos", (
        "cumple", "cumpleanos", "birthday", "teddy",
    )),
    ("Rutinas", (
        "periodo de inicio", "periodo de adaptacion", "settling", "adaptation",
        "adaptacion", "rutinas", "routines", "nap", "siesta", "lunch",
        "almuerzo", "snack", "merienda", "guarderia", "guardaria",
        "first day", "second day", "third day", "primer dia", "day 1",
        "new students", "new staff", "nuevos alumnos", "a new place",
        "brush your teeth", "cepillado", "daily plan", "vida diaria",
        "daily life", "guarde", "toddlers", "sala general",
    )),
    ("Aula", (
        "class", "clase", "aula", "corner", "corners", "juego", "jugando",
        "jugamos", "playing", "playground", "art", "arte", "pintando",
        "painting", "pintamos", "dibujo", "collage", "science", "ciencia",
        "ict", "library", "biblioteca", "book", "lectura", "reading",
        "leyendo", "taller", "workshop", "workshops", "proyecto", "project",
        # material plástico, que es la mitad de los títulos de Kinder
        "espuma", "afeitar", "tempera", "temperas", "masa", "moonsand",
        "moon sand", "playdough", "plastilina", "goma eva", "cartulina",
        "crayones", "pinceles", "rodillos", "punzon", "tijera", "papel",
        "mural", "sellamos", "esponjeado", "dactilopintura", "harina",
        "plasticola", "globos", "burbujas", "bubbles", "agua", "arena",
        # contenidos
        "phonics", "lecto", "lectoescritura", "maths", "matematicas",
        "matem", "numeros", "figuras", "geometric", "magnetismo",
        "experiments", "experimento", "volcanoes", "recycling", "reciclado",
        "huerta", "smartboard", "chromebook", "lab", "laboratorio",
        "motricidad", "cuento", "story", "libritos", "libros", "unidad",
        "unit", "cierre", "body", "circles", "cooking", "sorting",
        "seres vivos", "estaciones", "otono", "autumn", "spring", "primavera",
        "invierno", "verano", "sel", "fmc", "this is me", "me box",
        # segunda pasada, sobre lo que quedaba sin clasificar
        "team work", "trabajo en equipo", "drawing", "draw", "dibujando",
        "paint", "puffy paint", "paper", "colores", "making", "garden",
        "mapa", "edtech", "catequesis", "luz negra", "fluo", "easter",
        "pascua", "dulce de leche", "wonder", "challenge", "pruebas",
        "green", "rojo", "amarillo", "azul", "toddlers",
    )),
]

ACTIVITY_RULES = [
    ("Arte", ("espuma", "afeitar", "tempera", "temperas", "masa",
                  "moonsand", "moon sand", "playdough", "plastilina",
                  "goma eva", "cartulina", "crayones", "pinceles", "rodillos",
                  "punzon", "tijera", "mural", "collage", "dibujo", "pintando",
                  "pintamos", "painting", "paint", "arte", "art", "sellamos",
                  "esponjeado", "dactilopintura")),
    ("Lengua",   ("phonics", "lecto", "lectoescritura", "cuento", "story",
                  "leyendo", "libros", "libritos", "reading", "lectura",
                  "show and tell", "cancionero")),
    ("Matemática", ("maths", "matematicas", "matem", "numeros", "figuras",
                    "geometric", "volumen", "circles", "sorting")),
    ("Ciencia",  ("science", "ciencia", "experiments", "experimento",
                  "magnetismo", "volcanoes", "seres vivos", "huerta",
                  "recycling", "reciclado", "body", "estaciones")),
    ("ICT",      ("ict", "smartboard", "chromebook", "computacion", "lab",
                  "laboratorio", "online")),
    ("Música",   ("music", "musica", "coro", "choir", "songs", "cancionero",
                  "bailando", "baile")),
]


def classify(title):
    t = norm(title)
    if not t.strip() or ONLY_LEVEL.match(t):
        return "Aula", ["Jornada de aula"]

    types, tags = [], []
    for name, words in RULES:
        if any(w in t for w in words):
            types.append(name)
    for name, words in ACTIVITY_RULES:
        if any(w in t for w in words):
            tags.append(name)

    if not types:
        return None, tags
    return types[0], sorted(set(types + tags))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="../data/folders.json")
    ap.add_argument("--out", default="../data/tags.json")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    with open(a.inp, encoding="utf-8") as fh:
        rows = json.load(fh)

    events = [r for r in rows if r["depth"] >= 2 and not r["noise"]]
    by_id = {r["folder_id"]: r for r in rows}
    out, stats, nuevos, heredados = [], Counter(), 0, 0

    resolved = {}
    for r in events:
        base = [x for x in (r["event_types"].split("|") if r["event_types"] else []) if x]
        et, tags = classify(r["title"])
        if base and not et:
            et = base[0]
        if not base and et:
            nuevos += 1
        resolved[r["folder_id"]] = (et, base, tags or [])

    # Una subcarpeta sin titulo propio ("[Originals]", "K1S", "20X25") vive
    # dentro de un evento que si esta clasificado: hereda su tipo en vez de
    # quedar suelta. Se sube por el arbol hasta encontrar un ancestro con tipo.
    def inherit(fid, hops=0):
        node = by_id.get(fid)
        if not node or hops > 8:
            return None
        pid = node.get("parent_id")
        if not pid:
            return None
        got = resolved.get(pid)
        if got and got[0]:
            return got[0]
        return inherit(pid, hops + 1)

    for r in events:
        fid = r["folder_id"]
        et, base, tags = resolved[fid]
        inherited = False
        if not et:
            et = inherit(fid)
            if et:
                inherited = True
                heredados += 1
        merged = sorted(set(base + tags + ([et] if et else [])))
        if et or merged:
            out.append({"id": fid, "ai_event_type": et, "ai_tags": merged,
                        "inherited": inherited})
        stats[et or "(sin clasificar)"] += 1

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False)

    total = len(events)
    sin = stats["(sin clasificar)"]
    print("Eventos:               %d" % total)
    print("Ya tenían tipo:        %d" % sum(1 for r in events if r["event_types"]))
    print("Clasificados de nuevo: %d" % nuevos)
    print("Heredados del padre:   %d" % heredados)
    print("Quedan sin tipo:       %d  (%.1f%%)" % (sin, 100 * sin / total))
    print("Cobertura final:       %.1f%%\n" % (100 * (total - sin) / total))
    for k, v in stats.most_common():
        print("   %-16s %5d  %5.1f%%" % (k, v, 100 * v / total))
    print("\nEscrito: %s  (%d filas)" % (a.out, len(out)))


if __name__ == "__main__":
    main()
