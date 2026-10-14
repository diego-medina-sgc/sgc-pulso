# -*- coding: utf-8 -*-
"""
Fase 2 — Clasificación de los títulos que las reglas base no alcanzaron.

El vocabulario de acá salió de leer las 1.698 firmas distintas de los 3.204
títulos sin clasificar, no de adivinar sinónimos. Por eso incluye cosas que
ningún diccionario genérico traería: "espuma de afeitar", "moonsand",
"guarde motricidad fina", "abanderados", el nombre de un rancho.

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
from types import SimpleNamespace


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def norm(s):
    return strip_accents(str(s or "").lower())


# Un título que sólo nombra el nivel: "Year 4", "Kinder 2022", "Year 6 - 2024",
# "Kinder 3", "K3S". Admite el año pegado y el número suelto, que es como
# aparecen de verdad.
#
# NI EL TÍTULO VACÍO NI EL QUE ES SÓLO UN NIVEL SON "AULA" (12 y 13/10/2026).
# Los dos caían en "Aula / Jornada de aula": 349 carpetas con título vacío
# (15.206 fotos) y 412 que sólo dicen el nivel (7.321 fotos), con un tipo que
# nadie dio. Decisión del dueño de la biblioteca: quedan sin tipo y sin la
# etiqueta, y las completa el modelo de escenas. No se reemplaza por otro
# default, ni por el tipo de la carpeta madre. La regex queda para cortar
# antes: si siguieran a las reglas, el match por subcadena les inventaría
# algo.
ONLY_LEVEL = re.compile(
    r"^\s*(pre[\s-]?k(inder)?[gs]?|k(inder)?\s*[123]?[gs]?|y(ear)?\s*[1-6][gs]?|"
    r"[1-6][gs]|s[1-6]|primary|secondary|sala)"
    r"(\s*[-–]?\s*(\d{1,4}|[gs]|[123]))*\s*[-–]?\s*$"
)

# EX ALUMNOS (13/10/2026): cenas, encuentros y deportes de exalumnos. "OG
# dinner", "Old Georgian gathering", "Ex alumnos 2022", "OGC 108th dinner".
# Palabra entera: "og" por subcadena está en "blog", "yoga" y "biology".
# "Egresados" no entra: son los que terminan, no los que volvieron.
# El OGC como LUGAR no alcanza ("Encuentro ... en el OGC": staff que fue al
# club): "en el OGC", "al OGC", "at the OGC" se borran antes de buscar.
EX_ALUMNOS = re.compile(r"\bogc?s?\b|\bold georgians?\b|\bex[\s-]?alumn[oa]s?\b|\balumni\b")
OGC_LUGAR = re.compile(r"\b(?:en|al|at|in)\s+(?:el\s+|la\s+|the\s+)?ogc?\b")


def es_ex_alumnos(t):
    """El título (ya normalizado) es de exalumnos."""
    return bool(EX_ALUMNOS.search(OGC_LUGAR.sub(" ", t)))


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
        # sacramentos: "Comuniones 2024", "Primera comunión 5A"
        "comunion",
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
    # Antes que Deporte, y en el mismo lugar que en parse_folders.EVENT_RULES:
    # "OG sports day" lleva los dos tipos y la tarjeta muestra "Ex alumnos"
    # por los dos lados (event_types[0] y ai_event_type).
    ("Ex alumnos", (SimpleNamespace(search=es_ex_alumnos),)),
    ("Deporte", (
        "rugby", "hockey", "sports", "football", "futbol", "cricket",
        "athletics", "atletismo", "swimming", "natacion", "cross country",
        "crosscountry", "steeplechase", "diving", "relay",
        "tennis", "tenis", "basketball", "volley",
        # "Old Georgians deporte", "OG Seven" (rugby seven)
        "deporte", re.compile(r"\bsevens?\b"),
        "tournament", "torneo", "match",
        "bowling", "yoga", "gym", "gimnasio",
        "bicicleteada", "bicicletas", "wheels", "house sorting", "house captains",
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
        # giras (13/10/2026): "UK Tour día 3", "Gira Chile", "NZ Día 10".
        # Palabra entera: "touring team" es el equipo que nos visita.
        "tour", "gira", re.compile(r"\bnz(aus)?\d*\b"),
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
    # Con eñe: la base tenía las dos escrituras (62 y 30 carpetas) y el
    # filtro mostraba dos tipos.
    ("Cumpleaños", (
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


# PALABRAS CORTAS QUE VAN COMO PALABRA ENTERA (13/10/2026). Las reglas buscan
# por subcadena, y eso sirve para plurales y derivados ("mugshot" en
# "mugshots", "visit" en "visitamos"). Pero estas aparecen adentro de otras
# palabras que no tienen nada que ver, o empiezan otras de otro sentido:
#   art/arte/artes  party, start, smartboard, kindergarten, parte, martes, carton
#   play            display; playground y playing (eran Escenario)
#   cast            castillo, castle, encastres
#   gala            regalar
#   cas (CAS)       geometricas, casa, chicas, cast
#   sel (SEL)       seleccion, sellado, ourselves
#   ict             pictures, dictado
#   lab             palabras, reciclable, laberinto ("labo" sí cuenta)
#   unit/unidad     community, comunidad
#   aula            nombres que la contienen
#   agua            paraguas, Paraguay ("aguada" sí cuenta)
#   arena           nombres que la contienen
#   acto            factory, contacto, abstracto
#   trip            stripes
#   story, book     history, facebook (por la izquierda; "storytelling" sí)
#   mercado/market  supermercado, supermarket
#   camp            campus, campaña, camperas
#   sports          transports
#   fair            fairy
#   masa            masajes
#   nap             apellidos que empiezan así
#   day 1           day 10 a day 16 (días de un viaje)
#   retratos        autorretratos (es arte, no carnets)
PALABRA_ENTERA = {w: re.compile(r) for w, r in {
    "art": r"\bart(s|work|ist[a-z]*)?\b", "arte": r"\bartes?\b", "artes": r"\bartes\b",
    "play": r"\bplays?\b", "cast": r"\bcast\b", "gala": r"\bgala\b",
    "cas": r"\bcas\b", "sel": r"\bsel\b", "ict": r"\bict\b",
    "lab": r"\blab(o|s)?\b", "unit": r"\bunits?\b", "unidad": r"\bunidad(es)?\b",
    "aula": r"\baulas?\b", "agua": r"\bagua(s|da|das)?\b", "arena": r"\barenas?\b",
    "acto": r"\bactos?\b", "trip": r"\btrips?\b", "story": r"\bstory",
    "book": r"\bbook", "mercado": r"\bmercados?\b", "market": r"\bmarkets?\b",
    "camp": r"\bcamps?\b", "sports": r"\bsports\b", "fair": r"\bfairs?\b",
    "masa": r"\bmasas?\b", "nap": r"\bnaps?\b", "day 1": r"\bday 1\b",
    "retratos": r"\bretratos\b",
    "tour": r"\btours?\b", "gira": r"\bgiras?\b",
    "houses": r"\bhouses\b",
}.items()}

# HOUSES SÓLO SI NO HAY OTRA COSA (13/10/2026). "Houses" y "Captains" son
# Deporte cuando el título no dice nada más ("House sorting" y
# "House captains" siguen siendo Deporte siempre, como antes); si dice
# algo ("Houses Prep Expo Artesanos" -> Feria, "K3C houses unit" -> Aula),
# gana eso.
HOUSES_DEBIL = ("houses", "captains")


def _hay(w, t):
    if hasattr(w, "search"):
        return bool(w.search(t))
    rx = PALABRA_ENTERA.get(w)
    # el guion bajo cuenta como espacio: "K1S Art_ Shake the box"
    return bool(rx.search(t.replace("_", " "))) if rx else w in t


# "INTERHOUSE" NO ES UN DEPORTE (12/10/2026). Es una competencia entre houses
# de cualquier cosa: "Art interhouse", "Chess interhouse", "Music interhouse",
# "Physics interhouse". Como palabra de Deporte hacía deporte a 21 carpetas de
# arte y ajedrez (1.189 fotos). Se borra del título antes de matchear: el tipo
# sale de lo que se compite ("rugby" -> Deporte, "art" -> Aula, "music" ->
# Escenario, "chess" -> nada).
# Hay que borrarla y no sólo sacarla de la lista porque el match es por
# subcadena: "interhouses" contiene "houses", que sí está en Deporte.
#
# LA QUE NO DICE DE QUÉ ES, SÍ ES DEPORTE (13/10/2026): "Interhouse 2021",
# "Interhouses Prim", "Interhouses fin de año", "Interhouse walk" son jornadas
# deportivas. "No dice de qué" = lo que queda sin la palabra son sólo años,
# niveles, secciones o "walk".
INTERHOUSE = re.compile(r"\binter\s*-?\s*t?houses?\b")
INTERHOUSE_RELLENO = {
    "prim", "primary", "sec", "secondary", "college", "prep", "kinder", "senior",
    "junior", "fin", "de", "del", "ano", "cycle", "first", "second", "ih", "walk",
}
_NUMERO = re.compile(r"^(\d+(st|nd|rd|th)?|y[1-6]|[1-6][gs]?)$")


def interhouse_sin_tema(t):
    """El título (ya normalizado) es un interhouse que no dice de qué."""
    if not INTERHOUSE.search(t):
        return False
    resto = re.split(r"[^a-z0-9]+", INTERHOUSE.sub(" ", t))
    return all(not w or w in INTERHOUSE_RELLENO or _NUMERO.match(w) for w in resto)


def classify(title):
    t = norm(title)
    if not t.strip() or ONLY_LEVEL.match(t):
        return None, []
    if interhouse_sin_tema(t):
        return "Deporte", ["Deporte"]
    t = INTERHOUSE.sub(" ", t)

    types, tags = [], []
    for name, words in RULES:
        if any(_hay(w, t) for w in words):
            types.append(name)
    if not types and any(_hay(w, t) for w in HOUSES_DEBIL):
        types.append("Deporte")
    for name, words in ACTIVITY_RULES:
        if any(_hay(w, t) for w in words):
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
        # el título vacío queda sin tipo: heredar el de la madre sería otro
        # default (decisión del 12/10/2026)
        if not et and (r.get("title") or "").strip():
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
