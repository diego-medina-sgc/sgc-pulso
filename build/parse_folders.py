# -*- coding: utf-8 -*-
"""
Fase 1 — Parser de carpetas del archivo Server Media.

Lee manifest.json (salida del indexador de Drive) y produce la tabla de eventos
normalizada con todas las facetas derivadas del nombre de cada carpeta.

Drive no se toca: esto solo lee el manifiesto.

Uso:
    python parse_folders.py manifest.json --out ../data
"""
import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from datetime import date as _date
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify import interhouse_sin_tema, es_ex_alumnos

ROOT_ID = "0ADoh-DIvUMYeUk9PVA"

# ---------------------------------------------------------------- utilidades

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def norm(s):
    """Minúsculas sin acentos, para matchear reglas."""
    return strip_accents(s.lower())


def has(text, *words):
    """Palabra completa, sobre el texto ya normalizado."""
    return any(re.search(r"\b" + w + r"\b", text) for w in words)


# ---------------------------------------------------------------- fecha

# 2018-03-06 - Título   |   2018-03-06 – Título   |   2018-03-06 Título
RE_DAY = re.compile(r"^(\d{4})[-_.](\d{1,2})[-_.](\d{1,2})\s*(?:[-–—]\s*)?(.*)$")
# 2010-04 15 - Título  (día separado por espacio, aparece en 2010)
RE_DAY_SP = re.compile(r"^(\d{4})[-_.](\d{1,2})\s+(\d{1,2})\s*[-–—]\s*(.*)$")
# 2010-0825 - Título   (mes y día pegados)
RE_DAY_GLUED = re.compile(r"^(\d{4})[-_.](\d{2})(\d{2})\s*[-–—]\s*(.*)$")
# 2025-03 - Título
RE_MONTH = re.compile(r"^(\d{4})[-_.](\d{1,2})\s*[-–—]\s*(.*)$")
# 2017 - Título   |   2023 K3
RE_YEAR = re.compile(r"^(\d{4})\s*(?:[-–—]\s*)?(.*)$")
# 23 sep 2022 - Título
MESES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7,
         "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
         "jan": 1, "apr": 4, "aug": 8, "dec": 12}
# Un año suelto en cualquier parte del nombre: "Fotos 2023", "Landings 2024",
# "Uruguay 2023". Hasta el 21/9/2026 solo se leia al principio, y las
# carpetas de año de Quilmes ("Fotos 2026") quedaban sin año: 308 albumes y
# ~12.500 fotos fuera del filtro por año. Decisión: dejar sin año lo ambiguo.
# Por eso no cuenta lo que va entre parentesis ("Fotos staff (actualizado
# 2026)" es la fecha de actualizacion, no la de las fotos), y si hay dos años
# distintos ("International Awards 2017-2018") no se elige ninguno.
RE_ANIO_SUELTO = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
# y no del futuro: "IMG_2034" es un numero de archivo, no un año
_ANIO_TOPE = __import__("datetime").date.today().year + 1
RE_PARENTESIS = re.compile(r"\([^)]*\)")
RE_DMY = re.compile(r"^(\d{1,2})\s+([a-z]{3})[a-z]*\.?\s+(\d{4})\s*[-–—]\s*(.*)$")


def valid_date(y, m, d):
    """Rango razonable y fecha que exista de verdad en el calendario.

    Hay carpetas fechadas 2013-02-29: 2013 no fue bisiesto. Se aceptan igual
    como 'day' porque el nombre dice el día, pero el loader no arma event_date.
    """
    return 1990 <= y <= 2100 and 1 <= m <= 12 and 1 <= d <= 31


def real_date(y, m, d):
    try:
        _date(y, m, d)
        return True
    except (ValueError, TypeError):
        return False


def parse_date(name):
    """Devuelve (year, month, day, title, precision)."""
    n = name.strip()

    for rx in (RE_DAY, RE_DAY_SP, RE_DAY_GLUED):
        m = rx.match(n)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if valid_date(y, mo, d):
                return y, mo, d, m.group(4).strip(), "day"

    m = RE_DMY.match(norm(n))
    if m and m.group(2) in MESES:
        y, mo, d = int(m.group(3)), MESES[m.group(2)], int(m.group(1))
        if valid_date(y, mo, d):
            # el título original conserva mayúsculas y acentos
            return y, mo, d, n[n.find("-") + 1:].strip(), "day"

    m = RE_MONTH.match(n)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if valid_date(y, mo, 1):
            return y, mo, None, m.group(3).strip(), "month"

    m = RE_YEAR.match(n)
    if m and 1990 <= int(m.group(1)) <= 2100:
        return int(m.group(1)), None, None, m.group(2).strip(), "year"

    anios = {int(x) for x in RE_ANIO_SUELTO.findall(RE_PARENTESIS.sub(" ", n))
             if 1990 <= int(x) <= _ANIO_TOPE}
    if len(anios) == 1:
        return anios.pop(), None, None, n, "year"

    return None, None, None, n, "none"


def term(month):
    """Trimestre del calendario escolar argentino."""
    if month is None:
        return None
    if month in (2, 3, 4, 5):
        return "T1"
    if month in (6, 7, 8):
        return "T2"
    if month in (9, 10, 11, 12):
        return "T3"
    return None  # enero: receso


# ---------------------------------------------------------------- niveles

# Nomenclatura institucional (la misma que usa Salesforce y el campus de
# Quilmes): Kinder / Prep / College, con niveles K1-K3, P1-P6, C1-C6. En los
# nombres de carpeta de North aparecen como Y1-Y6 y S1-S6, asi que se leen en
# ese formato y se normalizan al escribir.
# K1G, K3S, PreKG, 1G, 4S, S6 ...
RE_KINDER = re.compile(r"\b(prek|pre\s*k|pre-k|k[123])\s*([gs])?\b")
RE_PRIMARY_Y = re.compile(r"\b(?:y|year)\s*([1-6])\s*([gs])?\b")
RE_PRIMARY_N = re.compile(r"\b([1-6])\s*([gs])\b")          # 1G, 4S
RE_SECONDARY = re.compile(r"\bs\s*([1-6])\b")               # S1..S6
# C1..C6, el codigo institucional tal cual (15/10/2026). Desde 2025 las
# carpetas de North se nombran asi ("2026-03-27 - C6 mugshots", "C3 Trip to
# Jujuy") y quedaban sin nivel: 18 carpetas, 6 de carnets, y sus carnets no se
# podian cruzar con la camada del alumno. "2c" (division) no entra: el \b
# pide que la c empiece la palabra.
RE_COLLEGE_C = re.compile(r"\bc\s*([1-6])\b")

# términos de sede, para no confundir "South" con la división S
RE_SITE_NORTH = re.compile(r"\bnorth\b")
RE_SITE_SOUTH = re.compile(r"\b(quilmes|south\s+site)\b")


def parse_levels(t):
    """t viene normalizado. Devuelve (section, levels, divisions)."""
    levels, divisions = [], set()
    sections = set()

    for m in RE_KINDER.finditer(t):
        lvl = m.group(1).replace(" ", "").replace("-", "")
        lvl = "prek" if lvl.startswith("prek") or lvl.startswith("pre") else lvl
        levels.append(lvl.upper())
        sections.add("Kinder")
        if m.group(2):
            divisions.add(m.group(2).upper())

    for m in RE_PRIMARY_Y.finditer(t):
        levels.append("P" + m.group(1))
        sections.add("Prep")
        if m.group(2):
            divisions.add(m.group(2).upper())

    for m in RE_PRIMARY_N.finditer(t):
        levels.append("P" + m.group(1))
        sections.add("Prep")
        divisions.add(m.group(2).upper())

    for rx in (RE_SECONDARY, RE_COLLEGE_C):
        for m in rx.finditer(t):
            levels.append("C" + m.group(1))
            sections.add("College")

    if has(t, "kinder", "jardin"):
        sections.add("Kinder")
    if has(t, "primary", "prep", "pyp", "ep"):
        sections.add("Prep")
    if has(t, "secondary", "college", "ib", "myp", "senior"):
        sections.add("College")
    if has(t, "staff", "teachers", "docentes", "personal"):
        sections.add("Staff")

    # una sola sección para el filtro principal
    if len(sections) > 1:
        section = "Whole School"
    elif sections:
        section = next(iter(sections))
    else:
        section = None

    return section, sorted(set(levels)), sorted(divisions)


def parse_site(t):
    if RE_SITE_NORTH.search(t):
        return "North"
    if RE_SITE_SOUTH.search(t):
        return "Quilmes"
    return None


# ---------------------------------------------------------------- tipo de evento

EVENT_RULES = [
    # North dice "mugshots"; Quilmes dice "individuales" o "fotos alumnos".
    # "grupales" a secas queda afuera: para identificar a una persona hace
    # falta una foto de una sola persona.
    # "Deporte individual" no es un retrato: es el deporte (13/10/2026).
    ("Retratos",   ("mugshot", "mugshots", "mug shots", "portrait", "portraits",
                    "retratos", "fotos carnet", r"(?<!deporte )individual",
                    r"(?<!deportes )individuales",
                    "fotos alumnos", "fotos de alumnos")),
    ("Ceremonia",  ("acto", "ceremony", "ceremonia", "graduation", "graduacion",
                    "valete", "prizegiving", "assembly", "founders", "founder's",
                    "speech day", "colacion")),
    ("Escenario",  ("play", "drama", "concert", "concierto", "music", "musica",
                    "coro", "choir", "songs", "gala", "recital", "artes",
                    "exhibition", "muestra")),
    # Ex alumnos: la regla vive en classify.es_ex_alumnos (None = usarla), y
    # va antes que Deporte en los dos lados para que "OG sports day" muestre
    # lo mismo en event_types[0] y en ai_event_type (13/10/2026).
    ("Ex alumnos", None),
    # "interhouse" no está: es una competencia entre houses de cualquier cosa
    # (arte, ajedrez, música); el deporte lo dice la otra palabra (12/10/2026).
    # Con ella se fue el Deporte de "Tennis interhouse" o "Diving interhouse":
    # por eso están acá todos los deportes de SPORTS.
    ("Deporte",    ("rugby", "hockey", "sports", "football", "futbol", "cricket",
                    "athletics", "atletismo", "swimming", "natacion", "cross",
                    "crosscountry", "steeplechase", "diving", "relay",
                    "tennis", "tenis", "basketball", "basquet",
                    "volleyball", "voley", "volley", "deporte", "sevens?",
                    "tournament", "torneo", "match",
                    "vs", "bowling", "yoga", "gym")),
    ("Viaje",      ("camp", "campamento", "trip", "excursion", "salida", "viaje",
                    "cordoba", "tandil", "angostura", "africa", "nz",
                    "new zealand", "bariloche", "visita", "visit",
                    # giras, como en classify.py (13/10/2026)
                    "tours?", "giras?", r"nz(aus)?\d*")),
    ("Feria",      ("fair", "feria", "expo", "open day", "openday", "science fair",
                    "book week")),
    ("Comunidad",  ("solidario", "solidarios", "solidaria", "community", "comunidad",
                    "service", "donacion", "campana")),
    ("Aula",       ("class", "clase", "aula", "corner", "corners", "juego", "jugando",
                    "playing", "snack", "playground", "art", "arte", "pintando",
                    "painting", "pintamos", "dibujo", "collage", "science",
                    "ciencia", "ict", "library", "biblioteca", "book", "lectura",
                    "reading", "taller", "workshop", "proyecto", "project")),
]

ACTIVITY_RULES = [
    ("Arte",      ("art", "arte", "pintando", "painting", "pintamos", "dibujo",
                   "collage", "escultura", "esculturas", "masa", "espuma")),
    ("Drama",     ("drama", "teatro", "play")),
    ("Música",    ("music", "musica", "coro", "choir", "songs", "concert", "banda")),
    ("Ciencia",   ("science", "ciencia", "experimento", "naturales")),
    ("ICT",       ("ict", "chromebook", "chromebooks", "computacion", "smartboard",
                   "tecnologia")),
    ("Lectura",   ("book", "books", "library", "biblioteca", "lectura", "reading",
                   "cuento", "story")),
    ("Deportes",  ("pe", "gym", "educacion fisica")),
]

# palabra del título -> deporte. UN SOLO VALOR POR DEPORTE (12/10/2026): la
# base tenía "football" y "futbol", "swimming" y "natacion", "athletics" y
# "atletismo", "basketball" y "basquet" como deportes distintos. Queda el
# inglés, que es el que ya tenía la mayoría de las carpetas.
SPORTS = {
    "rugby": "rugby", "hockey": "hockey", "cricket": "cricket",
    "football": "football", "futbol": "football",
    "athletics": "athletics", "atletismo": "athletics",
    "swimming": "swimming", "natacion": "swimming",
    "cross country": "cross country", "crosscountry": "cross country",
    "steeplechase": "steeplechase", "diving": "diving",
    "bowling": "bowling",
    "tennis": "tennis", "tenis": "tennis",
    "basketball": "basketball", "basquet": "basketball",
    "volleyball": "volleyball", "voley": "volleyball", "volley": "volleyball",
}

TRIPS = {"cordoba": "Córdoba", "tandil": "Tandil", "angostura": "Villa La Angostura",
         "africa": "Sudáfrica", "new zealand": "Nueva Zelanda", "nz": "Nueva Zelanda",
         "bariloche": "Bariloche", "mendoza": "Mendoza", "iguazu": "Iguazú"}

# Palabras de Retratos que alcanzan solas aunque el título diga otro tipo.
RETRATOS_FUERTES = ("mugshot", "mugshots", "mug shots", "portrait", "portraits",
                    "retratos", "fotos carnet", r"(?<!deporte )individual",
                    r"(?<!deportes )individuales")

RE_VS = re.compile(r"\bvs\.?\s+([a-z' ]{3,30})")


def parse_event(t):
    types = [name for name, words in EVENT_RULES
             if (es_ex_alumnos(t) if words is None else has(t, *words))]
    # "fotos alumnos" sola es la manera de Quilmes de decir individuales, pero
    # si el título también dice de qué es ("Videos y Fotos alumnos Music"), es
    # un álbum de eso y no una carpeta de carnets (19/10/2026).
    if ("Retratos" in types and len(types) > 1
            and not has(t, *RETRATOS_FUERTES)):
        types.remove("Retratos")
    # el interhouse que no dice de qué es una jornada deportiva (classify.py)
    if "Deporte" not in types and interhouse_sin_tema(t):
        types.append("Deporte")
    acts = [name for name, words in ACTIVITY_RULES if has(t, *words)]
    sports = [v for k, v in SPORTS.items() if has(t, k)]
    trips = sorted({label for key, label in TRIPS.items() if has(t, key)})

    opponent = None
    m = RE_VS.search(t)
    if m:
        opponent = m.group(1).strip().title()

    return types, acts, sorted(set(sports)), trips, opponent


# ---------------------------------------------------------------- ruido

RE_NOISE = [
    (re.compile(r"\.files$", re.I),                    "web_artifact"),
    (re.compile(r"^\[originals\]", re.I),              "originals"),
    (re.compile(r"^(nueva carpeta|new folder|untitled)", re.I), "untitled"),
    (re.compile(r"^\d{3}___\d+$|^dcim$", re.I),        "dcim"),
    (re.compile(r"^thumbs\.db$", re.I),                "thumbs"),
]


def noise_kind(name):
    for rx, kind in RE_NOISE:
        if rx.search(name.strip()):
            return kind
    return None


# ---------------------------------------------------------------- main

def build(manifest_path, out_dir):
    with open(manifest_path, encoding="utf-8") as fh:
        man = json.load(fh)

    folders = man["folders"]
    by_id = {f["id"]: f for f in folders}
    children = defaultdict(list)
    for f in folders:
        children[f.get("parent")].append(f)

    # carpeta de nivel 1 (año o colección) para cada carpeta
    def top_of(fid):
        seen = 0
        cur = by_id.get(fid)
        while cur and seen < 25:
            if cur.get("parent") == ROOT_ID:
                return cur
            cur = by_id.get(cur.get("parent"))
            seen += 1
        return None

    def path_of(fid):
        parts, seen = [], 0
        cur = by_id.get(fid)
        while cur and cur["id"] != ROOT_ID and seen < 25:
            parts.append(cur["name"].strip())
            cur = by_id.get(cur.get("parent"))
            seen += 1
        return "/".join(reversed(parts))

    def depth_of(fid):
        d, seen = 0, 0
        cur = by_id.get(fid)
        while cur and cur.get("parent") and seen < 25:
            cur = by_id.get(cur["parent"])
            d += 1
            seen += 1
        return d

    rows = []
    for f in folders:
        if f["id"] == ROOT_ID:
            continue

        name = f["name"].strip()
        top = top_of(f["id"])
        top_name = top["name"].strip() if top else None
        is_year_root = bool(top_name and re.fullmatch(r"\d{4}", top_name))

        nk = noise_kind(name)
        y, mo, d, title, prec = parse_date(name)
        t = norm(title) if title else ""

        section, levels, divisions = parse_levels(t)
        types, acts, sports, trips, opponent = parse_event(t)
        site = parse_site(t)

        # si el nombre no trae año pero cuelga de una carpeta de año, se hereda
        year = y if y else (int(top_name) if is_year_root else None)

        rows.append({
            "folder_id": f["id"],
            "folder_name": name,
            "parent_id": f.get("parent") or "",
            "top_folder": top_name or "",
            "collection": "" if is_year_root else (top_name or ""),
            "depth": depth_of(f["id"]),
            "path": path_of(f["id"]),
            "n_subfolders": len(children[f["id"]]),
            "year": year or "",
            "month": mo or "",
            "day": d or "",
            "date_precision": prec,
            "date_from_name": "yes" if y else "no",
            "term": term(mo) or "",
            "title": title,
            "section": section or "",
            "levels": "|".join(levels),
            "divisions": "|".join(divisions),
            "site": site or "",
            "event_types": "|".join(types),
            "activities": "|".join(acts),
            "sports": "|".join(sports),
            "trips": "|".join(trips),
            "opponent": opponent or "",
            "is_mugshot": "yes" if "Retratos" in types else "no",
            "noise": nk or "",
        })

    os.makedirs(out_dir, exist_ok=True)
    cols = list(rows[0].keys())

    csv_path = os.path.join(out_dir, "folders.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    json_path = os.path.join(out_dir, "folders.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)

    report(rows, man)
    print("\nEscrito:")
    print("  ", csv_path)
    print("  ", json_path)
    return rows


def report(rows, man):
    out = sys.stdout
    events = [r for r in rows if r["depth"] == 2 and not r["noise"]]
    total = len(rows)

    out.write("=" * 66 + "\n")
    out.write("PARSER DE CARPETAS — Server Media\n")
    out.write("=" * 66 + "\n")
    out.write("Manifiesto generado: %s\n" % man.get("generatedAt"))
    out.write("Carpetas totales:    %d\n" % total)
    out.write("Imágenes indexadas:  %d\n" % man["stats"]["images"])
    out.write("Eventos (nivel 2):   %d\n\n" % len(events))

    nz = Counter(r["noise"] for r in rows if r["noise"])
    out.write("RUIDO FILTRADO\n")
    for k, v in nz.most_common():
        out.write("   %-16s %5d\n" % (k, v))
    out.write("   %-16s %5d\n\n" % ("TOTAL", sum(nz.values())))

    prec = Counter(r["date_precision"] for r in events)
    out.write("PRECISIÓN DE FECHA (eventos)\n")
    for k in ("day", "month", "year", "none"):
        v = prec.get(k, 0)
        out.write("   %-16s %5d  %5.1f%%\n" % (k, v, 100 * v / len(events)))
    out.write("\n")

    def cover(field):
        n = sum(1 for r in events if r[field])
        return n, 100 * n / len(events)

    out.write("COBERTURA DE FACETAS (eventos)\n")
    for f in ("section", "levels", "divisions", "event_types", "activities",
              "sports", "trips", "site", "opponent"):
        n, p = cover(f)
        out.write("   %-16s %5d  %5.1f%%\n" % (f, n, p))
    out.write("\n")

    sec = Counter(r["section"] or "(sin clasificar)" for r in events)
    out.write("SECCIÓN\n")
    for k, v in sec.most_common():
        out.write("   %-16s %5d  %5.1f%%\n" % (k, v, 100 * v / len(events)))
    out.write("\n")

    ev = Counter()
    for r in events:
        for x in (r["event_types"].split("|") if r["event_types"] else ["(sin tipo)"]):
            ev[x] += 1
    out.write("TIPO DE EVENTO (una carpeta puede tener varios)\n")
    for k, v in ev.most_common():
        out.write("   %-16s %5d  %5.1f%%\n" % (k, v, 100 * v / len(events)))
    out.write("\n")

    mug = [r for r in events if r["is_mugshot"] == "yes"]
    with_id = [r for r in mug if r["year"] and int(r["year"]) >= 2021]
    out.write("GALERÍA DE RETRATOS\n")
    out.write("   carpetas de retratos      %5d\n" % len(mug))
    out.write("   con nombres (>=2021)      %5d\n" % len(with_id))
    out.write("   sin nombres (<=2020)      %5d   <- las etiqueta la fase 5\n\n"
              % (len(mug) - len(with_id)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default="../data")
    a = ap.parse_args()
    build(a.manifest, a.out)
