# -*- coding: utf-8 -*-
"""
Lee el nombre que ya viene escrito en el archivo del retrato.

    python retratos_nombres.py --ensayo      # dice a quien identificaria
    python retratos_nombres.py --aplicar

DE QUE SE TRATA

Buena parte de los retratos de Quilmes traen el nombre en el propio archivo y
nadie lo habia leido: quedaron en la cola del juego para que alguien los mire
uno por uno, cuando la respuesta estaba en el nombre del archivo.

    MDF_4445.jpg Persona AC.jpg          467 fotos de 2023
    ALVAREZ DE OLIVERA Persona AD.jpg     73 fotos de 2024
    Persona AE Sebastian.jpg              70 fotos de 2023
    Persona AF.jpg                  66 fotos de 2025
    Persona AG.jpg                   54 fotos de 2026
    MDF_6792Aguilera.jpg                   142 fotos de 2022, solo el apellido

Son cinco convenciones distintas y ninguna dice cual es el nombre y cual el
apellido. Da igual: no hace falta adivinarlo.

EL PADRON DESAMBIGUA

"Persona AC" puede ser Persona AH o Persona AC. En vez de decidirlo con
una regla por año -que es lo fragil, porque el año que viene cambia- se
prueban todas las permutaciones contra las 4.652 personas del padron y se
acepta la que existe. Si existe una sola, es esa. Si existen dos, no se toca.

Por eso este script depende de haber cargado antes las planillas del colegio
(padron_cargar.py): sin las 2.455 personas de Quilmes no hay contra que
probar.

CUANDO NO ESCRIBE

  - si el nombre del archivo da mas de una persona posible (hermanos con el
    mismo apellido en el caso de "MDF_6792Aguilera.jpg")
  - si la persona es de la otra sede
  - si el año de la foto cae fuera de su cursada, con un año de margen
  - si la foto ya tiene nombre, o si alguien ya rechazo ese nombre a mano
  - si lo que queda despues de limpiar no es un nombre sino el numero de la
    tanda: "Individuales-1", "PrepInd-100", "KinderInd-27"
"""
import argparse
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import sb

# Un año de margen: un retrato sacado en marzo puede quedar fechado el año
# anterior, y la cursada del padron es por año lectivo.
MARGEN = 1

# Prefijo de camara: MDF_4445, DSC_7192, IMG-2231. A veces con la extension
# pegada, porque el archivo se renombro agregando el nombre al final.
CAMARA = re.compile(r"^[A-Za-z]{2,4}[_\-]?\d{3,6}(\.\w{3,4})?[\s_\-]*")
# Numero de tanda al final: "Individuales-1", "PrepInd-100"
TANDA = re.compile(r"[\s_\-]*\d{1,4}$")
COPIA = re.compile(r"\s*\(\d+\)\s*$")
EXT = re.compile(r"\.(jpe?g|png|heic|heif|tiff?)$", re.I)
# Lo que queda cuando el archivo no trae nombre sino el rotulo de la sesion.
ROTULO = re.compile(r"^(prep|kinder|college|individuales?|ind|fotos?|grupal|"
                    r"grupales|es\d|k\d[gsc]?|p\d[abc]?|c\d|valete|alumnos?)$", re.I)


def sin_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", s or "")
                   if unicodedata.category(c) != "Mn")


def clave(s):
    t = sin_tildes(s).lower()
    return " ".join("".join(c if c.isalnum() or c == " " else " " for c in t).split())


def separar_camello(t):
    """'RuizDiasParquet' -> 'Persona AI'.

    Solo si no hay ningun espacio: con espacios, el archivo ya viene separado
    y meterse a cortar por mayusculas rompe los 'De la Fuente'.
    """
    if " " in t or not re.search(r"[a-z][A-Z]", t):
        return t
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)


def texto_del_archivo(nombre):
    """El trozo del nombre de archivo que deberia ser una persona, o None."""
    t = EXT.sub("", nombre or "")
    t = COPIA.sub("", t)
    t = CAMARA.sub("", t)
    t = t.replace("_", " ")
    # "APELLIDO, Nombre" y "APELLIDO,Nombre"
    if "," in t:
        izq, der = t.split(",", 1)
        t = "%s %s" % (der.strip(), izq.strip())
    t = TANDA.sub("", t)
    t = separar_camello(t.strip())
    t = " ".join(t.split())
    if not t or ROTULO.match(t) or not re.search(r"[A-Za-z]{2}", t):
        return None
    # Los digitos sueltos que quedan adelante -"2244RuizDias"- no son nombre.
    t = re.sub(r"^\d+\s*", "", t)
    return t or None


def indexar(padron):
    """Del padron: por juego de palabras, por palabra suelta, y por apellido."""
    por_palabras = defaultdict(list)
    por_apellido = defaultdict(list)
    por_token = defaultdict(set)
    juegos = {}
    for p in padron:
        pal = clave(p.get("display_name")).split()
        if not pal:
            continue
        s = frozenset(pal)
        por_palabras[s].append(p)
        juegos[int(p["id"])] = (s, p)
        for a in pal:
            por_token[a].add(int(p["id"]))
        # El primer token es el nombre de pila; el resto, apellidos.
        for a in pal[1:]:
            por_apellido[a].append(p)
    return por_palabras, por_apellido, por_token, juegos


def por_subconjunto(pal, por_token, juegos):
    """Personas cuyo nombre es el del archivo con un apellido de mas o de menos.

    El archivo dice "Persona AJ" y el padron "Persona AJ
    Rossi", o al reves. Se pide que uno contenga al otro y que compartan al
    menos dos palabras: con una sola, "Persona AK" entraria en cualquier Juan.

    No se mira el orden, que es justo lo que no se sabe: "Persona AC" y
    "Persona AH" son el mismo juego de palabras.
    """
    s = frozenset(pal)
    if len(s) < 2:
        return []
    posibles = set()
    for t in s:
        posibles |= por_token.get(t, set())
    out = []
    for pid in posibles:
        k, p = juegos[pid]
        if len(s & k) >= 2 and (s <= k or k <= s):
            out.append(p)
    return out


def titulo(s):
    """'Persona AL' -> 'Persona AL'."""
    return " ".join(p if (len(p) > 1 and p[1:].islower()) else p.capitalize()
                    for p in (s or "").split())


def anotar_nombres(cli, nuevos, fol):
    """Anota los nombres que el colegio escribio y que no estan en el padron.

    NO CREA PERSONAS, Y ANTES SI.

    La regla la fijo Diego: "no deberia crear personas, si traer foto + el
    nombre del archivo porque ahi esta la data importante, pero para buscar ese
    nombre y apellido en el padron. Si no esta, solo guardo foto y el nombre del
    archivo queda como referencia, no como persona".

    Tiene razon y es la direccion correcta del flujo. "Persona AM.jpg" sirve
    para encontrar a "Persona AM - Quilmes - 2004" en el padron. Si ese Diego
    Medina no esta en ninguna planilla, lo que hay es una foto sin identificar y
    un nombre anotado; convertirlo en alumno es dejar que una carpeta le agregue
    filas a la lista de quienes existen.

    El argumento viejo -"el nombre lo escribio el colegio en su propio archivo,
    asi que vale lo mismo que el de una planilla"- no es falso, pero no alcanza:
    una planilla es una lista CERRADA de personas y una carpeta es un monton de
    archivos, algunos de los cuales son personas. Por esa misma puerta entraron
    "Nini Supermercado" y "Persona W" desde el otro cargador.

    LO QUE SE GANA GUARDANDO EN VEZ DE CREAR

    Esta tabla es la pregunta al reves. "Padron sin foto" dice a quien le falta
    una foto; nombres_sin_ficha dice que fotos nombran a alguien que no esta en
    ninguna planilla, que casi siempre significa que a una planilla le falta una
    fila. Crear la ficha tapaba ese agujero y hacia desaparecer la senal.
    """
    filas = []
    for k, apariciones in sorted(nuevos.items()):
        txt, f = apariciones[0]
        d = fol[f["folder_id"]]
        anios = [x[1].get("year") or fol[x[1]["folder_id"]].get("year")
                 for x in apariciones]
        anios = [int(y) for y in anios if y]
        filas.append({
            "norm_name": k,
            "texto": titulo(txt),
            "apariciones": len(apariciones),
            "campus": d.get("campus"),
            "anio_min": min(anios) if anios else None,
            "anio_max": max(anios) if anios else None,
            "ejemplo": f["id"],
        })
    for j in range(0, len(filas), 200):
        cli.upsert("nombres_sin_ficha", filas[j:j + 200], on_conflict="norm_name")
    return len(filas)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--ensayo", action="store_true")
    a = ap.parse_args()

    cli = sb.SB()
    fol = {f["id"]: f for f in cli.select(
        "folders", select="id,title,campus,is_mugshot,year,noise")
        if f.get("is_mugshot") and f.get("noise") is None}
    ya = {r["photo_id"] for r in cli.select("photo_people", select="photo_id")}
    rechazo = defaultdict(set)
    for r in cli.select("photo_people_rejected", select="photo_id,person_id"):
        rechazo[r["photo_id"]].add(int(r["person_id"]))

    ids = list(fol)
    fotos = []
    for i in range(0, len(ids), 40):
        fotos += cli.select("photos", select="id,folder_id,name,year,is_group",
                            folder_id="in.(%s)" % ",".join(ids[i:i + 40]))
    pend = [f for f in fotos if f["id"] not in ya and not f.get("is_group")]
    print("Retratos sin nombre: %d" % len(pend))

    # Un rotulo de sesion se repite; una persona, no.
    #
    # "PrepInd-1", "PrepInd-2"... son 327 archivos que despues de sacarles el
    # numero quedan todos en "PrepInd". Ninguna lista de rotulos alcanza -cada
    # año el fotografo inventa uno nuevo- pero la repeticion los delata sola:
    # en una carpeta de retratos cada chico sale una vez.
    repetido = Counter()
    for f in pend:
        t = texto_del_archivo(f.get("name"))
        if t:
            repetido[(f["folder_id"], clave(t))] += 1

    # EL STAFF TAMBIEN, Y NO ESTABA.
    #
    # Esto miraba solo kind == "student". No hay ningun comentario que lo
    # justifique y la regla que si esta escrita -arriba, la de Diego- es otra
    # cosa: habla de no CREAR personas desde un nombre de archivo, no de
    # ignorar a la mitad del padron a la hora de buscarlo.
    #
    # Lo levanto Diego mirando una carpeta de ex staff donde cada archivo se
    # llama "Persona AN.JPG": "por lo menos tendrian que venir ya
    # confirmados". Hoy no hay ninguna foto de staff esperando por esto -se
    # midio: 12 retratos sin dueno con nombre de persona en el archivo, los 12
    # de alumnos- pero la carpeta que esta cargando es justamente de staff, asi
    # que el agujero se abre en cuanto se indexe.
    padron = [p for p in cli.select(
        "people", select="id,display_name,campus,first_seen,last_seen,kind")
        if p.get("kind") in ("student", "staff")]
    por_palabras, por_apellido, por_token, juegos = indexar(padron)
    print("Padron: %d (%d alumnos, %d staff)\n"
          % (len(padron),
             sum(1 for p in padron if p.get("kind") == "student"),
             sum(1 for p in padron if p.get("kind") == "staff")))

    filas, motivos, ejemplos = [], Counter(), []
    ambiguos = []
    nuevos = {}
    homonimos = []
    for f in pend:
        txt = texto_del_archivo(f.get("name"))
        if not txt:
            motivos["el archivo no trae nombre"] += 1
            continue
        pal = clave(txt).split()
        if not pal:
            motivos["el archivo no trae nombre"] += 1
            continue
        if repetido[(f["folder_id"], " ".join(pal))] > 2:
            motivos["el archivo no trae nombre"] += 1
            continue

        # Cualquier orden: el juego de palabras no distingue "Persona AC"
        # de "Persona AH", que es justo lo que no hace falta decidir.
        cand = list(por_palabras.get(frozenset(pal), []))
        como = "nombre completo"
        if not cand:
            cand = por_subconjunto(pal, por_token, juegos)
            como = "con un apellido de mas o de menos"
        if not cand and len(pal) == 1:
            cand = list(por_apellido.get(pal[0], []))
            como = "apellido solo"
        if not cand:
            motivos["el nombre no esta en el padron"] += 1
            if len(ejemplos) < 12:
                ejemplos.append((f.get("name"), txt))
            if len(pal) >= 2:
                nuevos.setdefault(" ".join(pal), []).append((txt, f))
            continue

        sede = fol[f["folder_id"]].get("campus")
        anio = f.get("year") or fol[f["folder_id"]].get("year")
        de_la_sede = [p for p in cand if not sede or not p.get("campus")
                      or p["campus"] == sede]
        cand = de_la_sede
        # La ventana de anios NO se le aplica al staff, y esto ya costo caro
        # una vez. En una ficha de staff first_seen/last_seen no son una
        # cursada: son el anio de la carpeta donde se leyo el nombre. Los de
        # "Staff Portraits 2018" figuran 2018..2018, asi que la ventana los
        # descarta de cualquier otro anio.
        #
        # Es exactamente el bug que encontro Diego preguntando por que no se
        # reconocio a Persona P en la fiesta de 2025: 19 caras de
        # referencia y la ficha decia 2021..2021. Medido entonces, unos 120 de
        # los 546 del staff quedaban afuera de CADA anio del archivo.
        # faces_eventos.py ya esta arreglado; este script no lo estaba porque
        # no miraba staff en absoluto.
        #
        # Un alumno si tiene cursada de verdad y ahi la ventana es lo que evita
        # confundirlo con su propio padre.
        if anio:
            cand = [p for p in cand
                    if p.get("kind") == "staff"
                    or ((p.get("first_seen") is None or p["first_seen"] - MARGEN <= anio)
                        and (p.get("last_seen") is None or p["last_seen"] + MARGEN >= anio))]
        if not cand:
            # El nombre existe pero ninguno de esos pudo estar en la foto.
            #
            # En un colegio con historia el mismo nombre se repite entre padres
            # e hijos: hay un Persona AO que se fue en 2003 y otro que
            # cursa en 2025. No es la misma ficha escrita de dos maneras, es
            # otra persona, asi que se da de alta en vez de descartar.
            #
            # No se dan de alta solos, y no por prudencia sino porque no se
            # puede: people tiene el nombre normalizado como clave unica, asi
            # que el alta pisaria la ficha del egresado en vez de crear otra.
            # Y algunos son dudosos de verdad: "Persona AP Arminio"
            # figura hasta 2020 y la foto es de 2025, que tanto puede ser la
            # hija como la planilla mal cargada. Se listan para mirar.
            if de_la_sede and anio and len(pal) >= 2:
                homonimos.append((txt, anio, [(p["display_name"], p.get("first_seen"),
                                               p.get("last_seen")) for p in de_la_sede[:2]]))
            else:
                motivos["esta en el padron pero de otra sede o de otros años"] += 1
            continue
        if len({p["id"] for p in cand}) > 1:
            motivos["mas de una persona posible"] += 1
            if len(ambiguos) < 8:
                ambiguos.append((txt, [p["display_name"] for p in cand[:4]]))
            continue

        p = cand[0]
        if int(p["id"]) in rechazo.get(f["id"], ()):
            motivos["ya lo habian rechazado a mano"] += 1
            continue
        filas.append({"photo_id": f["id"], "person_id": int(p["id"]),
                      "votes": 0, "source": "filename"})
        motivos[como] += 1

    print("Identificables por el nombre del archivo: %d\n" % len(filas))
    for k, v in motivos.most_common():
        print("   %-46s %d" % (k + ":", v))

    if ambiguos:
        print("\nAmbiguos, no se tocan:")
        for t, ns in ambiguos:
            print("   %-26s -> %s" % (t[:26], ", ".join(ns)))
    if ejemplos:
        print("\nCon nombre pero fuera del padron (los primeros):")
        for n, t in ejemplos:
            print("   %-44s -> %s" % (n[:44], t))

    if homonimos:
        print()
        print("Mismo nombre que alguien que ya se fue: %d" % len(homonimos))
        print("  (hay que mirarlos: o es el hijo, o la planilla tiene mal el ultimo año)")
        for t, y, quienes in homonimos:
            print("   %-30s foto de %s   padron: %s" % (t[:30], y, quienes))
    print()
    print("Nombres escritos por el colegio sin ficha en el padron: %d" % len(nuevos))
    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar para escribir.")
        return
    # Los nombres que no estan en el padron se ANOTAN, no se dan de alta.
    # Ya no hay un --crear: no es una opcion que convenga tener a mano.
    if nuevos:
        n = anotar_nombres(cli, nuevos, fol)
        print("Anotados %d nombres leidos que no estan en ninguna planilla." % n)
        print("Las fotos quedan sin identificar, que es lo que son.")
    for j in range(0, len(filas), 200):
        cli.upsert("photo_people", filas[j:j + 200], on_conflict="photo_id,person_id")
    print("\nEscritas %d identificaciones con source = filename" % len(filas))
    print("Para revertir esta tanda:")
    print("  delete from photo_people where source = 'filename';")


if __name__ == "__main__":
    main()
