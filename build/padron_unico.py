# -*- coding: utf-8 -*-
"""Lee el Padron Completo -el sheet unico- y dice que cambiaria en el padron.

    python padron_unico.py                # mide y no escribe (ensayo)
    python padron_unico.py --hoja base    # solo una pestaña

POR QUE (23/9/2026)

El padron de la app sale hoy de tres archivos distintos: el dashboard de
Marketing (alumnos, iSAMS), el Alumni Information Form (exalumnos) y el STAFF
report. Diego armo un cuarto archivo que los unifica -"Padron Completo |
Biblioteca de Fotos"- y pidio que pase a mandar ese.

Antes de mover 7.700 fichas hay que saber que cambia. Esto no escribe nada:
compara lo que dice el sheet contra lo que hay en la base y cuenta altas,
nombres distintos y sedes distintas. Sobre esos numeros se decide.

LAS TRES PESTAÑAS

  base              staff y ex staff, ya unificados por Diego. 1.765 filas de
                    tres origenes que colapsan por su clave: apellido + las
                    tres primeras letras del nombre.
  Alumnos actuales  el export de iSAMS, mismas columnas que la fuente de hoy.
  Alumni            exalumnos: nombre, apellido, sede y camada.

LO QUE LA APP CORRIGIO NO SE PISA

Diego: si el sheet dice una cosa y alguien ya la corrigio en la app, gana la
app. Las correcciones estan en padron_cambios (sql/padron_cambios.sql): si hay
una para esa persona y ese campo, y el valor de hoy es el que se corrigio, el
sheet no lo toca y se cuenta aparte.
"""
import argparse
import os
import sys
import unicodedata
from collections import Counter, defaultdict

import sb
from padron_sheet import Hojas, SA_PATH, SHEET

HERE = os.path.dirname(os.path.abspath(__file__))

# Cada hoja escribe la sede a su manera: la de staff en dos idiomas, la de
# alumni con "Site" pegado. Lo que no esta aca no se toca.
SEDES = {"norte / north": "North", "north": "North", "norte": "North",
         "north site": "North", "quilmes": "Quilmes", "quilmes site": "Quilmes",
         "ambas / both": None, "ambas": None, "both": None, "": None}


def clave(s):
    """minusculas, sin acentos, solo letras y numeros (igual que el padron)."""
    t = unicodedata.normalize("NFD", s or "")
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return " ".join("".join(c if c.isalnum() else " " for c in t).split())


def titulo(t):
    """Sube la primera letra de las palabras que vienen todas en minuscula o
    todas en mayuscula. "GONZÁLEZ FREA" -> "González Frea", y no toca
    "McGregor" ni "de la Torre", que alguien escribio a proposito."""
    out = []
    for p in (t or "").split():
        if p.isupper() or p.islower():
            out.append(p[:1].upper() + p[1:].lower())
        else:
            out.append(p)
    return " ".join(out)


def tiene_acentos(t):
    return any(unicodedata.combining(c) for c in unicodedata.normalize("NFD", t or ""))


def nombre_que_gana(app, sheet, por_alias=False):
    """Entre dos escrituras del mismo nombre, la que tiene acentos.

    Todas las diferencias de nombre son de acentos o mayusculas: la comparacion
    solo empareja cuando el nombre normalizado coincide. El sheet a veces los
    agrega ("Persona AD" -> "Persona AD" con tilde) y a veces los
    saca ("Persona AE" pierde las dos). Gana el que los tiene, venga
    de donde venga; si empatan, queda el de la app, que es lo que alguien vio.
    Decision de Diego, 23/9/2026."""
    # SI EL MATCH VINO POR UN ALIAS, EL NOMBRE NO SE TOCA (23/9/2026)
    #
    # Un alias existe porque alguien -o el propio sistema al contestar una
    # dudosa- ya dijo "esta ficha tambien se llama asi". Eso no lo convierte en
    # el nombre bueno: el sheet escribe "Persona AF" y la ficha se
    # llama "Persona AG". Al resolver esa dudosa, la regla vieja renombro la
    # ficha con el error de tipeo del sheet.
    #
    # Asi que por alias se completa (house, camada, sede) pero no se renombra.
    # El nombre solo cambia entre dos escrituras del MISMO nombre, y ahi manda
    # la que tiene acentos.
    if por_alias or clave(app) != clave(sheet):
        return app
    if tiene_acentos(sheet) and not tiene_acentos(app):
        return sheet
    if tiene_acentos(app) and not tiene_acentos(sheet):
        return app
    return app


def col(fila, i, nombre):
    # el encabezado se indexa normalizado, asi que la busqueda tambien: si no,
    # "First Name (s)" no se encuentra nunca y la hoja sale vacia
    k = i.get(clave(nombre))
    return (fila[k].strip() if k is not None and len(fila) > k and fila[k] else "")


def leer(h, hoja, hasta="Z"):
    import urllib.parse
    r = h._pedir("GET", SHEET + "/values/"
                 + urllib.parse.quote("%s!A1:%s20000" % (hoja, hasta)))
    v = r.get("values", [])
    if not v:
        return [], {}
    return v[1:], {clave(c): n for n, c in enumerate(v[0])}


# ---------------------------------------------------------------- pestañas

def staff(h):
    """La hoja base: una fila por origen, una persona por clave."""
    filas, i = leer(h, "base")
    gente = {}
    for f in filas:
        nom, ape = col(f, i, "nombre"), col(f, i, "apellido")
        if not nom or not ape:
            continue
        # la clave de Diego: apellido + tres letras del nombre. Es lo que hace
        # que las tres hojas de origen colapsen en una persona.
        k = clave(ape) + "|" + clave(nom)[:3]
        p = gente.setdefault(k, {"nombre": titulo(nom + " " + ape), "mail": "",
                                 "foto": "", "sede": None, "estado": "",
                                 "origenes": set()})
        p["origenes"].add(col(f, i, "origen"))
        p["mail"] = p["mail"] or col(f, i, "email institucional").lower() \
            or col(f, i, "email personal").lower()
        p["foto"] = p["foto"] or col(f, i, "foto")
        p["sede"] = p["sede"] or SEDES.get(col(f, i, "sede").lower())
        p["estado"] = p["estado"] or col(f, i, "status")
    return gente


def casa_actual(valor):
    """La house de hoy, sin la sede adelante: "N Stevenson" -> "Stevenson".

    iSAMS la escribe con la letra de la sede pegada (N / Q). En el padron la
    house es solo el nombre, igual que en las 1.400 de exalumnos. Un valor sin
    ese prefijo se deja como viene."""
    v = " ".join((valor or "").split())
    partes = v.split(" ", 1)
    if len(partes) == 2 and partes[0] in ("N", "Q"):
        return partes[1]
    return v


def alumnos(h):
    """Alumnos actuales: el export de iSAMS.

    HOUSE (29/9/2026). La columna AcademicHouse llega desde el 16/9 y no se
    leia, asi que ningun alumno actual tenia house: la ficha de un chico de
    Prep no mostraba la suya aunque el sheet la trajera. Decision de Diego: se
    carga una sola house, la de hoy, y solo donde la ficha la tiene vacia. El
    Kinder de Quilmes viene sin house en iSAMS y queda sin house.
    """
    filas, i = leer(h, "Alumnos actuales")
    gente = {}
    for f in filas:
        nom, ape = col(f, i, "forename"), col(f, i, "surname")
        if not nom or not ape:
            continue
        yc = col(f, i, "year code")
        gente[clave(nom + " " + ape)] = {
            "nombre": titulo(nom + " " + ape),
            "sede": {"Q": "Quilmes", "N": "North"}.get(yc[:1]),
            "year_code": yc or None,
            "house": casa_actual(col(f, i, "academichouse")) or None}
    return gente


def alumni(h):
    """Exalumnos: nombre, sede, CAMADA y house.

    De un exalumno no importan los años de cursada sino la camada de egreso
    (Diego, 23/9/2026), que es la columna E. La F trae la house, y quien paso
    por las dos etapas trae las dos: "Stevenson, Farran". En el padron se
    guardan con " / ", que es como ya estan las 1.400 que hay.
    """
    filas, i = leer(h, "Alumni")
    gente = {}
    for f in filas:
        nom, ape = col(f, i, "first name (s)"), col(f, i, "surname")
        if not nom or not ape:
            continue
        cam = col(f, i, "camada")
        casa = " / ".join(x.strip() for x in col(f, i, "house").split(",") if x.strip())
        gente[clave(nom + " " + ape)] = {
            "nombre": titulo(nom + " " + ape),
            "sede": SEDES.get(col(f, i, "campus").lower())
                    or (col(f, i, "campus") or None),
            "camada": int(cam) if cam.isdigit() else None,
            "house": casa or None}
    return gente


# ---------------------------------------------------------------- comparar

def padron(cli):
    """El padron de hoy, indexado por nombre y por alias."""
    por_clave = defaultdict(list)
    por_alias = defaultdict(list)
    gente = {}
    # house va en el select: sin el, "lo que ya esta" se leia como vacio y la
    # regla de completar terminaba pisando (23/9/2026)
    for p in cli.select("people", select="id,display_name,norm_name,kind,campus,"
                                         "cohort,house,aliases,ex_staff,year_code"):
        if p.get("kind") == "noise":
            continue
        gente[p["id"]] = p
        for k in {clave(p.get("display_name"))} | {clave(a) for a in (p.get("aliases") or [])}:
            if k:
                por_clave[k].append(p)
        for a in (p.get("aliases") or []):
            if clave(a):
                por_alias[clave(a)].append(p)
    return gente, por_clave, por_alias


def corregidas(cli):
    """(person_id, campo) -> el valor que alguien dejo a mano en la app."""
    out = {}
    for c in cli.select("padron_cambios", select="person_id,campo,despues,creado_at",
                        order="creado_at"):
        out[(int(c["person_id"]), c["campo"])] = c.get("despues")
    return out


def parecidas(altas, por_clave):
    """De las altas, cuales se parecen a alguien que ya esta.

    Es la pregunta que decide si el sheet se puede aplicar: "Agustina Cabllero"
    y "Persona AH" no son gente nueva, son Caballero y Canelotto mal
    tipeados. Darlos de alta seria crear el duplicado que acabamos de sacar.

    Se compara solo contra los que comparten alguna palabra -si no, son 1.600
    por 7.700 comparaciones- y se usa difflib, que alcanza para ver un error de
    tipeo."""
    import difflib
    por_pal = defaultdict(set)
    nombres_db = {}
    for k, ps in por_clave.items():
        for p in ps:
            nombres_db[p["id"]] = p["display_name"]
            for w in k.split():
                if len(w) > 2:
                    por_pal[w].add(p["id"])
    out = []
    for s in altas:
        k = clave(s["nombre"])
        cand = set()
        for w in k.split():
            cand |= por_pal.get(w, set())
        pal = set(k.split())
        mejor, puntaje = None, 0.0
        for pid in cand:
            k2 = clave(nombres_db[pid])
            r = difflib.SequenceMatcher(None, k, k2).ratio()
            # UNO CONTENIDO EN EL OTRO (23/9/2026)
            #
            # difflib mira las letras, asi que "Persona AI Echegaray Santomil"
            # y "Persona AJ" dan 0,68 y pasaban como gente nueva: el
            # sheet trae el nombre legal completo y la app el corto. Entraron
            # 140 duplicados asi antes de que esto estuviera. Dos palabras de
            # minimo para que "Persona U" no se coma a cualquier Juan.
            pal2 = set(k2.split())
            if len(pal) >= 2 and len(pal2) >= 2 and (pal <= pal2 or pal2 <= pal):
                r = max(r, 0.9)
            if r > puntaje:
                mejor, puntaje = nombres_db[pid], r
        if mejor and puntaje >= 0.82:
            out.append((s["nombre"], mejor, round(puntaje, 2)))
    return out


def por_un_alias(nombre, por_alias):
    """La ficha cuyo ALIAS es casi este nombre.

    POR QUE (23/9/2026). Diego corrigio su nombre en el sheet -"Diego Alejadro
    Medina" -> "Persona AK"- y la lectura siguiente no lo reconocio:
    el alias guardaba la version con el error, asi que el nombre corregido
    parecia una persona nueva y volvia a la cola de dudosas. Preguntar dos veces
    lo mismo, y encima por una correccion, es al reves de lo que tiene que
    pasar.

    Un alias no es un nombre cualquiera: esta ahi porque alguien contesto "es la
    misma". Asi que una variante de tipeo de un alias es esa misma ficha. El
    corte es alto -0,92- y tiene que haber una sola candidata.
    """
    import difflib
    k = clave(nombre)
    mejor, puntaje = None, 0.0
    empate = False
    for ka, ps in por_alias.items():
        r = difflib.SequenceMatcher(None, k, ka).ratio()
        if r < 0.92:
            continue
        for p in ps:
            if mejor and p["id"] != mejor["id"] and abs(r - puntaje) < 0.01:
                empate = True
            if r > puntaje:
                mejor, puntaje, empate = p, r, False
    return None if (empate or not mejor) else mejor


def comparar(nombre_hoja, gente, por_clave, corr, campo_sede=True, por_alias=None):
    altas, nombres, sedes, respetadas, ambiguas, completar = [], [], [], [], [], []
    nuevos_alias, distintas = [], []
    for k, s in gente.items():
        # el staff viene con su propia clave (apellido|nombre): para buscar en
        # el padron se usa el nombre completo
        cand = por_clave.get(clave(s["nombre"]), [])
        if not cand and por_alias:
            # el nombre corregido de alguien que ya tiene alias
            p_alias = por_un_alias(s["nombre"], por_alias)
            if p_alias:
                cand = [p_alias]
                nuevos_alias.append((p_alias, s["nombre"]))
        if not cand:
            altas.append(s)
            continue
        if len(cand) > 1:
            ambiguas.append((s, cand))
            continue
        p = cand[0]
        # por alias: el nombre del sheet no es el de la ficha ni una variante
        # de acentos, es otra escritura que alguien acepto como suya
        # OJO: variable aparte. Antes se llamaba igual que el indice de alias
        # que recibe la funcion y lo pisaba en la primera vuelta del for, asi
        # que el reconocimiento por alias solo funcionaba para la primera
        # persona de la hoja (23/9/2026).
        via_alias = clave(p.get("display_name")) != clave(s["nombre"])
        if s["nombre"] != p["display_name"]:
            # "hay correccion Y es la que esta puesta": sin el primer chequeo,
            # dos vacios se leen como "corregido a mano"
            if (p["id"], "nombre") in corr and corr[(p["id"], "nombre")] == p["display_name"]:
                respetadas.append((p, "nombre", p["display_name"], s["nombre"]))
            elif nombre_que_gana(p["display_name"], s["nombre"], via_alias) != p["display_name"]:
                nombres.append((p, p["display_name"], s["nombre"]))
        # LO QUE FALTA SE COMPLETA, LO QUE ESTA NO SE PISA
        #
        # La camada de egreso y la house de un exalumno son el dato que el
        # sheet trae y el padron no tiene (5.421 fichas sin house). Se llenan
        # solo si estan vacias: una house puesta a mano en la app no se toca,
        # igual que la sede.
        #
        # Lo que alguien toco a mano en la app tampoco se completa, aunque lo
        # haya dejado vacio a proposito (padron_cambios, campo "house" o
        # "camada").
        faltan = {}
        for campo, valor, corr_campo in (("house", s.get("house"), "house"),
                                         ("cohort", s.get("camada"), "camada")):
            if (p["id"], corr_campo) in corr:
                continue
            if campo == "cohort" and p.get("year_code"):
                # un alumno que hoy cursa no toma la camada de un exalumno con
                # su mismo nombre: la suya sale de su nivel (camada_ultimo_ano)
                continue
            if valor and not p.get(campo):
                faltan[campo] = valor
            elif valor and p.get(campo) and campo == "house" \
                    and valor not in p["house"].split(" / "):
                # la ficha ya tiene otra: no se pisa, se lista
                distintas.append((p, p["house"], valor))
        if faltan:
            completar.append((p, faltan))
        if campo_sede and s.get("sede") and s["sede"] != p.get("campus"):
            if (p["id"], "sede") in corr and corr[(p["id"], "sede")] == p.get("campus"):
                respetadas.append((p, "sede", p.get("campus"), s["sede"]))
            else:
                sedes.append((p, p.get("campus"), s["sede"]))
    print("\n== %s: %d personas en el sheet" % (nombre_hoja, len(gente)))
    print("   altas nuevas:          %d" % len(altas))
    print("   nombres distintos:     %d" % len(nombres))
    print("   sedes distintas:       %d" % len(sedes))
    print("   corregidas en la app:  %d  (el sheet no las pisa)" % len(respetadas))
    print("   ambiguas (2 fichas):   %d" % len(ambiguas))
    if completar:
        print("   fichas a completar:    %d  (%s)"
              % (len(completar), dict(Counter(c for _, f in completar for c in f))))
    if distintas:
        # la house de la ficha no es la del sheet: no se pisa, se muestra
        print("   house distinta a la del sheet (no se toca): %d" % len(distintas))
        for p, ahora, sheet in distintas:
            print("      house:  %-6s %-32s %s | sheet: %s"
                  % (p["id"], p["display_name"][:32], ahora, sheet))
    par = parecidas(altas, por_clave)
    print("   de esas altas, %d se parecen a alguien que YA esta (serian duplicados)"
          % len(par))
    for a, b, r in par[:10]:
        print("      parecida: %-32s ~ %-32s %.2f" % (a[:32], b[:32], r))
    # todas, no ocho: un cruce por nombre falla en silencio y esta es la lista
    # de los que no cruzaron
    for s in altas:
        print("      no cruza: %s%s" % (s["nombre"],
                                        "  (%s)" % s["year_code"] if s.get("year_code") else ""))
    for p, viejo, nuevo in nombres[:8]:
        print("      nombre: %-32s -> %s" % (viejo[:32], nuevo))
    for p, viejo, nuevo in sedes[:8]:
        print("      sede:   %-32s %s -> %s" % (p["display_name"][:32], viejo or "(vacia)", nuevo))
    for p, campo, valor, _ in respetadas[:5]:
        print("      queda:  %-32s %s = %s (corregido en la app)"
              % (p["display_name"][:32], campo, valor))
    if nuevos_alias:
        print("   nombres corregidos en el sheet que ya tenian alias: %d"
              % len(nuevos_alias))
    return {"altas": altas, "nombres": nombres, "sedes": sedes, "completar": completar,
            "respetadas": respetadas, "ambiguas": ambiguas, "por_clave": por_clave,
            "alias": nuevos_alias}


def aplicar(cli, res, kind, fuente, solo_completar=False):
    """Escribe lo medido: nombres, sedes y las altas CLARAS.

    Las altas que se parecen a alguien que ya esta no se crean (decision de
    Diego, 23/9/2026): "Agustina Cabllero" es Caballero mal tipeada y darla de
    alta seria el duplicado que se acaba de sacar del padron. Quedan listadas.

    Nada de esto se anota en padron_cambios: esa tabla es lo que la app le
    manda al sheet, y anotar aca lo que vino del sheet lo devolveria en
    circulos.
    """
    n_comp = 0
    for p, faltan in res.get("completar", []):
        # el filtro "is.null" es el "sin pisar" del lado de la base: si entre
        # la lectura y la escritura alguien la cargo, esta escritura no la toca
        cli.update("people", faltan, id="eq.%d" % p["id"],
                   **{c: "is.null" for c in faltan})
        n_comp += 1
    if solo_completar:
        # --solo-completar: llena lo vacio y nada mas. Para cargar un campo
        # nuevo del sheet sin arrastrar en la misma escritura nombres, sedes
        # ni altas que nadie reviso.
        print("   escrito: %d completadas (solo completar)" % n_comp)
        return set()
    n_nom = n_sede = n_alta = 0
    for p, viejo, nuevo in res["nombres"]:
        cli.update("people", {"display_name": nuevo}, id="eq.%d" % p["id"])
        n_nom += 1
    for p, viejo, nuevo in res["sedes"]:
        cli.update("people", {"campus": nuevo}, id="eq.%d" % p["id"])
        n_sede += 1
    # la escritura nueva del sheet queda como alias: la proxima lectura la
    # encuentra derecho, sin volver a pasar por el parecido
    for p, nom in res.get("alias", []):
        actuales = [a for a in (p.get("aliases") or []) if a]
        if nom in actuales or nom == p.get("display_name"):
            continue
        cli.update("people", {"aliases": actuales + [nom]}, id="eq.%d" % p["id"])

    pares = parecidas(res["altas"], res["por_clave"])
    dudosas = {a for a, _, _ in pares}
    # Quedan en padron_dudosas y no en un archivo suelto: es una pregunta
    # pendiente -"¿es la misma persona o son dos?"- y las preguntas del sistema
    # viven en la base, para poder contestarlas desde la app.
    if pares:
        ids = {b: None for _, b, _ in pares}
        for k, ps in res["por_clave"].items():
            for x in ps:
                if x["display_name"] in ids and ids[x["display_name"]] is None:
                    ids[x["display_name"]] = x["id"]
        filas = [{"nombre": a, "hoja": fuente, "parecida_id": ids.get(b),
                  "parecida_nombre": b, "puntaje": r} for a, b, r in pares]
        for i in range(0, len(filas), 200):
            cli.upsert("padron_dudosas", filas[i:i + 200], on_conflict="nombre")
    nuevas = []
    for x in res["altas"]:
        if x["nombre"] in dudosas:
            continue
        # todas las filas con las MISMAS claves: PostgREST rechaza un lote
        # donde una fila trae mail y otra no ("All object keys must match")
        nuevas.append({"display_name": x["nombre"], "norm_name": clave(x["nombre"]),
                       "kind": kind, "source": fuente, "campus": x.get("sede"),
                       "cohort": x.get("camada"), "year_code": x.get("year_code"),
                       "house": x.get("house"),
                       "email": (x.get("mail") or None),
                       "photo_url": (x.get("foto") or None)})

    # el mail tiene indice unico en people: si dos filas traen el mismo, entra
    # con mail una sola
    ya = {p.get("email", "").lower() for p in cli.select("people", select="email")
          if p.get("email")}
    vistos = set()
    for f in nuevas:
        m = (f.get("email") or "").lower()
        if not m:
            continue
        if m in vistos or m in ya:
            f["email"] = None
        else:
            vistos.add(m)

    for i in range(0, len(nuevas), 200):
        cli.upsert("people", nuevas[i:i + 200], on_conflict="norm_name,kind")
        n_alta += len(nuevas[i:i + 200])
    print("   escrito: %d nombres, %d sedes, %d completadas, %d altas "
          "(%d dudosas sin crear)" % (n_nom, n_sede, n_comp, n_alta, len(dudosas)))
    return dudosas


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hoja", default="", help="base | alumnos | alumni")
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--solo-completar", action="store_true",
                    help="con --aplicar: solo llena campos vacios (house, camada)")
    a = ap.parse_args()

    if not os.path.exists(SA_PATH):
        sys.exit("Falta el service account en %s" % SA_PATH)
    h = Hojas(SA_PATH)
    cli = sb.SB()
    gente_db, por_clave, por_alias = padron(cli)
    corr = corregidas(cli)
    print("Padron de hoy: %d fichas vivas.  Correcciones a mano: %d"
          % (len(gente_db), len(corr)))

    quiere = (a.hoja or "").lower()
    dudosas = []
    if quiere in ("", "base", "staff"):
        st = staff(h)
        print("\n(base: %d personas; por estado: %s)"
              % (len(st), dict(Counter(x["estado"] or "sin estado" for x in st.values()))))
        res = comparar("Staff (hoja base)", st, por_clave, corr, por_alias=por_alias)
        if a.aplicar:
            dudosas += sorted(aplicar(cli, res, "staff", "staff_sheet", a.solo_completar))
            # El Status de la hoja dice quien sigue y quien se fue, que hasta
            # hoy se adivinaba por ausencia en la planilla. ex_staff_forzado
            # -lo que Diego marco a mano- le gana igual.
            n_ex = 0
            for k, x in (st.items() if not a.solo_completar else []):
                if x["estado"] not in ("Staff actual", "Ex staff"):
                    continue
                cand = por_clave.get(clave(x["nombre"]), [])
                if len(cand) != 1:
                    continue
                p = cand[0]
                quiere_ex = x["estado"] == "Ex staff"
                if p.get("ex_staff") == quiere_ex:
                    continue
                cli.update("people", {"ex_staff": quiere_ex},
                           id="eq.%d" % p["id"], ex_staff_forzado="not.is.true")
                n_ex += 1
            print("   ex staff segun el Status de la hoja: %d fichas" % n_ex)
    if quiere in ("", "alumnos"):
        res = comparar("Alumnos actuales", alumnos(h), por_clave, corr, por_alias=por_alias)
        if a.aplicar:
            dudosas += sorted(aplicar(cli, res, "student", "planilla_alumnos", a.solo_completar))
    if quiere in ("", "alumni"):
        res = comparar("Alumni", alumni(h), por_clave, corr, por_alias=por_alias)
        if a.aplicar:
            dudosas += sorted(aplicar(cli, res, "student", "planilla_alumni", a.solo_completar))

    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar.")
        return
    if a.solo_completar:
        # no se miraron las dudosas: el archivo de la corrida completa queda
        return
    ruta = os.path.join(HERE, "_padron_unico_dudosas.txt")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(dudosas) + "\n")
    print("\nListo. Las %d dudosas quedaron en %s" % (len(dudosas), ruta))


if __name__ == "__main__":
    main()
