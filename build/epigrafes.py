# -*- coding: utf-8 -*-
"""Carga a photo_people lo que dicen los epigrafes que deja quien digitaliza en las carpetas.

QUE ES UN EPIGRAFE. Al digitalizar, quien digitaliza sube a la carpeta del archivo un .txt
que se llama igual que la foto (misma carpeta, mismo nombre base, otra
extension) con lo que la foto trae escrito atras: el curso en la primera linea
y despues bloques "Fila de arriba / del medio / de abajo (De izquierda a
derecha)." con los nombres separados por guion normal o largo (U+2013), con
Mrs./Ms./Mr. adelante cuando es staff. Desde el 3/11/2026 tambien el formato
sin encabezados: un renglon por fila de la foto (el primero, la de arriba),
nombres completos de izquierda a derecha separados por coma (ver nombres_de).
Los .txt los levanta sync_drive.py a
carpeta_notas; este script resuelve esos nombres contra el padron y escribe las
identificaciones con source = 'epigrafe'.

EL JUEZ ES QUIEN CARGA. Lo escrito atras de la foto lo puso una persona: se
carga. Lo que este script NO hace es adivinar: un nombre que da dos fichas o
ninguna no se carga ni crea ficha, va a padron_dudosas con su candidata, que es
donde el padron deja las preguntas.

REGLAS DE RESOLUCION (medidas el 26/10/2026 contra el padron vivo y acordadas
con el responsable del proyecto sobre las dos notas que hay hoy):

  1. El apellido del .txt tiene que ser SUFIJO del display_name comparado sin
     tildes, sin signos y SIN ESPACIOS. Sin espacios porque los apellidos de
     dos palabras del padron se escriben de las dos maneras, igual que el
     apostrofo (ver Trampas - padron).
  2. LA INICIAL DEL NOMBRE TIENE QUE COINCIDIR. Sin esto, una inicial de un
     apellido comun cae en el primer homonimo que aparece: medido el 26/10, un
     nombre de la nota resolvia en una persona de otra camada.
  3. Un nombre con Mrs./Ms./Mr. se busca en el staff (kind que no sea student
     ni alumni); uno sin titulo, en la camada que dice el nombre del archivo
     ("... - camada 2004.txt").
  4. SI EL ARCHIVO NO DICE LA CAMADA, LA DICE LA FOTO (27/10/2026). Una foto de
     curso es de una sola camada, y los nombres que SI resolvieron solos la
     delatan. Asi que se resuelve dos veces: la primera saca la camada de la
     foto (la moda de los alumnos que resolvieron en una sola ficha), la
     segunda la usa para desempatar. Es un desempate, no un descarte:
       - una sola candidata se carga igual, aunque su camada no sea la de la
         foto (en la nota de 6G hay dos asi, y son correctas);
       - con dos o mas candidatas manda la camada de la foto, y si ninguna es
         de esa camada NO se propone ninguna: va a dudosas sin candidata.
     Medido el 26/10 sin esta regla: un apellido con inicial, en una foto
     donde 24 de 26 eran camada 2004, proponia dos fichas de 2011 y de 2006,
     las dos falsas.
  5. Las fichas kind = 'noise' no son personas: nunca son candidatas.

  6. UNA DUDOSA PENDIENTE NO SE CARGA (3/11/2026). Si el nombre ya esta en
     padron_dudosas (hoja 'epigrafe', sin resolver) y su candidata es la
     misma ficha que resuelve hoy -o no tenia candidata-, no se carga: alguien
     tiene que contestar esa pregunta antes. Asi lo que se dejo afuera a mano
     (--dejar-afuera) no entra solo en la corrida siguiente.

Lo que resuelve en una sola ficha se carga; lo demas se informa y va a dudosas.

SE AGREGA, NO SE PISA (3/11/2026). Un par (foto, persona) que ya esta en
photo_people se deja como esta: si lo cargo una persona (game, manual), es el
juez; si lo cargo epigrafe_caras.py ('epigrafe_cara'), ya tiene su cara. Antes
el upsert hacia merge y le cambiaba el source a 'epigrafe' a cualquiera de
ellos. Ahora se inserta con ignore-duplicates y el ensayo dice cuantos ya
estaban y con que source.

Revertir la carga entera: delete from photo_people where source = 'epigrafe'.
Cada corrida deja su respaldo en build/_epigrafes_<fecha>.txt.

Uso:
    python epigrafes.py                       # todas las notas de carpeta_notas
    python epigrafes.py --carpeta <drive_id>  # solo esa carpeta
    python epigrafes.py --carpeta <id> --drive   # sin pasar por carpeta_notas
    python epigrafes.py --pulso               # solo notas nuevas o cambiadas
    ... --aplicar                             # escribe (ensayo si no esta)

--pulso: las notas cuyo md5 no esta en el cursor (corridas, tarea
'epigrafes', etiqueta 'nombres'; la etiqueta 'caras' es de epigrafe_caras.py).
Una nota se anota cuando todos sus nombres quedaron resueltos y escritos (o
frenados por un rechazo humano). La que tiene nombres en dudosas, sin foto, o
un par que no entro sin explicacion NO se anota: se vuelve a mirar en la
corrida siguiente (no baja nada de Drive, solo resuelve nombres).
"""
import argparse
import datetime as _dt
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sb  # noqa: E402

TITULO = re.compile(r"(?i)^(mrs?|ms|miss|dr|prof)\.?\s+")
INICIAL = re.compile(r"^[A-Za-zÁÉÍÓÚÑÜ]\.?$")
CAMADA_EN_NOMBRE = re.compile(r"(?i)camada\s*(\d{4})")
# Las filas del epigrafe: "Fila de arriba (De izquierda a derecha)."
FILA = re.compile(r"(?i)^fila\s+(de\s+arriba|del\s+medio|de\s+abajo)\b")
# EL CORTE DE NOMBRES ES DEL SCRIPT, NO DE LA BASE (26/10/2026). El parser de
# la base -carpeta_nota_nombres()- es el que tiene que mandar, pero hoy saca el
# titulo (Mrs./Ms.), que es el unico dato que dice si el nombre es del staff o
# de la camada, y deja el guion pegado cuando el renglon termina en guion
# largo: medido sobre la nota de 6G, 24 nombres y 2 con el guion adentro.
# Mientras base no devuelva el titulo, el corte se hace aca con el mismo
# criterio (guion normal y largo) y se compara contra el de la base.
CORTE = re.compile(r"\s*[-–—]\s+|\s+[-–—]\s*")
SOURCE = "epigrafe"
CURSOR = {"tarea": "epigrafes", "etiqueta": "nombres"}


def sin_tildes(s):
    s = unicodedata.normalize("NFD", (s or "").lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]", " ", s)


def pegado(s):
    return re.sub(r"\s+", "", sin_tildes(s))


# Que paso del pulso frena una escritura a mano esta en sb.pasos_del_pulso().


# ---------------------------------------------------------------- las notas

def notas_de_la_base(cli, carpeta=None):
    p = {"select": "id,folder_id,photo_id,drive_id,nombre,texto,ignorada",
         "ignorada": "is.false", "texto": "not.is.null"}
    if carpeta:
        p["folder_id"] = "eq." + carpeta
    return cli.select("carpeta_notas", **p) or []


def notas_de_drive(carpeta):
    """Plan B: bajar los .txt de una carpeta sin pasar por carpeta_notas (para
    poder resolver antes de que el sync los haya levantado)."""
    import requests
    from sync_drive import Drive
    d = Drive(os.path.join(HERE, "service-account.json"))
    arch = d.get("files", q="'%s' in parents and trashed=false" % carpeta,
                 fields="files(id,name,mimeType)", pageSize=1000,
                 supportsAllDrives="true", includeItemsFromAllDrives="true")["files"]
    out = []
    for f in arch:
        if not (f.get("name") or "").lower().endswith(".txt"):
            continue
        raw = requests.get(
            "https://www.googleapis.com/drive/v3/files/%s?alt=media&supportsAllDrives=true" % f["id"],
            headers={"Authorization": "Bearer " + d._token()}, timeout=60).content
        out.append({"id": None, "folder_id": carpeta, "photo_id": None,
                    "drive_id": f["id"], "nombre": f["name"],
                    "texto": raw.decode("utf-8", "replace"), "ignorada": False})
    return out


def foto_de(cli, nota):
    """La foto que describe la nota. La atadura la hace la base; si la nota
    todavia no la tiene (o venimos de Drive), se busca igual: misma carpeta,
    mismo nombre base."""
    if nota.get("photo_id"):
        return nota["photo_id"]
    base = re.sub(r"\.[^.]*$", "", nota["nombre"] or "").lower()
    fotos = cli.select("photos", select="id,name,is_primary",
                       folder_id="eq." + nota["folder_id"]) or []
    iguales = [f for f in fotos
               if re.sub(r"\.[^.]*$", "", (f["name"] or "")).lower() == base]
    iguales.sort(key=lambda f: (not f.get("is_primary"), f["id"]))
    return iguales[0]["id"] if iguales else None


def _partir_por_punto(s):
    """Corta en ". " salvo cuando el punto es de un titulo (Mr.) o de una
    inicial (M.): asi "Nombre Apellido. Otro Nombre" da dos nombres y
    "Mr. Nombre Apellido" o "J. M. Apellido" queda entero."""
    partes = re.split(r"\.\s+", s)
    out = []
    for p in partes:
        if out:
            ult = out[-1].split()[-1] if out[-1].split() else ""
            if TITULO.match(ult + " x") or INICIAL.match(ult):
                out[-1] = out[-1] + ". " + p
                continue
        out.append(p)
    return out


def _es_nombre(crudo):
    """Un pedazo que no puede ser un nombre: un link, un numero de curso
    ("6G"), algo sin letras. No se cuenta ni se busca."""
    if "://" in crudo or "www." in crudo.lower():
        return False
    if re.search(r"\d", crudo):
        return False
    return len(crudo) >= 3 and any(c.isalpha() for c in crudo)


def nombres_de(texto):
    """Los nombres del epigrafe, EN ORDEN: fila de arriba hacia abajo y, dentro
    de cada fila, de izquierda a derecha. Motor usa (fila, posicion) para unir
    cada nombre con su cara, asi que el orden de la salida es parte del
    contrato de esta funcion.

    Devuelve [(orden, fila, pos, crudo, es_staff, inicial, apellido, pila)]:
      orden     1..N sobre todo el epigrafe
      fila      "Fila de arriba" / "Fila del medio" / "Fila de abajo" en el
                formato con encabezados; "Fila 1", "Fila 2"... (1 = la de
                arriba) en el formato sin encabezados
      pos       1..M dentro de la fila, de izquierda a derecha
      inicial   la del primer nombre (de la inicial escrita o del nombre de pila)
      apellido  lo que se compara como sufijo del display_name
      pila      el primer nombre completo cuando el epigrafe lo trae ("" si
                trae iniciales): desempata homonimos con la misma inicial

    DOS FORMATOS (3/11/2026):
      - con encabezados "Fila de arriba / del medio / de abajo (De izquierda a
        derecha)." y los nombres de cada bloque, que pueden seguir en varios
        renglones. Los renglones fuera de un bloque (el curso, "6G") no son
        nombres.
      - sin encabezados (el que usa hoy quien digitaliza): CADA RENGLON NO
        VACIO ES UNA FILA de la foto, de arriba hacia abajo. Nombres
        completos, separados por coma (con o sin espacio despues).
    En los dos, los separadores entre nombres son el guion (normal o largo,
    con espacio de algun lado), la coma y el punto seguido de espacio cuando
    no es de un titulo ni de una inicial."""
    lineas = [l.strip() for l in (texto or "").splitlines()]
    bloques = []
    if any(FILA.match(s) for s in lineas):
        fila, buf = None, []
        for s in lineas:
            if FILA.match(s):
                if fila:
                    bloques.append((fila, " ".join(buf)))
                fila, buf = re.sub(r"\s*\(.*$", "", s), []
            elif s and fila:
                buf.append(s)
        if fila:
            bloques.append((fila, " ".join(buf)))
    else:
        k = 0
        for s in lineas:
            if not s:
                continue
            k += 1
            bloques.append(("Fila %d" % k, s))
    salida, n = [], 0
    for fila, cuerpo in bloques:
        pos = 0
        for trozo in CORTE.split(cuerpo):
            for trozo2 in trozo.split(","):
                for crudo in _partir_por_punto(trozo2):
                    crudo = crudo.strip(" .,;\t–—-")
                    if not _es_nombre(crudo):
                        continue
                    n += 1
                    pos += 1
                    es_staff = bool(TITULO.match(crudo))
                    limpio = TITULO.sub("", crudo).strip()
                    toks = limpio.split()
                    inis = [t for t in toks if INICIAL.match(t)]
                    if inis or len(toks) < 2:
                        apellido = " ".join(t for t in toks if t not in inis)
                        ini = sin_tildes(inis[0])[:1] if inis else ""
                        pila = ""
                    else:
                        # Nombre completo: el primero es el nombre de pila y
                        # el resto se prueba como apellido, del mas largo al
                        # mas corto (ver candidatas()).
                        pila = pegado(toks[0])
                        ini = pila[:1]
                        apellido = " ".join(toks[1:])
                    salida.append((n, fila, pos, crudo, es_staff, ini,
                                   apellido, pila))
    # Sin encabezados, un renglon sin ningun nombre (el curso, un link) no
    # cuenta como fila: los numeros de las que siguen no se corren.
    if not any(f.startswith("Fila de") for f, _c in bloques):
        renum, vistas = {}, []
        for t in salida:
            if t[1] not in renum:
                vistas.append(t[1])
                renum[t[1]] = "Fila %d" % len(vistas)
        salida = [(t[0], renum[t[1]]) + t[2:] for t in salida]
    return salida


# ---------------------------------------------------------------- el padron

def _base(gente, ini, ape, pila=""):
    base = []
    for p in gente:
        if (p.get("kind") or "") == "noise":
            continue                      # regla 5: no es una persona
        dn = p.get("display_name") or ""
        nombres = [dn] + [a for a in (p.get("aliases") or []) if a]
        if not any(pegado(x).endswith(ape) for x in nombres):
            continue
        if pila:
            # CON EL NOMBRE DE PILA ESCRITO, MANDA EL NOMBRE, NO LA INICIAL
            # (3/11/2026). En una foto de house hay hermanos con la misma
            # inicial: medido en la primera nota sin encabezados, la inicial
            # sola le daba a un hermano la ficha del otro (dos casos). Se
            # compara como prefijo sin espacios, asi un nombre escrito junto
            # encuentra la ficha que lo tiene separado.
            if not any(pegado(x).startswith(pila) for x in nombres):
                continue
        elif ini:
            primeras = {sin_tildes(x).split()[0][:1] for x in nombres
                        if sin_tildes(x).split()}
            if ini not in primeras:
                continue                  # regla 2: la inicial manda
        base.append(p)
    return base


# Para proponer (no cargar) una ficha a un nombre que no resolvio: parecido
# del nombre entero sin espacios. Es solo la candidata de padron_dudosas.
PARECIDO_MINIMO = 0.85


def parecida(gente, crudo):
    """(persona, puntaje) mas parecida a un nombre completo que no resolvio, o
    (None, 0). Compara el nombre escrito y el escrito sin el primer nombre
    contra la ficha y la ficha sin su primer nombre (no los dos recortados:
    eso deja solo el apellido y junta hermanos)."""
    import difflib
    toks = TITULO.sub("", crudo).split()
    if len(toks) < 2:
        return None, 0.0
    esc = [pegado(" ".join(toks)), pegado(" ".join(toks[1:]))]
    pil = [pegado(toks[0]), pegado(toks[1])]
    mejor, punt = None, 0.0
    for p in gente:
        if (p.get("kind") or "") == "noise":
            continue
        for x in [p.get("display_name") or ""] + [a for a in (p.get("aliases") or []) if a]:
            t = sin_tildes(x).split()
            if len(t) < 2:
                continue
            fic = [pegado(x), "".join(t[1:])]
            for i, e in enumerate(esc):
                for j, f in enumerate(fic):
                    if i and j:
                        continue
                    # El nombre con el que arranca cada lado tiene que ser el
                    # mismo (o una errata): si no, el recorte deja apellido
                    # contra apellido y propone a un hermano (medido: un
                    # nombre de pila distinto daba 0,88 contra el hermano).
                    a_, b_ = pil[i], t[j]
                    if not (a_.startswith(b_) or b_.startswith(a_) or
                            difflib.SequenceMatcher(None, a_, b_).ratio() >= 0.8):
                        continue
                    r = difflib.SequenceMatcher(None, e, f).ratio()
                    if r > punt:
                        mejor, punt = p, r
    if punt < PARECIDO_MINIMO:
        return None, 0.0
    return mejor, round(punt, 2)


def candidatas(gente, ini, apellido, es_staff, camada, camada_foto=None,
               pila=""):
    if not pegado(apellido):
        return [], ""
    if not pila:
        base = _base(gente, ini, pegado(apellido))
    else:
        # NOMBRE COMPLETO (3/11/2026). No se sabe donde termina el nombre y
        # empieza el apellido ("Persona S X", "Alejandro X Y"): se prueba
        # todo lo que sigue al nombre de pila como sufijo y, si nadie lo
        # tiene, se va acortando por la izquierda. Gana el sufijo mas largo
        # que alguien tenga: es el mas especifico.
        toks = apellido.split()
        base = []
        for i in range(len(toks)):
            base = _base(gente, ini, pegado(" ".join(toks[i:])), pila)
            if base:
                break
        if base and i > 0:
            # Lo que se cayo por la izquierda se dice, y main() no lo carga:
            # un apellido acortado puede ser el de otra persona.
            foco, etq = _filtrar(base, es_staff, camada, camada_foto)
            return foco, etq + " [apellido acortado a '%s']" % " ".join(toks[i:])
    return _filtrar(base, es_staff, camada, camada_foto)


def _filtrar(base, es_staff, camada, camada_foto):
    if es_staff:
        foco = [p for p in base if (p.get("kind") or "") not in ("student", "alumni")]
        etq = "staff"
    elif camada:
        foco = [p for p in base if str(p.get("cohort") or "") == str(camada)]
        etq = "camada %s" % camada
    elif camada_foto and len(base) > 1:
        # REGLA 4: desempate por la camada de la foto. Solo cuando hay mas de
        # una candidata, y aca el filtro NO cae al fallback de abajo: si
        # ninguna es de la camada de la foto, no hay candidata que proponer.
        foco = [p for p in base if str(p.get("cohort") or "") == str(camada_foto)]
        return foco, "camada %s (la de la foto)" % camada_foto
    else:
        foco, etq = base, "sin camada en el nombre del archivo"
    if foco:
        return foco, etq
    # NINGUNA CUMPLE EL FILTRO (27/10/2026). Antes esto devolvia la lista sin
    # filtrar, y era el agujero por el que salian las candidatas de otra
    # camada: con una sola, da igual -no hay a quien confundir- y conviene
    # cargarla (asi entra quien es del staff, que no tiene camada, y dos
    # alumnos repitentes de la nota de 6G). Con dos o mas, elegir seria
    # adivinar: no se propone ninguna y el nombre va a dudosas pelado.
    if not base:
        return [], etq                     # nadie con ese apellido: sin ficha
    if len(base) == 1:
        return base, etq + " (ninguna cumple: se carga igual, es la unica)"
    return [], etq + (" (ninguna cumple y hay %d homonimas de otras camadas: "
                      "no se propone ninguna)" % len(base))


# Para que la moda cuente como camada de la foto: al menos
# MINIMO_PARA_LA_MODA fichas y mas de la mitad de las que resolvieron.
MINIMO_PARA_LA_MODA = 3


def camada_de_la_foto(gente, texto, camada):
    """La camada que delatan los nombres del epigrafe que resolvieron solos.

    Devuelve (camada, cuantas la respaldan, cuantas resolvieron). Mira solo
    alumnos: el staff no tiene camada, y un nombre ambiguo no vota."""
    votos = defaultdict(int)
    resueltos = 0
    for (_n, _fila, _pos, _crudo, es_staff, ini, apellido, pila) in nombres_de(texto):
        if es_staff:
            continue
        cands, _etq = candidatas(gente, ini, apellido, es_staff, camada,
                                 pila=pila)
        if len(cands) != 1:
            continue
        resueltos += 1
        coh = cands[0].get("cohort")
        if coh:
            votos[int(coh)] += 1
    if not votos:
        return None, 0, resueltos
    cohorte, cuantas = max(votos.items(), key=lambda kv: kv[1])
    if cuantas < MINIMO_PARA_LA_MODA or cuantas * 2 <= resueltos:
        return None, cuantas, resueltos
    return cohorte, cuantas, resueltos


# ---------------------------------------------------------------- cursor y base

def leer_cursor(cli):
    """{id de nota (texto): md5 con que se cargo}."""
    filas = cli.select("corridas", select="mensaje", tarea="eq." + CURSOR["tarea"],
                       etiqueta="eq." + CURSOR["etiqueta"]) or []
    try:
        return json.loads(filas[0]["mensaje"] or "{}") if filas else {}
    except ValueError:
        return {}


def anotar_cursor(cli, cursor, hechas, escritas):
    """estado listo y no corriendo: salud() marca trabada una corrida que dice
    'corriendo' y no se mueve en dos horas (igual que epigrafe_caras.py)."""
    cli.upsert("corridas", [dict(CURSOR, estado="listo", total=len(cursor), hechas=hechas,
                                 encontradas=escritas, con_gente=0,
                                 mensaje=json.dumps(cursor, ensure_ascii=False),
                                 actualizado_at=_dt.datetime.now(_dt.timezone.utc).isoformat())],
               on_conflict="tarea,etiqueta")


def notas_nuevas(notas, md5, cursor):
    """Las notas cuyo contenido no se cargo todavia (md5 distinto del cursor)."""
    return [n for n in notas if cursor.get(str(n["id"])) != md5.get(int(n["id"]))]


def dudosas_pendientes(cli):
    """{nombre: id de la candidata o None} de las dudosas de epigrafe sin
    resolver (regla 6 del docstring)."""
    filas = cli.select("padron_dudosas", select="nombre,parecida_id,resuelto",
                       hoja="eq.epigrafe", resuelto="is.null") or []
    return {f["nombre"]: (int(f["parecida_id"]) if f.get("parecida_id") is not None
                          else None) for f in filas}


def frena_la_dudosa(pendientes, crudo, persona):
    """Si una dudosa sin contestar frena este nombre con esta ficha."""
    if crudo not in pendientes:
        return False
    cand = pendientes[crudo]
    return cand is None or cand == int(persona)


def pares_existentes(cli, fotos):
    """{(photo_id, person_id): source} de las fotos dadas."""
    out = {}
    fotos = sorted(set(fotos))
    for i in range(0, len(fotos), 40):
        for r in cli.select("photo_people", select="photo_id,person_id,source",
                            photo_id="in.(%s)" % ",".join(fotos[i:i + 40])) or []:
            out[(r["photo_id"], int(r["person_id"]))] = r.get("source")
    return out


def separar(filas, existentes):
    """(las que hay que insertar, las que ya estaban con su source)."""
    nuevas, ya = [], []
    for f in filas:
        k = (f["photo_id"], int(f["person_id"]))
        if k in existentes:
            ya.append((f, existentes[k]))
        else:
            nuevas.append(f)
    return nuevas, ya


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--carpeta", help="una sola carpeta de Drive")
    ap.add_argument("--drive", action="store_true",
                    help="bajar los .txt de Drive en vez de leer carpeta_notas")
    ap.add_argument("--aplicar", action="store_true")
    # UNA FICHA QUE RESUELVE PERO TODAVIA NO ESTA AUTORIZADA (26/10/2026). La
    # primera tanda se autorizo nombre por nombre: una ficha que resolvio
    # despues de afinar las reglas no entra de prepo, queda como dudosa hasta
    # que alguien diga que si. Es el mismo criterio de no inventar
    # definiciones, aplicado a la carga.
    ap.add_argument("--dejar-afuera", nargs="*", type=int, default=[],
                    metavar="ID", help="ids de persona que esta tanda no carga "
                                       "(quedan como dudosa, con su candidata)")
    ap.add_argument("--igual-con-el-pulso", action="store_true",
                    help="escribir aunque el pulso este en un paso que frena "
                         "(lo usaria el pulso)")
    ap.add_argument("--pulso", action="store_true",
                    help="solo las notas nuevas o cambiadas (cursor en corridas)")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                           # noqa: BLE001
        pass
    if a.pulso and (a.drive or a.carpeta or a.dejar_afuera):
        ap.error("--pulso va solo: recorre carpeta_notas entera contra el cursor")

    cli = sb.SB()
    if a.drive:
        if not a.carpeta:
            ap.error("--drive necesita --carpeta")
        notas = notas_de_drive(a.carpeta)
    else:
        notas = notas_de_la_base(cli, a.carpeta)
    print("Notas con texto: %d" % len(notas))
    cursor, md5 = {}, {}
    if a.pulso:
        cursor = leer_cursor(cli)
        md5 = {int(r["id"]): r.get("md5") for r in
               (cli.select("carpeta_notas", select="id,md5", ignorada="is.false") or [])}
        notas = notas_nuevas(notas, md5, cursor)
        print("Nuevas o cambiadas (cursor %s/%s): %d"
              % (CURSOR["tarea"], CURSOR["etiqueta"], len(notas)))
    if not notas:
        print("No hay nada que resolver.")
        sb.salida_pulso({"notas": 0, "hechas": 0, "escritas": 0, "fallidas": []})
        return

    gente = cli.select("people", select="id,display_name,kind,cohort,campus,house,aliases")
    print("Fichas del padron: %d" % len(gente))
    pendientes = dudosas_pendientes(cli)
    print("Dudosas de epigrafe sin contestar: %d" % len(pendientes))

    filas, dudosas, sueltas = [], {}, []
    # notas con algo que no se cargo (sin foto, dudosas): el cursor no las anota
    incompletas = set()
    resumen = defaultdict(int)
    for nota in sorted(notas, key=lambda n: n["nombre"] or ""):
        camada = None
        m = CAMADA_EN_NOMBRE.search(nota["nombre"] or "") or \
            CAMADA_EN_NOMBRE.search(nota["texto"] or "")
        if m:
            camada = int(m.group(1))
        foto = foto_de(cli, nota)
        camada_foto, votos, resueltos = (None, 0, 0)
        if not camada:
            camada_foto, votos, resueltos = camada_de_la_foto(
                gente, nota["texto"], camada)
        print("\n" + "=" * 78)
        print("%s" % nota["nombre"])
        print("  camada del archivo: %s   foto: %s"
              % (camada or "(no dice)", foto or "(sin foto atada)"))
        if not camada:
            print("  camada de la foto:  %s"
                  % ("%s (la votan %d de %d nombres que resolvieron solos)"
                     % (camada_foto, votos, resueltos) if camada_foto
                     else "no se puede sacar (%d de %d)" % (votos, resueltos)))
        print("=" * 78)
        if not foto:
            incompletas.add(nota["id"])
            resumen["notas sin foto"] += 1
            print("  La nota no tiene foto: no se carga nada (se ata cuando la foto se indexe).")
            continue
        fila_ant = None
        for (n, fila, pos, crudo, es_staff, ini, apellido, pila) in nombres_de(nota["texto"]):
            if fila != fila_ant:
                print("  -- %s" % fila)
                fila_ant = fila
            cands, etq = candidatas(gente, ini, apellido, es_staff, camada,
                                    camada_foto, pila=pila)
            if not (len(cands) == 1 and "acortado" not in etq
                    and cands[0]["id"] not in (a.dejar_afuera or [])
                    and not frena_la_dudosa(pendientes, crudo, cands[0]["id"])):
                # todo lo que no se carga deja la nota para la corrida siguiente
                incompletas.add(nota["id"])
            if len(cands) == 1 and frena_la_dudosa(pendientes, crudo, cands[0]["id"]):
                p = cands[0]
                resumen["dudosa sin contestar (no se carga)"] += 1
                print("  %-3d %-34s == %-34s [no se carga: dudosa sin contestar]"
                      % (n, crudo, p["display_name"]))
            elif len(cands) == 1 and cands[0]["id"] in (a.dejar_afuera or []):
                p = cands[0]
                resumen["dejados afuera a pedido"] += 1
                dudosas[crudo] = {
                    "nombre": crudo, "hoja": "epigrafe", "parecida_id": p["id"],
                    "parecida_nombre": p["display_name"], "puntaje": 0.9}
                print("  %-3d %-34s == %-34s [no se carga: --dejar-afuera]"
                      % (n, crudo, p["display_name"]))
            elif len(cands) == 1 and "acortado" in etq:
                # Una sola ficha, pero con parte del apellido escrito tirado:
                # puede ser una errata de la nota o puede ser otra persona.
                # No se adivina: va a dudosas con su candidata.
                p = cands[0]
                resumen["apellido acortado (a dudosas con candidata)"] += 1
                dudosas[crudo] = {
                    "nombre": crudo, "hoja": "epigrafe", "parecida_id": p["id"],
                    "parecida_nombre": p["display_name"], "puntaje": 0.7}
                print("  %-3d %-34s ~~ %-34s [%s %s]  <- no se carga: %s"
                      % (n, crudo, p["display_name"], p.get("kind") or "?",
                         p.get("cohort") or "-", etq))
            elif len(cands) == 1:
                p = cands[0]
                ok_filtro = ("ninguna cumple" not in etq)
                filas.append({"photo_id": foto, "person_id": p["id"],
                              "folder_id": nota["folder_id"], "source": SOURCE,
                              "_nota_id": nota["id"],
                              "_nota": nota["nombre"], "_crudo": crudo, "_fila": fila,
                              "_pos": pos,
                              "_nombre": p["display_name"], "_kind": p.get("kind"),
                              "_cohort": p.get("cohort"), "_firme": ok_filtro})
                resumen["resueltos" if ok_filtro else "resueltos sin el filtro"] += 1
                print("  %-3d %-34s -> %-34s [%s %s]%s"
                      % (n, crudo, p["display_name"], p.get("kind") or "?",
                         p.get("cohort") or "-", "" if ok_filtro else "  <- " + etq))
            elif cands:
                resumen["ambiguos"] += 1
                dudosas[crudo] = {
                    "nombre": crudo, "hoja": "epigrafe",
                    "parecida_id": cands[0]["id"],
                    "parecida_nombre": " / ".join(
                        "%s (%s)" % (c["display_name"], c["id"]) for c in cands[:4]),
                    "puntaje": 0.5}
                print("  %-3d %-34s ?? %d fichas: %s"
                      % (n, crudo, len(cands),
                         ", ".join("%s (%s, %s)" % (c["display_name"], c["id"],
                                                    c.get("cohort") or "-")
                                   for c in cands[:4])))
            elif "no se propone ninguna" in etq:
                # Habia candidatas por apellido, pero ninguna de la camada de
                # la foto: proponerlas seria proponer a otra persona.
                resumen["sin ficha de esa camada (habia homonimas)"] += 1
                sueltas.append((nota["nombre"], crudo))
                dudosas.setdefault(crudo, {
                    "nombre": crudo, "hoja": "epigrafe", "parecida_id": None,
                    "parecida_nombre": None, "puntaje": 0.0})
                print("  %-3d %-34s -- %s" % (n, crudo, etq))
            else:
                # Con nombre completo se busca una parecida para proponer en
                # dudosas (erratas de la nota, un segundo nombre usado como
                # primero). Es la candidata de la pregunta: no se carga.
                pp, punt = parecida(gente, crudo) if pila else (None, 0.0)
                resumen["sin ficha" if not pp
                        else "sin ficha (a dudosas con parecida)"] += 1
                sueltas.append((nota["nombre"], crudo))
                dudosas.setdefault(crudo, {
                    "nombre": crudo, "hoja": "epigrafe",
                    "parecida_id": pp["id"] if pp else None,
                    "parecida_nombre": pp["display_name"] if pp else None,
                    "puntaje": punt})
                print("  %-3d %-34s -- sin ficha en el padron%s"
                      % (n, crudo, "  (parecida: %s, %s, %s, %.2f)"
                         % (pp["display_name"], pp["id"], pp.get("cohort") or "-",
                            punt) if pp else ""))

    print("\n" + "-" * 78)
    # LO QUE YA ESTA NO SE TOCA: se mira antes, tambien en el ensayo, para que
    # el ensayo diga cuantos pares ya tenian dueño y con que source.
    existentes = pares_existentes(cli, [f["photo_id"] for f in filas]) if filas else {}
    nuevas_pp, ya_estaban = separar(filas, existentes)
    print("Resueltas: %d identificaciones" % len(filas))
    print("   para insertar:           %d" % len(nuevas_pp))
    por_source = defaultdict(int)
    for _f, s in ya_estaban:
        por_source[s or "(sin source)"] += 1
    print("   ya estaban (no se tocan): %d  %s"
          % (len(ya_estaban), dict(sorted(por_source.items())) if por_source else ""))
    for k, v in sorted(resumen.items()):
        print("   %-28s %d" % (k + ":", v))
    print("A dudosas (no se cargan ni se crean): %d" % len(dudosas))
    if a.pulso:
        print("Notas que quedarian al dia en el cursor: %d de %d"
              % (len({n["id"] for n in notas} - incompletas), len(notas)))

    sello = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    respaldo = os.path.join(HERE, "_epigrafes_%s.txt" % sello)
    with open(respaldo, "w", encoding="utf-8") as fh:
        fh.write("# epigrafes.py %s  %s\n" % (sello, "APLICADO" if a.aplicar else "ensayo"))
        fh.write("# revertir: delete from photo_people where source = 'epigrafe';\n")
        fh.write("# (solo se insertan pares nuevos: los que ya estaban no cambian de source)\n")
        fh.write("# nota\tnombre en el texto\tfila\tpersona\tid\tphoto_id\tpos en la fila\taccion\n")
        nuevos_k = {(f["photo_id"], f["person_id"]) for f in nuevas_pp}
        for f in filas:
            k = (f["photo_id"], f["person_id"])
            fh.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
                     % (f["_nota"], f["_crudo"], f["_fila"], f["_nombre"],
                        f["person_id"], f["photo_id"], f["_pos"],
                        "insertar" if k in nuevos_k
                        else "ya estaba (%s)" % existentes.get((f["photo_id"], int(f["person_id"])))))
        fh.write("\n# a dudosas\n")
        for d in dudosas.values():
            fh.write("%s\t%s\n" % (d["nombre"], d["parecida_nombre"] or "(sin ficha)"))
    print("Respaldo: %s" % respaldo)

    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar.")
        return

    frenan, otros = sb.pasos_del_pulso(cli)
    if otros:
        print("\n(el pulso esta en %s; no escribe identificaciones, sigo)"
              % ", ".join(otros))
    if frenan and not a.igual_con_el_pulso:
        print("\nNO escribo: el pulso esta en %s. Probar de nuevo despues."
              % ", ".join(frenan))
        sys.exit(2)

    limpias = [{k: v for k, v in f.items() if not k.startswith("_")} for f in nuevas_pp]
    # ignore-duplicates: si entre la lectura de arriba y ahora otro paso cargo
    # el mismo par, gana el que llego primero; este no lo pisa.
    cli.upsert("photo_people", limpias, on_conflict="photo_id,person_id", ignorar=True)

    # EL TRIGGER photo_people_guard DESCARTA EN SILENCIO (sin error) un insert
    # con source que no sea game ni manual si esa foto y esa persona estan en
    # photo_people_rejected. Asi que no alcanza con que el POST salga bien:
    # hay que contar lo que quedo.
    despues = pares_existentes(cli, [f["photo_id"] for f in nuevas_pp]) if nuevas_pp else {}
    quedaron, faltan, rechazadas = 0, [], []
    for f in nuevas_pp:
        k = (f["photo_id"], int(f["person_id"]))
        if k in despues:
            quedaron += 1
            if despues[k] != SOURCE:
                print("   entro con otro source (lo cargo otro paso en el medio): "
                      "%s en %s -> %s" % (f["person_id"], f["photo_id"], despues[k]))
            continue
        rech = cli.select("photo_people_rejected", select="*",
                          photo_id="eq." + f["photo_id"],
                          person_id="eq.%s" % f["person_id"]) or []
        (rechazadas if rech else faltan).append(f)
        print("   NO entro: %s (%s) en %s  %s"
              % (f["_nombre"], f["person_id"], f["photo_id"],
                 "hay rechazo humano previo: " + json.dumps(rech[0], default=str)
                 if rech else "sin rechazo previo: revisar"))
    print("\nInsertadas y verificadas: %d de %d   (rechazo humano previo: %d, "
          "sin explicacion: %d)" % (quedaron, len(nuevas_pp), len(rechazadas), len(faltan)))
    total = cli.select("photo_people", select="photo_id", source="eq." + SOURCE) or []
    print("Filas con source = '%s' en la base: %d (incluye las copias que propaga "
          "photo_people_copias por dedupe_key)" % (SOURCE, len(total)))

    if dudosas:
        ya = {d["nombre"] for d in (cli.select("padron_dudosas", select="nombre") or [])}
        nuevas = [d for d in dudosas.values() if d["nombre"] not in ya]
        if nuevas:
            cli.upsert("padron_dudosas", nuevas, on_conflict="nombre")
        print("Dudosas nuevas: %d (ya estaban %d)" % (len(nuevas), len(dudosas) - len(nuevas)))

    # El cursor: una nota queda al dia si no le quedo nada pendiente y todos
    # sus pares estan (insertados, ya estaban, o frenados por un rechazo
    # humano, que es una respuesta). Un par que no entro sin explicacion deja
    # la nota para la corrida siguiente y el paso sale con error: no se tapa.
    fallidas = sorted({f["_nota_id"] for f in faltan}, key=str)
    if a.pulso:
        hechas = 0
        for n in notas:
            if n["id"] in incompletas or n["id"] in fallidas:
                continue
            cursor[str(n["id"])] = md5.get(int(n["id"]))
            hechas += 1
        anotar_cursor(cli, cursor, hechas, quedaron)
        print("Cursor %s/%s: %d notas anotadas en esta corrida, %d en total"
              % (CURSOR["tarea"], CURSOR["etiqueta"], hechas, len(cursor)))
    sb.salida_pulso({"notas": len(notas), "escritas": quedaron,
                     "incompletas": len(incompletas), "fallidas": fallidas})
    if faltan:
        sys.exit(1)

if __name__ == "__main__":
    main()
