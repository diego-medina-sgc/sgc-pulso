# -*- coding: utf-8 -*-
"""
Suma las caras de las carpetas Valete del Drive de Marketing.

    python valete_cargar.py --ensayo
    python valete_cargar.py --aplicar

DE DONDE SALE

Dentro del desarrollo editorial del Georgian, cada camada tiene una carpeta
"Valete" con el retrato individual en alta de cada egresado, y el nombre en el
propio archivo. Son doce carpetas, de 2019 a 2025, de las dos sedes.

Es la mejor fuente que hay: el nombre lo escribio el colegio, no hay que
transcribir nada a ojo como en los anuarios de papel, y son fotos de estudio
donde la cara se detecta sola.

POR QUE IMPORTA

Los 36 retratos que quedaron sin resolver son de gente que esta en el padron
pero sin una sola foto identificada, asi que la cara no puede decidir si es el
hijo o el padre. Los Valete les dan esa primera cara.

CADA CAMADA ESCRIBE DISTINTO

    Persona BB.jpg                     APELLIDO, Nombre
    ALVAREZ DE OLIVERA Persona AO.jpg      igual, sin espacio
    Persona BC.jpg                     nombre.apellido
    Persona BD 329.JPG               con el numero de la toma
    Agosti JazminENB_1015 - copia.JPG        con el codigo de camara pegado
    Persona BE Uniform ES 6 Lockwood.JPG  con el uniforme y la casa
    S6-A0014 FINAL.jpg                       solo codigo, se descarta

Igual que en los retratos, el orden no se decide con una regla: se prueban las
permutaciones contra el padron y gana la que existe.

Y las fotos no siempre estan en la carpeta Valete: en Quilmes cuelgan uno o
dos niveles mas abajo, en subcarpetas que se llaman "Valete ES6", "ES 6" o
"Valete individuales". Se baja hasta tres niveles.

QUE ESCRIBE

Las huellas van a _referencias_valete_<motor>.npz, que faces_sugerir.py suma a
las demas referencias. No entran a la biblioteca como fotos: para eso habria
que indexarlas, generar miniaturas y deduplicarlas contra las que ya estan, y
es otra tarea.
"""
import argparse
import io
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict

import numpy as np

import faces
import sb
from index_drive import Drive, SA_PATH
from retratos_nombres import (clave, indexar, por_subconjunto,
                              texto_del_archivo, titulo)

HERE = os.path.dirname(os.path.abspath(__file__))
LISTA = os.path.join(HERE, "_valetes.tsv")
SUF = "" if faces.MOTOR == "sface" else "_" + faces.MOTOR
SALIDA = os.path.join(HERE, "_referencias_valete%s.npz" % SUF)

FOL = "application/vnd.google-apps.folder"
# Georgian 2023, Valete 2025, Valete ES 6
ANIO = re.compile(r"(20\d\d)")
CAMARA = re.compile(r"^(s6|es6|dsc|img|mdf|final)?[\s\-_]*[a-z]{0,3}\d{3,6}", re.I)
# Lo que viene pegado al nombre y no es el nombre: el codigo de camara, el
# numero de la toma, el uniforme, la casa, el rotulo del grupo.
# Son dos regex y no uno porque el codigo de camara hay que buscarlo con
# las mayusculas puestas: "ENB_1015" va pegado al nombre en "Agosti
# JazminENB_1015", y buscandolo sin distinguir mayusculas [A-Z]{2,4} se
# come la "n" de Jazmin.
CODIGO_CAM = re.compile(r"[A-Z]{2,4}[_\-]?\d{3,6}")
# El numero de la toma, el uniforme, la casa, el rotulo del grupo.
RUIDO = re.compile(
    r"(\s*-\s*copia\b|\s*\(\d+\)"
    r"|\buniform\b|\bes\s?6\b|\bs6\b"
    r"|\b(agar|cutts|farran|lockwood|stevenson)\b"
    r"|\b(prefects?|boarders?|captains?|valete)\b"
    r"|\s+\d{1,4}\s*$)", re.I)


# Lo que sobra despues de limpiar y no es una persona. Dar de alta a alguien
# es lo unico de esto que no se puede revertir solo, asi que ante la duda no.
BASURA = {"copia", "de", "prefects", "prefect", "boarders", "amon", "grupal",
          "grupales", "unifrom", "uniforme", "foto", "fotos", "listado",
          # las fotos de staff que llegaron por WhatsApp traen su nombre puesto
          "whatsapp", "image", "img"}


def campus_de(ruta):
    r = ruta.lower()
    if "quilmes" in r:
        return "Quilmes"
    if "north" in r or "norte" in r:
        return "North"
    return None


def cohorte_de(ruta):
    """El año de la camada. El de la carpeta Valete manda sobre el del Georgian.

    "Georgian 2023 / ... / Valete 2025" no existe, pero "Georgian 2025 / ... /
    Valete S6" si: ahi el año esta en el nombre del Georgian. Se toma el
    ultimo que aparezca en la ruta, que es el mas especifico.
    """
    todos = ANIO.findall(ruta)
    return int(todos[-1]) if todos else None


# El grado, escrito de todas las formas que usa el colegio.
#
# Iba adentro de RUIDO como "un numero al final", y eso dejaba la letra
# colgada: "Persona BF Y1" quedaba "Persona BF Y". Aparte
# porque la letra y el numero son una sola cosa y hay que sacar las dos.
GRADO = re.compile(
    r"\b(y|ep|es|k|s|c|p)\s?[1-6][a-z]?\b|\bprek\b|\byear\s?[1-6]\b"
    r"|\bsala\s+\w+\b", re.I)


# La casa, cuando viene detras de un punto.
#
# Lo vio Diego en la lista de candidatos del juego: "Persona BG Kim",
# "Persona BH Guest", "Persona BI Skrypnyk". Son veintidos chicos de
# una misma carpeta con la casa metida adentro del apellido.
#
# La convencion del archivo es "Apellido, Nombre. Casa - Grado", y hasta ahora
# la casa se sacaba por el nombre de la CARPETA, que es donde suele estar
# escrita. En "Individuales 2024 Q" no esta, asi que no se sacaba nada.
#
# No se puede sacar por lista a secas: Jackson, Roberts, Stevenson, Agar y
# Haxell son casas del colegio Y apellidos de verdad, y borrarlos a ciegas
# arruina a un Jackson real. Lo que las separa es el punto: un apellido se
# escribe "Persona BJ" o "Persona BI", nunca detras de un punto. La
# casa, en esta convencion, siempre.
CASA_TRAS_PUNTO = re.compile(
    r"\.\s*(agar|cutts|farran|haxell|jackson|lockwood|roberts|stevenson"
    r"|school\s+house)\b", re.I)


def nombre_de(archivo, contexto=""):
    """El nombre de la persona, o None si lo que queda no es un nombre.

    LAS CONVENCIONES QUE USA EL COLEGIO

    Lo explico Diego mirando un archivo que yo habia leido mal:

        Persona BK. Jackson - EP 1      Apellido, Nombre. Casa - Grado
        Persona BF Y1                Nombre Apellido Grado
        Persona BB                      APELLIDO, Nombre
        Persona BC                      nombre.apellido

    El archivo esta bien escrito; el que no entendia era este lector. De
    "Persona BK. Jackson - EP 1" sacaba "Persona BI Skrypnyk" y
    creaba una ficha con la casa adentro del nombre.

    EL CONTEXTO ES EL NOMBRE DE LA CARPETA

    La casa no se puede sacar por lista: Jackson, Roberts y Stevenson son
    casas del colegio Y apellidos de verdad, y borrarlos a ciegas arruina a un
    Jackson real. Pero la carpeta de donde sale la foto SE LLAMA como la casa,
    asi que se saca lo que coincide con ella y nada mas. Un apellido igual al
    de la carpeta en la que esta es el unico falso positivo posible, y es
    preferible a inventarle una casa de apellido a todos los demas.
    """
    base = archivo or ""
    # la extension puede venir dos veces y con un espacio en el medio:
    # "Persona BL. Jpg.jpg". Se saca hasta que no quede ninguna.
    for _ in range(3):
        base = re.sub(r"[\s.]*\.?(jpe?g|png|tiff?|heic)\s*$", "", base, flags=re.I)
    base = re.sub(r"\b(jpe?g|png)\b", " ", base, flags=re.I)
    base = re.sub(r"\s*(final|copia|copy|editada?)\s*$", "", base, flags=re.I)
    # "Copy of 002 Persona BM": lo que puso el que duplico la
    # carpeta, mas el numero de orden de la sesion. Ninguno de los dos es el
    # nombre, y el nombre esta completo detras.
    base = re.sub(r"^\s*cop(y\s+of|ia\s+de)\s*\d*\s*", "", base, flags=re.I)
    # el grado, en cualquier lado del nombre
    base = GRADO.sub(" ", base)
    # la casa, cuando viene detras de un punto
    base = CASA_TRAS_PUNTO.sub(" ", base)
    # y lo que se llama igual que la carpeta: ahi tambien vive la casa
    for palabra in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{3,}", contexto or ""):
        base = re.sub(r"\b%s\b" % re.escape(palabra), " ", base, flags=re.I)
    # En las fotos de staff el archivo suele traer el puesto detras de un
    # guion: "Persona BN - Teaching Assistant", "Persona Z2
    # - K2G Teacher", "Barbara - Yoga". El nombre de una persona no lleva un
    # guion con espacios, asi que lo que sigue no es nombre.
    base = re.split(r"\s+-\s+", base)[0]
    if CAMARA.match(base.strip()):
        return None
    # "Persona BC" -> "Persona BC". Solo cuando los puntos separan
    # palabras y no queda ninguna otra separacion.
    if "." in base and " " not in base.replace(",", ""):
        base = base.replace(".", " ")
    # se limpia ANTES de invertir por la coma, porque el ruido va al final
    base = CODIGO_CAM.sub(" ", base)
    base = RUIDO.sub(" ", base)
    base = " ".join(base.split())
    if not base:
        return None
    return texto_del_archivo(base + ".jpg")


# Las fotos de staff vienen con el mail en el archivo:
# "persona@ejemplo.org.JPG". Es la mejor clave que puede traer una
# foto -el padron tiene el mail de casi todo el staff y no hay dos personas con
# el mismo- pero hay que leerla antes de limpiar el nombre: nombre_de() la
# convierte en "Persona BO@stgeorges edu ar" y ahi ya no sirve para nada.
MAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def mail_de(archivo):
    base = re.sub(r"\.(jpe?g|png|tiff?|heic)$", "", archivo or "", flags=re.I)
    m = MAIL.fullmatch(base.strip())
    return m.group(0).lower() if m else None


def imagenes_recursivo(d, fid, prof=0):
    """Las imagenes de la carpeta y de sus subcarpetas, hasta tres niveles.

    En Quilmes la carpeta Valete casi nunca tiene las fotos: cuelgan de una
    subcarpeta que se llama "Valete ES6", "ES 6" o "Valete individuales".
    """
    out = []
    if prof > 3:
        return out
    for x in contenido(d, fid):
        if (x.get("mimeType") or "").startswith("image/"):
            out.append(x)
        elif x["mimeType"] == FOL:
            out += imagenes_recursivo(d, x["id"], prof + 1)
    return out


def contenido(d, fid, solo_imagenes=False):
    out, tok = [], None
    q = "'%s' in parents and trashed=false" % fid
    if solo_imagenes:
        q += " and mimeType contains 'image/'"
    while True:
        p = {"q": q, "fields": "nextPageToken,files(id,name,mimeType)",
             "pageSize": 1000, "includeItemsFromAllDrives": "true",
             "supportsAllDrives": "true"}
        if tok:
            p["pageToken"] = tok
        r = d.get("files", **p)
        out += r.get("files", [])
        tok = r.get("nextPageToken")
        if not tok:
            return out


def indexar_padron(cli):
    """El padron indexado por todas las formas en que un archivo lo puede nombrar.

    Se separo de main() para que fuentes_procesar.py pueda leer una carpeta
    cualquiera con esta misma logica. Era lo unico que ataba este lector a las
    doce carpetas Valete de la lista.
    """
    padron = [p for p in cli.select(
        "people", select="id,display_name,campus,first_seen,last_seen,kind,email")
        if p.get("kind") != "noise"]
    pw, pa, pt, ju = indexar(padron)
    # "nina.dipietro" da "nina dipietro" y el padron dice "Persona BP":
    # ningun juego de palabras coincide. Comparando sin espacios si.
    # La camada 2022 de North pega los dos apellidos: "RamirezIribarren
    # Valeria" contra "Persona BQ" del padron. No coincide
    # ningun juego de palabras ni sacando los espacios, porque el orden cambia.
    #
    # Lo que si coincide es cortar el nombre del padron en dos y pegar cada
    # mitad. Se indexan todos los cortes posibles, asi entra tanto
    # "Ramireziribarren / Valeria" como "Juancruz / Vizcay".
    pegado = defaultdict(list)
    for p in padron:
        pal = clave(p.get("display_name")).split()
        if len(pal) < 2:
            continue
        for k in range(1, len(pal)):
            pegado[frozenset(["".join(pal[:k]), "".join(pal[k:])])].append(p)
    pormail = {}
    for p in padron:
        m = (p.get("email") or "").strip().lower()
        if m:
            pormail[m] = p
    return {"pw": pw, "pt": pt, "ju": ju, "pegado": pegado,
            "pormail": pormail, "n": len(padron)}


def confiable(pal, txt):
    """Si el nombre leido alcanza para CREAR una ficha nueva.

    Lo marco Diego: "esta carga de nombre es muy manual y lo hicieron
    distintas personas, sin convencion". Teniendo eso, ninguna regla de lectura
    va a acertar siempre, y la pregunta no es como leer mejor sino que hacer
    cuando la lectura no convence.

    La respuesta: un nombre dudoso sirve para RECONOCER a alguien que ya esta
    en el padron -ahi el padron valida la lectura- pero nunca para crear una
    ficha. Crear es lo unico que no se revierte solo, y una ficha basura
    ensucia para siempre la lista de candidatos del juego.

    Yo cargue asi 214 fichas con el grado o la casa adentro del nombre
    ("Persona Z3", "Persona BI Skrypnyk"), y aparecieron
    recien cuando la aprobacion automatica quiso escribirlas sobre una foto.
    """
    if len(pal) < 2:
        return False
    # "de" esta en BASURA porque suelto no es un nombre, pero adentro de uno
    # si: "Persona BR", "Persona BS", "Persona BT".
    # Se mira si lo que queda sacando las particulas es basura, no si la toca.
    PARTICULAS = {"de", "del", "la", "las", "los", "y", "da", "di", "van", "von"}
    propias = [w for w in pal if w not in PARTICULAS]
    if len(propias) < 2 or (set(propias) & BASURA):
        return False
    # un digito en cualquier parte: lo que queda de un grado, de una toma o de
    # un rotulo de carpeta ("Individuales 26b"). Un nombre no lleva numeros.
    if any(c.isdigit() for w in pal for c in w):
        return False
    # Una palabra de una sola letra puede ser dos cosas distintas.
    #
    # En el medio es una inicial y es parte del nombre: "Persona BU",
    # "Persona BV". Al final es lo que deja un grado a medio sacar
    # -"Buktenica Y"- o un apellido escrito en inicial -"Persona BW"-, y en los
    # dos casos no alcanza para crear una ficha.
    # Solo la ULTIMA. Una inicial adelante es corriente en castellano -"M
    # Persona BX", "M Persona BY", "Persona BZ"- y una al
    # final no: ahi es un grado a medio sacar o un apellido escrito en inicial.
    if len(pal[-1]) < 2:
        return False
    # un grado que sobrevivio a la limpieza
    if GRADO.search(txt or ""):
        return False
    # El epigrafe de una obra, no una persona. Salieron de las carpetas de
    # muestras de arte, donde el archivo se llama como el cuadro:
    # "acuarela Y lapiz sobre papel Martina Machia", ". Persona CA Fatch.
    # Obra de titeres El tesoro pirata".
    #
    # El largo se puede usar de corte porque el padron dice donde esta el
    # techo: de 6.128 fichas, la mas larga tiene seis palabras -"Persona CB
    # Diaz Valdez Vaz Pinto"- y son cinco fichas. Ninguna tiene siete.
    if len(pal) > 6:
        return False
    # Y un nombre no empieza con puntuacion: "-angeles Grosz- cronofotografia".
    if (txt or "").strip()[:1] in ".-,;:|":
        return False
    return True


def a_quien_apunta(idx, fotos):
    """Decide de quien es cada foto por el nombre del archivo.

    fotos: (fid, archivo, campus, cohorte)
    Devuelve (resueltas, sin_padron, motivos). No escribe nada: quien llama
    decide si da de alta a los que no estan.
    """
    resuelto, motivos, sin_padron = [], Counter(), []
    for fid, arch, camp, coh in fotos:
        # El mail primero: es exacto y no hay que adivinar nada. Si el archivo
        # ES un mail y ese mail no esta en el padron, no se da de alta: de
        # "j.perez@" no se puede sacar como se escribe el nombre, y una ficha
        # con el nombre mal escrito es peor que ninguna.
        m = mail_de(arch)
        if m:
            q = (idx.get("pormail") or {}).get(m)
            if q:
                resuelto.append((fid, arch, q, camp, coh))
                motivos["encontrada por el mail del archivo"] += 1
            else:
                motivos["el mail del archivo no esta en el padron"] += 1
            continue
        txt = nombre_de(arch)
        if not txt:
            motivos["el archivo es un codigo de camara"] += 1
            continue
        pal = clave(txt).split()
        if len(pal) < 2:
            motivos["el archivo no trae nombre y apellido"] += 1
            continue
        cand = (list(idx["pw"].get(frozenset(pal), []))
                or por_subconjunto(pal, idx["pt"], idx["ju"])
                or list(idx["pegado"].get(frozenset(pal), [])))
        cand = [p for p in cand if not camp or not p.get("campus")
                or p["campus"] == camp]
        unicos = {int(p["id"]): p for p in cand}
        if len(unicos) == 1:
            resuelto.append((fid, arch, list(unicos.values())[0], camp, coh))
            motivos["ya esta en el padron"] += 1
        elif len(unicos) > 1:
            motivos["mas de una persona posible, no se toca"] += 1
        elif confiable(pal, txt):
            sin_padron.append((fid, arch, titulo(txt), camp, coh))
            motivos["hay que darlo de alta"] += 1
        elif len(pal) >= 2:
            motivos["el nombre no da para crear una ficha"] += 1
        else:
            motivos["no parece un nombre, no se crea"] += 1
    return resuelto, sin_padron, motivos


def medir_caras(d, resuelto, limite=0, log=print):
    """Baja cada foto y le saca la huella. -> (vecs, personas, anios, errores)

    Una sola cara por foto: son retratos individuales, y si aparece alguien
    detras no es de quien dice el archivo.
    """
    c = faces.Caras()
    vecs, personas, anios = [], [], []
    err = Counter()
    for n, (fid, arch, p, camp, coh) in enumerate(resuelto, 1):
        if limite and n > limite:
            break
        try:
            link = d.get("files/" + fid, fields="thumbnailLink").get("thumbnailLink")
            if not link:
                err["sin miniatura"] += 1
                continue
            with urllib.request.urlopen(link.replace("=s220", "=s1600"), timeout=60) as r:
                data = r.read()
            hs = c.huellas(data, max_caras=1)
        except Exception:
            err["error al bajar"] += 1
            continue
        if not hs:
            err["no se detecto cara"] += 1
            continue
        # huellas() devuelve (huella, caja, confianza); solo importa la huella
        vecs.append(hs[0][0])
        personas.append(int(p["id"]))
        anios.append(coh or -1)
        if n % 50 == 0:
            log("   %d/%d" % (n, len(resuelto)))
            d._refresh()
    return vecs, personas, anios, err


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--ensayo", action="store_true")
    ap.add_argument("--limite", type=int, default=0)
    a = ap.parse_args()
    print("Motor: %s" % faces.MOTOR)

    if not os.path.exists(LISTA):
        sys.exit("falta %s: correr antes el recorrido del Drive" % LISTA)
    carpetas = []
    for l in io.open(LISTA, encoding="utf-8"):
        if not l.strip():
            continue
        fid, ruta = l.rstrip("\n").split("\t", 1)
        carpetas.append((fid, ruta))
    print("Carpetas Valete: %d" % len(carpetas))

    d = Drive(SA_PATH)
    # una foto por egresado, y algunas carpetas tienen las fotos un nivel mas
    # abajo (Valete S6 / S6)
    fotos = []
    for fid, ruta in carpetas:
        img = imagenes_recursivo(d, fid)
        camp, coh = campus_de(ruta), cohorte_de(ruta)
        for x in img:
            fotos.append((x["id"], x["name"], camp, coh, ruta))
        print("   %-58s %s %s  %d fotos"
              % (ruta[-58:], str(camp), str(coh), len(img)))
    print("Fotos en total: %d\n" % len(fotos))

    cli = sb.SB()
    idx = indexar_padron(cli)

    resuelto, sin_padron, motivos = a_quien_apunta(
        idx, [(f[0], f[1], f[2], f[3]) for f in fotos])

    for k, v in motivos.most_common():
        print("   %-46s %d" % (k + ":", v))
    if sin_padron:
        print("\nPara dar de alta (los primeros):")
        for _, arch, nom, camp, coh in sin_padron[:10]:
            print("   %-34s %-8s %s" % (nom[:34], str(camp), str(coh)))

    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar para escribir.")
        return

    # las que faltan se crean con la misma funcion que usa el anuario, que ya
    # sabe no duplicar cuando el nombre esta escrito de otra manera
    for _, arch, nom, camp, coh in sin_padron:
        pid = cli.rpc("upsert_yearbook_person", {
            "p_display_name": nom, "p_norm_name": clave(nom),
            "p_house": None, "p_cohort": coh, "p_desde": None,
            "p_campus": camp})
        resuelto.append((_, arch, {"id": pid, "display_name": nom}, camp, coh))

    print("\nBajando y midiendo %d caras…" % len(resuelto))
    vecs, personas, anios, err = medir_caras(d, resuelto, a.limite)

    for k, v in err.most_common():
        print("   %-30s %d" % (k + ":", v))
    if not vecs:
        sys.exit("no se saco ninguna cara")
    np.savez_compressed(SALIDA,
                        vecs=np.array(vecs, dtype=np.float32),
                        personas=np.array(personas),
                        anios=np.array(anios))
    print("\nEscrito %s: %d caras de %d personas"
          % (SALIDA, len(vecs), len(set(personas))))
    print("Correr despues:  faces_sugerir.py --rehacer  y  faces_aprobar.py --aplicar")


if __name__ == "__main__":
    main()
