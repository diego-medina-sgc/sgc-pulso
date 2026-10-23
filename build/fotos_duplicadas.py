# -*- coding: utf-8 -*-
"""
Encuentra la misma foto cargada dos veces y deja una sola en la cola.

    python fotos_duplicadas.py                     # ensayo de una vuelta del pulso
    python fotos_duplicadas.py --aplicar           # (en el pulso) lo nuevo + el año de turno
    python fotos_duplicadas.py --todo              # todo el archivo de una vez, sin tope
    python fotos_duplicadas.py --todo --aplicar
    python fotos_duplicadas.py --retratos [--aplicar]   # solo carpetas de retrato
    python fotos_duplicadas.py --anio 2019 [--aplicar]  # un año entero

POR QUE

Se vio usando la app: "en el etiquetado y en las confirmaciones aparecen
muchas fotos duplicadas para confirmar, literal la misma foto". La causa: hay
albumes repetidos. "Individuales-148.jpg" esta en "Fotos individuales" y otra
vez en "Kinder"; un carnet estaba tres veces.

En las carpetas de retrato solas eran 953 copias de mas sobre 13.198: el 7% de
las preguntas del juego eran una foto que ya se contesto.

La columna dedupe_key ya existia pero solo la llenaba index_zenfolio.py, que
cruza Zenfolio contra Drive (con el nombre de archivo normalizado). La copia de
Drive contra Drive -el mismo archivo en dos carpetas, que en Drive son dos ids
distintos- no la miraba nadie.

COMO SE RECONOCE UNA COPIA

Primero se agrupa por (bytes, ancho, alto), que es gratis porque ya esta en la
base. Eso PROPONE: dos JPEG de la misma camara pueden pesar exactamente lo
mismo sin ser la misma foto, y pasa -en 2009 hay dos tomas distintas de la
misma sesion, BJN_0399 y BJN_0550, con los mismos 754.585 bytes-.

Despues se le pide a Drive el md5 de cada candidata y se unen solo las que
tienen el mismo md5. Ahi no queda lugar a duda: es el mismo archivo byte por
byte. Solo se preguntan las que ya coinciden en tamaño, no las 490.000.

La clave que queda en dedupe_key es ese md5 (32 caracteres hexadecimales). Asi
una foto ya unida no se le vuelve a preguntar a Drive: su md5 esta en la base.

CUAL SE QUEDA

La que ya tiene gente identificada, para no dejar huerfano el trabajo hecho.
Despues, la que tiene caras o propuestas (face_suggestions, caras_mapa): las
propuestas de las copias no se borran pero tampoco se ven, y si la primaria
fuera una sin caras, ese trabajo del reconocimiento quedaria escondido
(23/10/2026). Despues, la que ya era la primaria de su grupo (una copia nueva no le roba el
lugar a la que ya estaba). Si no, la de Drive antes que la de Zenfolio
-Zenfolio tiene marca de agua encima de la cara-, la que tiene año antes que
la que no, la que entro primero a la base (indexed_at) y a igual todo la de
mas pixeles. Las dos del medio son del 22/10/2026: los discos externos subidos
a Server Media (_EXTERNAL_IMPORT_STAGING_*) entran sin año y son copias del
archivo; sin ellas, con el mismo tamaño, desempataba el id y la copia del disco
le podia ganar a la original.

Las otras quedan is_primary=false. El juego ya filtra por is_primary; la
pantalla de confirmar la filtra desde fotos_copias.sql.

LO QUE YA SE HABIA CONTESTADO SE COPIA

Si una de las copias ya tenia nombres puestos, se los pasa a las demas: son la
misma foto, y buscar a una persona tiene que encontrarla en las dos. De ahi en
adelante lo hace solo el trigger de fotos_copias.sql.

EN EL PULSO, CADA HORA (22/10/2026)

Se habia corrido a mano y nunca mas: las fotos que fueron llegando quedaron
sin mirar (~35.400 candidatas por tamaño sin dedupe_key). Ahora es un paso del
pulso ("duplicados") y cada vuelta mira dos cosas:

  1. LO NUEVO: las fotos de Drive sin dedupe_key que entraron en las ultimas
     VENTANA_H horas (indexed_at), contra TODO el archivo con su mismo tamaño.
     La ventana se superpone de una vuelta a la otra a proposito (una vuelta
     del pulso puede durar 5 horas): si una no corre, la siguiente igual lo ve.
  2. EL BARRIDO: todo el archivo, en orden de (bytes, ancho, alto), un tramo
     por vuelta (TOPE_BARRIDO consultas o TOPE_BARRIDO_MIN minutos). Es lo que
     se junto antes de que el paso existiera (154.441 fotos el 23/10/2026) y
     la red de seguridad de lo que lo nuevo no vio.

EL CURSOR DEL BARRIDO (23/10/2026). Un grupo del mismo tamaño cuyo md5 da
distinto no recibe dedupe_key: si el barrido se guiara por "lo que no tiene
clave", lo volveria a preguntar en cada vuelta y no terminaria nunca. Por eso
avanza por un orden estable y guarda hasta donde llego en la base, en
corridas (tarea 'duplicados', etiqueta 'barrido'), no en un archivo: el
runner del pulso arranca vacio. El cursor solo se mueve si la escritura salio
entera y hasta antes del primer grupo con una consulta fallida: lo que no se
guardo vuelve a salir. Al llegar al final anota "terminado"; desde ahi las
vueltas son solo lo nuevo (TOPE_CONSULTAS, TOPE_MIN), y a los REPASO_DIAS el
barrido arranca solo otra vez desde el principio (una pasada de repaso, que
ya no tiene casi nada que preguntar).

Se cortan grupos enteros, nunca la mitad de uno.

Ningun nombre de persona en este archivo: se publica en el repo publico.
"""
import argparse
import datetime as dt
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import sb
from sync_drive import Drive

HERE = os.path.dirname(os.path.abspath(__file__))
SA_PATH = os.path.join(HERE, "service-account.json")

CAMPOS = ("id,folder_id,name,size_bytes,width,height,source,is_primary,"
          "dedupe_key,year,indexed_at")

# Lo nuevo: lo que entro en estas horas (ver EN EL PULSO).
VENTANA_H = 24
# El paso no se puede comer la vuelta. Sin barrido pendiente:
TOPE_CONSULTAS = 4000
TOPE_MIN = 10
# Con barrido: ~20 min de Drive a 18 consultas por segundo (medido el
# 23/10/2026); con la lectura y la escritura la vuelta queda en ~25 min.
TOPE_BARRIDO = 20000
TOPE_BARRIDO_MIN = 19
REPASO_DIAS = 7
CURSOR = {"tarea": "duplicados", "etiqueta": "barrido"}
# Lecturas a Drive a la vez: el punto medido en sync_drive.py (30/9/2026).
HILOS = 8
# Si falla mas que esto (y al menos ERRORES_MIN), no son fotos rotas: es Drive o
# la red. No se escribe nada y el paso falla (lo mismo que escenas_procesar.py).
ERRORES_MAX = 0.5
ERRORES_MIN = 50
LOTE = 100


def es_md5(k):
    """dedupe_key puesto por este script (el md5 del archivo), no el nombre de
    archivo normalizado que pone index_zenfolio.py."""
    return bool(k) and len(k) == 32 and all(c in "0123456789abcdef" for c in k)


def candidatas(fotos):
    """Grupos que comparten bytes y medidas. Propone, no decide."""
    g = defaultdict(dict)
    for p in fotos:
        try:
            if p.get("size_bytes") and p.get("width") and p.get("height"):
                g[(int(p["size_bytes"]), int(p["width"]), int(p["height"]))][p["id"]] = p
        except (TypeError, ValueError):
            # una fila rara no tira el lote
            continue
    return {k: list(v.values()) for k, v in g.items() if len(v) > 1}


def libre(p):
    """Una foto de Drive que este script puede unir: sin clave o con un md5.

    LAS QUE YA TIENEN OTRA CLAVE NO SE TOCAN (23/10/2026). 98.951 fotos de
    Drive tienen un dedupe_key que no es md5 (el nombre de archivo con que se
    las cruzo con Zenfolio y con otras copias), 16.943 de ellas como copia. El
    ensayo las preguntaba, les pisaba la clave con el md5 -cortando el lazo con
    su gemela de Zenfolio- y subia a primaria 293 por vuelta que eran copia de
    una de Zenfolio: el mismo duplicado al reves."""
    return p.get("source") == "drive" and (not p.get("dedupe_key") or es_md5(p["dedupe_key"]))


def a_preguntar(grupo):
    """De un grupo por tamaño, las fotos de Drive cuyo md5 no se sabe todavia.
    Vacio si no hay nada que hacer: menos de dos libres, o todas ya unidas."""
    drive = [p for p in grupo if libre(p)]
    if len(drive) < 2:
        return []
    return [p["id"] for p in drive if not es_md5(p.get("dedupe_key"))]


class Preguntador:
    """El md5 de cada foto, a Drive, con tope de consultas y de tiempo."""

    def __init__(self, tope, hasta):
        self.d = Drive(SA_PATH)
        self.tope = tope
        self.hasta = hasta
        self.md5 = {}
        self.no_estan = 0      # 404: se borro de Drive y la base no se entero
        self.errores = 0
        self.fallidas = set()
        self.consultas = 0
        self.segundos = 0.0

    def uno(self, fid):
        try:
            r = self.d._pedir("GET", "files/" + fid, ok_404=True,
                              params={"fields": "md5Checksum", "supportsAllDrives": "true"})
        except Exception:
            return fid, "error"
        if r is None:
            return fid, "404"
        return fid, r.get("md5Checksum")

    def preguntar(self, ids):
        ids = [x for x in ids if x not in self.md5]
        t0 = time.time()
        with ThreadPoolExecutor(HILOS) as pool:
            for fid, m in pool.map(self.uno, ids):
                self.consultas += 1
                if m == "error":
                    self.errores += 1
                    self.fallidas.add(fid)
                    continue
                if m == "404":
                    self.no_estan += 1
                    m = None
                self.md5[fid] = m
        self.segundos += time.time() - t0

    def lleno(self, cuantas):
        return self.consultas + cuantas > self.tope or time.time() > self.hasta


def fecha(iso):
    try:
        return dt.datetime.fromisoformat(str(iso).replace("Z", "+00:00")) if iso else None
    except ValueError:
        return None


def clave_txt(k):
    return "%d:%d:%d" % k if k else None


def txt_clave(s):
    try:
        return tuple(int(x) for x in s.split(":")) if s else None
    except ValueError:
        return None


def leer_cursor(cli):
    """El estado del barrido, de la base: {"cursor", "terminado", "pasada"}."""
    filas = cli.select("corridas", select="mensaje", tarea="eq.%s" % CURSOR["tarea"],
                       etiqueta="eq.%s" % CURSOR["etiqueta"])
    try:
        return json.loads(filas[0]["mensaje"] or "{}") if filas else {}
    except ValueError:
        return {}


def anotar_cursor(cli, estado, total, hechas, encontradas):
    """estado listo y no corriendo: salud() marca trabada una corrida que dice
    'corriendo' y no se mueve en dos horas."""
    cli.upsert("corridas", [dict(CURSOR, estado="listo", total=total, hechas=hechas,
                                 encontradas=encontradas, con_gente=0,
                                 mensaje=json.dumps(estado, ensure_ascii=False),
                                 actualizado_at=dt.datetime.now(dt.timezone.utc).isoformat())],
               on_conflict="tarea,etiqueta")


def leer_nuevas(cli, ahora, horas):
    """Las fotos de Drive sin dedupe_key que entraron en las ultimas horas, y
    todas las del archivo que tienen su mismo tamaño (de cualquier año)."""
    desde = (ahora - dt.timedelta(hours=horas)).strftime("%Y-%m-%dT%H:%M:%SZ")
    nuevas = cli.select("photos", select=CAMPOS, source="eq.drive",
                        dedupe_key="is.null", indexed_at="gte.%s" % desde)
    tamanos = sorted({int(p["size_bytes"]) for p in nuevas if p.get("size_bytes")})
    mismas = []
    for i in range(0, len(tamanos), 150):
        mismas += cli.select("photos", select=CAMPOS,
                             size_bytes="in.(%s)" % ",".join(str(x) for x in tamanos[i:i + 150]))
    return nuevas, mismas


def elegir(grupo, con_gente, md5, con_caras=()):
    """Cual de las copias se queda como la buena."""
    return sorted(grupo, key=lambda p: (
        p["id"] not in con_gente,                                  # la que ya tiene nombres
        p["id"] not in con_caras,                                  # la que tiene caras o propuestas
        not (p.get("dedupe_key") == md5 and p.get("is_primary")),  # la que ya era la buena
        p["source"] != "drive",                                    # Drive antes que Zenfolio
        p.get("year") is None,                                     # la que tiene año
        p.get("indexed_at") or "9999",                             # la que estaba primero
        -((p["width"] or 0) * (p["height"] or 0)),                 # la de mas pixeles
        p["id"]))[0]                                               # y algo estable para empatar


def ids_q(ids):
    return "in.(%s)" % ",".join('"%s"' % x for x in ids)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--todo", action="store_true",
                    help="todo el archivo de una vez, agrupado entre todos los años, sin tope")
    ap.add_argument("--retratos", action="store_true",
                    help="solo las carpetas de retrato, que es donde molesta en el juego")
    ap.add_argument("--anio", type=int, default=0)
    ap.add_argument("--ventana-h", type=float, default=VENTANA_H)
    ap.add_argument("--tope-consultas", type=int, default=None,
                    help="default %d en la vuelta del pulso; sin tope en --todo/--anio/--retratos" % TOPE_CONSULTAS)
    ap.add_argument("--tope-min", type=float, default=None)
    ap.add_argument("--ejemplos", type=int, default=5)
    ap.add_argument("--desde", default=None,
                    help="ensayo: barrer desde este cursor (bytes:ancho:alto; '' = el principio) "
                         "en vez del anotado en la base")
    ap.add_argument("--barrer", action="store_true",
                    help="barrer aunque el barrido este terminado")
    a = ap.parse_args()
    barrido = False

    t0 = time.time()
    ahora = dt.datetime.now(dt.timezone.utc)
    cli = sb.SB()

    # --- que se mira -------------------------------------------------------
    # prioridad: los grupos de "lo nuevo" van primero; si el tope corta, corta
    # en el año de turno, que vuelve a tocar.
    tandas = []   # (etiqueta, grupos por tamaño)
    leidas = 0
    if a.retratos:
        folders = [x["id"] for x in cli.select("folders", select="id", is_mugshot="is.true")]
        fotos = []
        for i in range(0, len(folders), 40):
            fotos += cli.select("photos", select=CAMPOS,
                                folder_id="in.(%s)" % ",".join(folders[i:i + 40]))
        tandas.append(("retratos", candidatas(fotos)))
        leidas = len(fotos)
        filas_leidas = fotos
    elif a.anio:
        fotos = cli.select("photos", select=CAMPOS, year="eq.%d" % a.anio)
        tandas.append((str(a.anio), candidatas(fotos)))
        leidas = len(fotos)
        filas_leidas = fotos
    elif a.todo:
        fotos = cli.select("photos", select=CAMPOS)
        tandas.append(("todo", candidatas(fotos)))
        leidas = len(fotos)
        filas_leidas = fotos
    else:
        # ¿barrido o solo lo nuevo? Lo dice el cursor en la base.
        est = leer_cursor(cli)
        if a.desde is not None:
            est = {"cursor": a.desde or None, "pasada": est.get("pasada") or 1}
        terminado = fecha(est.get("terminado"))
        barrido = a.barrer or not terminado or \
            ahora - terminado > dt.timedelta(days=REPASO_DIAS)
        if barrido and terminado:
            est = {"cursor": None, "pasada": int(est.get("pasada") or 1) + 1}
        desde = (ahora - dt.timedelta(hours=a.ventana_h)).strftime("%Y-%m-%dT%H:%M:%S")
        if barrido:
            fotos = cli.select("photos", select=CAMPOS)
            nuevas = [p for p in fotos if p.get("source") == "drive" and not p.get("dedupe_key")
                      and (p.get("indexed_at") or "")[:19] >= desde]
            todas = candidatas(fotos)
            filas_leidas = fotos
        else:
            nuevas, filas_leidas = leer_nuevas(cli, ahora, a.ventana_h)
            todas = candidatas(filas_leidas)
        ids_nuevas = {p["id"] for p in nuevas}
        g = {k: v for k, v in todas.items() if any(p["id"] in ids_nuevas for p in v)}
        print("lo nuevo: %d fotos de Drive sin clave en las ultimas %g h"
              % (len(nuevas), a.ventana_h), flush=True)
        tandas.append(("nuevas", g))
        if barrido:
            cur = txt_clave(est.get("cursor"))
            print("barrido: pasada %s, desde %s" % (est.get("pasada") or 1,
                                                    clave_txt(cur) or "el principio"), flush=True)
            tandas.append(("barrido", {k: v for k, v in todas.items()
                                       if cur is None or k > cur}))
        else:
            print("barrido: terminado el %s; el proximo arranca a los %d dias"
                  % (est.get("terminado"), REPASO_DIAS), flush=True)
        leidas = len(filas_leidas)
    t_leer = time.time() - t0

    # en la vuelta del pulso siempre hay tope; a mano, solo si se lo pide
    vuelta = not (a.todo or a.anio or a.retratos)
    largo = vuelta and barrido
    tope_c = a.tope_consultas if a.tope_consultas is not None else (
        (TOPE_BARRIDO if largo else TOPE_CONSULTAS) if vuelta else 10 ** 9)
    tope_m = a.tope_min if a.tope_min is not None else (
        (TOPE_BARRIDO_MIN if largo else TOPE_MIN) if vuelta else None)
    hasta = t0 + tope_m * 60 if tope_m else float("inf")
    pr = Preguntador(tope_c, hasta)

    vistos = set()
    pendientes_grupos = pendientes_fotos = 0
    reales = []    # (md5, [fotos])
    cand_total = fotos_cand = 0
    orden_barrido, hechos_barrido = [], set()
    for etiqueta, grupos in tandas:
        claves = [k for k in grupos if k not in vistos]
        if etiqueta == "barrido":
            claves.sort()            # el orden del cursor
        else:
            random.shuffle(claves)   # que el tope no deje siempre la misma cola
        n_grupos = n_preg = 0
        lote, lote_grupos = [], []
        hechos = []
        cortado = False

        def cerrar_lote():
            pr.preguntar(lote)
            hechos.extend(g for _, g in lote_grupos)
            if etiqueta == "barrido":
                hechos_barrido.update(k for k, _ in lote_grupos)
            lote.clear()
            lote_grupos.clear()

        # Con barrido, lo nuevo tiene su propio tope (TOPE_CONSULTAS): el
        # 23/10/2026 entraron 9.616 fotos de discos externos en un dia y se
        # comian toda la vuelta, sin que el barrido avanzara un grupo.
        tope_tanda = pr.consultas + min(TOPE_CONSULTAS, tope_c) \
            if (largo and etiqueta == "nuevas") else None
        for k in claves:
            g = grupos[k]
            preg = a_preguntar(g)
            if not preg:
                vistos.add(k)
                continue
            if tope_tanda is not None and pr.consultas + len(lote) + len(preg) > tope_tanda:
                # no se marca como visto: el barrido lo toma en su orden
                continue
            vistos.add(k)
            if etiqueta == "barrido":
                orden_barrido.append(k)
            n_grupos += 1
            n_preg += len(preg)
            # en el barrido, el primer grupo que no entra corta: el cursor
            # no puede saltear nada
            if cortado or pr.lleno(len(lote) + len(preg)):
                cortado = cortado or etiqueta == "barrido"
                pendientes_grupos += 1
                pendientes_fotos += len(preg)
                continue
            lote.extend(preg)
            lote_grupos.append((k, g))
            if len(lote) >= 400:
                cerrar_lote()
        if lote:
            cerrar_lote()
        cand_total += n_grupos
        fotos_cand += n_preg

        n_reales = 0
        for g in hechos:
            sub = defaultdict(list)
            for p in g:
                if not libre(p):
                    continue
                m = p["dedupe_key"] if es_md5(p.get("dedupe_key")) else pr.md5.get(p["id"])
                if m:
                    sub[m].append(p)
            for m, v in sub.items():
                if len(v) > 1 and any(p.get("dedupe_key") != m for p in v):
                    reales.append((m, v))
                    n_reales += 1
        print("%-12s %5d grupos por tamaño con algo que preguntar (%d fotos) -> %d grupos con copias"
              % (etiqueta, n_grupos, n_preg, n_reales), flush=True)

    # hasta donde llego el barrido: el ultimo grupo en orden tal que todos
    # los de antes se preguntaron sin ninguna consulta fallida
    nuevo_est = None
    if barrido and vuelta:
        cur = txt_clave(est.get("cursor"))
        termino = True
        for k in orden_barrido:
            if k not in hechos_barrido or any(x in pr.fallidas for x in a_preguntar(todas[k])):
                termino = False
                break
            cur = k
        nuevo_est = {"pasada": est.get("pasada") or 1, "empezo": est.get("empezo") or ahora.isoformat()}
        if termino:
            nuevo_est.update(cursor=None, terminado=ahora.isoformat())
            print("Barrido: termina en esta vuelta.")
        else:
            nuevo_est.update(cursor=clave_txt(cur))
            print("Barrido: el cursor quedaria en %s; faltan %d grupos (%d fotos), ~%d vueltas mas"
                  % (clave_txt(cur), pendientes_grupos, pendientes_fotos,
                     -(-pendientes_fotos // max(tope_c, 1))))

    if pr.consultas:
        print("Drive: %d consultas en %.0f s (%.1f por segundo), %d ya no estan en Drive, %d errores"
              % (pr.consultas, pr.segundos, pr.consultas / max(pr.segundos, 0.01),
                 pr.no_estan, pr.errores), flush=True)
    if pendientes_grupos:
        print("Quedaron para otra vuelta por el tope: %d grupos (%d fotos)"
              % (pendientes_grupos, pendientes_fotos))

    if pr.errores >= ERRORES_MIN and pr.errores > ERRORES_MAX * max(pr.consultas, 1):
        sys.exit("Fallaron %d de %d consultas a Drive: no es una foto rota, es Drive o la red. "
                 "No se escribe nada." % (pr.errores, pr.consultas))

    # --- grupos que se quedaron sin primaria -------------------------------
    # Si la primaria de un grupo se borra de Drive, sync_drive.py la saca de
    # photos y las copias, todas is_primary=false, desaparecen del juego y de
    # confirmar. Se mira en lo que esta vuelta ya leyo: un grupo con md5 cuyas
    # fotos leidas son todas copias se trae entero, y si de verdad no le queda
    # ninguna primaria se sube una (con la misma regla de elegir).
    sospechosas = defaultdict(list)
    for p in filas_leidas:
        if es_md5(p.get("dedupe_key")):
            sospechosas[p["dedupe_key"]].append(p)
    sospechosas = sorted(m for m, v in sospechosas.items()
                         if not any(p.get("is_primary") is not False for p in v))
    huerfanos = []
    for i in range(0, len(sospechosas), LOTE):
        g = defaultdict(list)
        for r in cli.select("photos", select=CAMPOS,
                            dedupe_key=ids_q(sospechosas[i:i + LOTE])):
            g[r["dedupe_key"]].append(r)
        huerfanos += [(m, v) for m, v in g.items()
                      if not any(p.get("is_primary") is not False for p in v)]
    if huerfanos:
        print("Grupos sin ninguna primaria (se les sube una): %d" % len(huerfanos))
    reales += huerfanos

    # --- que cambiaria -----------------------------------------------------
    ids = sorted({p["id"] for _, v in reales for p in v})
    con_gente = defaultdict(set)
    for i in range(0, len(ids), LOTE):
        for r in cli.select("photo_people", select="photo_id,person_id",
                            photo_id=ids_q(ids[i:i + LOTE])):
            con_gente[r["photo_id"]].add(int(r["person_id"]))
    # quien tiene caras: propuestas (face_suggestions) o caras sin grupo con
    # recuadro (caras_mapa). Ver CUAL SE QUEDA.
    propuestas = defaultdict(set)
    con_caras = set()
    rechazos = set()
    for i in range(0, len(ids), LOTE):
        trozo = ids_q(ids[i:i + LOTE])
        for r in cli.select("face_suggestions", select="photo_id,person_id", photo_id=trozo):
            propuestas[r["photo_id"]].add(int(r["person_id"]))
        for r in cli.select("caras_mapa", select="photo_id", photo_id=trozo):
            con_caras.add(r["photo_id"])
        for r in cli.select("photo_people_rejected", select="photo_id,person_id", photo_id=trozo):
            rechazos.add((r["photo_id"], int(r["person_id"])))
    con_caras |= set(propuestas)

    def pendientes(pid):
        return [q for q in propuestas.get(pid, ())
                if q not in con_gente.get(pid, ()) and (pid, q) not in rechazos]

    claves, bajar, subir, filas = [], [], [], []
    en_copias = sin_en_primaria = 0
    for m, v in reales:
        buena = elegir(v, con_gente, m, con_caras)
        for p in v:
            if p["id"] != buena["id"]:
                n = len(pendientes(p["id"]))
                en_copias += n
                if not propuestas.get(buena["id"]):
                    sin_en_primaria += n
        claves.append((m, [p["id"] for p in v if p.get("dedupe_key") != m]))
        bajar += [p["id"] for p in v if p["id"] != buena["id"] and p.get("is_primary") is not False]
        if buena.get("is_primary") is False:
            subir.append(buena["id"])
        juntos = set()
        for p in v:
            juntos |= con_gente.get(p["id"], set())
        filas += [{"photo_id": p["id"], "person_id": q, "votes": 1, "source": "dedupe"}
                  for p in v for q in juntos if q not in con_gente.get(p["id"], set())]

    copias = sum(len(v) - 1 for _, v in reales)
    print("\nLeidas %d fotos en %.0f s. %d grupos con copias (%d copias de mas)."
          % (leidas, t_leer, len(reales), copias))
    print("Escritura: dedupe_key en %d fotos (%d grupos), is_primary=false en %d, "
          "is_primary=true en %d, %d nombres copiados."
          % (sum(len(x) for _, x in claves), len(claves), len(bajar), len(subir), len(filas)))
    print("Propuestas pendientes que quedan sobre copias (no se borran): %d; de esas, en "
          "copias cuya primaria no tiene ninguna propuesta: %d." % (en_copias, sin_en_primaria))

    if not a.aplicar:
        nombre_carpeta = {}
        if a.ejemplos and reales:
            fids = sorted({p["folder_id"] for _, v in reales[:a.ejemplos] for p in v})
            for r in cli.select("folders", select="id,title", id=ids_q(fids)):
                nombre_carpeta[r["id"]] = r.get("title") or ""
        for m, v in reales[:a.ejemplos]:
            print("   -- %s" % m[:12])
            for p in v:
                print("      %-32s  %s" % ((p.get("name") or "")[:32],
                                           nombre_carpeta.get(p["folder_id"], "")[:40]))
        if nuevo_est:
            print("Ensayo: el cursor no se anota. Para seguir: --desde \"%s\""
                  % (nuevo_est.get("cursor") or ""))
        print("Ensayo: no se escribio nada (%.0f s). Agregar --aplicar." % (time.time() - t0))
        return

    # --- escribir ------------------------------------------------------------
    # El orden importa: primero la clave -que no esconde nada- y despues se
    # bajan las otras. Al reves, un corte entre las dos escrituras dejaria al
    # grupo sin ninguna primaria y esas fotos desaparecerian del juego. Un
    # grupo que falla no tira los demas; al final, si algo fallo, el paso
    # falla (el pulso lo reintenta y salud() lo ve).
    fallas = 0
    tw = time.time()

    def poner_clave(par):
        m, sin_clave = par
        try:
            cli.update("photos", {"dedupe_key": m}, id=ids_q(sin_clave))
            return None
        except Exception as e:
            return "  no se pudo poner la clave %s: %s" % (m[:12], str(e)[:160])

    # una por grupo (cada uno tiene su md5): de a HILOS a la vez, porque en el
    # barrido son ~10.000 por vuelta
    with ThreadPoolExecutor(HILOS) as pool:
        for err in pool.map(poner_clave, [c for c in claves if c[1]]):
            if err:
                fallas += 1
                print(err)
    print("dedupe_key: %d grupos en %.0f s" % (sum(1 for c in claves if c[1]), time.time() - tw),
          flush=True)
    for i in range(0, len(subir), LOTE):
        try:
            cli.update("photos", {"is_primary": True}, id=ids_q(subir[i:i + LOTE]))
        except Exception as e:
            fallas += 1
            print("  no se pudo subir un lote: %s" % str(e)[:160])
    for i in range(0, len(bajar), LOTE):
        trozo = bajar[i:i + LOTE]
        try:
            cli.update("photos", {"is_primary": False}, id=ids_q(trozo))
        except Exception as e:
            fallas += 1
            print("  no se pudo bajar un lote: %s" % str(e)[:160])
    # Las propuestas de las copias NO se borran (23/10/2026, "nada se borra"):
    # las funciones que leen face_suggestions filtran is_primary
    # (sql/propuestas_solo_primarias.sql), asi que una copia no aparece en
    # ninguna cola ni contador; y si vuelve a ser primaria, las tiene.
    # lo que ya se contesto en cualquiera de las copias vale para todas
    try:
        cli.upsert("photo_people", filas, on_conflict="photo_id,person_id")
    except Exception as e:
        fallas += 1
        print("  no se pudieron copiar los nombres: %s" % str(e)[:160])

    # El cursor, al final y solo si todo se guardo: si no, la vuelta siguiente
    # retoma desde el anterior y vuelve a ver estos grupos (un error no se tapa).
    if nuevo_est and not fallas:
        try:
            anotar_cursor(cli, nuevo_est, pendientes_fotos, fotos_cand, len(reales))
        except Exception as e:
            fallas += 1
            print("  no se pudo anotar el cursor: %s" % str(e)[:160])

    print("Listo en %.0f s: %d grupos, %d copias de mas, %d nombres copiados, %d fallas."
          % (time.time() - t0, len(reales), copias, len(filas), fallas))
    if fallas:
        sys.exit(1)


if __name__ == "__main__":
    main()
