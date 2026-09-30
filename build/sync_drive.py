# -*- coding: utf-8 -*-
"""
Sync incremental de las fuentes de fotos de la biblioteca (tabla fuentes_fotos).

Hasta el 14/9/2026 solo existia la parte de la unidad Server Media (North), y la
carpeta de Quilmes ("FOTOS POR AÑO", un My Drive compartido con la cuenta de
servicio) no se sincronizaba: un album nuevo de Quilmes no aparecia solo. Ahora
se recorren todas las fuentes con sincroniza = true del panel de Fuentes:
  - unidad compartida -> Drive Changes API (lo de abajo, sync_unidad)
  - carpeta comun     -> por fecha y por niveles (sync_carpeta)
Cada fuente guarda su estado en sync_state ('drive' para Server Media, que ya
tenia su token; 'drive:<id>' para las demas).

Sobre la unidad Server Media contra la Drive Changes API:

Por que no un crawl: recorrer las 7.449 carpetas cada semana tarda una hora y
no detecta borrados. La Changes API devuelve exactamente lo que cambio desde
el ultimo token, incluidos los archivos que se fueron.

El token vive en la tabla sync_state de Supabase, no en disco: esto corre en
GitHub Actions, que arranca de cero cada vez.

La primera corrida no tiene token. En vez de sincronizar el vacio, pide el
token de arranque y sale sin tocar nada: lo que ya esta cargado vino del
crawl completo, y desde ahi en adelante alcanza con los cambios.

PUSH DE DRIVE (30/9/2026, docs/drive-push.md). Al final de cada corrida se
crean o renuevan los canales de changes.watch (uno por unidad y uno de la
cuenta de servicio) que avisan a la Edge Function drive-aviso, y si llego un
aviso mientras la pasada corria se da otra (hasta 5). Sin DRIVE_AVISO_TOKEN en
el ambiente, o sin las columnas canal_* en sync_state, no se hace nada de eso
y el sync sigue como antes.

POR QUE ES RAPIDO (30/9/2026). Una corrida sin novedades tardaba ~207 s y no
era trabajo: eran ~770 llamadas HTTPS en fila, cada una con conexion nueva.
Ahora las conexiones se reusan (una sesion por hilo), sync_state se lee una
vez por pasada y se escribe una vez, Server Media pregunta que carpetas faltan
en un solo viaje (rpc carpetas_faltantes, con la cuenta vieja si la funcion no
existe) y las fuentes tipo carpeta se leen en paralelo (HILOS). Las
escrituras de esas fuentes siguen siendo de a una, como antes.

Uso:
    python sync_drive.py                # incremental
    python sync_drive.py --bootstrap    # solo fija el token, no sincroniza
    python sync_drive.py --dry-run      # muestra que haria
"""
import argparse
import datetime as _dt
import json
import os
import sys
import threading
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests
from google.oauth2 import service_account
import google.auth.transport.requests as gart

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Las claves salen de build/.env, igual que en todo el resto. Esto nacio para
# GitHub Actions, donde vienen de secrets, y a mano moria con "Faltan
# SUPABASE_URL y SUPABASE_SERVICE_KEY" aunque el .env estuviera al lado.
# load_env usa setdefault, asi que lo que ya este en el ambiente manda.
from sb import DEFAULT_URL, load_env
from parse_folders import (parse_date, parse_levels, parse_event, parse_site,
                           noise_kind, term, norm)
from parse_photos import exif_iso
from classify import classify

ROOT_ID = "0ADoh-DIvUMYeUk9PVA"
FOLDER_MIME = "application/vnd.google-apps.folder"
HERE = os.path.dirname(os.path.abspath(__file__))

FIELDS = ("nextPageToken,newStartPageToken,changes("
          "removed,fileId,file(id,name,mimeType,parents,trashed,webViewLink,"
          "modifiedTime,size,imageMediaMetadata(width,height,time)))")


load_env()


def env(name, *alts):
    for k in (name,) + alts:
        v = os.environ.get(k)
        if v:
            return v
    return None


# UNA SESION POR HILO (30/9/2026). urllib abria una conexion TLS nueva por
# llamada; con ~770 llamadas por corrida eso era casi todo el tiempo. Una
# requests.Session reusa la conexion, pero no se comparte entre hilos: cada hilo
# tiene la suya.
_HILO = threading.local()


def _sesion(nombre):
    s = getattr(_HILO, nombre, None)
    if s is None:
        s = requests.Session()
        setattr(_HILO, nombre, s)
    return s


# Lo que un hilo imprime se junta y sale entero al terminar su fuente: si no,
# las lineas de ocho fuentes en paralelo quedan mezcladas en el log.
def _log(msg=""):
    buf = getattr(_HILO, "buf", None)
    if buf is None:
        print(msg)
    else:
        buf.append(msg)


class Drive:
    RED = (requests.exceptions.RequestException, OSError)

    def __init__(self, sa_path):
        self.creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=["https://www.googleapis.com/auth/drive.readonly"])
        self._lock = threading.Lock()
        self.expires = 0
        self._token()

    def _token(self):
        """El token vigente; lo renueva uno solo aunque pregunten ocho hilos.

        LA RENOVACION TAMBIEN REINTENTA. Las llamadas a la API reintentaban y
        la renovacion no: un SSLEOFError de un segundo al renovar mato el
        nocturno entero a los 8 minutos."""
        with self._lock:
            if time.time() > self.expires:
                for i in range(6):
                    try:
                        self.creds.refresh(gart.Request())
                        break
                    except Exception:
                        if i == 5:
                            raise
                        time.sleep(min(2 ** i, 30))
                self.expires = time.time() + 3000
            return self.creds.token

    def _pedir(self, method, path, body=None, params=None, ok_404=False):
        url = "https://www.googleapis.com/drive/v3/" + path
        for i in range(6):
            h = {"Authorization": "Bearer " + self._token()}
            try:
                r = _sesion("drive").request(method, url, params=params, json=body,
                                             headers=h, timeout=90)
            except self.RED:
                if i < 5:
                    time.sleep(min(2 ** i, 30))
                    continue
                raise
            if r.status_code < 300:
                return r.json() if r.content else None
            if ok_404 and r.status_code == 404:
                return None
            if r.status_code in (403, 429, 500, 502, 503) and i < 5:
                time.sleep(2 ** i)
                continue
            raise RuntimeError("Drive %s en %s: %s" % (r.status_code, path, r.text[:400]))

    def get(self, path, **params):
        return self._pedir("GET", path, params=params)

    def post(self, path, body, ok_404=False, **params):
        return self._pedir("POST", path, body=body, params=params, ok_404=ok_404)


class NoExiste(RuntimeError):
    """PostgREST no encuentra la tabla, la columna o la funcion: lo del paso 1
    de sql/drive_aviso.sql todavia no esta aplicado."""


# codigos de PostgREST y de Postgres para "eso no existe"
_NO_EXISTE = ("PGRST202", "PGRST204", "PGRST205", "42P01", "42703", "42883")


class Supa:
    RED = (requests.exceptions.RequestException, OSError)

    def __init__(self, url, key):
        self.url = url.rstrip("/")
        self.h = {"apikey": key, "Authorization": "Bearer " + key,
                  "Content-Type": "application/json"}

    def _call(self, method, path, body=None, prefer=None, timeout=240):
        h = dict(self.h)
        if prefer:
            h["Prefer"] = prefer
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        for i in range(4):
            try:
                r = _sesion("supa").request(method, self.url + "/rest/v1/" + path,
                                            data=data, headers=h, timeout=timeout)
            except self.RED:
                if i < 3:
                    time.sleep(2 * (i + 1))
                    continue
                raise
            if r.status_code < 300:
                return r.json() if r.content else None
            detalle = r.text[:300]
            if any(c in detalle for c in _NO_EXISTE):
                raise NoExiste("%s %s -> %s %s" % (method, path.split("?")[0],
                                                   r.status_code, detalle))
            # lo que es del pedido (400, 401, 404, 409) no cambia reintentando
            if r.status_code in (408, 429) or r.status_code >= 500:
                if i < 3:
                    time.sleep(2 * (i + 1))
                    continue
            raise RuntimeError("%s %s -> %s %s" % (method, path, r.status_code, detalle))

    def select(self, path):
        return self._call("GET", path)

    def upsert(self, table, rows, on_conflict="id"):
        if not rows:
            return
        for i in range(0, len(rows), 500):
            self._call("POST", "%s?on_conflict=%s" % (table, on_conflict),
                       rows[i:i + 500],
                       "resolution=merge-duplicates,return=minimal")

    def rpc_filas(self, fn, args, orden="id"):
        """Una rpc que devuelve filas, entera. PostgREST corta en 1.000 sin
        avisar: fuente_carpetas de "FOTOS POR AÑO" (1.218 carpetas) volvia con
        1.000, y el sync tomaba las otras 218 por desconocidas (21/9/2026)."""
        filas, off = [], 0
        while True:
            pag = self.rpc("%s?order=%s&limit=1000&offset=%d" % (fn, orden, off), args) or []
            filas += pag
            if len(pag) < 1000:
                return filas
            off += 1000

    def rpc(self, fn, args):
        return self._call("POST", "rpc/" + fn, args, timeout=300)


# ---------------------------------------------------------------- mapeo

def campus_de(sede):
    """La sede de la fuente, tal como va en cada carpeta.

    'Ambas' no es una sede: es una carpeta que tiene fotos de los dos campus.
    Decision del 16/9/2026, sobre las de Alumni: "hay de los dos campus". Estamparle
    North o Quilmes seria repartir mal la mitad de las fotos, y eso no se nota
    nunca. Va nulo, que es como la app ya escribe "no se de que sede es":
    labeling_by_kind y confirmar_candidatas filtran con
    "p_campus is null or f.campus = p_campus", y el emparejador trata el nulo
    como compatible con cualquiera ("a.campus is null or b.campus is null").
    """
    return None if (sede or "").strip().lower() == "ambas" else sede


def folder_row(f, parent_id, path_of, campus):
    """Misma derivacion de facetas que parse_folders, sobre una sola carpeta."""
    name = (f.get("name") or "").strip()
    y, mo, d, title, prec = parse_date(name)
    t = norm(title) if title else ""
    section, levels, divisions = parse_levels(t)
    types, acts, sports, trips, opponent = parse_event(t)
    et, ai_tags = classify(title)

    event_date = None
    if prec == "day" and y and mo and d:
        try:
            from datetime import date as _d
            _d(y, mo, d)
            event_date = "%04d-%02d-%02d" % (y, mo, d)
        except ValueError:
            pass
    elif prec == "month" and y and mo:
        event_date = "%04d-%02d-01" % (y, mo)

    return {
        "id": f["id"],
        "name": name,
        "parent_id": parent_id,
        # la sede de la fuente: sin esto un album nuevo quedaba sin sede y fuera
        # de los filtros y de los accesos (sql/carpetas_heredan_sede.sql)
        "campus": campus,
        "path": path_of,
        "year": y, "month": mo, "day": d,
        "event_date": event_date,
        "date_precision": prec,
        "term": term(mo),
        "title": title,
        "section": section,
        "levels": levels, "divisions": divisions,
        "site": parse_site(t),
        "event_types": types, "activities": acts,
        "sports": sports, "trips": trips,
        "opponent": opponent,
        "is_mugshot": "Retratos" in types,
        "noise": noise_kind(name),
        "ai_event_type": et,
        "ai_tags": ai_tags or [],
    }


def photo_row(f, parent_id):
    md = f.get("imageMediaMetadata") or {}
    return {
        "id": f["id"],
        "folder_id": parent_id,
        "name": f.get("name") or "",
        "mime_type": f.get("mimeType"),
        "web_view_link": f.get("webViewLink"),
        "drive_modified_time": f.get("modifiedTime"),
        "exif_time": exif_iso(md.get("time")),
        "width": md.get("width"),
        "height": md.get("height"),
        "size_bytes": int(f["size"]) if str(f.get("size") or "").isdigit() else None,
    }


# ---------------------------------------------------------------- main

def estado_id(fuente):
    """La fila de sync_state de una fuente. Server Media conserva la de siempre,
    que tiene el token de la Changes API guardado."""
    return "drive" if fuente["id"] == ROOT_ID else "drive:" + fuente["id"]


# Lo que falta del paso 1 de sql/drive_aviso.sql se avisa una vez por corrida
# y se sigue como antes: el push no puede romper el sync.
_AVISADO = set()


def avisar_una_vez(clave, msg):
    if clave not in _AVISADO:
        _AVISADO.add(clave)
        _log("  aviso: " + msg)


def faltan_en_folders(supa, ids):
    """De una lista de ids de carpetas, las que no estan en folders.

    Un viaje por cada 1.000 ids a carpetas_faltantes (el POST lleva la lista en
    el cuerpo), en vez de un GET cada 100. De a 1.000 porque PostgREST corta la
    respuesta en 1.000 filas sin avisar: con 1.000 ids nunca vuelven mas. Si la
    funcion no existe todavia, la cuenta de siempre."""
    ids = sorted(set(ids))
    if "carpetas_faltantes" not in _AVISADO:
        try:
            faltan = set()
            for i in range(0, len(ids), 1000):
                faltan |= set(supa.rpc("carpetas_faltantes", {"p_ids": ids[i:i + 1000]}) or [])
            return faltan
        except NoExiste as e:
            avisar_una_vez("carpetas_faltantes",
                           "no existe rpc carpetas_faltantes (falta sql/drive_aviso.sql): "
                           "se pregunta de a 100 como antes (%s)" % str(e)[:160])
    estan = set()
    for i in range(0, len(ids), 100):
        estan |= {r["id"] for r in supa.select(
            "folders?select=id&id=in.(%s)" % ",".join(ids[i:i + 100])) or []}
    return {c for c in ids if c not in estan}


def sync_unidad(drive, supa, fuente, a, state):
    """Una unidad compartida, por la Changes API. Devuelve si escribio algo."""
    root = fuente["id"]
    sid = estado_id(fuente)
    token = state.get("page_token")

    if a.bootstrap or not token:
        # UNA UNIDAD NUEVA SE RECORRE ENTERA (22/9/2026)
        #
        # Antes esto solo fijaba el token y decia "el contenido vino del crawl
        # completo", que era cierto para Server Media -alguien lo habia
        # indexado a mano antes- y falso para cualquier unidad que se sume
        # despues. El 22/9 se sumo "Multimedia Drive", donde ya se venian
        # cargando albumes: la primera corrida fijo el token y dejo afuera las
        # 27 carpetas que ya estaban. Desde el token solo se ve lo que cambie
        # DESPUES, asi que lo viejo no entraba nunca.
        #
        # Con --bootstrap no se recorre: esa bandera existe justo para decir
        # "no mires el contenido, solo fija el token".
        escrito = False
        if not a.bootstrap:
            print("Unidad nueva: se recorre entera…")
            carpetas, fotos = recorrer(drive, root)
            print("  %d carpetas, %d fotos" % (len(carpetas), len(fotos)))
            if a.dry_run:
                print("\n(dry-run: no se escribio nada)")
                return False

            def nivel_nuevo(cid):
                n, actual = 0, cid
                while actual in carpetas and carpetas[actual][1] in carpetas and n < 30:
                    actual = carpetas[actual][1]
                    n += 1
                return n

            # la raiz primero y sin madre, como las otras fuentes: las carpetas
            # de arriba cuelgan de ella y folders tiene FK contra si misma
            unidad = drive.get("drives/" + root, fields="id,name")
            frows = [folder_row({"id": root, "name": unidad.get("name") or fuente["nombre"]},
                                None, "", campus_de(fuente["sede"]))]
            for cid in sorted(carpetas, key=nivel_nuevo):
                f, parent = carpetas[cid]
                frows.append(folder_row(f, parent, (f.get("name") or "").strip(),
                                        campus_de(fuente["sede"])))
            supa.upsert("folders", frows)
            known = {r["id"] for r in frows}
            prows = [photo_row(f, parent) for (f, parent) in fotos.values()
                     if parent in known]
            supa.upsert("photos", prows)
            print("  escritas: %d carpetas, %d fotos" % (len(frows), len(prows)))
            escrito = bool(frows or prows)

        r = drive.get("changes/startPageToken", driveId=root,
                      supportsAllDrives="true")
        tok = r["startPageToken"]
        if not a.dry_run:
            supa.upsert("sync_state", [{
                "id": sid, "page_token": tok,
                "last_run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "last_status": "bootstrap",
                "note": ("token inicial despues del recorrido completo"
                         if escrito else "token inicial, sin recorrer (--bootstrap)"),
            }])
        print("Token inicial fijado: %s" % tok)
        print("La proxima corrida ya sincroniza solo los cambios.")
        return escrito

    print("Sync desde el token guardado…")

    changed_folders, changed_photos = {}, {}
    gone = []
    pages = 0
    new_token = None

    while True:
        r = drive.get("changes", pageToken=token, driveId=root,
                      includeItemsFromAllDrives="true",
                      supportsAllDrives="true", pageSize=1000,
                      includeRemoved="true", fields=FIELDS)
        pages += 1
        for ch in r.get("changes", []):
            fid = ch.get("fileId")
            f = ch.get("file")
            # removed, o mandado a la papelera, o sacado de la unidad
            if ch.get("removed") or not f or f.get("trashed"):
                gone.append(fid)
                continue
            parents = f.get("parents") or []
            parent = parents[0] if parents else None
            if f.get("mimeType") == FOLDER_MIME:
                changed_folders[fid] = (f, parent)
            elif (f.get("mimeType") or "").startswith("image/"):
                changed_photos[fid] = (f, parent)

        token = r.get("nextPageToken")
        if not token:
            new_token = r.get("newStartPageToken")
            break

    print("  paginas de cambios: %d" % pages)
    print("  carpetas afectadas: %d" % len(changed_folders))
    print("  fotos afectadas:    %d" % len(changed_photos))
    print("  desaparecidas:      %d" % len(gone))

    # LO QUE LA CHANGES API NO CUENTA (21/9/2026). Una carpeta MOVIDA adentro
    # de la unidad llega como un cambio suyo, pero sus fotos no: no cambiaron.
    # Asi "2022 Deportes Grupales" (de 2022, movida el 21/9) iba a quedar como
    # un album vacio. Y si el movimiento se escapo del token, ni la carpeta.
    # Es la misma red que sync_carpeta ya tenia:
    #   1. los dos primeros niveles (años y albumes) se listan enteros;
    #   2. toda carpeta de ahi o de los cambios que no este en la base se
    #      recorre entera, con sus fotos.
    solo_carpetas = "trashed=false and mimeType='%s'" % FOLDER_MIME
    arriba = {}
    anios = list(listar(drive, q="'%s' in parents and %s" % (root, solo_carpetas)))
    for anio in anios:
        arriba[anio["id"]] = (anio, root)
    # los albumes de cada año, varios años a la vez (solo lectura)
    with ThreadPoolExecutor(HILOS) as pool:
        hijos = pool.map(lambda an: list(listar(
            drive, q="'%s' in parents and %s" % (an["id"], solo_carpetas))), anios)
        for anio, albs in zip(anios, hijos):
            for alb in albs:
                arriba[alb["id"]] = (alb, anio["id"])
    candidatas = dict(arriba)
    candidatas.update(changed_folders)
    ids = sorted(candidatas)
    faltan = faltan_en_folders(supa, ids)
    for cid in sorted(faltan):
        if candidatas[cid][1] in faltan:
            continue          # la trae el recorrido de su madre
        changed_folders[cid] = candidatas[cid]
        c2, f2 = recorrer(drive, cid)
        changed_folders.update(c2)
        changed_photos.update(f2)
        # candidatas y no arriba: una carpeta que llego por los cambios no esta
        # en los dos primeros niveles, y el aviso cortaba la corrida entera con
        # KeyError (22/9/2026, Server Media)
        print("     faltaba en la base: %s (%d carpetas, %d fotos adentro)"
              % (candidatas[cid][0].get("name"), len(c2), len(f2)))

    if a.dry_run:
        print("\n(dry-run: no se escribio nada)")
        return False

    # --- carpetas primero: las fotos tienen FK contra ellas. Y la madre antes
    # que la hija: el trigger folders_hereda_de_madre saca profundidad, ruta y
    # año de la madre, y solo la ve si ya se escribio
    # (sql/carpetas_heredan_ubicacion.sql). La Changes API no trae orden.
    def nivel(fid):
        n, actual = 0, fid
        while changed_folders[actual][1] in changed_folders and n < 30:
            actual = changed_folders[actual][1]
            n += 1
        return n

    frows = []
    for fid in sorted(changed_folders, key=nivel):
        f, parent = changed_folders[fid]
        frows.append(folder_row(f, parent, (f.get("name") or "").strip(),
                                campus_de(fuente["sede"])))
    supa.upsert("folders", frows)

    # una foto nueva puede colgar de una carpeta que no cambio y que ya esta
    # cargada, pero tambien de una recien creada que quedo en frows
    known = {r["id"] for r in frows}
    if changed_photos:
        need = {p for (_, p) in changed_photos.values() if p and p not in known}
        if need:
            have = supa.select("folders?select=id&id=in.(%s)"
                               % ",".join(sorted(need))) or []
            known |= {r["id"] for r in have}

    prows, huerfanas = [], 0
    for fid, (f, parent) in changed_photos.items():
        if not parent or parent not in known:
            huerfanas += 1
            continue
        prows.append(photo_row(f, parent))
    supa.upsert("photos", prows)
    if huerfanas:
        print("  fotos sin carpeta conocida (se veran en el proximo crawl): %d" % huerfanas)

    removed = 0
    if gone:
        removed = (supa.rpc("purge_photos", {"ids": gone}) or 0)
        removed += (supa.rpc("purge_folders", {"ids": gone}) or 0)

    # el recuento de totales se hace una vez, al final de todas las fuentes (main)
    cambios = bool(frows or prows or removed)

    supa.upsert("sync_state", [{
        "id": sid,
        "page_token": new_token or token,
        "last_run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "last_status": "ok",
        "folders_seen": len(changed_folders),
        "photos_seen": len(changed_photos),
        "added": len(prows),
        "updated": len(frows),
        "removed": removed,
        "note": None,
    }])

    print("\nCarpetas escritas: %d" % len(frows))
    print("Fotos escritas:    %d" % len(prows))
    print("Eliminadas:        %d" % removed)
    print("Nuevo token guardado.")
    return cambios


LISTA_FIELDS = ("nextPageToken,files(id,name,mimeType,parents,trashed,webViewLink,"
                "modifiedTime,createdTime,size,imageMediaMetadata(width,height,time))")


def listar(drive, **params):
    """files.list paginado, en todo lo que ve la cuenta de servicio."""
    token = None
    while True:
        p = dict(params, pageSize=1000, includeItemsFromAllDrives="true",
                 supportsAllDrives="true", fields=LISTA_FIELDS)
        if token:
            p["pageToken"] = token
        r = drive.get("files", **p)
        for f in r.get("files", []):
            yield f
        token = r.get("nextPageToken")
        if not token:
            return


# UN OR DE "in parents" PIERDE CARPETAS (30/9/2026). Se probo listar los dos
# primeros niveles de todas las fuentes con "('a' in parents or 'b' in parents
# ...)": con cinco madres que tienen 33 subcarpetas, Drive devolvio 12, con y
# sin corpora=allDrives, y sin error. Cada carpeta se lista con su propia
# consulta, aunque cueste una llamada por carpeta: lo que no se lista no entra.


# LOS CAMBIOS DE DRIVE SE PIDEN UNA VEZ POR CORRIDA, NO UNA POR FUENTE (21/9/2026)
#
# Cada fuente tipo carpeta le preguntaba a Drive "que cambio desde ayer en todo
# lo que ve la cuenta" y despues se quedaba con lo suyo. Son ~70 fuentes: la
# misma lista, 70 veces. Un dia tranquilo no se nota; el 21/9 las cuentas de
# los celulares de North subieron ~2.000 fotos, cada fuente tardo ~50 s y la
# corrida se corto a los 30 minutos sin recontar los totales.
#
# Ahora la primera fuente trae la lista y las demas la filtran en memoria con
# su propio corte. Si una pide un corte mas viejo que el que se trajo, se
# vuelve a pedir desde ese (pasa poco: las fuentes corren siempre en el mismo
# orden). Las madres que se van consultando tambien se guardan: son datos de
# Drive, no dependen de la fuente.
#
# Con las fuentes en paralelo (30/9/2026) la lista se pide una vez ANTES de
# repartirlas, con el corte mas viejo, y un lock cuida que dos hilos no la
# pidan a la vez. Cada pasada empieza con la lista vacia: la de la pasada
# anterior ya no tiene lo que llego despues.
_CAMBIOS = {"corte": None, "archivos": []}
_MADRES = {}
_CAMBIOS_LOCK = threading.Lock()


def cambios_desde(drive, corte):
    """Carpetas e imagenes creadas o modificadas despues de corte (UTC, sin zona)."""
    with _CAMBIOS_LOCK:
        if _CAMBIOS["corte"] is None or corte < _CAMBIOS["corte"]:
            q = ("(modifiedTime > '%s' or createdTime > '%s') and trashed = false and "
                 "(mimeType = '%s' or mimeType contains 'image/')") % (corte, corte, FOLDER_MIME)
            _CAMBIOS["archivos"] = list(listar(drive, q=q, corpora="allDrives"))
            _CAMBIOS["corte"] = corte
            _log("  (lista de cambios de Drive traida desde %s: %d)" % (corte, len(_CAMBIOS["archivos"])))
        archivos = _CAMBIOS["archivos"]
    # Drive devuelve "2026-09-21T17:40:43.882Z": los primeros 19 caracteres se
    # comparan como texto contra el corte, que tiene el mismo formato
    return [f for f in archivos
            if (f.get("modifiedTime") or "")[:19] > corte
            or (f.get("createdTime") or "")[:19] > corte]


def corte_de(state, base=None):
    """El corte de una fuente carpeta: desde cuando mira los cambios, con una
    hora de margen (los relojes de Drive y el de la corrida no son el mismo).
    None si no hay de donde sacarlo."""
    desde = state.get("last_run_at") if state.get("last_status") == "ok" else None
    if not desde and base and state.get("last_status") != "error":
        fechas = [r["indexed_at"] for r in base if r.get("indexed_at")]
        desde = max(fechas) if fechas else None
    corte = desde or "2000-01-01T00:00:00Z"
    try:
        t = _dt.datetime.fromisoformat(str(corte).replace("Z", "+00:00")) - _dt.timedelta(hours=1)
        corte = t.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        pass
    return desde, corte


def recorrer(drive, carpeta_id):
    """Todo lo que cuelga de una carpeta: ({id: (carpeta, madre)}, {id: (foto, carpeta)})."""
    carpetas, fotos = {}, {}
    pendientes = [carpeta_id]
    while pendientes:
        c = pendientes.pop()
        for f in listar(drive, q="'%s' in parents and trashed=false" % c):
            if f.get("mimeType") == FOLDER_MIME:
                carpetas[f["id"]] = (f, c)
                pendientes.append(f["id"])
            elif (f.get("mimeType") or "").startswith("image/"):
                fotos[f["id"]] = (f, c)
    return carpetas, fotos


# Las fuentes carpeta se LEEN en paralelo; lo que escriben va de a una, como
# cuando corrian en fila: dos fuentes pueden compartir carpetas y el trigger
# folders_hereda_de_madre necesita la madre escrita antes que la hija.
_ESCRIBIR = threading.Lock()
# Lecturas a la vez. Medido el 30/9/2026 con --dry-run: 4 hilos 66 s, 8 hilos
# 60 s, 16 hilos 172 s (sin ningun 403 ni 429: la cola de Drive se estira, p90
# de 2,6 s a 8,7 s por consulta). Mas no es mejor.
HILOS = 8


def sync_carpeta(drive, supa, fuente, a, state):
    """Una carpeta comun: un My Drive compartido con la cuenta de servicio.

    La Changes API por unidad no sirve: la carpeta no es una unidad, y la de
    toda la cuenta mezclaria cualquier cosa compartida. Se combinan tres cosas:

      1. lo creado o modificado desde la ultima corrida en todo lo que ve la
         cuenta, quedandose con lo que cuelga de esta carpeta;
      2. los dos primeros niveles -años y albumes- listados enteros, porque un
         album MOVIDO adentro conserva su fecha vieja y el punto 1 no lo ve;
      3. toda carpeta que aparece y no esta en la base se recorre entera.

    Una fuente sin nada cargado se recorre completa. Una que ya vino del
    indexado inicial (Quilmes, 7/9/2026) arranca desde esa fecha. Lo que se
    borra en Drive no se detecta aca: queda para un recorrido completo.

    Devuelve (si escribio algo, la fila 'ok' de sync_state o None). La fila no
    se escribe aca: main junta las de todas las fuentes y las sube en un viaje.
    """
    root, sede, sid = fuente["id"], fuente["sede"], estado_id(fuente)
    inicio = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base = supa.rpc_filas("fuente_carpetas", {"p_id": root})
    conocidas = {r["id"] for r in base}
    # LA QUE FALLO SE RECORRE ENTERA, NO DESDE SU PROPIA MARCA.
    #
    # Las carpetas se escriben antes que las fotos. Si el POST de fotos falla,
    # las carpetas ya quedaron con indexed_at = hoy, y el atajo de abajo -"si
    # no hay corrida buena, arranca desde lo mas nuevo que ya tengo"- lee esa
    # marca y concluye que no hay nada nuevo. La fuente se declara al dia sin
    # tener una sola foto.
    #
    # Paso el 16/9/2026 con seis fuentes que se cayeron por un EXIF roto: la
    # segunda corrida dijo "0 cambios" y no las miro nunca mas. Un error que
    # se tapa solo es peor que el error.
    #
    # "Entera" quiere decir BAJANDO EL ARBOL DE LA FUENTE, que es el camino que
    # ya existe para una fuente nueva y esta acotado a sus carpetas. El primer
    # intento de este arreglo hacia otra cosa: dejaba la fecha en nulo, que cae
    # en "cambios desde 1999-12-31" y eso lista todo lo que la cuenta de
    # servicio puede ver, entero, por cada fuente. Quedo dando vueltas sin
    # escribir una foto y hubo que cortarlo.
    fallo_antes = state.get("last_status") == "error"
    if fallo_antes:
        _log("  la corrida anterior fallo: se recorre el arbol entero")
        conocidas = set()
    desde, corte = corte_de(state, base)

    carpetas, fotos = {}, {}
    if not conocidas:
        _log("  fuente nueva: se recorre entera")
        meta = drive.get("files/" + root, fields="id,name,mimeType,parents,webViewLink,modifiedTime",
                         supportsAllDrives="true")
        carpetas[root] = (meta, None)
        c2, f2 = recorrer(drive, root)
        carpetas.update(c2)
        fotos.update(f2)
    else:
        _log("  cambios desde %s" % corte)

        madres = _MADRES

        def cuelga(pid, prof=0):
            """Si esa carpeta cuelga de la fuente; de paso anota las intermedias nuevas."""
            if pid is None or prof > 12:
                return False
            if pid == root or pid in conocidas or pid in carpetas:
                return True
            if pid not in madres:
                try:
                    m = drive.get("files/" + pid,
                                  fields="id,name,mimeType,parents,webViewLink,modifiedTime",
                                  supportsAllDrives="true")
                except RuntimeError:
                    madres[pid] = (None, None)
                    return False
                madres[pid] = (m, (m.get("parents") or [None])[0])
            m, p = madres[pid]
            if m is None or not cuelga(p, prof + 1):
                return False
            if m.get("mimeType") == FOLDER_MIME:
                carpetas.setdefault(pid, (m, p))
            return True

        vistos = 0
        for f in cambios_desde(drive, corte):
            vistos += 1
            parent = (f.get("parents") or [None])[0]
            if not cuelga(parent):
                continue
            if f.get("mimeType") == FOLDER_MIME:
                carpetas[f["id"]] = (f, parent)
            else:
                fotos[f["id"]] = (f, parent)
        _log("  modificados en todo lo que ve la cuenta: %d" % vistos)

        solo_carpetas = "trashed=false and mimeType='%s'" % FOLDER_MIME
        for anio in listar(drive, q="'%s' in parents and %s" % (root, solo_carpetas)):
            if anio["id"] not in conocidas:
                carpetas.setdefault(anio["id"], (anio, root))
            for alb in listar(drive, q="'%s' in parents and %s" % (anio["id"], solo_carpetas)):
                if alb["id"] not in conocidas:
                    carpetas.setdefault(alb["id"], (alb, anio["id"]))

        nuevas = [c for c in list(carpetas) if c not in conocidas]
        for cid in nuevas:
            c2, f2 = recorrer(drive, cid)
            carpetas.update(c2)
            fotos.update(f2)

    nuevas = [c for c in carpetas if c not in conocidas]
    _log("  carpetas afectadas: %d (%d nuevas)" % (len(carpetas), len(nuevas)))
    _log("  fotos afectadas:    %d" % len(fotos))
    for c in nuevas[:10]:
        _log("     nueva: %s" % (carpetas[c][0].get("name") or c))

    if a.dry_run:
        _log("\n(dry-run: no se escribio nada)")
        return False, None

    # de arriba hacia abajo: la madre tiene que estar antes que la hija
    def profundidad(cid, vistos=None):
        n, actual = 0, cid
        while actual in carpetas and carpetas[actual][1] in carpetas and n < 30:
            actual = carpetas[actual][1]
            n += 1
        return n

    orden = sorted(carpetas, key=profundidad)
    frows = [folder_row(carpetas[c][0], carpetas[c][1], (carpetas[c][0].get("name") or "").strip(), campus_de(sede))
             for c in orden]
    known = conocidas | set(carpetas)
    prows = [photo_row(f, parent) for (f, parent) in fotos.values() if parent in known]
    with _ESCRIBIR:
        supa.upsert("folders", frows)
        supa.upsert("photos", prows)

    fila = {
        "id": sid,
        "page_token": None,
        "last_run_at": inicio,
        "last_status": "ok",
        "folders_seen": len(carpetas),
        "photos_seen": len(fotos),
        "added": len(prows),
        "updated": len(frows),
        "removed": 0,
        "note": "carpeta: cambios desde %s" % (desde or "el principio (recorrido completo)"),
    }
    _log("  escritas: %d carpetas, %d fotos" % (len(frows), len(prows)))
    return bool(frows or prows), fila


# ---------------------------------------------------------------- pasadas

def leer_estados(supa):
    """sync_state entero, una vez por pasada (antes: un GET por fuente)."""
    return {r["id"]: r for r in (supa.select("sync_state?select=*") or [])}


def anotar_error(supa, a, fuente, e):
    """El error de una fuente se escribe en el acto: la corrida siguiente la
    recorre entera (ver sync_carpeta)."""
    if a.dry_run:
        return
    try:
        supa.upsert("sync_state", [{"id": estado_id(fuente), "last_status": "error",
                                    "note": str(e)[:300]}])
    except Exception:
        pass


def _una_carpeta(drive, supa, f, a, state):
    """Una fuente carpeta dentro de un hilo: lo que imprime queda en su buffer
    y una falla vuelve como dato, no tira el lote."""
    _HILO.buf = ["\n== %s (%s, %s)" % (f["nombre"], f["sede"], f["raiz"])]
    try:
        cambios, fila = sync_carpeta(drive, supa, f, a, state)
        return f, cambios, fila, None, _HILO.buf
    except Exception as e:
        _HILO.buf.append("  FALLO: %s" % str(e)[:400])
        return f, False, None, e, _HILO.buf
    finally:
        _HILO.buf = None


def pasada(drive, supa, fuentes, a):
    """Todas las fuentes una vez. Devuelve (si algo cambio, nombres que fallaron)."""
    _CAMBIOS["corte"], _CAMBIOS["archivos"] = None, []
    _MADRES.clear()
    estados = leer_estados(supa)
    cambios, fallas = False, []

    # las unidades primero y en fila: son dos y cada una ya es un solo registro
    # de cambios
    for f in [f for f in fuentes if f["raiz"] == "unidad"]:
        print("\n== %s (%s, %s)" % (f["nombre"], f["sede"], f["raiz"]))
        try:
            cambios = sync_unidad(drive, supa, f, a, estados.get(estado_id(f), {})) or cambios
        except Exception as e:
            fallas.append(f["nombre"])
            print("  FALLO: %s" % str(e)[:400])
            anotar_error(supa, a, f, e)

    carpetas = [f for f in fuentes if f["raiz"] != "unidad"]
    if carpetas:
        # la lista de cambios de Drive se trae una vez, con el corte mas viejo
        # de las fuentes que ya tienen una corrida buena
        cortes = [corte_de(estados[estado_id(f)])[1] for f in carpetas
                  if estados.get(estado_id(f), {}).get("last_status") == "ok"
                  and estados[estado_id(f)].get("last_run_at")]
        if cortes:
            try:
                cambios_desde(drive, min(cortes))
            except Exception as e:
                # cada fuente la vuelve a pedir y falla por su cuenta
                print("  aviso: no se pudo traer la lista de cambios (%s)" % str(e)[:300])
        filas_ok = []
        with ThreadPoolExecutor(HILOS) as pool:
            futuros = [pool.submit(_una_carpeta, drive, supa, f, a,
                                   estados.get(estado_id(f), {})) for f in carpetas]
            for fut in futuros:
                f, cambio, fila, error, buf = fut.result()
                print("\n".join(buf))
                cambios = cambio or cambios
                if error is not None:
                    fallas.append(f["nombre"])
                    anotar_error(supa, a, f, error)
                elif fila:
                    filas_ok.append(fila)
        # un solo viaje con todas las filas 'ok' (todas con la misma forma,
        # como pide PostgREST para un upsert en lote)
        if filas_ok:
            supa.upsert("sync_state", filas_ok)
    return cambios, fallas


def _instante(s):
    """timestamptz de PostgREST ('2026-09-30T13:05:00.12+00:00') a epoch.
    fromisoformat de Python < 3.11 no acepta fracciones de 2 digitos."""
    if not s:
        return None
    s = str(s).replace("Z", "+00:00")
    try:
        return _dt.datetime.fromisoformat(s).timestamp()
    except ValueError:
        try:
            base = _dt.datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
            return base.replace(tzinfo=_dt.timezone.utc).timestamp()
        except ValueError:
            return None


def aviso_despues(supa, desde):
    """Si Drive aviso algo despues del instante desde (epoch). La Edge Function
    no dispara dos veces en 120 s pero anota cada aviso en ultimo_aviso_at: lo
    que llego mientras la pasada corria no tiene quien lo vaya a buscar."""
    if "drive_aviso" in _AVISADO:
        return False
    try:
        filas = supa.select("drive_aviso?select=ultimo_aviso_at&id=eq.1") or []
    except Exception as e:
        avisar_una_vez("drive_aviso", "no se pudo leer drive_aviso, sin segunda pasada "
                       "(falta sql/drive_aviso.sql?): %s" % str(e)[:160])
        return False
    t = _instante(filas[0].get("ultimo_aviso_at")) if filas else None
    return t is not None and t > desde


# ---------------------------------------------------------------- canales

DRIVE_AVISO_URL = "https://gfctvsxgulpftiytaxrn.supabase.co/functions/v1/drive-aviso"
CANAL_DURA_MS = 7 * 86400000 - 3600000        # 7 dias menos 1 hora
CANAL_RENOVAR_S = 24 * 3600


def canales_deseados(fuentes):
    """(fila de sync_state, driveId o None) por canal: uno por unidad con
    sincroniza y uno de la cuenta de servicio, que ve lo compartido con ella
    (las fuentes carpeta). Salen de fuentes_fotos: una unidad nueva suma su
    canal sola."""
    return ([(estado_id(f), f["id"]) for f in fuentes if f["raiz"] == "unidad"]
            + [("cuenta", None)])


def renovar_canales(drive, supa, fuentes, a):
    """Crea o renueva los canales de changes.watch que avisan a drive-aviso.

    Un canal vence a los 7 dias como mucho y no se renueva solo: se crea otro
    (con otro id), se guarda y se para el viejo. Cualquier error se imprime
    entero -ahi se ve si Drive pide dominio verificado- y no corta nada: el
    sync ya hizo su trabajo."""
    token = env("DRIVE_AVISO_TOKEN")
    url = env("DRIVE_AVISO_URL") or DRIVE_AVISO_URL
    try:
        estados = leer_estados(supa)
    except Exception as e:
        print("  push de Drive: no se pudo leer sync_state (%s)" % str(e)[:200])
        return
    if estados and not any("canal_id" in r for r in estados.values()):
        print("  push de Drive apagado: sync_state no tiene las columnas canal_* "
              "(falta sql/drive_aviso.sql)")
        return
    if not a.dry_run and not token:
        print("  push de Drive apagado: falta DRIVE_AVISO_TOKEN en el ambiente")
        return

    ahora = time.time()
    for sid, drive_id in canales_deseados(fuentes):
        st = estados.get(sid, {})
        vence = _instante(st.get("canal_vence"))
        if vence is not None and vence - ahora > CANAL_RENOVAR_S:
            continue
        motivo = ("vence %s" % st.get("canal_vence")) if vence is not None else "no tiene"
        if a.dry_run:
            print("  canal %s: %s, se crearia uno nuevo%s" % (
                sid, motivo, " y se pararia el viejo" if st.get("canal_id") else ""))
            continue
        try:
            unidad = {"driveId": drive_id, "supportsAllDrives": "true"} if drive_id else {}
            tok = drive.get("changes/startPageToken", **unidad)["startPageToken"]
            params = {"pageToken": tok, "supportsAllDrives": "true",
                      "includeItemsFromAllDrives": "true"}
            if drive_id:
                params["driveId"] = drive_id
            nuevo_id = str(uuid.uuid4())
            resp = drive.post("changes/watch", {
                "id": nuevo_id, "type": "web_hook", "address": url, "token": token,
                "expiration": int(ahora * 1000) + CANAL_DURA_MS}, **params)
        except Exception as e:
            print("  canal %s: FALLO el watch: %s" % (sid, e))
            continue
        exp = resp.get("expiration")
        fila = {"id": sid, "canal_id": nuevo_id, "canal_recurso": resp.get("resourceId"),
                "canal_vence": (_dt.datetime.fromtimestamp(int(exp) / 1000, _dt.timezone.utc)
                                .strftime("%Y-%m-%dT%H:%M:%SZ") if exp else None)}
        try:
            # upsert: la fila 'cuenta' no existe la primera vez; en las demas
            # solo toca las tres columnas del canal
            supa.upsert("sync_state", [fila])
        except Exception as e:
            # sin guardarlo nadie lo renueva ni lo para: se para ya
            print("  canal %s: no se pudo guardar (%s); se para el nuevo" % (sid, e))
            try:
                drive.post("channels/stop", {"id": nuevo_id, "resourceId": resp.get("resourceId")},
                           ok_404=True)
            except Exception as e2:
                print("  canal %s: tampoco se pudo parar: %s" % (sid, e2))
            continue
        print("  canal %s: creado %s, vence %s" % (sid, nuevo_id, fila["canal_vence"]))
        if st.get("canal_id") and st.get("canal_recurso"):
            try:
                drive.post("channels/stop", {"id": st["canal_id"],
                                             "resourceId": st["canal_recurso"]}, ok_404=True)
                print("  canal %s: parado el viejo %s" % (sid, st["canal_id"]))
            except Exception as e:
                print("  canal %s: no se pudo parar el viejo %s: %s" % (sid, st["canal_id"], e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sa", default=os.path.join(HERE, "service-account.json"))
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    inicio_corrida = time.time()

    # El .env tiene la clave pero no la URL: es la misma para todo el
    # proyecto y vive como constante en sb.py. Sin esto, correrlo a mano
    # moria pidiendo una variable que nadie escribio nunca.
    sb_url = env("SUPABASE_URL") or DEFAULT_URL
    sb_key = env("SUPABASE_SERVICE_KEY", "SUPABASE_SERVICE_ROLE_KEY")
    if not sb_url or not sb_key:
        sys.exit("Faltan SUPABASE_URL y SUPABASE_SERVICE_KEY")
    if not os.path.exists(a.sa):
        sys.exit("Falta el service account en %s" % a.sa)

    drive = Drive(a.sa)
    supa = Supa(sb_url, sb_key)

    fuentes = supa.select("fuentes_fotos?select=*&proveedor=eq.drive&sincroniza=eq.true&order=creado_at") or []
    if not fuentes:
        # sin la tabla cargada, lo de siempre: solo Server Media
        fuentes = [{"id": ROOT_ID, "nombre": "Server Media", "raiz": "unidad", "sede": "North"}]

    # LA SEGUNDA PASADA. Si Drive aviso algo despues de que empezo la pasada,
    # ese aviso no dispara otra corrida (esta, que ya esta andando, se la
    # queda por el concurrency del workflow, o la ventana de 120 s de la Edge
    # Function lo junto con otro). Se da otra vuelta, hasta 5 por corrida:
    # un album que se sube durante 10 minutos va entrando por tandas.
    cambios, fallas = False, []
    for n in range(1, 6):
        comienzo = time.time()
        if n > 1:
            print("\n#### pasada %d: Drive aviso algo mientras corria la anterior" % n)
        c, fl = pasada(drive, supa, fuentes, a)
        cambios = c or cambios
        fallas += [x for x in fl if x not in fallas]
        if a.bootstrap or not aviso_despues(supa, comienzo):
            break
        if a.dry_run:
            print("\n(dry-run: hubo un aviso durante la pasada; se daria otra)")
            break
    print("\ntiempo de las pasadas: %.1f s" % (time.time() - inicio_corrida))

    # Solo si algo cambio: el rollup recorre todas las carpetas y tarda unos
    # segundos. Una foto nueva cambia el total de todos sus ancestros.
    if cambios and not a.dry_run:
        try:
            supa.rpc("refresh_photo_counts", {})
        except Exception as e:
            print("  aviso: no se pudo recalcular los totales (%s)" % e)

    # los canales, una vez por corrida y aunque alguna fuente haya fallado
    print("\n== push de Drive")
    try:
        renovar_canales(drive, supa, fuentes, a)
    except Exception as e:
        print("  push de Drive: FALLO %s" % e)

    if fallas:
        sys.exit("fallaron: %s" % ", ".join(fallas))


if __name__ == "__main__":
    main()
