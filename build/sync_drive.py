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

Uso:
    python sync_drive.py                # incremental
    python sync_drive.py --bootstrap    # solo fija el token, no sincroniza
    python sync_drive.py --dry-run      # muestra que haria
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

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


class Drive:
    def __init__(self, sa_path):
        self.creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=["https://www.googleapis.com/auth/drive.readonly"])
        self._refresh()

    def _refresh(self):
        self.creds.refresh(gart.Request())
        self.expires = time.time() + 3000

    def get(self, path, **params):
        if time.time() > self.expires:
            self._refresh()
        url = ("https://www.googleapis.com/drive/v3/" + path + "?"
               + urllib.parse.urlencode(params))
        req = urllib.request.Request(
            url, headers={"Authorization": "Bearer " + self.creds.token})
        for i in range(6):
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                if e.code in (403, 429, 500, 502, 503) and i < 5:
                    time.sleep(2 ** i)
                    continue
                detalle = e.read().decode("utf-8", "replace")[:400]
                raise RuntimeError("Drive %s en %s: %s" % (e.code, path, detalle))
            except (urllib.error.URLError, TimeoutError, OSError):
                if i < 5:
                    time.sleep(min(2 ** i, 30))
                    continue
                raise


class Supa:
    def __init__(self, url, key):
        self.url = url.rstrip("/")
        self.h = {"apikey": key, "Authorization": "Bearer " + key,
                  "Content-Type": "application/json"}

    def _call(self, method, path, body=None, prefer=None):
        h = dict(self.h)
        if prefer:
            h["Prefer"] = prefer
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        req = urllib.request.Request(self.url + "/rest/v1/" + path,
                                     data=data, method=method, headers=h)
        for i in range(4):
            try:
                with urllib.request.urlopen(req, timeout=240) as r:
                    raw = r.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                if i == 3:
                    raise RuntimeError("%s %s -> %s %s" % (
                        method, path, e.code, e.read().decode("utf-8", "replace")[:300]))
                time.sleep(2 * (i + 1))

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
        req = urllib.request.Request(self.url + "/rest/v1/rpc/" + fn,
                                     data=json.dumps(args).encode(),
                                     method="POST", headers=self.h)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            raise RuntimeError("rpc %s -> %s %s" % (
                fn, e.code, e.read().decode("utf-8", "replace")[:300]))


# ---------------------------------------------------------------- mapeo

def campus_de(sede):
    """La sede de la fuente, tal como va en cada carpeta.

    'Ambas' no es una sede: es una carpeta que tiene fotos de los dos campus.
    Diego, 16/9/2026, sobre las de Alumni: "hay de los dos campus". Estamparle
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


def sync_unidad(drive, supa, fuente, a):
    """Una unidad compartida, por la Changes API. Devuelve si escribio algo."""
    root = fuente["id"]
    sid = estado_id(fuente)
    state = (supa.select("sync_state?id=eq.%s&select=*" % urllib.parse.quote(sid)) or [{}])[0]
    token = state.get("page_token")

    if a.bootstrap or not token:
        r = drive.get("changes/startPageToken", driveId=root,
                      supportsAllDrives="true")
        tok = r["startPageToken"]
        if not a.dry_run:
            supa.upsert("sync_state", [{
                "id": sid, "page_token": tok,
                "last_run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "last_status": "bootstrap",
                "note": "token inicial; el contenido vino del crawl completo",
            }])
        print("Token inicial fijado: %s" % tok)
        print("La proxima corrida ya sincroniza solo los cambios.")
        return False

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
    for anio in listar(drive, q="'%s' in parents and %s" % (root, solo_carpetas)):
        arriba[anio["id"]] = (anio, root)
        for alb in listar(drive, q="'%s' in parents and %s" % (anio["id"], solo_carpetas)):
            arriba[alb["id"]] = (alb, anio["id"])
    candidatas = dict(arriba)
    candidatas.update(changed_folders)
    ids = sorted(candidatas)
    estan = set()
    for i in range(0, len(ids), 100):
        estan |= {r["id"] for r in supa.select(
            "folders?select=id&id=in.(%s)" % ",".join(ids[i:i + 100])) or []}
    faltan = {c for c in ids if c not in estan}
    for cid in sorted(faltan):
        if candidatas[cid][1] in faltan:
            continue          # la trae el recorrido de su madre
        changed_folders[cid] = candidatas[cid]
        c2, f2 = recorrer(drive, cid)
        changed_folders.update(c2)
        changed_photos.update(f2)
        print("     faltaba en la base: %s (%d carpetas, %d fotos adentro)"
              % (arriba[cid][0].get("name"), len(c2), len(f2)))

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
_CAMBIOS = {"corte": None, "archivos": []}
_MADRES = {}


def cambios_desde(drive, corte):
    """Carpetas e imagenes creadas o modificadas despues de corte (UTC, sin zona)."""
    if _CAMBIOS["corte"] is None or corte < _CAMBIOS["corte"]:
        q = ("(modifiedTime > '%s' or createdTime > '%s') and trashed = false and "
             "(mimeType = '%s' or mimeType contains 'image/')") % (corte, corte, FOLDER_MIME)
        _CAMBIOS["archivos"] = list(listar(drive, q=q, corpora="allDrives"))
        _CAMBIOS["corte"] = corte
        print("  (lista de cambios de Drive traida desde %s: %d)" % (corte, len(_CAMBIOS["archivos"])))
    # Drive devuelve "2026-09-21T17:40:43.882Z": los primeros 19 caracteres se
    # comparan como texto contra el corte, que tiene el mismo formato
    return [f for f in _CAMBIOS["archivos"]
            if (f.get("modifiedTime") or "")[:19] > corte
            or (f.get("createdTime") or "")[:19] > corte]


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


def sync_carpeta(drive, supa, fuente, a):
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
    """
    root, sede, sid = fuente["id"], fuente["sede"], estado_id(fuente)
    inicio = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state = (supa.select("sync_state?id=eq.%s&select=*" % urllib.parse.quote(sid)) or [{}])[0]
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
        print("  la corrida anterior fallo: se recorre el arbol entero")
        conocidas = set()
    desde = state.get("last_run_at") if state.get("last_status") == "ok" else None
    if not desde and base and not fallo_antes:
        fechas = [r["indexed_at"] for r in base if r.get("indexed_at")]
        desde = max(fechas) if fechas else None

    carpetas, fotos = {}, {}
    if not conocidas:
        print("  fuente nueva: se recorre entera")
        meta = drive.get("files/" + root, fields="id,name,mimeType,parents,webViewLink,modifiedTime",
                         supportsAllDrives="true")
        carpetas[root] = (meta, None)
        c2, f2 = recorrer(drive, root)
        carpetas.update(c2)
        fotos.update(f2)
    else:
        corte = desde or "2000-01-01T00:00:00Z"
        # una hora de margen: los relojes de Drive y el de la corrida no son el mismo
        try:
            import datetime as _dt
            t = _dt.datetime.fromisoformat(str(corte).replace("Z", "+00:00")) - _dt.timedelta(hours=1)
            corte = t.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            pass
        print("  cambios desde %s" % corte)

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
        print("  modificados en todo lo que ve la cuenta: %d" % vistos)

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
    print("  carpetas afectadas: %d (%d nuevas)" % (len(carpetas), len(nuevas)))
    print("  fotos afectadas:    %d" % len(fotos))
    for c in nuevas[:10]:
        print("     nueva: %s" % (carpetas[c][0].get("name") or c))

    if a.dry_run:
        print("\n(dry-run: no se escribio nada)")
        return False

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
    supa.upsert("folders", frows)
    known = conocidas | set(carpetas)
    prows = [photo_row(f, parent) for (f, parent) in fotos.values() if parent in known]
    supa.upsert("photos", prows)

    supa.upsert("sync_state", [{
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
    }])
    print("  escritas: %d carpetas, %d fotos" % (len(frows), len(prows)))
    return bool(frows or prows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sa", default=os.path.join(HERE, "service-account.json"))
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

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

    cambios, fallas = False, []
    for f in fuentes:
        print("\n== %s (%s, %s)" % (f["nombre"], f["sede"], f["raiz"]))
        try:
            hacer = sync_unidad if f["raiz"] == "unidad" else sync_carpeta
            cambios = hacer(drive, supa, f, a) or cambios
        except Exception as e:
            fallas.append(f["nombre"])
            print("  FALLO: %s" % str(e)[:400])
            if not a.dry_run:
                try:
                    supa.upsert("sync_state", [{"id": estado_id(f), "last_status": "error",
                                                "note": str(e)[:300]}])
                except Exception:
                    pass

    # Solo si algo cambio: el rollup recorre todas las carpetas y tarda unos
    # segundos. Una foto nueva cambia el total de todos sus ancestros.
    if cambios and not a.dry_run:
        try:
            supa.rpc("refresh_photo_counts", {})
        except Exception as e:
            print("  aviso: no se pudo recalcular los totales (%s)" % e)
    if fallas:
        sys.exit("fallaron: %s" % ", ".join(fallas))


if __name__ == "__main__":
    main()
