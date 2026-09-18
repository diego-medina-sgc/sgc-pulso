# -*- coding: utf-8 -*-
"""
Fase 0 — Indexador de la unidad Server Media, contra la Drive API.

Reemplaza al Apps Script `indexador`. Recorre la unidad compartida completa y
escribe dos archivos que después consumen parse_folders.py y load_*.py.

Ventajas sobre el índice viejo:
  - Ve la unidad entera, incluido 2026 (el índice de enero no lo tenía).
  - Trae `imageMediaMetadata.time`, la fecha EXIF real de la toma.
  - Es reanudable: guarda el estado y retoma donde quedó.

Necesita build/service-account.json y que la unidad esté compartida con esa
cuenta como Lector.

Uso:
    python index_drive.py                 # completo, reanudable
    python index_drive.py --reset         # empieza de cero
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from google.oauth2 import service_account
import google.auth.transport.requests as gart

# Cada unidad de Drive es un campus. North es la unidad compartida
# "Server Media"; Quilmes es la carpeta "FOTOS POR AÑO", que vive en otro lado.
ROOTS = {
    "North":   "0ADoh-DIvUMYeUk9PVA",
    "Quilmes": "1XPeP4R06cC228tiwGkuahp5p4zELm1ng",
}
ROOT_ID = ROOTS["North"]
HERE = os.path.dirname(os.path.abspath(__file__))
SA_PATH = os.path.join(HERE, "service-account.json")
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
STATE = os.path.join(DATA, "index_state.json")

FOLDER_MIME = "application/vnd.google-apps.folder"
# Solo imágenes: los videos del Movie Bank quedan fuera de la biblioteca de fotos.
IMAGE_PREFIX = "image/"

FIELDS = ("nextPageToken,files(id,name,mimeType,parents,webViewLink,"
          "modifiedTime,size,imageMediaMetadata(width,height,time))")


class Drive:
    def __init__(self, sa_path):
        self.creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=["https://www.googleapis.com/auth/drive.readonly"])
        self.drive_id = None
        self._refresh()

    def _refresh(self):
        """Renueva el token, reintentando.

        Sin reintento, un parpadeo de red mata la corrida entera. Paso: el
        nocturno del 10 de septiembre se cayo a los 8 minutos con

            SSLEOFError: EOF occurred in violation of protocol

        renovando el token contra oauth2.googleapis.com. Las llamadas a la API
        de Drive ya reintentaban -una corrida de 7.000 carpetas cruza cortes de
        red tarde o temprano- pero la renovacion del token no, y es la unica
        parte que no se puede saltear.

        Seis intentos con espera creciente. Si despues de eso sigue sin haber
        red, ahi si corresponde fallar: no es un parpadeo.
        """
        for intento in range(6):
            try:
                self.creds.refresh(gart.Request())
                self.expires = time.time() + 3000
                return
            except Exception:
                if intento == 5:
                    raise
                time.sleep(min(2 ** intento, 30))

    def detect_drive(self, root_id):
        """driveId si la raiz esta en una unidad compartida; None si es My Drive."""
        m = self.get("files/" + root_id, fields="id,name,driveId")
        self.drive_id = m.get("driveId")
        return m

    def get(self, path, **params):
        if time.time() > self.expires:
            self._refresh()
        params.setdefault("supportsAllDrives", "true")
        url = ("https://www.googleapis.com/drive/v3/" + path + "?"
               + urllib.parse.urlencode(params))
        req = urllib.request.Request(
            url, headers={"Authorization": "Bearer " + self.creds.token})
        for intento in range(7):
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    return json.load(r)
            except urllib.error.HTTPError as e:
                # 403 acá es casi siempre rate limit, no permisos
                if e.code in (403, 429, 500, 502, 503) and intento < 6:
                    time.sleep(2 ** intento)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                # DNS caído, corte de red, timeout: una corrida de 7.000
                # carpetas los cruza tarde o temprano y no son fatales.
                if intento < 6:
                    time.sleep(min(2 ** intento, 30))
                    self._refresh_safe()
                    continue
                raise

    def _refresh_safe(self):
        try:
            self._refresh()
        except Exception:
            pass

    def children(self, folder_id):
        """Todos los hijos de una carpeta, paginando.

        corpora=drive + driveId solo valen si la raiz es una unidad
        compartida. La carpeta de Quilmes vive en un My Drive, y pasarle esos
        parametros hace que Drive devuelva 404.
        """
        token = None
        while True:
            p = dict(q="'%s' in parents and trashed=false" % folder_id,
                     fields=FIELDS, pageSize=1000,
                     includeItemsFromAllDrives="true")
            if self.drive_id:
                p.update(corpora="drive", driveId=self.drive_id)
            if token:
                p["pageToken"] = token
            r = self.get("files", **p)
            for f in r.get("files", []):
                yield f
            token = r.get("nextPageToken")
            if not token:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--sa", default=SA_PATH)
    ap.add_argument("--campus", default="North", choices=list(ROOTS))
    ap.add_argument("--suffix", default="",
                    help="sufijo para los archivos de salida, para no pisar otra corrida")
    a = ap.parse_args()

    global ROOT_ID, STATE
    ROOT_ID = ROOTS[a.campus]
    suf = a.suffix or ("" if a.campus == "North" else "_" + a.campus.lower())
    STATE = os.path.join(DATA, "index_state%s.json" % suf)

    if not os.path.exists(a.sa):
        sys.exit("Falta %s" % a.sa)
    os.makedirs(DATA, exist_ok=True)

    drive = Drive(a.sa)
    root = drive.detect_drive(ROOT_ID)
    print("Campus %s: %s (%s)" % (a.campus, root.get("name"), ROOT_ID))
    print("Origen: %s\n" % ("unidad compartida" if drive.drive_id
                            else "carpeta en My Drive"))

    if a.reset or not os.path.exists(STATE):
        state = {"queue": [ROOT_ID],
                 "folders": [{"id": ROOT_ID, "name": "(ROOT)", "parent": None}],
                 "done": [], "photos_file": os.path.join(DATA, "drive_photos%s.ndjson" % suf)}
        open(state["photos_file"], "w", encoding="utf-8").close()
    else:
        with open(STATE, encoding="utf-8") as fh:
            state = json.load(fh)
        print("Reanudando: %d carpetas en cola, %d ya recorridas\n"
              % (len(state["queue"]), len(state["done"])))

    done = set(state["done"])
    queue = state["queue"]
    folders = state["folders"]
    n_photos = 0
    t0 = time.time()

    out = open(state["photos_file"], "a", encoding="utf-8")
    try:
        while queue:
            fid = queue.pop(0)
            if fid in done:
                continue
            for f in drive.children(fid):
                if f["mimeType"] == FOLDER_MIME:
                    folders.append({"id": f["id"], "name": f["name"], "parent": fid})
                    queue.append(f["id"])
                elif f["mimeType"].startswith(IMAGE_PREFIX):
                    md = f.get("imageMediaMetadata") or {}
                    out.write(json.dumps({
                        "id": f["id"],
                        "name": f["name"],
                        "mimeType": f["mimeType"],
                        "parent": fid,
                        "webViewLink": f.get("webViewLink"),
                        "modifiedTime": f.get("modifiedTime"),
                        "size": f.get("size"),
                        "exifTime": md.get("time"),
                        "width": md.get("width"),
                        "height": md.get("height"),
                    }, ensure_ascii=False) + "\n")
                    n_photos += 1
            done.add(fid)

            if len(done) % 100 == 0:
                out.flush()
                state.update(queue=queue, folders=folders, done=sorted(done))
                with open(STATE, "w", encoding="utf-8") as fh:
                    json.dump(state, fh)
                print("  carpetas %5d | cola %5d | fotos %7d | %5.0fs"
                      % (len(done), len(queue), n_photos, time.time() - t0))
    finally:
        out.close()
        state.update(queue=queue, folders=folders, done=sorted(done))
        with open(STATE, "w", encoding="utf-8") as fh:
            json.dump(state, fh)

    manifest = {
        "version": 2,
        "rootFolderId": ROOT_ID,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "drive-api-service-account",
        "campus": a.campus,
        "stats": {"folders": len(folders), "images": n_photos},
        "folders": folders,
    }
    dest = os.path.join(DATA, "manifest%s.json" % suf)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False)

    print("\nTerminado en %.0fs" % (time.time() - t0))
    print("  carpetas: %d" % len(folders))
    print("  imagenes: %d" % n_photos)
    print("  %s" % dest)
    print("  %s" % state["photos_file"])


if __name__ == "__main__":
    main()
