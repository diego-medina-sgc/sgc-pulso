# -*- coding: utf-8 -*-
"""
Calcula y GUARDA la huella de cada retrato del catalogo.

Hasta ahora cada corrida bajaba la foto, sacaba la huella, la comparaba y la
tiraba. Volver a hacer cualquier cosa costaba horas de nuevo. Guardarlas
cambia eso: agrupar por cara, reprocesar con otro umbral o sumar una fuente
nueva pasan a ser segundos.

Son 128 numeros por cara: 11.700 retratos entran en 6 MB. No van a la base
—no hacen falta ahi y ocuparian espacio que no sobra— sino a un archivo al
lado del resto del cache.

    python faces_huellas.py              # todos los retratos que falten
    python faces_huellas.py --limite 500

Es reanudable: guarda cada tanda y al reanudar saltea lo hecho.
"""
import argparse
import os
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import faces
import sb
import zenfolio
from index_drive import Drive, SA_PATH

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = faces.npz("_huellas_retratos.npz")
HILOS = 6
TANDA = 400


def cargar():
    if not os.path.exists(DEST):
        return {}
    z = np.load(DEST, allow_pickle=True)
    # Cada acceso z["campo"] sobre un .npz DESCOMPRIME el array entero de
    # nuevo. Dentro de un bucle eso es una copia completa por elemento: con
    # 9.900 huellas eran 47 GB y el proceso moria pidiendo 4,8 MB. Se leen
    # una sola vez y despues se indexa en memoria.
    ids, vecs = z["photo_ids"], z["vecs"]
    carpetas, anios = z["carpetas"], z["anios"]
    return {str(p): (vecs[i], str(carpetas[i]), int(anios[i]))
            for i, p in enumerate(ids)}


def guardar(d):
    ids = list(d)
    np.savez(DEST,
             photo_ids=np.array(ids),
             vecs=np.array([d[i][0] for i in ids], dtype=np.float32),
             carpetas=np.array([d[i][1] for i in ids]),
             anios=np.array([d[i][2] for i in ids]))


def catalogo(cli):
    """Todos los retratos individuales, identificados o no.

    Las dos fuentes: Drive (North) y Zenfolio (Quilmes). Quilmes quedaba afuera
    porque sus imagenes necesitan sesion, y son 1.566 retratos sin una sola
    identificacion.
    """
    fol = {f["id"]: f for f in cli.select("folders", select="id,year,campus,is_mugshot,noise")
           if f.get("is_mugshot") and f.get("noise") is None}
    ids = list(fol)
    fotos = []
    for i in range(0, len(ids), 40):
        fotos += cli.select("photos",
                            select="id,folder_id,source,is_group,url_host,url_core,external_realm",
                            folder_id="in.(%s)" % ",".join(ids[i:i + 40]))
    out = [(f["id"], f["folder_id"], fol[f["folder_id"]].get("year") or -1, f)
           for f in fotos
           if f.get("source") in ("drive", "zenfolio") and not f.get("is_group")]
    out.sort(key=lambda t: t[0])
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0)
    a = ap.parse_args()

    cli = sb.SB()
    hechas = cargar()
    print("Huellas ya guardadas: %d" % len(hechas))

    todos = catalogo(cli)
    pend = [t for t in todos if t[0] not in hechas]
    if a.limite:
        pend = pend[:a.limite]
    print("Retratos en el catalogo: %d   a procesar: %d" % (len(todos), len(pend)))
    if not pend:
        print("Nada que hacer.")
        return

    d = Drive(SA_PATH)
    lock = threading.Lock()
    local = threading.local()
    cuenta = {"n": 0, "sin_cara": 0, "error": 0, "grupales": 0}

    # Zenfolio solo se abre si hace falta: pide login y abrir realms.
    zf = None
    realms = {t[3].get("external_realm") for t in pend if t[3].get("source") == "zenfolio"}
    realms.discard(None)
    if realms:
        print("Abriendo Zenfolio (%d realms)…" % len(realms))
        zf = zenfolio.Zenfolio()
        zf.abrir(realms)

    # Las caras se detectan con el filtro de tamaño bajo, para poder contar
    # cuantas hay: una carpeta de retratos con una foto de diez caras es una
    # grupal mal ubicada, y el juego no deberia preguntar "quien es este".
    GRUPAL_DESDE = 4

    def mis_caras():
        # los modelos de OpenCV no son thread-safe: uno por hilo
        if not hasattr(local, "c"):
            local.c = faces.Caras()
        return local.c

    def procesar(t):
        pid, carpeta, anio, meta = t
        try:
            if meta.get("source") == "zenfolio":
                data = zf.bajar(meta.get("url_host"), meta.get("url_core"))
                if not data:
                    raise ValueError("imagen protegida")
            else:
                link = d.get("files/" + pid, fields="thumbnailLink").get("thumbnailLink")
                if not link:
                    return None
                with urllib.request.urlopen(link.replace("=s220", "=s1600"), timeout=60) as r:
                    data = r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError, ValueError):
            with lock:
                cuenta["error"] += 1
            return None

        c = mis_caras()
        img = c.leer(data)
        if img is None:
            with lock:
                cuenta["error"] += 1
            return None
        caras = c.detectar(img)

        with lock:
            cuenta["n"] += 1
            if cuenta["n"] % 500 == 0:
                print("  %d/%d" % (cuenta["n"], len(pend)))
            if len(caras) >= GRUPAL_DESDE:
                cuenta["grupales"] += 1
            elif not caras:
                cuenta["sin_cara"] += 1

        # una foto con varias caras en una carpeta de retratos es grupal
        if len(caras) >= GRUPAL_DESDE:
            return ("GRUPAL", pid)
        if not caras:
            return None
        h = c.huella(img, caras[0])
        return None if h is None else (pid, h, carpeta, anio)

    try:
        for i in range(0, len(pend), TANDA):
            d._refresh()
            grupales = []
            with ThreadPoolExecutor(max_workers=HILOS) as ex:
                for r in ex.map(procesar, pend[i:i + TANDA]):
                    if not r:
                        continue
                    if r[0] == "GRUPAL":
                        grupales.append(r[1])
                    else:
                        hechas[r[0]] = (r[1], r[2], r[3])
            guardar(hechas)      # por tanda, para no perder si se corta
            if grupales:
                # UPDATE y no upsert: las filas ya existen y el upsert las
                # trataria como INSERT, exigiendo columnas que no vienen aca.
                for j in range(0, len(grupales), 100):
                    cli.update("photos", {"is_group": True},
                               id="in.(%s)" % ",".join(grupales[j:j + 100]))
    except KeyboardInterrupt:
        print("\nInterrumpido: se guarda lo que haya.")
        guardar(hechas)

    print("\nHuellas: %d   sin cara: %d   errores: %d"
          % (len(hechas), cuenta["sin_cara"], cuenta["error"]))
    print("Escrito: %s  (%.1f MB)" % (DEST, os.path.getsize(DEST) / 1e6))


if __name__ == "__main__":
    main()
