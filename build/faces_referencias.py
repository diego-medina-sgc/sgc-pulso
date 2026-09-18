# -*- coding: utf-8 -*-
"""
Arma las caras de referencia: una huella por cada retrato ya identificado.

Es contra esto que después se comparan las caras de las fotos de evento. Sin
referencia no hay reconocimiento posible, y hoy sólo 960 personas tienen una.

Se guarda UNA HUELLA POR RETRATO, no un promedio por persona. La calibración
mostró que la misma persona baja de 0,94 a 0,72 de parecido de un año a
otro: promediar el mugshot de 1er grado con el de 5to da una cara que no es
ninguna de las dos. Cada huella conserva su año y la comparación después se
acota a los años cercanos.

    python faces_referencias.py              # todas
    python faces_referencias.py --limite 50  # prueba corta

Escribe _referencias.npz. Es reanudable: si se corta, la próxima corrida
sigue desde donde estaba.
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
DEST = faces.npz("_referencias.npz")
HUELLAS = faces.npz("_huellas_retratos.npz")
HILOS = 6          # más que esto y Drive empieza a devolver 403 por rate limit


def cajas_guardadas(cli):
    """El recuadro de cada (foto, persona), en fracciones de la imagen.

    Es lo que convierte "esta persona esta en esta foto" en "esta persona es
    ESTA cara de esta foto". Sin eso no hay forma de saber cual de las cuarenta
    caras de un acto es la que alguien confirmo.
    """
    d = {}
    for r in cli.select("face_suggestions",
                        select="photo_id,person_id,bx,by,bw,bh"):
        if r.get("bw") and r.get("bh"):
            d[(r["photo_id"], int(r["person_id"]))] = (
                float(r["bx"]), float(r["by"]), float(r["bw"]), float(r["bh"]))
    return d


def solape(a, b):
    """Cuanto se pisan dos recuadros (x, y, w, h), de 0 a 1."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / float(aw * ah + bw * bh - inter)


# Cuanto se tienen que pisar el recuadro guardado y el detectado para aceptar
# que son la misma cara. Es el mismo numero que usa caras_huellas_evento.py:
# mismo detector sobre la misma imagen, los recuadros dan casi identicos.
IOU_MIN = 0.5


def cara_de(c, data, caja, hecho, lock):
    """La cara de LA PERSONA en esta foto. None si no se puede saber cual es.

    ESTE ERA EL AGUJERO POR DONDE ENTRABA TODO

    Antes esto era una linea: huellas(data, max_caras=1), la cara mas grande,
    con el comentario "en un retrato la cara del sujeto es la mas grande". El
    comentario es cierto y la funcion no: catalogo() no trae solo retratos,
    trae TODAS las identificaciones confirmadas, y la mayoria son fotos de
    acto. En una foto de cuarenta personas la cara mas grande es la del que
    quedo mas cerca de la camara.

    Asi que el juego preguntaba "¿esta Roberto en esta foto?", Diego decia que
    si -y tenia razon, Roberto estaba-, y el sistema archivaba como cara de
    Roberto la del senor de adelante. Esa cara pasaba a ser referencia, atraia
    mas caras que no eran de Roberto, que tambien se confirmaban porque Roberto
    tambien estaba ahi. Roberto termino con 185 referencias de las que 109 no
    eran el, y su nombre aparecia en toda la biblioteca.

    LA REGLA AHORA

    Una sola cara en la foto: es esa, no hay ambiguedad.
    Varias caras y hay recuadro guardado: la que se pisa con el recuadro.
    Varias caras y no hay recuadro: NINGUNA.

    El ultimo caso es el importante. Perder una referencia cuesta una cara;
    inventarla cuesta una ficha entera y todo lo que esa ficha despues atrae.
    """
    img = c.leer(data)
    if img is None:
        return None
    caras = c.detectar(img)
    if not caras:
        return None
    if len(caras) == 1:
        return c.huella(img, caras[0])
    if caja is None:
        with lock:
            hecho["varias_sin_caja"] += 1
        return None
    alto, ancho = img.shape[:2]
    objetivo = (caja[0] * ancho, caja[1] * alto, caja[2] * ancho, caja[3] * alto)
    mejor, cuanto = None, 0.0
    for cara in caras:
        s = solape(objetivo, [float(v) for v in cara[:4]])
        if s > cuanto:
            mejor, cuanto = cara, s
    if mejor is None or cuanto < IOU_MIN:
        with lock:
            hecho["caja_sin_cara"] += 1
        return None
    return c.huella(img, mejor)


def catalogo(cli):
    """(photo_id, person_id, año) de cada retrato con identidad confirmada."""
    print("Leyendo identificaciones confirmadas…")
    pp = cli.select("photo_people",
                    select="photo_id,person_id,"
                           "photos(folder_id,source,url_host,url_core,external_realm)")
    fol = cli.select("folders", select="id,year,campus")
    anio = {f["id"]: f.get("year") for f in fol}
    campus = {f["id"]: f.get("campus") for f in fol}

    # Zenfolio tambien: sus huellas salen del cache de faces_huellas.py, y si
    # falta alguna se baja con zenfolio.py. Quedaba afuera de aca desde antes
    # de que ese modulo existiera, y el efecto era que Quilmes nunca podia
    # reciclar sus propias identificaciones como caras de referencia.
    from collections import Counter
    todo = []
    for r in pp:
        ph = r.get("photos") or {}
        if ph.get("source") not in ("drive", "zenfolio"):
            continue
        fid = ph.get("folder_id")
        todo.append((r["photo_id"], r["person_id"], anio.get(fid), campus.get(fid), ph))

    # La marca de agua de Zenfolio cae encima de la cara y la arruina como
    # referencia. Medido sobre los 622 retratos que estan en las dos fuentes,
    # la misma foto limpia y marcada se parecen 0,857 de mediana, y 73 de esos
    # 622 quedan por debajo de 0,50, que es el umbral con el que se aprueba: el
    # motor no reconoce que son la misma persona.
    #
    # Asi que una foto marcada solo se usa cuando esa persona no tiene ninguna
    # limpia. Es mejor que nada, pero peor que cualquier otra cosa.
    limpias = {t[1] for t in todo if t[4].get("source") == "drive"}
    out = [t for t in todo
           if t[4].get("source") != "zenfolio" or t[1] not in limpias]
    tirados = len(todo) - len(out)
    print("  identificaciones: %d   utilizables: %d  (%s)"
          % (len(pp), len(out), ", ".join("%s %d" % (k, v) for k, v in
                                          sorted(Counter(t[4].get("source")
                                                         for t in out).items()))))
    if tirados:
        print("  con marca de agua descartadas por tener una limpia: %d" % tirados)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--rehacer-con-caja", action="store_true",
                    help="recalcular las referencias que salieron de una foto "
                         "de evento: son las que se archivaron tomando la cara "
                         "mas grande de la foto en vez de la de la persona")
    a = ap.parse_args()

    cli = sb.SB()
    filas = catalogo(cli)
    if a.limite:
        filas = filas[:a.limite]

    # Reanudar: no volver a bajar lo que ya se proceso.
    ya = {}
    if os.path.exists(DEST):
        z = np.load(DEST, allow_pickle=True)
        # cada acceso z["campo"] descomprime el array entero: se leen una vez
        ids, vecs = z["photo_ids"], z["vecs"]
        pers, anios, camp = z["personas"], z["anios"], z["campus"]
        for i, pid in enumerate(ids):
            ya[str(pid)] = (vecs[i], int(pers[i]), anios[i], str(camp[i]))
        print("Ya procesados de antes: %d" % len(ya))

    # LA PODA. De una cara de referencia no se sale.
    #
    # Este diccionario se sembraba con el npz anterior y no se limpiaba nunca:
    # lo unico que hacia era no volver a bajar lo ya bajado. El efecto es que
    # una cara que entro mal se queda para siempre, aunque despues alguien
    # conteste "No es" o la saque una auditoria.
    #
    # Se vio la noche del 11/9/2026: caras_mal_archivadas.py saco 481 caras
    # ajenas de photo_people a las 18:58 y las referencias las seguian teniendo
    # a las 21:30. Rehacer el archivo tampoco las habria sacado, porque salian
    # de aca y no de la base. Persona Q tenia 226 caras de referencia
    # contra 135 identificaciones vivas: la diferencia era pura acumulacion.
    #
    # Ahora sobrevive solo lo que sigue confirmado HOY, y con la misma persona:
    # si la foto se reasigno, la huella vieja quedo colgada de un nombre que ya
    # no es el suyo.
    vigentes = {(f[0], int(f[1])) for f in filas}
    antes = len(ya)
    ya = {k: v for k, v in ya.items() if (k, int(v[1])) in vigentes}
    if antes != len(ya):
        print("Podadas: %d caras que ya no estan confirmadas" % (antes - len(ya)))

    cajas = cajas_guardadas(cli)
    print("Recuadros guardados: %d" % len(cajas))

    # Rehacer las que salieron de una foto de evento.
    #
    # La poda saca lo que ya no esta confirmado, pero no arregla lo que sigue
    # confirmado y se calculo mal. Esas son las que tienen recuadro guardado:
    # tener recuadro significa que la cara salio de una foto de acto, y hasta
    # hoy de esas fotos se archivaba la cara mas grande. Se las tira del cache
    # para que se vuelvan a bajar y ahora si se recorte la cara que indica el
    # recuadro.
    #
    # No se hace siempre porque cuesta bajar una imagen por cada una. Los
    # retratos de ficha no hace falta tocarlos: ahi la cara mas grande es la
    # de la persona, que es lo que decia el comentario original.
    if getattr(a, "rehacer_con_caja", False):
        antes = len(ya)
        ya = {k: v for k, v in ya.items() if (k, int(v[1])) not in cajas}
        print("A rehacer por venir de una foto de evento: %d" % (antes - len(ya)))

    pend = [f for f in filas if f[0] not in ya]
    print("A procesar ahora: %d" % len(pend))

    # Las huellas de los retratos ya estan calculadas: para esas no hay que
    # bajar nada, ni de Drive ni de Zenfolio. Reconstruir las referencias pasa
    # a costar segundos, que es lo que permite rehacerlas cada vez que el
    # juego confirma nombres nuevos.
    cache = {}
    if os.path.exists(HUELLAS):
        z = np.load(HUELLAS, allow_pickle=True)
        ids_c, vecs_c = z["photo_ids"], z["vecs"]
        cache = {str(pid): vecs_c[i] for i, pid in enumerate(ids_c)}
        print("Huellas en cache: %d" % len(cache))

    # Zenfolio solo se abre si queda algo suyo fuera del cache: pide login.
    zf = None
    realms = {f[4].get("external_realm") for f in pend
              if f[4].get("source") == "zenfolio" and f[0] not in cache}
    realms.discard(None)
    if realms:
        print("Abriendo Zenfolio (%d realms)…" % len(realms))
        zf = zenfolio.Zenfolio()
        zf.abrir(realms)

    d = Drive(SA_PATH)
    lock = threading.Lock()
    hecho = {"n": 0, "sin_cara": 0, "error": 0, "de_cache": 0,
             "varias_sin_caja": 0, "caja_sin_cara": 0}

    # Los modelos de OpenCV NO son thread-safe: compartir un detector entre
    # hilos corrompe su estado y tira "buf.shape() == m.shape()". Cada hilo
    # carga el suyo; son ~40 MB cada uno, y el detector es lo que aprovecha
    # los varios núcleos.
    local = threading.local()

    def mis_caras():
        if not hasattr(local, "c"):
            local.c = faces.Caras()
        return local.c

    def procesar(fila):
        pid, persona, anio, camp, meta = fila
        caja = cajas.get((pid, int(persona)))
        # El cache de retratos guarda la cara mas grande de cada foto, que es
        # la buena en una ficha y cualquiera en un acto. Si hay recuadro, esta
        # foto es una foto de evento: se ignora el cache y se va a buscar esa
        # cara y no otra.
        if caja is None:
            v = cache.get(pid)
            if v is not None:
                with lock:
                    hecho["n"] += 1
                    hecho["de_cache"] += 1
                return (pid, v, persona, anio, camp)
        try:
            if meta.get("source") == "zenfolio":
                data = zf.bajar(meta.get("url_host"), meta.get("url_core")) if zf else None
                if not data:
                    raise ValueError("imagen protegida o sin sesion")
            else:
                link = d.get("files/" + pid, fields="thumbnailLink").get("thumbnailLink")
                if not link:
                    return None
                # =s1600: la miniatura por defecto es de 220 px y una cara ahí
                # no tiene píxeles suficientes para reconocerla
                with urllib.request.urlopen(link.replace("=s220", "=s1600"), timeout=60) as r:
                    data = r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError, ValueError):
            with lock:
                hecho["error"] += 1
            return None

        v = cara_de(mis_caras(), data, caja, hecho, lock)
        with lock:
            hecho["n"] += 1
            if hecho["n"] % 100 == 0:
                print("  %d/%d" % (hecho["n"], len(pend)))
            if v is None:
                hecho["sin_cara"] += 1
        return None if v is None else (pid, v, persona, anio, camp)

    # De a tandas, refrescando el token entre una y otra desde el hilo
    # principal: google-auth tampoco es thread-safe, y un refresh disparado
    # por dos hilos a la vez deja la credencial en un estado raro.
    resultados = []
    TANDA = 400
    try:
        for i in range(0, len(pend), TANDA):
            d._refresh()
            with ThreadPoolExecutor(max_workers=HILOS) as ex:
                for r in ex.map(procesar, pend[i:i + TANDA]):
                    if r:
                        resultados.append(r)
    except KeyboardInterrupt:
        print("\nInterrumpido: se guarda lo que haya.")

    for pid, vec, persona, anio, camp in resultados:
        ya[pid] = (vec, persona, anio, camp)

    if not ya:
        sys.exit("No se pudo armar ninguna referencia.")

    ids = list(ya)
    np.savez(DEST,
             photo_ids=np.array(ids),
             vecs=np.array([ya[i][0] for i in ids], dtype=np.float32),
             personas=np.array([ya[i][1] for i in ids]),
             anios=np.array([ya[i][2] if ya[i][2] is not None else -1 for i in ids]),
             campus=np.array([ya[i][3] if ya[i][3] else "" for i in ids]))

    personas = {ya[i][1] for i in ids}
    print("\nReferencias: %d caras de %d personas" % (len(ids), len(personas)))
    print("Desde el cache de huellas: %d (sin bajar nada)" % hecho["de_cache"])
    print("Sin cara detectada: %d   errores de descarga: %d"
          % (hecho["sin_cara"], hecho["error"]))
    # Las que NO se archivaron a proposito. Conviene mirarlas: si este numero
    # es grande, hay muchas identificaciones confirmadas sobre fotos de grupo
    # sin recuadro, y esas son justo las que antes envenenaban la ficha.
    print("Descartadas por no poder saber cual es la cara: %d sin recuadro, "
          "%d con recuadro que no coincide"
          % (hecho["varias_sin_caja"], hecho["caja_sin_cara"]))
    print("Escrito: %s  (%.1f MB)" % (DEST, os.path.getsize(DEST) / 1e6))


if __name__ == "__main__":
    main()
