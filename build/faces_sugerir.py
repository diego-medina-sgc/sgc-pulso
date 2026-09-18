# -*- coding: utf-8 -*-
"""
Recorre los retratos pendientes y guarda lo que la cara propone para cada uno.

No etiqueta a nadie: escribe en face_suggestions, que el juego muestra como
sugerencia para que una persona confirme. A 0,65 acierta el 94%, o sea que
1 de cada 16 estaria mal, y por eso nunca se aplica solo.

    python faces_sugerir.py                 # todos los pendientes
    python faces_sugerir.py --limite 500    # una tanda
    python faces_sugerir.py --umbral 0.70   # mas exigente
    python faces_sugerir.py --rehacer       # tira las viejas y recalcula

Es reanudable: no vuelve a mirar los retratos que ya tienen sugerencia.

Por eso mismo, al cambiar de motor hace falta --rehacer. Las sugerencias que
quedaron son de otro reconocedor, con otra escala de puntajes: un 0,66 de
sface y uno de arcface no quieren decir lo mismo, y mezclarlos en la misma
tabla rompe cualquier decision que mire el score, empezando por la aprobacion
en lote. Solo se tiran las de fotos que siguen sin nombre; las sugerencias son
un derivado que se recalcula en minutos desde el cache de huellas.
"""
import argparse
import glob
import os
import re
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import faces
import sb
from faces_piloto import cargar_refs
from index_drive import Drive, SA_PATH

HERE = os.path.dirname(os.path.abspath(__file__))
REFS_ANUARIO = faces.npz("_referencias_anuario.npz")
# Las fotos de curso del anuario en PDF: son los chicos de todos los años, no
# solo los egresados, que es lo que las hace valiosas para la cola actual.
REFS_ANUARIO_PDF = faces.npz("_referencias_anuario_pdf.npz")
REFS_STAFF = faces.npz("_referencias_staff.npz")
# Las carpetas Valete del Drive de Marketing: el retrato de egreso de cada
# camada, en alta y con el nombre en el archivo. Ver valete_cargar.py.
REFS_VALETE = faces.npz("_referencias_valete.npz")
HUELLAS = faces.npz("_huellas_retratos.npz")
HILOS = 6
TOPE = 3          # cuantas sugerencias por foto


def todas_las_referencias(cli):
    """Junta las tres fuentes de caras conocidas.

    Los mugshots identificados cubren 2021-2025, los anuarios 1992-2017 y la
    planilla de staff, a los adultos que aparecen en casi todos los actos.
    Se complementan, asi que se comparan todas juntas.
    """
    vecs, personas, anios, _ = cargar_refs(cli)
    vivas = {p["id"] for p in cli.select("people", select="id,kind")
             if p.get("kind") != "noise"}

    # Las caras de fuente que alguien resolvio a mano en "Revisar incoherencias".
    #
    # Una cara equivocada que viene de una foto del repositorio se saca de
    # photo_people y no vuelve. Una que viene de una fuente -la planilla de
    # staff, el anuario, un Valete- no esta en photo_people: esta en el .npz, y
    # se regenera de la fuente en cada corrida. Sin esta lista, resolver la
    # ficha no serviria para nada: la cara mala volveria a entrar aca.
    malas = set()
    try:
        for r in cli.select("caras_fuente_mala", select="person_id,fuente"):
            malas.add((int(r["person_id"]), r["fuente"]))
    except Exception:
        pass
    if malas:
        print("  - %d caras de fuente marcadas como equivocadas" % len(malas))

    def sumar(path, etiqueta, con_anio=True):
        nonlocal vecs, personas, anios
        if not os.path.exists(path):
            return
        z = np.load(path, allow_pickle=True)
        pers = z["personas"].astype(int)
        base = os.path.basename(path)
        # una persona borrada o fusionada dejaria una referencia apuntando a nadie
        keep = [i for i, p in enumerate(pers)
                if p in vivas and (int(p), base) not in malas]
        if not keep:
            return
        vecs = np.concatenate([vecs, z["vecs"][keep].astype(np.float32)])
        personas = np.concatenate([personas, pers[keep]])
        # el staff no tiene año: su foto oficial vale para cualquiera
        ax = z["anios"].astype(int)[keep] if con_anio else np.full(len(keep), -1)
        anios = np.concatenate([anios, ax])
        print("  + %d caras de %s" % (len(keep), etiqueta))

    sumar(REFS_VALETE, "Valete")
    sumar(REFS_ANUARIO, "anuario")
    sumar(REFS_ANUARIO_PDF, "anuario en PDF")
    sumar(REFS_STAFF, "staff", con_anio=False)

    # Las fuentes que se anotan desde la pantalla del padron, una por archivo.
    # Van por glob y no enumeradas: son las que Diego y John van a ir sumando,
    # y tener que agregar una linea aca por cada una era justamente lo que la
    # cola de fuentes vino a sacarse de encima.
    #
    # El patron pasa por faces.npz() para que traiga el sufijo del motor: las
    # huellas de sface son de 128 dimensiones y las de arcface de 512, y
    # mezclarlas no da un error, da resultados sin sentido.
    patron = faces.npz(os.path.join(HERE, "_referencias_fuente_*.npz"))
    for path in sorted(glob.glob(patron)):
        # con sface el patron no lleva sufijo y tomaria tambien los de arcface
        if faces.MOTOR == "sface" and re.search(r"_[a-z]+\.npz$", path):
            continue
        sumar(path, "fuente " + os.path.basename(path))
    return vecs, personas, anios


def pendientes(cli, solo=None):
    """Retratos sin identidad y sin sugerencia previa."""
    fol = {f["id"]: f for f in cli.select("folders", select="id,year,campus,is_mugshot")
           if f.get("is_mugshot") and (solo is None or f["id"] == solo)}
    ya = {r["photo_id"] for r in cli.select("photo_people", select="photo_id")}
    hechos = {r["photo_id"] for r in cli.select("face_suggestions", select="photo_id")}

    ids = list(fol)
    fotos = []
    for i in range(0, len(ids), 40):
        fotos += cli.select("photos", select="id,folder_id,source",
                            folder_id="in.(%s)" % ",".join(ids[i:i + 40]))

    # Zenfolio (Quilmes) tambien: sus huellas ya estan en el cache, asi que
    # entran sin bajar nada. Quedaban afuera por el filtro a "drive" y eran
    # 1.566 retratos que nunca se compararon contra ninguna referencia.
    # Sin fotos repetidas: select() pagina sin orden garantizado, y con otra
    # corrida escribiendo al mismo tiempo una fila puede salir en dos paginas.
    # Aguas abajo eso terminaba en dos sugerencias de la misma (foto, persona)
    # dentro del mismo lote, que Postgres rechaza entero con 21000.
    vistas = {}
    for f in fotos:
        if (f["id"] in ya or f["id"] in hechos
                or f.get("source") not in ("drive", "zenfolio")):
            continue
        vistas[f["id"]] = (f["id"], fol[f["folder_id"]].get("year"), f.get("source"))
    if len(vistas) != sum(1 for f in fotos
                          if f["id"] not in ya and f["id"] not in hechos
                          and f.get("source") in ("drive", "zenfolio")):
        print("  (la lista traia fotos repetidas: se dejo una de cada una)")
    pend = sorted(vistas.values(), key=lambda t: t[0])
    from collections import Counter
    print("Retratos pendientes sin sugerencia: %d  (%s)"
          % (len(pend), ", ".join("%s %d" % (k, v) for k, v in
                                  sorted(Counter(t[2] for t in pend).items()))))
    return pend


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--umbral", type=float, default=faces.umbral("sugerir"))
    ap.add_argument("--carpeta", default=None,
                    help="procesar solo esta carpeta de Drive")
    ap.add_argument("--rehacer", action="store_true",
                    help="borrar las sugerencias de las fotos que siguen "
                         "pendientes y volver a calcularlas (al cambiar de motor)")
    a = ap.parse_args()

    cli = sb.SB()
    vecs, personas, anios = todas_las_referencias(cli)
    print("Referencias: %d caras de %d personas" % (len(vecs), len(set(personas))))

    if a.rehacer:
        # Solo las de retrato, que son las que este script vuelve a calcular.
        #
        # Borraba las de cualquier foto sin nombre, pero pendientes() solo mira
        # carpetas de retrato: las de evento se perdian y no las recalculaba
        # nadie. Las escribe faces_eventos.py, año por año y a hora y media
        # por año, asi que dos --rehacer se llevaron puestas nueve horas de
        # cuenta sin que nada lo dijera.
        retrato = {f["id"] for f in cli.select("folders", select="id,is_mugshot")
                   if f.get("is_mugshot")}
        ya = {r["photo_id"] for r in cli.select("photo_people", select="photo_id")}
        candidatas = {r["photo_id"] for r in
                      cli.select("face_suggestions", select="photo_id")} - ya
        viejas = set()
        ids = list(candidatas)
        for i in range(0, len(ids), 40):
            for p in cli.select("photos", select="id,folder_id",
                                id="in.(%s)" % ",".join(ids[i:i + 40])):
                if p.get("folder_id") in retrato:
                    viejas.add(p["id"])
        if viejas:
            print("Tirando las sugerencias de %d fotos todavia sin nombre…"
                  % len(viejas))
            ids = list(viejas)
            for i in range(0, len(ids), 100):
                cli.delete("face_suggestions",
                           photo_id="in.(%s)" % ",".join(ids[i:i + 100]))

    pend = pendientes(cli, a.carpeta)
    if a.limite:
        pend = pend[:a.limite]
    if not pend:
        print("Nada que hacer.")
        return

    # Las huellas de los retratos ya estan calculadas y guardadas: para esos no
    # hace falta bajar nada. Antes cada corrida rebajaba las 5.000 fotos de
    # Drive y tardaba horas; con el cache es cuestion de segundos, que es lo
    # que permite reprocesar cada vez que crece el padron.
    cache = {}
    if os.path.exists(HUELLAS):
        z = np.load(HUELLAS, allow_pickle=True)
        ids_c, vecs_c = z["photo_ids"], z["vecs"]
        cache = {str(p): vecs_c[i] for i, p in enumerate(ids_c)}
        print("Huellas en cache: %d" % len(cache))

    d = Drive(SA_PATH)
    lock = threading.Lock()
    local = threading.local()
    cuenta = {"n": 0, "sin_cara": 0, "error": 0, "con_sug": 0, "de_cache": 0,
              "zf_sin_cache": 0}

    def mis_caras():
        # los modelos de OpenCV no son thread-safe: uno por hilo
        if not hasattr(local, "c"):
            local.c = faces.Caras()
        return local.c

    def procesar(t):
        pid, anio, src = t
        if pid in cache:
            with lock:
                cuenta["n"] += 1
                cuenta["de_cache"] += 1
            return sugerir(pid, cache[pid], anio)
        # una foto de Zenfolio no se baja de Drive: si no tiene huella en el
        # cache hay que correr faces_huellas.py, no fallar contra la otra API
        if src == "zenfolio":
            with lock:
                cuenta["zf_sin_cache"] += 1
            return []
        try:
            link = d.get("files/" + pid, fields="thumbnailLink").get("thumbnailLink")
            if not link:
                return []
            with urllib.request.urlopen(link.replace("=s220", "=s1600"), timeout=60) as r:
                data = r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            with lock:
                cuenta["error"] += 1
            return []

        hs = mis_caras().huellas(data, max_caras=1)
        with lock:
            cuenta["n"] += 1
            if cuenta["n"] % 200 == 0:
                print("  %d/%d   con sugerencia: %d" % (cuenta["n"], len(pend), cuenta["con_sug"]))
            if not hs:
                cuenta["sin_cara"] += 1
        if not hs:
            return []

        return sugerir(pid, hs[0][0], anio)

    def sugerir(pid, huella, anio):
        s = vecs @ huella
        # Una persona puede tener varias caras de referencia (un mugshot por
        # año): se queda el mejor puntaje de cada una, no todas.
        mejor = {}
        for k in np.argsort(-s)[:60]:
            if s[k] < a.umbral:
                break
            p = int(personas[k])
            if p not in mejor:
                mejor[p] = float(s[k])

        filas = [{"photo_id": pid, "person_id": p, "score": round(v, 4), "rank": i + 1}
                 for i, (p, v) in enumerate(sorted(mejor.items(), key=lambda x: -x[1])[:TOPE])]
        if filas:
            with lock:
                cuenta["con_sug"] += 1
        return filas

    acumulado = []
    TANDA = 400
    try:
        for i in range(0, len(pend), TANDA):
            d._refresh()
            with ThreadPoolExecutor(max_workers=HILOS) as ex:
                for filas in ex.map(procesar, pend[i:i + TANDA]):
                    acumulado += filas
            # se escribe por tanda: si se corta, no se pierde lo hecho
            if acumulado:
                cli.upsert("face_suggestions", acumulado, on_conflict="photo_id,person_id")
                acumulado = []
    except KeyboardInterrupt:
        print("\nInterrumpido: se guarda lo que haya.")
        if acumulado:
            cli.upsert("face_suggestions", acumulado, on_conflict="photo_id,person_id")

    print("\nProcesados: %d   con sugerencia: %d (%.0f%%)"
          % (cuenta["n"], cuenta["con_sug"],
             100.0 * cuenta["con_sug"] / max(cuenta["n"], 1)))
    print("Desde el cache de huellas: %d (sin bajar nada)" % cuenta["de_cache"])
    print("Sin cara detectada: %d   errores de descarga: %d"
          % (cuenta["sin_cara"], cuenta["error"]))
    if cuenta["zf_sin_cache"]:
        print("Zenfolio sin huella en cache: %d  (correr faces_huellas.py)"
              % cuenta["zf_sin_cache"])


if __name__ == "__main__":
    main()
