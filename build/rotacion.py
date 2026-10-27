# -*- coding: utf-8 -*-
"""Llena photos.rotation con lo que informa Drive, por carpeta.

POR QUE HACE FALTA. Las fotos de celular tomadas en vertical se ven acostadas
en la app: Drive entrega el bitmap -y la miniatura- SIN girar y dice cuanto hay
que girarlo en imageMediaMetadata.rotation (0 a 3, cuartos de vuelta en sentido
del reloj). sync_drive.py no pedia ese campo, asi que nunca llego a la base.
Desde el 26/10/2026 lo pide y lo escribe en photos.rotation para las fotos que
pasan por el sync; este script es para las que ya estaban indexadas.

NULO NO ES 0. Nulo = todavia no se leyo. 0 = no hay que girarla. Drive a veces
tiene la foto y NO informa rotation (escaneos, PNG, fotos sin EXIF; medido el
26/10/2026: 8 de 13 en una carpeta de escaneos). Diego decidio ese dia que
esas valen 0: para la app es lo mismo -no girar- y la foto deja de mirarse en
cada tramo del barrido. Si quedaran en nulo, el barrido las volveria a pedir a
Drive para siempre y salud() las contaria como pendientes para siempre.
Este script NUNCA escribe nulo arriba de un dato.

SE PIDE POR CARPETA, NO POR FOTO. Un files.list por carpeta trae hasta 1.000
fotos con su rotation en una sola llamada. Medido el 26/10/2026 con 8 hilos
(el punto medido del sync): 0,32 s por carpeta, 40 carpetas y 889 fotos en
12,7 s. Las 10.073 carpetas con fotos de Drive sin leer son ~54 minutos de
listados, que es por que el barrido grande va de a tramos y no de una.

Uso:
    python rotacion.py --carpeta <drive_id>          # ensayo de un album
    python rotacion.py --carpeta <drive_id> --aplicar
    python rotacion.py --tope 200                    # un tramo de 200 carpetas
    python rotacion.py --tope 200 --aplicar
"""
import argparse
import collections
import datetime as _dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sb                                                        # noqa: E402
from sync_drive import Drive, HILOS, giro                         # noqa: E402

CAMPOS = "nextPageToken,files(id,name,mimeType,imageMediaMetadata(rotation))"
# Cuantos ids entran en un PATCH. El id de Drive son 33 caracteres: 300 ids son
# ~10 KB de URL, que PostgREST acepta sin chistar.
POR_PATCH = 300


# Que paso del pulso frena una escritura a mano esta en sb.pasos_del_pulso().

# EL BARRIDO VA EN TRAMOS, Y EL CURSOR VIVE EN LA BASE (27/10/2026)
#
# Son 10.073 carpetas con fotos. El paso "rotacion" hace un tramo por vuelta y
# anota donde quedo en corridas (tarea 'rotacion', etiqueta 'barrido'), igual
# que el barrido de duplicados: en la base y no en un archivo, porque el runner
# del pulso arranca vacio cada vez. De a tramos y no de corrido porque una
# vuelta del pulso tiene mas pasos atras y nadie espera un paso largo: el
# barrido entero son ~17 vueltas, unos 12 minutos de listados sumados.
#
# El tramo se toma de la lista ORDENADA de carpetas con fotos (1,9 s de
# lectura), no de las pendientes (354 s y 28,8 MB para dar las mismas): el
# cursor es el id de la ultima carpeta mirada y la vuelta siguiente sigue
# desde ahi. Cuando llega al final queda terminado=True; a los REPASO_DIAS se
# arranca otra pasada, para las fotos que entraron despues.
CURSOR = {"tarea": "rotacion", "etiqueta": "barrido"}
CARPETAS_POR_VUELTA = 600          # 42 s de listados, medido el 27/10/2026
                                   # (600 carpetas, 26.421 fotos, 8 hilos:
                                   # 0,07 s por carpeta). El tope lo pone la
                                   # escritura, ~88 PATCH de 300 ids.
REPASO_DIAS = 7


def leer_cursor(cli):
    """El estado del barrido, de la base: {"cursor", "terminado", "pasada"}."""
    filas = cli.select("corridas", select="mensaje", tarea="eq.%s" % CURSOR["tarea"],
                       etiqueta="eq.%s" % CURSOR["etiqueta"]) or []
    if not filas:
        return {}
    m = filas[0].get("mensaje")
    if isinstance(m, str):
        try:
            m = json.loads(m)
        except ValueError:
            return {}
    return m or {}


def anotar_cursor(cli, est, total, hechas):
    """Deja el cursor en la base. Solo se llama si la escritura salio bien: si
    el tramo fallo, la vuelta siguiente repite ese tramo y no saltea carpetas."""
    cli.upsert("corridas", [dict(CURSOR, estado="listo", total=total, hechas=hechas,
                                 mensaje=json.dumps(est),
                                 actualizado_at=_dt.datetime.now(
                                     _dt.timezone.utc).isoformat())],
               on_conflict="tarea,etiqueta")


def tramo_del_pulso(cli):
    """(carpetas del tramo, estado nuevo). Lista vacia = no hay nada que hacer."""
    est = leer_cursor(cli)
    terminado = bool(est.get("terminado"))
    if terminado:
        desde_cuando = est.get("terminado_at")
        if desde_cuando:
            try:
                fin = _dt.datetime.fromisoformat(desde_cuando)
                if (_dt.datetime.now(_dt.timezone.utc) - fin).days < REPASO_DIAS:
                    return [], est
            except ValueError:
                pass
        est = {"pasada": int(est.get("pasada") or 1) + 1, "cursor": None}
    todas = carpetas_con_fotos(cli, est.get("cursor"))
    # carpetas_con_fotos incluye la del cursor: se saltea, ya se miro
    if est.get("cursor"):
        todas = [c for c in todas if c[0] > est["cursor"]]
    tramo = todas[:CARPETAS_POR_VUELTA]
    nuevo = {"pasada": int(est.get("pasada") or 1), "cursor": est.get("cursor")}
    if tramo:
        nuevo["cursor"] = tramo[-1][0]
    if len(todas) <= CARPETAS_POR_VUELTA:
        nuevo["terminado"] = True
        nuevo["terminado_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    return tramo, nuevo


def fotos_de(drive, carpeta):
    """{id de foto: rotation} de una carpeta de Drive. Solo imagenes."""
    out, tok = {}, None
    while True:
        p = {"q": "'%s' in parents and trashed=false and mimeType contains 'image/'" % carpeta,
             "pageSize": "1000", "fields": CAMPOS, "supportsAllDrives": "true",
             "includeItemsFromAllDrives": "true"}
        if tok:
            p["pageToken"] = tok
        r = drive._pedir("GET", "files", params=p)
        for f in r.get("files", []):
            # Drive la lista como imagen: si no dice rotation, es 0 (ver arriba).
            rot = giro(f.get("imageMediaMetadata") or {})
            out[f["id"]] = 0 if rot is None else rot
        tok = r.get("nextPageToken")
        if not tok:
            return out


def carpetas_con_fotos(cli, desde=None):
    """Las carpetas que tienen fotos, por id. Ordenadas: un barrido que se
    corta se sigue con --desde <el ultimo id que imprimio>.

    Sale de folders.photo_count, no de leer las fotos pendientes: la lista de
    las 565.145 fotos sin rotation leida tarda 354 s y baja 28,8 MB (medido el
    26/10/2026), y da las mismas carpetas."""
    filas = cli.select("folders", select="id,photo_count", photo_count="gt.0") or []
    ids = sorted(f["id"] for f in filas)
    if desde:
        ids = [i for i in ids if i >= desde]
    return [(i, None) for i in ids]


def carpetas_pendientes(cli, tope):
    """Carpetas con fotos de Drive sin rotation leida, las mas chicas primero.

    Las mas chicas primero no es un capricho: un tramo entra entero y el
    siguiente arranca limpio, en vez de cortarse en el medio de un album de
    3.000 fotos."""
    # No hay funcion en la base que las devuelva y no se pide una por un
    # barrido: se arma leyendo las fotos pendientes, que es una sola lectura
    # paginada por clave.
    cuenta = collections.Counter()
    for f in cli.select("photos", select="folder_id", source="eq.drive",
                        rotation="is.null"):
        if f.get("folder_id"):
            cuenta[f["folder_id"]] += 1
    orden = sorted(cuenta.items(), key=lambda kv: (kv[1], kv[0]))
    return orden[:tope] if tope else orden


def escribir(cli, por_valor, a):
    """Un PATCH por valor y por tanda de ids. Nunca escribe nulo."""
    escritas = 0
    for valor in sorted(por_valor):
        ids = por_valor[valor]
        for i in range(0, len(ids), POR_PATCH):
            tanda = ids[i:i + POR_PATCH]
            if a.aplicar:
                cli.update("photos", {"rotation": valor},
                           id="in.(%s)" % ",".join(tanda))
            escritas += len(tanda)
    return escritas


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--carpeta", action="append", default=[],
                    help="una carpeta de Drive (se puede repetir)")
    ap.add_argument("--tope", type=int, default=0,
                    help="cuantas carpetas pendientes mirar (tramo)")
    ap.add_argument("--todas", action="store_true",
                    help="todas las carpetas con fotos, de corrido")
    ap.add_argument("--desde", help="seguir un barrido cortado desde ese id")
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--pulso", action="store_true",
                    help="un tramo del barrido, siguiendo el cursor de la base")
    ap.add_argument("--igual-con-el-pulso", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                             # noqa: BLE001
        pass
    if not a.carpeta and not a.tope and not a.todas and not a.pulso:
        ap.error("--carpeta, --tope, --todas o --pulso")

    cli = sb.SB()
    drive = Drive(os.path.join(HERE, "service-account.json"))

    est_nuevo = None
    if a.pulso:
        t = time.time()
        objetivo, est_nuevo = tramo_del_pulso(cli)
        print("Barrido: pasada %s, tramo de %d carpetas (%.1f s de lectura)"
              % (est_nuevo.get("pasada") or 1, len(objetivo), time.time() - t))
        if est_nuevo.get("terminado"):
            print("   (este es el ultimo tramo de la pasada)")
    elif a.carpeta:
        objetivo = [(c, None) for c in a.carpeta]
    elif a.todas:
        t = time.time()
        objetivo = carpetas_con_fotos(cli, a.desde)
        if a.tope:
            objetivo = objetivo[:a.tope]
        print("Carpetas con fotos: %d (%.1f s de lectura)"
              % (len(objetivo), time.time() - t))
    else:
        t = time.time()
        objetivo = carpetas_pendientes(cli, a.tope)
        print("Carpetas pendientes en este tramo: %d (%.1f s de lectura)"
              % (len(objetivo), time.time() - t))
    if not objetivo:
        print("No hay nada pendiente.")
        return

    t0 = time.time()
    por_valor = collections.defaultdict(list)
    resumen = collections.Counter()
    detalle = []
    with ThreadPoolExecutor(HILOS) as pool:
        for (cid, _), fotos in zip(objetivo, pool.map(
                lambda kv: fotos_de(drive, kv[0]), objetivo)):
            for pid, r in sorted(fotos.items()):
                # fotos_de ya resolvio el "Drive no lo informa" en 0: aca no
                # llega nulo nunca, y toda foto listada se escribe.
                resumen["rot=%s" % r] += 1
                por_valor[r].append(pid)
                detalle.append((cid, pid, r))
    segundos = time.time() - t0
    print("Carpetas miradas: %d   fotos: %d   %.1f s (%.2f s por carpeta, %d hilos)"
          % (len(objetivo), sum(resumen.values()), segundos,
             segundos / max(1, len(objetivo)), HILOS))
    for k, v in sorted(resumen.items()):
        print("   %-34s %d" % (k + ":", v))
    print("Para escribir: %d fotos" % sum(len(v) for v in por_valor.values()))

    sello = _dt.datetime.now().strftime("%Y%m%d_%H%M")
    respaldo = os.path.join(HERE, "_rotacion_%s.txt" % sello)
    with open(respaldo, "w", encoding="utf-8") as fh:
        fh.write("# rotacion.py %s  %s\n" % (sello, "APLICADO" if a.aplicar else "ensayo"))
        fh.write("# revertir: update photos set rotation = null where id in (...)\n")
        fh.write("# carpeta\tfoto\trotation\n")
        for cid, pid, r in detalle:
            fh.write("%s\t%s\t%s\n" % (cid, pid, r))
    print("Respaldo: %s" % respaldo)

    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar.")
        return
    frenan, otros = sb.pasos_del_pulso(cli)
    if otros:
        print("\n(el pulso esta en %s; no toca photos.rotation, sigo)" % ", ".join(otros))
    if frenan and not a.igual_con_el_pulso:
        print("\nNO escribo: el pulso esta en %s. Probar de nuevo despues."
              % ", ".join(frenan))
        sys.exit(2)
    n = escribir(cli, por_valor, a)
    print("Escritas: %d" % n)
    # SE VERIFICA, NO SE SUPONE: un PATCH que no toco ninguna fila devuelve 204
    # igual que uno que toco 300.
    ids = [pid for _, pid, _ in detalle]
    quedaron = 0
    for i in range(0, len(ids), POR_PATCH):
        tanda = ids[i:i + POR_PATCH]
        hay = cli.select("photos", select="id,rotation",
                         id="in.(%s)" % ",".join(tanda)) or []
        quedaron += sum(1 for f in hay if f.get("rotation") is not None)
    print("Verificadas con rotation en la base: %d de %d" % (quedaron, len(ids)))
    # El cursor se mueve DESPUES de escribir y verificar: si el tramo se
    # corta, la vuelta siguiente lo repite en vez de saltear carpetas.
    if a.pulso and est_nuevo is not None:
        anotar_cursor(cli, est_nuevo, len(objetivo), quedaron)
        print("Cursor: %s%s" % (est_nuevo.get("cursor"),
                                "  (pasada terminada)" if est_nuevo.get("terminado") else ""))


if __name__ == "__main__":
    main()
