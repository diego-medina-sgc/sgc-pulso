# -*- coding: utf-8 -*-
"""Un carnet es de UNA persona. Cuando figuran dos, se decide por la cara.

    python carnets_dos_personas.py              # mide y no escribe
    python carnets_dos_personas.py --aplicar

POR QUE

Diego, 13/9/2026, con el captcha de Persona J abierto: "la de la foto de
la portada es Persona K". Un grupo mezclado le habia puesto el
nombre de Lucila a carnets de Lanfranco. Las etiquetas de mas autoridad (nombre
de archivo, un "si") ya se limpiaron con cara_de_otro
(sql/resolver_captcha_respeta.sql). Quedaron 81 carnets no grupales con dos
personas donde las dos etiquetas son de maquina -face_auto, face, cluster,
grupo- y ninguna regla de origen alcanza para decidir.

COMO SE DECIDE

La cara del carnet se compara con el centro de las referencias de cada
persona, calculado SIN las referencias que salen de ese mismo carnet (la cara
no puede ser su propio juez). Gana una si llega a 0,50 -el umbral con que el
proyecto aprueba una cara- y le saca 0,10 a la otra: los mismos numeros que
recuadros_faltantes.py, validados ahi contra 5.102 recuadros humanos (98,3% con
varias caras en la foto).

A la que pierde se le anota el rechazo y se le borra la identificacion, en ese
orden (ver caras_sueltas.py): sin el rechazo la volveria a proponer la
maquina.

LO QUE NO SE TOCA

  - Una etiqueta con autoridad: nombre de archivo, game o manual, o un "si"
    guardado. Si la cara dice otra cosa es un desacuerdo y se informa; lo
    decide una persona (Revisar grupo, el visor).
  - Un carnet donde la cara no decide (ninguna llega o quedan parejas).
  - Un carnet sin huella medida.

La salida dice cuantas veces la cara coincide con la etiqueta de autoridad en
los carnets que tienen una: es la medida de que el juez sirve.
"""
import argparse
import os
import sys
import time
from collections import Counter, defaultdict

if os.environ.get("FACES_MOTOR", "").lower() != "arcface":
    sys.exit("Falta FACES_MOTOR. En PowerShell: $env:FACES_MOTOR='arcface'")

import numpy as np  # noqa: E402

import faces  # noqa: E402
import sb  # noqa: E402
from caras_agrupar_archivo import cargar  # noqa: E402

PARECIDO = 0.50
MARGEN = 0.10
AUTORIDAD = {"mugshot_filename", "filename", "filename_face", "game", "manual"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cli = sb.SB()

    print("Leyendo carnets…", flush=True)
    carnet_folder = {f["id"] for f in cli.select("folders", select="id,is_mugshot") if f["is_mugshot"]}
    ruido = {int(p["id"]) for p in cli.select("people", select="id,kind") if p["kind"] == "noise"}
    nombre = {int(p["id"]): p["display_name"] for p in cli.select("people", select="id,display_name")}
    rechazadas = {(r["photo_id"], int(r["person_id"])) for r in
                  cli.select("photo_people_rejected", select="photo_id,person_id")}
    si = {(l["photo_id"], int(l["person_id"])) for l in
          cli.select("photo_labels", select="photo_id,person_id,answer") if l["answer"] == "yes"}

    por_foto = defaultdict(list)   # carnet -> [(persona, source)]
    for r in cli.select("photo_people", select="photo_id,person_id,source,folder_id"):
        p = int(r["person_id"])
        if r["folder_id"] in carnet_folder and p not in ruido \
                and (r["photo_id"], p) not in rechazadas:
            por_foto[r["photo_id"]].append((p, r["source"] or ""))
    grupales = set()
    ids = [f for f, v in por_foto.items() if len({p for p, _ in v}) > 1]
    for k in range(0, len(ids), 150):
        for ph in cli.select("photos", select="id,is_group", id="in.(%s)" % ",".join(ids[k:k + 150])):
            if ph.get("is_group"):
                grupales.add(ph["id"])
    dudosos = {f: v for f, v in por_foto.items()
               if f not in grupales and len({p for p, _ in v}) > 1}
    print("  carnets no grupales con dos o mas personas: %d" % len(dudosos))
    if not dudosos:
        print("(%.0f s)" % (time.time() - t0))
        return

    print("Leyendo huellas y referencias…", flush=True)
    fotos, cajas, vecs, _, _, _ = cargar()
    huella = {}                    # carnet -> huella de su cara
    area_de = {}
    for i, f in enumerate(fotos):
        if f in dudosos:
            area = float(cajas[i][2]) * float(cajas[i][3])
            if f not in huella or area > area_de[f]:
                huella[f], area_de[f] = vecs[i], area
    # LOS CARNETS VIVEN EN _huellas_retratos, NO EN cargar().
    # La primera corrida (13/9/2026) dio "sin huella medida" a 42 de 52: todos
    # estaban en _huellas_retratos_arcface.npz, que cargar() no lee porque
    # junta caras de eventos y referencias. La huella del retrato es la cara
    # entera del carnet, que es lo que se quiere juzgar.
    ruta_r = faces.npz("_huellas_retratos.npz")
    if os.path.exists(ruta_r):
        with np.load(ruta_r, allow_pickle=True) as z:
            for i, f in enumerate(z["photo_ids"].astype(str)):
                if f in dudosos and f not in huella:
                    huella[f] = z["vecs"][i]
    refs = np.load(faces.npz("_referencias.npz"), allow_pickle=True)
    rv = refs["vecs"].astype(np.float32)
    rv /= np.maximum(np.linalg.norm(rv, axis=1, keepdims=True), 1e-9)
    ref_foto = refs["photo_ids"].astype(str)
    por_persona = defaultdict(list)
    for k, p in enumerate(refs["personas"]):
        por_persona[int(p)].append(k)

    def centro_sin(p, f):
        ix = [k for k in por_persona.get(p, []) if ref_foto[k] != f]
        if not ix:
            return None
        c = rv[ix].sum(axis=0)
        return c / max(float(np.linalg.norm(c)), 1e-9)

    motivo, saca, desacuerdo = Counter(), [], []
    acuerdo = Counter()
    for f, etiquetas in dudosos.items():
        if f not in huella:
            motivo["sin huella medida"] += 1
            continue
        v = np.asarray(huella[f], dtype=np.float32)
        v /= max(float(np.linalg.norm(v)), 1e-9)
        fuentes = defaultdict(set)
        for p, s in etiquetas:
            fuentes[p].add(s)
        puntaje = []
        for p in fuentes:
            c = centro_sin(p, f)
            puntaje.append((float(v @ c) if c is not None else -1.0, p))
        puntaje.sort(reverse=True)
        (s1, gana), s2 = puntaje[0], puntaje[1][0]

        def autoridad(p):
            return bool(fuentes[p] & AUTORIDAD) or (f, p) in si

        con_aut = [p for p in fuentes if autoridad(p)]
        # UNA SOLA ETIQUETA CON AUTORIDAD: gana ella, sin juez de cara.
        #
        # Diego, 13/9/2026, con el carnet "Persona L Alcala.jpg" a nombre
        # de Persona M (archivo) y de Persona N (cara): "confio en
        # lo que dice el nombre de archivo". La cara no podia decidir: las
        # referencias de "Pardini" eran fotos de Victoria. Si el nombre del
        # archivo dice quien es, una etiqueta de maquina en contra no es un
        # desacuerdo, es un error de la maquina.
        if len(con_aut) == 1:
            for p in fuentes:
                if p != con_aut[0]:
                    saca.append((f, p, con_aut[0], 0.0, 0.0))
                    motivo["una sola etiqueta con autoridad: se saca la de maquina"] += 1
            continue
        decide = s1 >= PARECIDO and s1 - s2 >= MARGEN
        if len(con_aut) == 1:
            acuerdo["la cara decide y coincide con la etiqueta de autoridad" if decide and gana == con_aut[0]
                    else "la cara decide OTRA persona que la de autoridad" if decide
                    else "la cara no decide (hay etiqueta de autoridad)"] += 1
        if not decide:
            motivo["la cara no decide (ninguna llega a 0,50 o quedan parejas)"] += 1
            continue
        for s, p in puntaje[1:]:
            if autoridad(p):
                desacuerdo.append((f, p, gana, s1, s))
                motivo["pierde una etiqueta con autoridad: se informa, no se toca"] += 1
            else:
                saca.append((f, p, gana, s1, s))
                motivo["se saca la etiqueta de maquina que pierde"] += 1

    print("\nRESULTADO sobre %d carnets" % len(dudosos))
    for k, n in motivo.most_common():
        print("  %4d  %s" % (n, k))
    print("\nEL JUEZ, CONTRA LAS ETIQUETAS DE AUTORIDAD")
    for k, n in acuerdo.most_common():
        print("  %4d  %s" % (n, k))
    for f, p, g, s1, s in desacuerdo[:20]:
        print("  desacuerdo  %s  figura %s (%.2f), la cara dice %s (%.2f)"
              % (f, nombre.get(p), s, nombre.get(g), s1))
    for f, p, g, s1, s in saca[:30]:
        print("  saca  %-28s (%.2f)  queda %-28s (%.2f)  %s"
              % (nombre.get(p), s, nombre.get(g), s1, f))

    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar. (%.0f s)" % (time.time() - t0))
        return
    if not saca:
        print("\nNada que sacar. (%.0f s)" % (time.time() - t0))
        return

    nom = "_carnets_dos_%s.txt" % time.strftime("%Y%m%d_%H%M")
    with open(nom, "w", encoding="utf-8") as fh:
        fh.write("# photo_id\tsale person_id\tsale nombre\tparecido\tqueda\tparecido\n")
        for f, p, g, s1, s in saca:
            fh.write("%s\t%d\t%s\t%.4f\t%s\t%.4f\n" % (f, p, nombre.get(p), s, nombre.get(g), s1))
    print("\nrespaldo: %s" % nom)
    # primero el rechazo, despues el borrado (ver caras_sueltas.py)
    cli.upsert("photo_people_rejected",
               [{"photo_id": f, "person_id": p} for f, p, _, _, _ in saca],
               on_conflict="photo_id,person_id")
    for f, p, _, _, _ in saca:
        cli.delete("photo_people", person_id="eq.%d" % p, photo_id="eq." + f)
    print("sacadas %d etiquetas (%.0f s)" % (len(saca), time.time() - t0))


if __name__ == "__main__":
    main()
