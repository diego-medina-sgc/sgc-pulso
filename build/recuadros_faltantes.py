# -*- coding: utf-8 -*-
"""Le busca la cara a las identificaciones que no saben donde estan.

    python recuadros_faltantes.py              # mide y no escribe
    python recuadros_faltantes.py --aplicar

POR QUE

Diego: "algunos recuadros de las etiquetas estan en cualquier lado de la foto
en vez de en la cara". Mandó una foto con "Persona AL" arriba a la
izquierda, sobre el pasto.

No era un error de dibujo: faltaba el dato. El 13/9/2026, de 25.984
identificaciones en fotos de evento, 6.459 no tenian recuadro -la sugerencia
no existe o esta en 0,0,0,0- y la app, sin saber donde, las pone en la
esquina. 4.742 venian de nombrar un grupo: resolver_captcha anota la persona
en la foto pero no copia cual de sus caras es.

COMO SE ELIGE LA CARA, DE MAS SEGURO A MENOS

  1. exacta      en la foto hay una cara que esta en un grupo con ESE nombre:
                 esa es. No se adivina nada.
  2. parecido    se compara cada cara detectada en la foto con el centro de
                 las referencias de la persona. Se asigna si la mejor llega a
                 0,50 -el umbral con el que el proyecto aprueba una cara- y,
                 si hay mas de una cara, le gana a la segunda por 0,10.
                 Tambien con una sola cara en la foto: la persona puede estar
                 de espaldas y la unica cara detectada ser la de otro.

Nunca se asigna una cara que ya tiene el recuadro de OTRA persona identificada
en esa foto. Lo que no se puede ubicar queda sin recuadro, y la app lo muestra
en la lista de "En esta foto" y no en la esquina.

QUE ESCRIBE

El recuadro en face_suggestions, que es de donde lo lee la app (photo_faces).
Las parejas estan identificadas, asi que no entran a ninguna cola. El
respaldo -foto, persona, recuadro anterior y nuevo, metodo- se escribe antes.
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--validar", action="store_true",
                    help="probar la regla contra recuadros ya confirmados por una persona")
    a = ap.parse_args()
    t0 = time.time()
    cli = sb.SB()

    print("Leyendo identificaciones y recuadros…", flush=True)
    mugshot = {f["id"] for f in cli.select("folders", select="id,is_mugshot") if f["is_mugshot"]}
    pp = [(r["photo_id"], int(r["person_id"]), r["folder_id"]) for r in
          cli.select("photo_people", select="photo_id,person_id,folder_id")]
    sug = {}
    for s in cli.select("face_suggestions", select="photo_id,person_id,bx,by,bw,bh,score"):
        sug[(s["photo_id"], int(s["person_id"]))] = s
    identificadas = {(f, p) for f, p, _ in pp}
    sin = [(f, p) for f, p, carpeta in pp
           if carpeta not in mugshot and not (float((sug.get((f, p)) or {}).get("bw") or 0) > 0)]
    print("  identificaciones sin recuadro en fotos de evento: %d" % len(sin))

    # VALIDACION: identificaciones cuyo recuadro confirmo una persona. Se les
    # esconde el recuadro y se ve si la regla elige esa misma cara.
    verdad = {}
    if a.validar:
        carpeta_de = {f: c for f, _, c in pp}
        si = {(l["photo_id"], int(l["person_id"])) for l in
              cli.select("photo_labels", select="photo_id,person_id,answer")
              if l["answer"] == "yes"}
        for k in identificadas:
            s = sug.get(k)
            if k in si and s and float(s.get("bw") or 0) > 0 \
                    and carpeta_de.get(k[0]) not in mugshot:
                verdad[k] = (round(float(s["bx"]), 4), round(float(s["by"]), 4))
        print("  para validar: %d recuadros confirmados por una persona" % len(verdad))

    fotos_sin = {f for f, _ in sin} | {f for f, _ in verdad}
    caras = defaultdict(list)      # foto -> [(bx, by, bw, bh, grupo)]
    for g in cli.select("face_groups", select="photo_id,bx,by,bw,bh,grupo"):
        if g["photo_id"] in fotos_sin and float(g["bw"] or 0) > 0:
            caras[g["photo_id"]].append(g)
    nombre_grupo = {int(n["grupo"]): int(n["person_id"]) for n in
                    cli.select("face_group_names", select="grupo,person_id")}

    # "Esta cara no es X", dicho por una persona (grupos_no_son, por cara; ver
    # sql/no_es_esta_cara.sql). Desde el 13/9/2026 corregir un recuadro deja a
    # X en la foto sin recuadro y este script le busca la cara: no puede volver
    # a elegir la que alguien ya dijo que no es.
    no_es = {(r["photo_id"], round(float(r["bx"]), 4), round(float(r["by"]), 4), int(r["person_id"]))
             for r in cli.select("grupos_no_son", select="photo_id,bx,by,person_id")}

    # las caras de cada foto que ya tienen el recuadro de otra persona identificada
    tomada = defaultdict(set)
    for (f, p), s in sug.items():
        if f in fotos_sin and (f, p) in identificadas and float(s.get("bw") or 0) > 0:
            tomada[f].add((round(float(s["bx"]), 4), round(float(s["by"]), 4)))

    print("Leyendo huellas y referencias…", flush=True)
    fotos, cajas, vecs, _, _, _ = cargar()
    clave = {}
    for i, f in enumerate(fotos):
        if f in fotos_sin:
            clave.setdefault((f, round(float(cajas[i][0]), 4), round(float(cajas[i][1]), 4)), i)
    refs = np.load(faces.npz("_referencias.npz"), allow_pickle=True)
    rv = refs["vecs"].astype(np.float32)
    rv /= np.maximum(np.linalg.norm(rv, axis=1, keepdims=True), 1e-9)
    ref_foto = refs["photo_ids"].astype(str)
    centro, suma_p = {}, {}
    por_persona = defaultdict(list)
    for k, p in enumerate(refs["personas"]):
        por_persona[int(p)].append(k)
    for p, ix in por_persona.items():
        c = rv[ix].sum(axis=0)
        suma_p[p] = c
        centro[p] = c / max(float(np.linalg.norm(c)), 1e-9)

    def centro_sin_foto(p, f):
        """El centro de la persona sin las referencias que salen de esa foto:
        al validar, la cara que se juzga no puede ser parte de su propio juez."""
        if p not in suma_p:
            return None
        ix = [k for k in por_persona[p] if ref_foto[k] == f]
        if not ix:
            return centro[p]
        if len(ix) == len(por_persona[p]):
            return None
        c = suma_p[p] - rv[ix].sum(axis=0)
        return c / max(float(np.linalg.norm(c)), 1e-9)
    print("  %d personas con referencias (%.0f s)" % (len(centro), time.time() - t0), flush=True)

    def elegir(f, p, libres, centro_p):
        """(cara, metodo, parecido) o (None, motivo, None). La MISMA regla para
        validar y para escribir: lo que se mide es lo que se aplica."""
        # "El grupo con ese nombre" SOLO no alcanza: validado el 13/9/2026
        # acerto 91,7% (1.529 de 1.668), contra 98,4% y 100% del parecido. Los
        # grupos nombrados arrastran caras de otra persona -Persona J tenia
        # 21 carnets de Persona K-. Se usa solo como desempate
        # cuando el parecido no alcanza el margen pero la cara del grupo es la
        # mas parecida.
        exactas = [c for c in libres if nombre_grupo.get(int(c["grupo"])) == p]
        if centro_p is None:
            return None, "la persona no tiene referencias", None
        puntaje = []
        for c in libres:
            i = clave.get((f, round(float(c["bx"]), 4), round(float(c["by"]), 4)))
            if i is None:
                continue
            v = vecs[i].astype(np.float32)
            v /= max(float(np.linalg.norm(v)), 1e-9)
            puntaje.append((float(v @ centro_p), c))
        puntaje.sort(key=lambda x: -x[0])
        if not puntaje:
            return None, "caras sin huella", None
        s1 = puntaje[0][0]
        s2 = puntaje[1][0] if len(puntaje) > 1 else -1.0
        if s1 >= PARECIDO and s1 - s2 >= MARGEN:
            return (puntaje[0][1],
                    "parecido (una cara)" if len(puntaje) == 1 else "parecido (varias caras)", s1)
        if s1 >= PARECIDO and len(exactas) == 1 and exactas[0] is puntaje[0][1]:
            return puntaje[0][1], "parecido + grupo con ese nombre", s1
        return None, "parecido insuficiente o dos caras parejas", None

    if a.validar:
        res = Counter()
        for (f, p), real in verdad.items():
            detectadas = caras.get(f, [])
            claves_det = {(round(float(c["bx"]), 4), round(float(c["by"]), 4)) for c in detectadas}
            if real not in claves_det:
                res["(fuera: la cara confirmada no esta en el mapa de caras)"] += 1
                continue
            # tomadas por OTRA persona: la propia no cuenta como tomada
            otras = tomada[f] - {real}
            libres = [c for c in detectadas
                      if (round(float(c["bx"]), 4), round(float(c["by"]), 4)) not in otras]
            elegida, metodo, _ = elegir(f, p, libres, centro_sin_foto(p, f))
            if elegida is None:
                res["no asigna: " + metodo] += 1
                continue
            ok = (round(float(elegida["bx"]), 4), round(float(elegida["by"]), 4)) == real
            res[metodo + (" -> ACIERTA" if ok else " -> ERRA")] += 1
        print("\nVALIDACION sobre %d recuadros confirmados por una persona" % len(verdad))
        for k in sorted(res):
            print("  %6d  %s" % (res[k], k))
        for m in ("exacta (grupo con ese nombre)", "parecido (una cara)", "parecido (varias caras)"):
            bien, mal = res[m + " -> ACIERTA"], res[m + " -> ERRA"]
            if bien + mal:
                print("  %-32s acierto %.1f%%  (%d de %d)" % (m, 100.0 * bien / (bien + mal), bien, bien + mal))
        print("\n(%.0f s)" % (time.time() - t0))
        return

    asignar, motivo = [], Counter()
    for f, p in sin:
        libres = [c for c in caras.get(f, [])
                  if (round(float(c["bx"]), 4), round(float(c["by"]), 4)) not in tomada[f]
                  and (f, round(float(c["bx"]), 4), round(float(c["by"]), 4), p) not in no_es]
        if not caras.get(f):
            motivo["sin ninguna cara detectada"] += 1
            continue
        if not libres:
            motivo["todas las caras ya son de otra persona identificada"] += 1
            continue
        elegida, metodo, s1 = elegir(f, p, libres, centro.get(p))
        if elegida is None:
            motivo[metodo] += 1
            continue
        motivo[metodo] += 1
        tomada[f].add((round(float(elegida["bx"]), 4), round(float(elegida["by"]), 4)))
        viejo = sug.get((f, p)) or {}
        asignar.append({"photo_id": f, "person_id": p,
                        "score": round(s1, 4) if s1 is not None else float(viejo.get("score") or 1.0),
                        "rank": 1,
                        "bx": elegida["bx"], "by": elegida["by"],
                        "bw": elegida["bw"], "bh": elegida["bh"],
                        "_metodo": metodo,
                        "_antes": "%s,%s,%s,%s" % (viejo.get("bx"), viejo.get("by"),
                                                   viejo.get("bw"), viejo.get("bh"))})

    print("\nRESULTADO sobre %d identificaciones sin recuadro" % len(sin))
    for k, n in motivo.most_common():
        print("  %5d  %s" % (n, k))
    print("  -> se asignarian %d (%.0f%%)" % (len(asignar), 100.0 * len(asignar) / max(len(sin), 1)))

    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar. (%.0f s)" % (time.time() - t0))
        return

    nom = "_recuadros_%s.txt" % time.strftime("%Y%m%d_%H%M")
    with open(nom, "w", encoding="utf-8") as fh:
        fh.write("# photo_id\tperson_id\tmetodo\tparecido\tantes (bx,by,bw,bh)\tnuevo\n")
        for r in asignar:
            fh.write("%s\t%s\t%s\t%s\t%s\t%s,%s,%s,%s\n" % (
                r["photo_id"], r["person_id"], r["_metodo"], r["score"], r["_antes"],
                r["bx"], r["by"], r["bw"], r["bh"]))
    print("\nrespaldo: %s" % nom)
    filas = [{k: v for k, v in r.items() if not k.startswith("_")} for r in asignar]
    cli.upsert("face_suggestions", filas, on_conflict="photo_id,person_id")
    print("escritos %d recuadros (%.0f s)" % (len(filas), time.time() - t0))


if __name__ == "__main__":
    main()
