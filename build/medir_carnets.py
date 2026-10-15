# -*- coding: utf-8 -*-
"""Cuanto se equivoca la maquina cuando nombra un carnet. Solo mide.

    python medir_carnets.py              # mide y muestra totales; no escribe
    python medir_carnets.py --anotar     # ademas deja el resultado en corridas

Corre en el pulso solo a pedido:
    gh workflow run pulso.yml --repo <dueño>/sgc-pulso -f paso=medir_carnets

POR QUE (15/10/2026)

Diego decide con este numero si los carnets que nombro la maquina pueden ser
foto de perfil: tolera ~1,6% de error. La primera medicion (14/10) se hizo en
la notebook con huellas de septiembre y a 2.922 carnets les faltaba la cara;
las huellas al dia viven solo en el pulso (cifradas), asi que se mide aca.

QUE MIDE

Universo: carnets (fotos primarias de carpetas is_mugshot) que la maquina le
puso a una persona (source face_auto, cluster, face o grupo) sin un "si" ni
el nombre del archivo que lo respalde, de personas que tienen algo confirmado
por una persona. Cada carnet se compara contra las caras de ESA ficha que
confirmo una persona -game/manual con "si", o carnet por nombre de archivo-,
sin la propia foto y sin las caras en disputa abierta. Coincide si el parecido
maximo llega a faces.umbral("sugerir") (0,50 con arcface); se informan tambien
0,60 y 0,63 y los que quedan debajo de 0,30 (otra persona sin discusion).

Ademas: por source, por kind, por desfase de camada (la escalera de
enrollments: PREK, K1-K3, P1-P6, C1-C6; el nivel L se cursa en
cohort - (15 - indice(L))), y la portada de hoy (people.foto_perfil que es un
carnet de maquina).

QUE DEJA

El repo del pulso y los logs de Actions son publicos: en el log van solo
totales y porcentajes. El detalle (ids de ficha y de foto de los que no
coinciden) va a corridas (tarea 'medicion', etiqueta 'carnets_maquina'), en
la base. Nunca un nombre.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

import faces
import sb

MAQ = ("face_auto", "cluster", "face", "grupo")
ARCHIVO = ("filename", "mugshot_filename", "filename_face")
# Debajo de esto es otra persona sin discusion: el maximo medido entre dos
# personas distintas en la calibracion fue 0,473; 0,30 deja un margen ancho.
CLARAMENTE_OTRA = 0.30
ESCALERA = ["PREK", "K1", "K2", "K3"] + ["P%d" % n for n in range(1, 7)] + \
    ["C%d" % n for n in range(1, 7)]
EJEMPLOS = 60          # cuantos "no coincide" se guardan en corridas


def leer(cli):
    gente = {int(p["id"]): p for p in cli.select("people", select="id,kind,cohort,foto_perfil")}
    pp = cli.select("photo_people", select="photo_id,person_id,source")
    si = {(r["photo_id"], int(r["person_id"])) for r in
          cli.select("photo_labels", select="photo_id,person_id,answer", answer="eq.yes")
          if r.get("person_id") is not None}
    rech = {(r["photo_id"], int(r["person_id"])) for r in
            cli.select("photo_people_rejected", select="photo_id,person_id")}
    carp = {f["id"]: f for f in cli.select("folders", select="id,is_mugshot,levels,year")
            if f.get("is_mugshot")}
    ids = list(carp)
    carnet = {}
    for i in range(0, len(ids), 40):
        for ph in cli.select("photos", select="id,folder_id,is_primary",
                             folder_id="in.(%s)" % ",".join(ids[i:i + 40])):
            if ph.get("is_primary"):
                carnet[ph["id"]] = ph["folder_id"]
    return gente, pp, si, rech, carp, carnet, faces.disputas_abiertas(cli)


def medir(gente, pp, si, rech, carp, carnet, disputadas, huellas, referencias):
    """Devuelve (filas, faltan) con una fila por carnet del universo:
    (persona, foto, source, kind, estado, parecido, desfase)."""
    hfotos, hV = huellas
    rf, rp, rV = referencias
    huella = {f: i for i, f in enumerate(hfotos)}
    ref_idx = {(f, int(p)): i for i, (f, p) in enumerate(zip(rf, rp))}

    hum_de = defaultdict(set)
    for r in pp:
        f, p, s = r["photo_id"], int(r["person_id"]), r["source"]
        if (f, p) in rech or (f, p) in disputadas:
            continue
        if (s in ("game", "manual") and (f, p) in si) or s in ARCHIVO:
            hum_de[p].add(f)

    def refs(p, sin):
        out = []
        for f in hum_de.get(p, ()):
            if f == sin:
                continue
            i = ref_idx.get((f, p))
            if i is not None:
                out.append(rV[i])
            elif f in carnet and f in huella:
                out.append(hV[huella[f]])
        return np.stack(out) if out else None

    def desfase(p, f):
        c = (gente.get(p) or {}).get("cohort")
        fo = carp.get(carnet.get(f))
        if not c or not fo or not fo.get("year") or not fo.get("levels"):
            return None
        ds = [abs(int(fo["year"]) - (int(c) - (15 - ESCALERA.index(l))))
              for l in fo["levels"] if l in ESCALERA]
        return min(ds) if ds else None

    filas = []
    for r in pp:
        f, p, s = r["photo_id"], int(r["person_id"]), r["source"]
        if s not in MAQ or f not in carnet or (f, p) in rech or (f, p) in si:
            continue
        kind = (gente.get(p) or {}).get("kind")
        if kind in (None, "noise") or not hum_de.get(p):
            continue
        d = desfase(p, f)
        if f not in huella:
            filas.append((p, f, s, kind, "sin cara del carnet en las huellas", None, d))
            continue
        R = refs(p, f)
        if R is None:
            filas.append((p, f, s, kind, "persona sin caras humanas en las huellas", None, d))
            continue
        filas.append((p, f, s, kind, "juzgable", float((R @ hV[huella[f]]).max()), d))
    return filas


def resumen(filas, clave):
    out = {}
    grupos = defaultdict(list)
    for x in filas:
        grupos[str(clave(x))].append(x)
    grupos["TOTAL"] = list(filas)
    for k, g in grupos.items():
        j = [x for x in g if x[4] == "juzgable"]

        def bajo(u):
            return sum(1 for x in j if x[5] < u)
        no = bajo(faces.umbral("sugerir"))
        out[k] = {
            "total": len(g), "juzgables": len(j), "coincide": len(j) - no, "no_coincide": no,
            "error_pct": round(100.0 * no / len(j), 2) if j else None,
            "error_060_pct": round(100.0 * bajo(0.60) / len(j), 2) if j else None,
            "error_063_pct": round(100.0 * bajo(0.63) / len(j), 2) if j else None,
            "claramente_otra": bajo(CLARAMENTE_OTRA),
            "claramente_otra_pct": round(100.0 * bajo(CLARAMENTE_OTRA) / len(j), 2) if j else None,
            "no_juzgables": dict(Counter(x[4] for x in g if x[4] != "juzgable")),
        }
    return out


def tramo_desfase(x):
    d = x[6]
    return "sin dato" if d is None else "0" if d == 0 else "1" if d == 1 else "2+"


def imprimir(titulo, res):
    print("\n%s" % titulo)
    print("   %-26s %7s %9s %8s %7s %7s %7s %7s %8s"
          % ("", "total", "juzgables", "coincide", "no", "err%", "@0,60", "@0,63", "<0,30%"))
    for k in sorted(res, key=lambda y: (y == "TOTAL", y)):
        v = res[k]
        f = (lambda z: "-" if z is None else "%.2f" % z)
        print("   %-26s %7d %9d %8d %7d %7s %7s %7s %8s"
              % (k[:26], v["total"], v["juzgables"], v["coincide"], v["no_coincide"],
                 f(v["error_pct"]), f(v["error_060_pct"]), f(v["error_063_pct"]),
                 f(v["claramente_otra_pct"])))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--anotar", action="store_true",
                    help="deja el resultado en corridas (tarea medicion)")
    a = ap.parse_args()
    if faces.MOTOR != "arcface":
        sys.exit("Falta FACES_MOTOR=arcface")
    t0 = time.time()
    cli = sb.SB()

    with np.load(faces.npz("_huellas_retratos.npz"), allow_pickle=True) as z:
        hfotos = [str(x) for x in z["photo_ids"]]
        hV = z["vecs"].astype(np.float32)
    hV /= np.maximum(np.linalg.norm(hV, axis=1, keepdims=True), 1e-9)
    with np.load(faces.npz("_referencias.npz"), allow_pickle=True) as z:
        rf = [str(x) for x in z["photo_ids"]]
        rp = z["personas"].astype(int)
        rV = z["vecs"].astype(np.float32)
    rV /= np.maximum(np.linalg.norm(rV, axis=1, keepdims=True), 1e-9)

    gente, pp, si, rech, carp, carnet, disputadas = leer(cli)
    filas = medir(gente, pp, si, rech, carp, carnet, disputadas, (hfotos, hV), (rf, rp, rV))

    # la portada de hoy: fichas cuya foto de perfil es un carnet de maquina
    portada = [x for x in filas if (gente.get(x[0]) or {}).get("foto_perfil") == x[1]]

    alumnos = [x for x in filas if x[3] in ("student", "alumni")]
    res = {
        "medido_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "motor": faces.MOTOR, "corte": faces.umbral("sugerir"),
        "huellas_retratos": len(hfotos), "referencias": len(rf),
        "carnets_primarios": len(carnet), "disputas_abiertas": len(disputadas),
        "universo": len(filas), "personas": len({x[0] for x in filas}),
        "por_source": resumen(filas, lambda x: x[2]),
        "por_kind": resumen(filas, lambda x: x[3]),
        "alumnos_por_desfase": resumen(alumnos, tramo_desfase),
        "portada": {"fichas": len(portada), "por_source": resumen(portada, lambda x: x[2])},
    }
    malos = sorted([x for x in filas if x[4] == "juzgable" and x[5] < faces.umbral("sugerir")],
                   key=lambda x: x[5])
    res["no_coincide"] = [{"person_id": x[0], "photo_id": x[1], "source": x[2],
                           "parecido": round(x[5], 3), "desfase": x[6],
                           "portada": x in portada}
                          for x in malos[:EJEMPLOS]]

    # al log, solo numeros
    print("Huellas de carnets: %d   referencias: %d   carnets primarios: %d   disputas abiertas: %d"
          % (len(hfotos), len(rf), len(carnet), len(disputadas)))
    print("Universo: %d carnets de maquina de %d fichas con algo humano"
          % (res["universo"], res["personas"]))
    imprimir("POR SOURCE (corte %.2f)" % faces.umbral("sugerir"), res["por_source"])
    imprimir("POR KIND", res["por_kind"])
    imprimir("ALUMNOS POR DESFASE DE CAMADA (anios)", res["alumnos_por_desfase"])
    imprimir("PORTADA DE HOY: %d fichas con un carnet de maquina" % len(portada),
             res["portada"]["por_source"])
    print("\n(%.0f s)" % (time.time() - t0))

    if not a.anotar:
        print("Ensayo: no se escribio nada. Con --anotar queda en corridas (tarea medicion).")
        return 0
    cli.upsert("corridas", [{
        "tarea": "medicion", "etiqueta": "carnets_maquina", "estado": "listo",
        "mensaje": json.dumps(res, ensure_ascii=False),
        "total": res["universo"], "hechas": res["por_source"]["TOTAL"]["juzgables"],
        "encontradas": res["por_source"]["TOTAL"]["no_coincide"], "con_gente": len(portada),
        "actualizado_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }], on_conflict="tarea,etiqueta")
    print("Anotado en corridas: tarea 'medicion', etiqueta 'carnets_maquina'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
