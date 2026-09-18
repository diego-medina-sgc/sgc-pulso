# -*- coding: utf-8 -*-
"""Los candidatos del juego Fusionar, con el parecido de sus caras.

    python fichas_parecidas.py              # mide y no escribe
    python fichas_parecidas.py --aplicar

POR QUE

El juego Fusionar (sql/fusionar_juego.sql) pone dos fichas lado a lado para que
una persona decida si son la misma. Los candidatos salen del nombre
(fichas_candidatas: un nombre contenido en el otro, o nombres muy parecidos),
pero el nombre solo no alcanza: "Persona R" y "Persona S" pueden
ser padre e hijo. La cara si ayuda: dos fichas de la misma persona tienen caras
que se parecen.

Este script agrega a cada par cuanto se parecen los CENTROS de las caras de
referencia de las dos fichas (_referencias.npz). El juego muestra primero los
pares de caras parecidas. No decide nada: ordena.

Con --aplicar reescribe fichas_parecidas entera.
"""
import argparse
import os
import sys
import time
from collections import defaultdict

if os.environ.get("FACES_MOTOR", "").lower() != "arcface":
    sys.exit("Falta FACES_MOTOR. En PowerShell: $env:FACES_MOTOR='arcface'")

import numpy as np  # noqa: E402

import faces  # noqa: E402
import sb  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cli = sb.SB()

    print("Buscando candidatos por nombre…", flush=True)
    pares = cli.rpc("fichas_candidatas") or []
    print("  pares: %d   (%.0f s)" % (len(pares), time.time() - t0), flush=True)

    refs = np.load(faces.npz("_referencias.npz"), allow_pickle=True)
    rv = refs["vecs"].astype(np.float32)
    rv /= np.maximum(np.linalg.norm(rv, axis=1, keepdims=True), 1e-9)
    por = defaultdict(list)
    for k, p in enumerate(refs["personas"]):
        por[int(p)].append(k)
    centro = {}
    for p, ix in por.items():
        c = rv[ix].sum(axis=0)
        centro[p] = c / max(float(np.linalg.norm(c)), 1e-9)

    filas, con_cara = [], 0
    for r in pares:
        x, y = int(r["a_id"]), int(r["b_id"])
        s = None
        if x in centro and y in centro:
            s = round(float(centro[x] @ centro[y]), 4)
            con_cara += 1
        filas.append({"a_id": x, "b_id": y, "regla": r["regla"],
                      "parecido_nombre": r["parecido_nombre"], "parecido_cara": s})

    print("  con caras en las dos fichas: %d" % con_cara)
    altos = sorted((f for f in filas if f["parecido_cara"] is not None),
                   key=lambda f: -f["parecido_cara"])
    print("  caras parecidas (>= 0,5): %d" % sum(1 for f in altos if f["parecido_cara"] >= 0.5))
    nombres = {int(p["id"]): p["display_name"] for p in cli.select("people", select="id,display_name")}
    for f in altos[:15]:
        print("   %.2f  %-30s  %-30s  %s" % (f["parecido_cara"], nombres.get(f["a_id"], "")[:30],
                                          nombres.get(f["b_id"], "")[:30], f["regla"]))

    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar. (%.0f s)" % (time.time() - t0))
        return
    cli.delete("fichas_parecidas", a_id="gte.0")
    for i in range(0, len(filas), 500):
        cli.upsert("fichas_parecidas", filas[i:i + 500], on_conflict="a_id,b_id")
    print("\nescritos %d pares (%.0f s)" % (len(filas), time.time() - t0))


if __name__ == "__main__":
    main()
