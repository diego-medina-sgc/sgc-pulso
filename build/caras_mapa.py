# -*- coding: utf-8 -*-
"""Los recuadros de las caras que no estan en ningun grupo, para el visor.

    python caras_mapa.py              # mide cuanto ocuparia y no escribe
    python caras_mapa.py --aplicar

POR QUE

El visor deja nombrar cualquier cara detectada ("+ ¿Quién es?"), pero leia los
recuadros de face_groups, que solo guarda las caras que cayeron en un grupo de
2 o mas. Las demas -una cara que aparece una sola vez en el archivo- no tenian
recuadro. Estan en _caras_todas.npz; aca se suben compactas a caras_mapa, una
fila por foto (sql/caras_mapa.sql).

Corre en el pulso despues de "grupos": lo que no quedo en un grupo es lo que
va aca. Reescribe la tabla entera (TRUNCATE y carga), asi no se infla.
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

TOPE_MB = 25          # si ocuparia mas, no escribe: la base esta cerca de los 500 MB


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cli = sb.SB()

    print("Leyendo las caras agrupadas…", flush=True)
    agrupadas = {(r["photo_id"], round(float(r["bx"]), 4), round(float(r["by"]), 4))
                 for r in cli.select("face_groups", select="photo_id,bx,by")}
    print("  en grupos: %d   (%.0f s)" % (len(agrupadas), time.time() - t0), flush=True)

    with np.load(faces.npz("_caras_todas.npz"), allow_pickle=True) as z:
        F = z["photo_ids"].astype(str)
        C = z["cajas"].astype(np.float32)
    por_foto = defaultdict(list)
    sueltas = 0
    for i in range(len(F)):
        bx, by, bw, bh = (round(float(x), 4) for x in C[i])
        if bw <= 0 or bh <= 0:
            continue
        if (F[i], bx, by) in agrupadas:
            continue
        por_foto[F[i]].extend([bx, by, bw, bh])
        sueltas += 1

    filas = [{"photo_id": f, "cajas": c} for f, c in por_foto.items()]
    # fila: ~24 de cabecera + id de 33 + arreglo (24 + 4 por numero); indice ~45 por fila
    bytes_tabla = sum(24 + 34 + 24 + 4 * len(c) for c in por_foto.values())
    mb = (bytes_tabla + 45 * len(filas)) / 1048576
    print("  caras detectadas: %d   sin grupo: %d   en %d fotos" % (len(F), sueltas, len(filas)))
    print("  ocuparia ~%.1f MB (tope %d MB)" % (mb, TOPE_MB))

    if mb > TOPE_MB:
        print("\nNO SE ESCRIBE: ocuparia mas del tope.")
        return
    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar. (%.0f s)" % (time.time() - t0))
        return
    cli.rpc("vaciar_caras_mapa")
    for i in range(0, len(filas), 2000):
        cli.upsert("caras_mapa", filas[i:i + 2000], on_conflict="photo_id")
        if (i // 2000) % 10 == 0:
            print("  %d/%d" % (min(i + 2000, len(filas)), len(filas)), flush=True)
    print("\nescritas %d fotos con %d caras (%.0f s)" % (len(filas), sueltas, time.time() - t0))


if __name__ == "__main__":
    main()
