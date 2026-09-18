# -*- coding: utf-8 -*-
"""Pasa por el detector los albumes de evento que no pasaron enteros.

    python caras_revisar.py              # lista los pendientes y no corre nada
    python caras_revisar.py --aplicar    # los manda a faces_eventos.py

POR QUE

Diego, 13/9/2026: la cara de Persona E en Whole School Acto 2024 no tenia
recuadro. El detector la ve; el album nunca habia pasado entero por
faces_eventos.py despues de que se empezaran a guardar todas las caras, y nada
lo decia. Eran 13 albumes con el mismo agujero y 59 sin ninguna cara.

La regla vive en la base (sql/caras_revisadas.sql): faces_eventos.py anota cada
album que termina entero, y carpetas_caras_pendientes() devuelve los que no
estan anotados o recibieron fotos despues. Este script es el que los repasa:
paso "caras" del pulso.

Repasar no pisa nada: faces_eventos.py no toca las propuestas que ya existen ni
propone a quien ya esta identificado (sin_pisar), y la base no deja proponer lo
rechazado ni una cara marcada "esta cara no es".

Va por años porque faces_eventos.py filtra las referencias por el año, y en
tandas con tope de fotos para que una vuelta del pulso no tarde horas.
"""
import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict

if os.environ.get("FACES_MOTOR", "").lower() != "arcface":
    sys.exit("Falta FACES_MOTOR. En PowerShell: $env:FACES_MOTOR='arcface'")

import sb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOPE_FOTOS = 6000      # por vuelta: ~0,25 s por foto, unos 25 minutos
POR_LLAMADA = 150      # carpetas por llamada: la linea de comandos tiene tope


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--tope", type=int, default=TOPE_FOTOS)
    a = ap.parse_args()
    cli = sb.SB()

    pendientes = cli.rpc("carpetas_caras_pendientes") or []
    total = sum(int(p["fotos"] or 0) for p in pendientes)
    print("Albumes pendientes: %d (%d fotos)" % (len(pendientes), total))
    if not pendientes:
        return

    # los mas chicos primero: con el tope, mas albumes arreglados por vuelta
    elegidos, suma = [], 0
    for p in sorted(pendientes, key=lambda p: (int(p["fotos"] or 0), p["folder_id"])):
        if elegidos and suma + int(p["fotos"] or 0) > a.tope:
            continue
        elegidos.append(p)
        suma += int(p["fotos"] or 0)
    por_anio = defaultdict(list)
    for p in elegidos:
        por_anio[int(p["year"])].append(p["folder_id"])
    print("Esta vuelta: %d albumes, %d fotos, %d años" % (len(elegidos), suma, len(por_anio)))
    for anio in sorted(por_anio):
        print("  %d: %d albumes" % (anio, len(por_anio[anio])))

    if not a.aplicar:
        print("\nEnsayo: no se corrio nada. Agregar --aplicar.")
        return

    fallos = 0
    t0 = time.time()
    for anio in sorted(por_anio):
        ids = por_anio[anio]
        for i in range(0, len(ids), POR_LLAMADA):
            lote = ids[i:i + POR_LLAMADA]
            print("\n=== %d: %d albumes ===" % (anio, len(lote)), flush=True)
            codigo = subprocess.run(
                [sys.executable, "-u", os.path.join(HERE, "faces_eventos.py"),
                 "--anio", str(anio), "--carpeta", ",".join(lote)],
                cwd=HERE, env=dict(os.environ, PYTHONUTF8="1")).returncode
            if codigo != 0:
                fallos += 1
                print("faces_eventos.py termino con codigo %s" % codigo, flush=True)

    quedan = cli.rpc("carpetas_caras_pendientes") or []
    print("\nQuedan pendientes: %d albumes (%.0f min)" % (len(quedan), (time.time() - t0) / 60))
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    main()
