# -*- coding: utf-8 -*-
"""Vuelve a cargar las listas de personas del panel de Fuentes, en orden.

    python listas_cargar.py --ensayo
    python listas_cargar.py --aplicar

Lo corre el pulso (paso "listas") cuando cambia el link o la hoja de una lista,
o Diego toca "Leer de nuevo" (pulso_firmas -> listas, sql/fuentes_panel.sql).

El orden importa: primero alumnos y exalumnos (altas y cursada), despues las
camadas de Total Archive (que buscan a esas personas por nombre) y al final el
staff. Si uno falla, los siguientes no corren: el pulso lo reintenta.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--ensayo", action="store_true")
    a = ap.parse_args()
    ensayo = a.ensayo or not a.aplicar

    pasos = [
        ["padron_cargar.py"] + (["--ensayo"] if ensayo else []),
        ["alumni_camadas.py"] + (["--ensayo"] if ensayo else ["--aplicar"]),
        ["staff_cargar.py"] + (["--ensayo"] if ensayo else []),
    ]
    for p in pasos:
        print("\n=== %s ===" % " ".join(p), flush=True)
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, p[0])] + p[1:], cwd=HERE)
        if r.returncode != 0:
            sys.exit("fallo %s (codigo %s)" % (p[0], r.returncode))


if __name__ == "__main__":
    main()
