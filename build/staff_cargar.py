# -*- coding: utf-8 -*-
"""
Carga el padron de staff desde la lista Staff del panel de Fuentes, con su foto oficial.

Todo el staff que teniamos salio de nombres de archivo de mugshots, que es
donde estaban los errores: "Persona Z4" por "Nombre", "Lanusse" por
"Lanus", nombres con la eñe rota. La planilla trae apellido, nombre, mail y
un link a la foto de cada uno, asi que reemplaza esa fuente por una buena.

El mail es la clave: dos personas pueden llamarse igual, pero no compartir
casilla. Cuando no hay mail se cae al nombre normalizado.

DESDE EL 14/9/2026

Lee la planilla que dice el panel (padron_fuentes, clave staff_sheet, hoja
"staff": STAFF report) y no un _staff.csv del disco. Las filas marcadas en
"Duplicado?" no se cargan. upsert_staff ya no pisa el nombre de una ficha
existente (sql/fuentes_panel.sql): Diego los corrige a mano.

    python staff_cargar.py --ensayo
    python staff_cargar.py

Despues, para bajar las caras:
    python staff_caras.py
"""
import argparse
import datetime as dt
import re
import sys

import sb
from listas import Listas, indice, texto
from retratos_nombres import clave


def titulo(s):
    """'Persona AQ' -> 'Persona AQ', sin tocar lo que ya viene bien escrito."""
    out = []
    for p in (s or "").split():
        out.append(p if (len(p) > 1 and p[1:].islower() and p[0].isupper()) else p.capitalize())
    return " ".join(out)


def leer(L):
    filas, lecturas = [], []
    for lista in L.de_tipo("staff"):
        hoja = L.filas(lista)
        i = indice(hoja[0])

        n = 0
        for f in hoja[1:]:
            def d(col):
                k = i.get(col)
                return texto(f[k]) if k is not None and k < len(f) else ""
            if d("duplicado?").lower() not in ("", "no", "false", "0"):
                continue
            nombre = titulo("%s %s" % (d("name"), d("last name"))).strip()
            if not nombre:
                continue
            foto = d("photo")
            m = re.search(r"/d/([A-Za-z0-9_-]{20,})|[?&]id=([A-Za-z0-9_-]{20,})", foto)
            filas.append({"nombre": nombre, "norm": clave(nombre),
                          "email": d("email").lower(),
                          "drive_id": (m.group(1) or m.group(2)) if m else "",
                          "url": foto})
            n += 1
        lecturas.append((lista, n))
    return filas, lecturas


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensayo", action="store_true")
    a = ap.parse_args()

    cli = sb.SB()
    L = Listas(cli)
    # EX STAFF = EL QUE NO ESTA EN ESTA LECTURA (15/9/2026)
    #
    # Diego pidio el filtro de ex staff en los juegos. No hay lista de ex staff
    # y mantenerla a mano se desactualiza sola: la planilla STAFF es la de los
    # que trabajan hoy, asi que el que no aparece en esta lectura es ex staff.
    # upsert_staff marca staff_visto_at; exstaff_recalcular compara con esta
    # hora (sql/juegos_filtros.sql).
    desde = dt.datetime.now(dt.timezone.utc).isoformat()
    filas, lecturas = leer(L)
    print("Personas en la planilla: %d" % len(filas))

    antes = {p["id"] for p in cli.select("people", select="id,kind") if p.get("kind") == "staff"}
    print("Staff que ya habia en el padron: %d" % len(antes))

    if a.ensayo:
        por_mail = {p.get("email", "").lower() for p in cli.select("people", select="id,email")
                    if p.get("email")}
        por_nombre = {p["norm_name"] for p in cli.select("people", select="id,norm_name")}
        nuevos = [f for f in filas
                  if f["email"].lower() not in por_mail and f["norm"] not in por_nombre]
        print("Se crearian %d personas nuevas; el resto se actualiza." % len(nuevos))
        for f in nuevos[:15]:
            print("   nueva: %-34s %s" % (f["nombre"][:34], f["email"]))
        return

    creados = 0
    for i, f in enumerate(filas, 1):
        url = ("https://drive.google.com/file/d/%s/view" % f["drive_id"]) if f["drive_id"] else f["url"]
        cli.rpc("upsert_staff", {
            "p_display_name": f["nombre"], "p_norm_name": f["norm"],
            "p_email": f["email"], "p_photo_url": url})
        creados += 1
        if i % 100 == 0:
            print("  %d/%d" % (i, len(filas)))

    marca = cli.rpc("exstaff_recalcular", {"p_desde": desde}) or []
    ex = marca[0] if isinstance(marca, list) and marca else (marca or {})

    despues = [p for p in cli.select("people", select="id,kind,email,photo_url")
               if p.get("kind") == "staff"]
    for lista, n in lecturas:
        L.anotar(lista, n, "%d personas procesadas; staff en el padron: %d"
                 % (n, len(despues)))
    print("\nProcesadas: %d" % creados)
    print("Staff en el padron ahora: %d" % len(despues))
    print("  con mail:  %d" % sum(1 for p in despues if p.get("email")))
    print("  con foto:  %d" % sum(1 for p in despues if p.get("photo_url")))
    print("  staff actual: %s   ex staff: %s" % (ex.get("staff"), ex.get("ex_staff")))


if __name__ == "__main__":
    main()
