# -*- coding: utf-8 -*-
"""Lleva al sheet del padron las correcciones hechas desde la app.

    python padron_sheet.py --ensayo     # dice que escribiria
    python padron_sheet.py --aplicar

POR QUE (23/9/2026)

Diego corrige un nombre o una sede desde la pantalla del Padron y eso arregla
la app, pero no la planilla de la que sale el padron: la proxima lectura vuelve
a traer el dato viejo y alguien lo corrige otra vez. Diego: "me gustaria que
las modificaciones como estas en el nombre de alguien del padron, tambien se
graben en el sheet".

DONDE ESCRIBE, Y POR QUE AHI

En la hoja "Correcciones" del Padron Completo, una fila por cambio. No pisa las
hojas que son export de iSAMS: si Diego vuelve a pegar el export -que es lo que
hace- una correccion escrita encima se perderia sin que nadie se entere. Aca
queda a la vista para arreglar la fuente de verdad.

El nombre va entero, como quedo en la app, y no partido en Forename/Surname:
partirlo es adivinar donde termina el nombre y empieza el apellido, que es
justo el error que trajo "Persona AC".

QUE NECESITA

El sheet compartido como EDITOR con la cuenta de servicio del proyecto, y el
alcance de Sheets en el token (el resto del proyecto pide drive.readonly). Si
no lo tiene, este paso falla diciendo con quien hay que compartirlo: una cola
que se vacia sola sin escribir seria peor.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from google.oauth2 import service_account
import google.auth.transport.requests as gart

import sb

HERE = os.path.dirname(os.path.abspath(__file__))
SA_PATH = os.path.join(HERE, "service-account.json")

# Padron Completo | Biblioteca de Fotos
SHEET = "1wlAT6t-0f7m7ZvxbC4k_zT4etCH_Pl8HRYIU-_U7hDU"
HOJA = "Correcciones"
CABECERA = ["Cuando", "Quien", "Ficha", "Que", "Antes", "Despues"]
ALCANCE = ["https://www.googleapis.com/auth/spreadsheets"]


class Hojas:
    """Lo minimo de la API de Sheets: leer una hoja, crearla y agregar filas."""

    def __init__(self, sa_path):
        self.creds = service_account.Credentials.from_service_account_file(
            sa_path, scopes=ALCANCE)
        self.expires = 0

    def _token(self):
        if time.time() > self.expires:
            self.creds.refresh(gart.Request())
            self.expires = time.time() + 3000
        return self.creds.token

    def _pedir(self, metodo, ruta, cuerpo=None, **params):
        url = ("https://sheets.googleapis.com/v4/spreadsheets/" + ruta)
        if params:
            url += "?" + urllib.parse.urlencode(params)
        datos = json.dumps(cuerpo).encode("utf-8") if cuerpo is not None else None
        req = urllib.request.Request(url, data=datos, method=metodo, headers={
            "Authorization": "Bearer " + self._token(),
            "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)

    def hojas(self):
        r = self._pedir("GET", SHEET, fields="sheets/properties/title")
        return [h["properties"]["title"] for h in r.get("sheets", [])]

    def crear_hoja(self, titulo):
        self._pedir("POST", SHEET + ":batchUpdate",
                    {"requests": [{"addSheet": {"properties": {"title": titulo}}}]})

    def agregar(self, filas):
        self._pedir("POST", SHEET + "/values/" + urllib.parse.quote(HOJA + "!A1")
                    + ":append", {"values": filas},
                    valueInputOption="USER_ENTERED",
                    insertDataOption="INSERT_ROWS")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--ensayo", action="store_true")
    a = ap.parse_args()

    cli = sb.SB()
    pend = cli.select("padron_cambios",
                      select="id,person_id,campo,antes,despues,quien,creado_at",
                      escrito_at="is.null", order="id")
    if not pend:
        print("No hay correcciones para escribir.")
        return

    gente = {}
    ids = sorted({str(c["person_id"]) for c in pend})
    for i in range(0, len(ids), 100):
        for p in cli.select("people", select="id,display_name",
                            id="in.(%s)" % ",".join(ids[i:i + 100])):
            gente[int(p["id"])] = p["display_name"]

    filas = []
    for c in pend:
        filas.append([str(c.get("creado_at") or "")[:19].replace("T", " "),
                      c.get("quien") or "",
                      gente.get(int(c["person_id"]), "ficha %s" % c["person_id"]),
                      c["campo"],
                      c.get("antes") or "",
                      c.get("despues") or ""])

    print("Correcciones para escribir: %d" % len(filas))
    for f in filas[:10]:
        print("   %s  %s: %s -> %s" % (f[0], f[2], f[4] or "(vacio)", f[5] or "(vacio)"))
    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar.")
        return

    if not os.path.exists(SA_PATH):
        sys.exit("Falta el service account en %s" % SA_PATH)
    with open(SA_PATH, encoding="utf-8") as fh:
        cuenta = json.load(fh).get("client_email", "la cuenta del proyecto")

    h = Hojas(SA_PATH)
    try:
        hojas = h.hojas()
        if HOJA not in hojas:
            h.crear_hoja(HOJA)
            h.agregar([CABECERA])
        h.agregar(filas)
    except urllib.error.HTTPError as e:
        detalle = e.read().decode("utf-8", "replace")[:200]
        if "has not been used in project" in detalle or "is disabled" in detalle:
            sys.exit("Falta habilitar la API de Google Sheets en el proyecto de "
                     "Google Cloud (sgc-photo-library). El resto del sistema usa "
                     "solo Drive, por eso nunca hizo falta. (%s)" % detalle)
        if e.code in (403, 404):
            sys.exit("El sheet del padron no se puede escribir con la cuenta del "
                     "sistema. Compartilo como EDITOR con %s. (%s)" % (cuenta, detalle))
        sys.exit("Google no dejo escribir (%s): %s" % (e.code, detalle))

    ahora = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for c in pend:
        cli.update("padron_cambios", {"escrito_at": ahora}, id="eq.%s" % c["id"])
    print("Escritas %d filas en la hoja \"%s\"." % (len(filas), HOJA))


if __name__ == "__main__":
    main()
