# -*- coding: utf-8 -*-
"""La foto de perfil de cada ficha, al dia: llama a foto_de_perfil_refrescar().

    python foto_perfil_refrescar.py                    # ensayo: cuantas revisaria
    python foto_perfil_refrescar.py --aplicar          # las que cambiaron (cada vuelta)
    python foto_perfil_refrescar.py --aplicar --todos  # todas (una vez por dia)

POR QUE (13/10/2026)

La ficha, el buscador y la camada elegian la foto de perfil con tres criterios
distintos. La regla quedo en un solo lugar, en la base (sql/foto_de_perfil.sql,
fotos_de_perfil), y se guarda en people.foto_perfil*: calcularla para todos
tarda de 24 a 62 s y no puede correr cada vez que alguien abre una ficha.

Esto es lo unico que la escribe, y no decide nada: la regla y la escritura son
de la base. Aca se elige cuantas fichas y se deja el resultado en el log.

  incremental  p_todos = false. Las fichas con algo nuevo desde su ultima
               vuelta (fotos_de_perfil_pendientes): una identificacion, un si o
               un no, un rechazo, un recuadro, una fusion. Cada vuelta del pulso.
  completo     p_todos = true. Todas las fichas con fotos: ve lo que el
               incremental no ve (una carpeta que pasa a ser de carnets, el
               anio, la cuenta de caras al reagrupar). Una vez por dia. La
               primera vez es el llenado: hoy ninguna ficha esta calculada.

UNA SOLA LLAMADA, SIN REINTENTOS

sb.SB.rpc reintenta los 500 y los cortes de red, y espera 120 s. Para esto no
sirve: la funcion escribe y tiene su propio tope de 300 s (statement_timeout de
la funcion; medido el 13/10/2026 que el tope de la funcion SI vale por PostgREST:
salud(), con 60 s, contesto en 44 s cuando el rol tiene 8 s). Si el cliente
cortara a los 120 s y reintentara, la base seguiria con la primera llamada y
arrancaria otra encima. Asi que va una sola, esperando mas que el tope de la
base. Si falla, el pulso lo anota y reintenta en la vuelta siguiente.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

import sb

FUNCION = "foto_de_perfil_refrescar"
# Mas que el statement_timeout de la funcion (300 s, sql/foto_de_perfil.sql):
# que corte la base y no el cliente, asi nunca quedan dos llamadas a la vez.
ESPERA_S = 330


def contar_pendientes(cli):
    """Cuantas fichas revisaria el incremental. Solo lee (funcion STABLE).
    PostgREST devuelve a lo sumo 1.000 filas: el total sale de Content-Range."""
    req = urllib.request.Request(
        "%s/rest/v1/rpc/fotos_de_perfil_pendientes?limit=1" % cli.url,
        data=b"{}", method="POST",
        headers=cli._headers({"Prefer": "count=exact"}))
    with urllib.request.urlopen(req, timeout=60) as r:
        rango = r.headers.get("Content-Range") or ""
    total = rango.rsplit("/", 1)[-1]
    return int(total) if total.isdigit() else None


def contar_todas(cli):
    """Cuantas fichas no ruido hay: el completo revisa las que tienen fotos,
    que son menos (la base las elige). Es el techo, para el ensayo."""
    return sum(1 for p in cli.select("people", select="id,kind") if p.get("kind") != "noise")


def refrescar(cli, todos):
    req = urllib.request.Request(
        "%s/rest/v1/rpc/%s" % (cli.url, FUNCION),
        data=json.dumps({"p_todos": bool(todos)}).encode("utf-8"),
        method="POST", headers=cli._headers())
    try:
        with urllib.request.urlopen(req, timeout=ESPERA_S) as r:
            cuerpo = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                import gzip
                cuerpo = gzip.decompress(cuerpo)
    except urllib.error.HTTPError as e:
        crudo = e.read()
        if e.headers.get("Content-Encoding") == "gzip":
            import gzip
            try:
                crudo = gzip.decompress(crudo)
            except Exception:
                pass
        raise RuntimeError("HTTP %s: %s" % (e.code, crudo.decode("utf-8", "replace")[:600]))
    return json.loads(cuerpo) if cuerpo else {}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--todos", action="store_true",
                    help="todas las fichas con fotos, no solo las que cambiaron")
    a = ap.parse_args()
    cli = sb.SB()
    cual = "completo" if a.todos else "incremental"

    if not a.aplicar:
        t0 = time.time()
        try:
            if a.todos:
                print("Ensayo %s: revisaria hasta %d fichas (las no ruido con fotos)"
                      % (cual, contar_todas(cli)))
            else:
                n = contar_pendientes(cli)
                print("Ensayo %s: revisaria %s fichas pendientes"
                      % (cual, "?" if n is None else n))
        except Exception as e:
            print("Ensayo %s: no se pudo contar (%s)" % (cual, str(e)[:200]))
            return 1
        print("(%.1f s) No se escribio nada. Agregar --aplicar." % (time.time() - t0))
        return 0

    t0 = time.time()
    try:
        r = refrescar(cli, a.todos)
    except Exception as e:
        # el pulso lo ve como falla y reintenta: no se marca al dia algo que fallo
        print("FALLO %s de la foto de perfil en %.0f s: %s" % (cual, time.time() - t0, str(e)[:600]))
        return 1
    try:
        print("Foto de perfil, %s: revisadas %s, cambiadas %s, %s ms en la base (%.1f s en total)"
              % (cual, r.get("revisadas"), r.get("cambiadas"), r.get("ms"), time.time() - t0))
    except Exception:
        print("Foto de perfil, %s: %s" % (cual, r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
