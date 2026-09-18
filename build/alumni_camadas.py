# -*- coding: utf-8 -*-
"""
Carga la camada de cada exalumno desde la hoja "Total Archive".

    python alumni_camadas.py --ensayo
    python alumni_camadas.py --aplicar

POR QUE EXISTE

Diego miro la camada 2004 en la pantalla del padron y dijo que estaba mal.
Tenia razon, y el error era mio de raiz: la camada salia de "NS/QS hasta" de la
hoja Alumni, que no es un año de cursada. De 2.228 filas, solo 593 tienen una
edad coherente ahi. Hay gente nacida en 2006 que figura dejando el colegio en
2003, y una chica nacida en 2004 con "hasta 2004".

Medido contra las personas cuya camada sale de un anuario -dato duro, porque el
anuario de un año fotografia a los que se egresan ese año-:

    "NS/QS hasta" de la hoja Alumni          21% dentro de +-1 año
    año de nacimiento + 17 o 18 segun el mes 98% dentro de +-1  (74% exacta)
    "Total Archive"                          98% dentro de +-1  (96% EXACTA)

Asi que se usa Total Archive, que ademas llega hasta 1957 y trae la sede.

LA VENTANA DE CURSADA TAMBIEN SE REHACE

first_seen y last_seen salieron de las mismas columnas rotas, y no son un
adorno: faces_eventos.py los usa para decidir en que fotos puede aparecer cada
persona. Un exalumno de los años sesenta con ventana 1996-2004 se vuelve
candidato de fotos donde no pudo estar.

La ventana nueva es la union de dos cosas:

    la escalera        de K1 a C6 hay 17 años, asi que quien se egresa en Y
                       pudo entrar en Y-16 como muy temprano
    sus propias fotos  los años de las fotos que ya tiene identificadas, que
                       son un hecho y mandan sobre cualquier calculo

Se toma la union y no solo la escalera porque una foto identificada fuera de la
ventana calculada no es un error de la foto: es que la persona estuvo.

QUE NO PISA

Si la persona ya tiene camada de anuario, esa gana: Total Archive coincide en
el 96% y en el 4% restante el anuario es el que vio la cara.

Desde el 14/9/2026 "de anuario" se mira por el origen de la ficha (yearbook,
yearbook_pdf). Antes cualquier camada distinta se tomaba como de anuario, y eso
impedia justo lo que Diego pidio: pasar al archivo corregido ("tenia errores en
las camadas") y que corrigiera las que habia puesto la version vieja. Y el
archivo ya no es una descarga en la carpeta Descargas: es la hoja Total Archive
de la lista Exalumnos del panel de Fuentes (build/listas.py).
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import sb
from listas import Listas
from retratos_nombres import clave, indexar, por_subconjunto

HERE = os.path.dirname(os.path.abspath(__file__))
POR_DEFECTO = os.path.join(
    os.path.expanduser("~"), "Downloads",
    "St George's College Alumni Information Form (Responses) (1).xlsx")
HOJA = "Total Archive"
# De K1 a C6 hay diecisiete años de escalera.
ESCALERA = 17


def leer(ruta):
    import openpyxl
    ws = openpyxl.load_workbook(ruta, data_only=True)[HOJA]
    return interpretar([list(f) for f in ws.iter_rows(values_only=True)])


def interpretar(filas):
    out, motivos = {}, Counter()
    for f in filas[1:]:
        f = list(f) + [None] * 4
        ape, nom, cam, sede = f[0], f[1], f[2], f[3]
        if not ape or not nom:
            motivos["fila sin nombre"] += 1
            continue
        m = re.search(r"(19\d\d|20\d\d)", str(cam or ""))
        if not m:
            motivos["sin camada"] += 1
            continue
        k = clave("%s %s" % (nom, ape))
        if not k:
            motivos["nombre vacio al normalizar"] += 1
            continue
        anio = int(m.group(1))
        if k in out and out[k][0] != anio:
            # dos personas distintas con el mismo nombre y distinta camada: no
            # hay forma de saber cual es cual, asi que ninguna
            motivos["mismo nombre con dos camadas, se descarta"] += 1
            out[k] = None
            continue
        if out.get(k) is None and k in out:
            continue
        out[k] = (anio, str(sede or "") or None)
    return {k: v for k, v in out.items() if v}, motivos


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    # sin --archivo se lee la planilla del panel de Fuentes (lo normal)
    ap.add_argument("--archivo", default=None)
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--ensayo", action="store_true")
    a = ap.parse_args()

    cli = sb.SB()
    if a.archivo:
        if not os.path.exists(a.archivo):
            sys.exit("no esta %s" % a.archivo)
        reg, motivos = leer(a.archivo)
        origen = a.archivo
    else:
        L = Listas(cli)
        ex = L.de_tipo("exalumnos")
        if not ex:
            sys.exit("No hay una lista de exalumnos con link en el panel de Fuentes.")
        reg, motivos = interpretar(L.filas(ex[0], HOJA))
        origen = "%s / hoja %s" % (ex[0]["titulo"], HOJA)
    print("%s: %d personas con camada" % (origen, len(reg)))
    for k, v in motivos.most_common():
        print("   %-44s %d" % (k + ":", v))

    gente = [g for g in cli.select(
        "people", select="id,display_name,kind,campus,cohort,first_seen,last_seen,year_code,source")
        if g["kind"] != "noise"]

    # los años de las fotos que cada persona ya tiene: son un hecho
    anios_foto = defaultdict(set)
    pp = cli.select("photo_people", select="person_id,photos(year)")
    for r in pp:
        y = (r.get("photos") or {}).get("year")
        if y:
            anios_foto[int(r["person_id"])].add(int(y))

    # El nombre no coincide palabra por palabra entre las dos listas. El padron
    # dice "Persona A" y Total Archive "Persona Z1":
    # el mismo apellido con uno mas, que es como varia entre fuentes. Se usa el
    # mismo criterio que el resto del proyecto -uno contiene al otro y comparten
    # al menos dos palabras- y solo cuando hay una unica candidata, porque con
    # dos no hay forma de saber cual es.
    como_padron = [{"id": i, "display_name": k} for i, k in enumerate(reg)]
    claves = list(reg)
    _, _, por_token, juegos = indexar(como_padron)

    def buscar(nombre):
        k = clave(nombre)
        if k in reg:
            return reg[k]
        cand = por_subconjunto(k.split(), por_token, juegos)
        if len(cand) == 1:
            return reg[claves[int(cand[0]["id"])]]
        return None

    cambios, motivos2 = [], Counter()
    for g in gente:
        v = buscar(g["display_name"])
        if not v:
            motivos2["no esta en Total Archive"] += 1
            continue
        camada, sede = v
        if g.get("year_code"):
            # alumno actual: su nivel de hoy manda, no una camada de exalumno
            motivos2["alumno actual, no se toca"] += 1
            continue
        if (g.get("cohort") and g["cohort"] != camada
                and g.get("source") in ("yearbook", "yearbook_pdf")):
            motivos2["ya tiene camada de anuario, esa gana"] += 1
            continue
        if g.get("kind") == "staff" and not g.get("cohort"):
            # staff sin camada: si ademas fue alumno, son dos fichas
            motivos2["staff, no se toca"] += 1
            continue

        fotos = anios_foto.get(int(g["id"]), set())
        desde = min([camada - ESCALERA + 1] + list(fotos))
        hasta = max([camada] + list(fotos))
        campos = {}
        if g.get("cohort") != camada:
            campos["cohort"] = camada
        if g.get("first_seen") != desde:
            campos["first_seen"] = desde
        if g.get("last_seen") != hasta:
            campos["last_seen"] = hasta
        if not g.get("campus") and sede in ("North", "Quilmes"):
            campos["campus"] = sede
        if not campos:
            motivos2["ya estaba bien"] += 1
            continue
        campos["id"] = g["id"]
        campos["_antes"] = (g.get("cohort"), g.get("first_seen"), g.get("last_seen"))
        campos["_nombre"] = g["display_name"]
        cambios.append(campos)

    print()
    for k, v in motivos2.most_common():
        print("   %-44s %d" % (k + ":", v))
    print("\nPersonas a corregir: %d" % len(cambios))
    mueve = [c for c in cambios if c["_antes"][0] not in (None, c.get("cohort"))]
    print("   con camada distinta a la que tenian: %d" % len(mueve))
    for c in cambios[:12]:
        print("   %-30s camada %-6s -> %-6s   ventana %s-%s -> %s-%s"
              % (c["_nombre"][:30], c["_antes"][0], c.get("cohort", c["_antes"][0]),
                 c["_antes"][1], c["_antes"][2],
                 c.get("first_seen", c["_antes"][1]), c.get("last_seen", c["_antes"][2])))

    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar.")
        return
    for c in cambios:
        cid = c.pop("id")
        c.pop("_antes", None)
        c.pop("_nombre", None)
        cli.update("people", c, id="eq.%d" % cid)
    print("\nCorregidas %d personas." % len(cambios))


if __name__ == "__main__":
    main()
