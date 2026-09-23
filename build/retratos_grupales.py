# -*- coding: utf-8 -*-
"""
Las fotos de varias personas que están en la cola de retratos.

    python retratos_grupales.py --ensayo
    python retratos_grupales.py --aplicar

POR QUE

Diego lo vio jugando: el juego le mostró una foto de tres mujeres y le preguntó
"¿Quién es?", proponiendo a Persona AL. Su comentario fue exacto: "esta
foto hay 3 personas, no tiene sentido reconocer una si hay 3". Y de paso las
nombró a las tres -Persona AM, Persona F y Persona AL- que
ya quedaron puestas.

El juego pregunta "de quién es este retrato" y eso sólo tiene sentido si hay
una persona. La pantalla de confirmar es la que sabe preguntar por una cara
dentro de una foto de varias, porque dibuja el recuadro.

La columna para separarlos ya existía -photos.is_group, que el juego filtra-
pero nadie la llenaba mirando la imagen: venía de las carpetas de acto. En una
carpeta de retratos se asumía que todo era retrato, y casi siempre lo es: en
una sesión de mugshots pasa un chico por vez. La excepción son las fotos que el
fotógrafo saca al final, de los profesores juntos, y ésas quedaban en la cola
como si fueran el retrato de alguien.

COMO SE DECIDE

Contando caras, pero no cualquiera y no con el piso de siempre.

El piso de la casa son 44 px de lado, que es lo que hace falta para RECONOCER
una cara. Contarlas es otra cosa. Lo destapó Diego con la foto entera de K5
-treinta chicos y tres maestras en las gradas- que el juego le preguntó "¿quién
es?": el detector ve las treinta caras, miden entre 32 y 43 px, y el filtro las
descartaba todas. La foto contaba cero caras y pasaba por retrato. Por eso acá
se cuenta desde 20 px.

Y no alcanza con "dos o más", porque a 20 px aparece cualquier cabeza del
fondo, y un retrato con un chico pasando atrás sigue siendo un retrato. Lo que
separa los dos casos es el TAMAÑO RELATIVO: en un retrato la cara del sujeto es
mucho más grande que cualquier otra cosa en la foto; en una foto de grupo todas
miden parecido. Así que es grupal si la segunda cara más grande llega al 45% de
la primera.

    K5, treinta chicos          43 y 43 px   ->  100%, grupal
    las tres maestras juntas    parecidas    ->  grupal
    un retrato con alguien atrás 200 y 30    ->   15%, retrato

El error barato sigue siendo el mismo: si se queda corta, la foto se queda
donde estaba.

QUE PASA DESPUES

Marcadas como grupales salen de la cola del juego, que es la mitad del
problema. La otra mitad es que puedan contestarse: para eso hay que pasarlas
por faces_eventos.py, que es el que guarda el recuadro de cada cara, y ahí
caen en la pantalla de confirmar con su cuadrado. Eso es otro cambio; esto no
lo hace.
"""
import argparse
import sys
import threading
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import faces
import sb
from index_drive import Drive, SA_PATH

HILOS = 5
CAMPOS = "id,name,folder_id,is_primary,is_group"

# Para CONTAR caras, no para reconocerlas: ver el comentario de arriba.
LADO_CONTAR = 20
# Cuanto tiene que medir la segunda cara respecto de la primera para que la
# foto sea de un grupo y no un retrato con alguien de fondo.
PROPORCION = 0.45


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--ensayo", action="store_true")
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--todas", action="store_true",
                    help="tambien las que ya tienen nombre puesto")
    a = ap.parse_args()

    cli = sb.SB()
    fol = {f["id"]: f for f in cli.select(
        "folders", select="id,title,display_title,is_mugshot,noise")}
    mg = [k for k, v in fol.items() if v.get("is_mugshot") and not v.get("noise")]
    fo = []
    for i in range(0, len(mg), 40):
        fo += cli.select("photos", select=CAMPOS,
                         folder_id="in.(%s)" % ",".join(mg[i:i + 40]))
    fo = [p for p in fo if p["is_primary"] and not p["is_group"]]

    if not a.todas:
        # Las que ya tienen nombre no molestan en la cola, y son 11.000: mirar
        # la cara de todas cuesta horas de descarga para arreglar algo que no
        # esta roto. Con --todas se hace igual.
        tenidas = {r["photo_id"] for r in cli.select("photo_people", select="photo_id")}
        fo = [p for p in fo if p["id"] not in tenidas]
    if a.limite:
        fo = fo[:a.limite]
    print("Motor: %s   retratos a mirar: %d" % (faces.MOTOR, len(fo)))

    d = Drive(SA_PATH)
    local = threading.local()
    lock = threading.Lock()
    cuenta = Counter()
    grupales = []

    def mirar(p):
        if not hasattr(local, "c"):
            local.c = faces.Caras()
        try:
            link = d.get("files/" + p["id"], fields="thumbnailLink").get("thumbnailLink")
            if not link:
                return None
            with urllib.request.urlopen(link.replace("=s220", "=s1600"), timeout=60) as r:
                data = r.read()
        except Exception:
            with lock:
                cuenta["no se pudo bajar"] += 1
            return None
        # Se cuentan las caras, no se las reconoce: no hace falta la huella
        # -que es lo caro- ni el piso de 44 px.
        img = local.c.leer(data)
        det = local.c.detectar(img, lado_min=LADO_CONTAR) if img is not None else []
        lados = sorted((min(f[2], f[3]) for f in det), reverse=True)
        grupal = len(lados) > 1 and lados[1] >= lados[0] * PROPORCION
        with lock:
            cuenta["n"] += 1
            cuenta["%d cara(s)" % len(lados)] += 1
            if grupal:
                cuenta["grupales"] += 1
            if cuenta["n"] % 200 == 0:
                print("  %d/%d   grupales hasta ahora: %d"
                      % (cuenta["n"], len(fo), cuenta["grupales"]))
        return p if grupal else None

    for i in range(0, len(fo), 300):
        d._refresh()
        with ThreadPoolExecutor(max_workers=HILOS) as ex:
            grupales += [x for x in ex.map(mirar, fo[i:i + 300]) if x]

    print()
    for k, n in sorted(cuenta.items()):
        if k != "n":
            print("   %-20s %d" % (k, n))
    print("\nFotos con mas de una cara en carpetas de retrato: %d" % len(grupales))
    porcarp = Counter(fol[p["folder_id"]].get("display_title")
                      or fol[p["folder_id"]].get("title") or "?" for p in grupales)
    for k, n in porcarp.most_common(12):
        print("   %-46s %d" % (k[:46], n))

    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar.")
        return
    ids = [p["id"] for p in grupales]
    for i in range(0, len(ids), 100):
        cli.update("photos", {"is_group": True},
                   id="in.(%s)" % ",".join('"%s"' % x for x in ids[i:i + 100]))
    print("\nMarcadas %d como grupales. Salen de la cola del juego." % len(ids))
    print("Para que se puedan contestar hay que pasarlas por faces_eventos.py,")
    print("que es el que guarda el recuadro de cada cara.")


if __name__ == "__main__":
    main()
