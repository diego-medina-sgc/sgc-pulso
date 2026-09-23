# -*- coding: utf-8 -*-
"""
Aprueba las sugerencias que son la MISMA fotografía que la persona ya tiene.

    python identicas_aprobar.py
    python identicas_aprobar.py --aplicar

POR QUE

Lo preguntó Diego viendo la pantalla de confirmar: "¿por qué pasa esto de
confirmar, si el dato ya lo tenías en el Drive con la foto con nombre?".

El caso que miró: le preguntaba si una foto de la carpeta PROFESORES es Maureen
Bond Field. Y sí, el dato estaba: en "Staff Mugshots" hay dos archivos que se
llaman "Persona Y", ya identificados. Lo que pasa es que son ARCHIVOS
DISTINTOS del mismo retrato -el fotógrafo entrega los suyos con código de
cámara, y alguien renombra una copia- y pesan distinto, 4,8 MB contra 5,5, así
que fotos_duplicadas.py no los une: su md5 no coincide.

La cara sí lo sabe. El parecido da 1,0000: no es "se parece mucho", es la misma
fotografía.

POR QUE NO LO HACIA faces_aprobar.py

Porque descarta de entrada todo lo que no está en una carpeta de retrato, y con
razón: sus reglas están calibradas para retratos -"una vez por carpeta" no vale
en un acto donde la misma persona sale veinte veces- y sin ese filtro metería
decenas de miles de sugerencias de evento en un molde que no es el suyo.

Pero hay un caso que no depende del molde. Un parecido de 0,97 para arriba no
es reconocer a alguien: es reconocer la misma imagen. Eso vale igual en una
carpeta de acto que en una de retrato.

Son 395 preguntas que nadie tendría que contestar, y están concentradas en
carpetas de curso -ES 6, K3, 1S, 6G- que no figuran como carpetas de retrato.
Eso último es el problema de fondo y se arregla aparte; esto levanta lo que ya
se puede levantar.

LA MEDIDA TIENE QUE COINCIDIR

Es la misma exigencia que usa faces_aprobar para su regla de idénticas: el
puntaje dice que la cara es la misma y el ancho y el alto confirman que la
imagen también. Una foto distinta de la misma sesión no garantiza las dos
cosas; una copia del mismo archivo, sí.
"""
import argparse
import sys
from collections import defaultdict

import sb

IDENTICA = 0.97


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--umbral", type=float, default=IDENTICA)
    a = ap.parse_args()
    cli = sb.SB()

    fs = [x for x in cli.select("face_suggestions", select="photo_id,person_id,score")
          if x["score"] >= a.umbral]
    print("Sugerencias con parecido >= %.2f: %d" % (a.umbral, len(fs)))

    ya = defaultdict(set)
    for r in cli.select("photo_people", select="photo_id,person_id"):
        ya[r["photo_id"]].add(int(r["person_id"]))
    rech = defaultdict(set)
    for r in cli.select("photo_people_rejected", select="photo_id,person_id"):
        rech[r["photo_id"]].add(int(r["person_id"]))
    fs = [x for x in fs
          if int(x["person_id"]) not in ya.get(x["photo_id"], set())
          and int(x["person_id"]) not in rech.get(x["photo_id"], set())]
    print("   sin contestar y sin rechazar: %d" % len(fs))

    # las medidas de todo lo que interviene
    ids = sorted({x["photo_id"] for x in fs} | set(ya))
    med = {}
    for i in range(0, len(ids), 60):
        for y in cli.select("photos", select="id,width,height",
                            id="in.(%s)" % ",".join('"%s"' % z for z in ids[i:i + 60])):
            if y.get("width"):
                med[y["id"]] = (y["width"], y["height"])

    # las medidas que cada persona ya tiene identificadas
    suyas = defaultdict(set)
    for foto, gente in ya.items():
        m = med.get(foto)
        if m:
            for p in gente:
                suyas[p].add(m)

    buenas, sin_medida, distintas = [], [], []
    for x in fs:
        m = med.get(x["photo_id"])
        p = int(x["person_id"])
        if not m:
            sin_medida.append(x)
        elif m in suyas.get(p, set()):
            buenas.append(x)
        else:
            distintas.append(x)
    print("\n   la persona ya tiene una foto de la misma medida : %d   <- se aprueban" % len(buenas))
    print("   la medida no coincide                           : %d   <- se dejan" % len(distintas))
    print("   sin medida guardada                             : %d   <- se dejan" % len(sin_medida))

    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar.")
        return
    filas = [{"photo_id": x["photo_id"], "person_id": int(x["person_id"]),
              "votes": 1, "source": "face_auto"} for x in buenas]
    for i in range(0, len(filas), 200):
        cli.upsert("photo_people", filas[i:i + 200], on_conflict="photo_id,person_id")
    print("\nAprobadas %d." % len(filas))
    print("Para revertir esta tanda:")
    print("  delete from photo_people where source = 'face_auto';")


if __name__ == "__main__":
    main()
