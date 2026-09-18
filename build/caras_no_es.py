# -*- coding: utf-8 -*-
"""
Aprende de los "No es": una cara rechazada no se vuelve a preguntar.

    python caras_no_es.py                # mide y no escribe
    python caras_no_es.py --aplicar

POR QUE

Lo pidió Diego, y es la cosa más importante que pidió: "lo importante es que el
sistema aprenda. Si me pregunta si Persona G es el segundo mejor 'x persona'
me la muestra y le digo que no es, que no me vuelva a preguntar, porque ese 'x
persona' ya no es Persona G".

Hoy el "No es" se guarda como (esa foto, esa persona) y nada más. Si el mismo
señor pelado aparece en otras veinte fotos de la misma fiesta, son veinte
preguntas nuevas de "¿Es Persona G?", porque para el sistema esas veinte
caras no tienen ninguna relación entre sí.

LA REGLA

Si la cara F no es de María, y la cara G es la misma persona que F, entonces G
tampoco es de María. Eso es todo, y es válido: no hace falta saber quién es F.

Para saber que F y G son la misma persona se comparan sus huellas, que
faces_eventos.py ahora guarda. El corte es alto -0,80- a propósito: acá un error
no cuesta una pregunta molesta, cuesta una identificación buena que nunca se
ofrece. Dos caras de la misma persona en la misma sesión pasan de 0,85 sin
esfuerzo, y el máximo medido entre dos personas distintas fue 0,473.

Y se compara sólo contra las propuestas de la MISMA persona: la pregunta que se
quiere borrar es "¿es Persona G?", así que sólo importan las caras a las que
se les propuso María.

QUE NO HACE

No toca lo que alguien contestó que sí. Si una cara está confirmada como de
María, no se rechaza por parecerse a una rechazada: eso lo decidió una persona
mirando, y gana.

Tampoco propaga los rechazos del juego de retratos, que no tienen recuadro ni
huella guardada.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

import faces
import sb

# Arriba de esto son la misma persona y no hay discusión. Ver arriba.
MISMA = 0.80


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--corte", type=float, default=MISMA)
    a = ap.parse_args()
    cli = sb.SB()

    cache = faces.npz("_caras_evento.npz")
    if not os.path.exists(cache):
        print("Todavía no hay huellas de caras de evento guardadas.")
        print("Las guarda faces_eventos.py; hace falta que corra al menos un año.")
        return
    with np.load(cache, allow_pickle=True) as z:
        fotos = z["photo_ids"].tolist()
        personas = z["personas"].astype(int).tolist()
        vecs = z["vecs"].astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    print("Huellas de cara guardadas: %d" % len(fotos))

    # dónde está cada (foto, persona) dentro del cache
    fila = {(f, p): i for i, (f, p) in enumerate(zip(fotos, personas))}
    # las propuestas de cada persona, para comparar sólo contra las suyas
    de_persona = defaultdict(list)
    for i, p in enumerate(personas):
        de_persona[p].append(i)

    rech = cli.select("photo_people_rejected", select="photo_id,person_id")
    si = {(r["photo_id"], int(r["person_id"])) for r in
          cli.select("photo_people", select="photo_id,person_id")}
    ya_no = {(r["photo_id"], int(r["person_id"])) for r in rech}
    print("Rechazos anotados: %d   con huella guardada: %d"
          % (len(rech), sum(1 for r in rech
                            if (r["photo_id"], int(r["person_id"])) in fila)))

    nuevos, por_persona = [], Counter()
    for r in rech:
        p = int(r["person_id"])
        i = fila.get((r["photo_id"], p))
        if i is None:
            continue
        hermanas = de_persona.get(p, [])
        if len(hermanas) < 2:
            continue
        s = vecs[hermanas] @ vecs[i]
        for j, parecido in zip(hermanas, s):
            if j == i or parecido < a.corte:
                continue
            k = (fotos[j], p)
            if k in ya_no or k in si:
                continue
            ya_no.add(k)
            nuevos.append({"photo_id": fotos[j], "person_id": p})
            por_persona[p] += 1

    print("\nPreguntas que se dejan de hacer: %d   sobre %d fichas"
          % (len(nuevos), len(por_persona)))
    if por_persona:
        g = {int(x["id"]): x["display_name"] for x in
             cli.select("people", select="id,display_name")}
        for pid, n in por_persona.most_common(15):
            print("   %-34s %d" % (g.get(pid, "?")[:34], n))

    if not a.aplicar:
        print("\nEnsayo: no se escribió nada. Agregar --aplicar.")
        return
    if not nuevos:
        return
    # rejected_by queda nulo: no lo contestó una persona, lo dedujo esto. La
    # columna dice quién, y mentir ahí sería peor que dejarla vacía.
    for i in range(0, len(nuevos), 200):
        cli.upsert("photo_people_rejected", nuevos[i:i + 200],
                   on_conflict="photo_id,person_id")
    print("\nAnotados %d rechazos deducidos." % len(nuevos))
    print("Para revertir esta tanda hay que borrarlos por (foto, persona);")
    print("quedan sin rejected_by, que es como se distinguen de los contestados.")


if __name__ == "__main__":
    main()
