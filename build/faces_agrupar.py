# -*- coding: utf-8 -*-
"""
Agrupa los retratos por cara y propaga el nombre dentro de cada grupo.

El mismo chico tiene retrato en cinco o seis años. Si en 2018 su archivo se
llamaba "Persona O.jpg" y en 2015 se llamaba "AXN_0957.jpg", hoy el de
2015 queda huerfano aunque sea la misma cara. Comparar cada pendiente contra
los identificados no alcanza: hay que unir los pendientes ENTRE SI y despues
dejar que el nombre viaje por el grupo.

Lo que hace esto seguro sin que nadie apruebe son tres reglas:

  UNA VEZ POR CARPETA   En una carpeta de retratos cada chico aparece una sola
                        vez. Dos fotos del mismo grado y año son
                        necesariamente personas distintas: nunca se unen. Sola,
                        esta regla elimina la mayoria de los errores posibles.

  SIN CONFLICTO         Si a un grupo llegan dos identidades distintas, el
                        grupo esta mal armado y no se propaga nada. El
                        conflicto es la señal, no un empate a resolver.

  UMBRAL ALTO           Con sface 0,80; con arcface 0,63. Medido: la misma
                        persona el mismo año da 0,94 y en otro año 0,72. Se
                        agarra lo seguro y se deja pasar lo dudoso, que es la
                        direccion correcta del error. El valor sale de
                        faces.umbral("agrupar"), que lo elige por motor.

La primera regla tiene una excepcion: la rafaga. El fotografo saca dos o tres
tomas seguidas del mismo chico, y esas dos fotos SI son la misma persona en la
misma carpeta. Por encima de faces.RAFAGA (0,90) se unen igual, porque a ese
parecido ya no hay forma de que sean dos personas: el maximo entre personas
distintas es 0,47 con arcface y 0,64 con sface. Sin esto, el juego pregunta
dos veces por la misma cara y la respuesta de la primera no alcanza a la
segunda.

    python faces_agrupar.py                 # solo informa
    python faces_agrupar.py --aplicar
"""
import argparse
import os
import sys

import numpy as np

import faces
import sb

HERE = os.path.dirname(os.path.abspath(__file__))
HUELLAS = faces.npz("_huellas_retratos.npz")
UMBRAL = faces.umbral("agrupar")
# Un chico tiene a lo sumo un retrato por año y el colegio son 14 años; un
# grupo mucho mas grande que eso es una fusion mal hecha, no una persona.
GRUPO_MAX = 20


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=UMBRAL)
    ap.add_argument("--aplicar", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(HUELLAS):
        sys.exit("Faltan las huellas. Corré antes:  python faces_huellas.py")
    z = np.load(HUELLAS, allow_pickle=True)
    ids = [str(x) for x in z["photo_ids"]]
    vecs = z["vecs"].astype(np.float32)
    carpetas = [str(x) for x in z["carpetas"]]
    print("Huellas cargadas: %d" % len(ids))

    cli = sb.SB()
    dueno = {r["photo_id"]: r["person_id"]
             for r in cli.select("photo_people", select="photo_id,person_id")}
    vivas = {p["id"] for p in cli.select("people", select="id,kind")
             if p.get("kind") != "noise"}
    rechazos = set()
    for r in cli.select("photo_people_rejected", select="photo_id,person_id"):
        rechazos.add((r["photo_id"], r["person_id"]))
    print("Identificados: %d de %d" % (sum(1 for i in ids if i in dueno), len(ids)))

    # union-find sobre los pares que superan el umbral
    padre = list(range(len(ids)))

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    print("\nComparando todos contra todos…")
    unidos = 0
    BL = 512
    for i0 in range(0, len(ids), BL):
        sim = vecs[i0:i0 + BL] @ vecs.T
        for fila in range(sim.shape[0]):
            i = i0 + fila
            for j in np.where(sim[fila] >= a.umbral)[0]:
                j = int(j)
                if j <= i:
                    continue
                # misma carpeta = personas distintas, salvo que sea tan
                # parecido que solo puede ser la misma toma repetida
                if carpetas[i] == carpetas[j] and sim[fila][j] < faces.RAFAGA:
                    continue
                ri, rj = raiz(i), raiz(j)
                if ri != rj:
                    padre[rj] = ri
                    unidos += 1
        if (i0 // BL) % 5 == 0:
            print("  %d/%d" % (min(i0 + BL, len(ids)), len(ids)))
    print("Uniones: %d" % unidos)

    grupos = {}
    for i in range(len(ids)):
        grupos.setdefault(raiz(i), []).append(i)
    grupos = {k: v for k, v in grupos.items() if len(v) > 1}

    propuestas, conflictos, sin_dueno, gigantes = [], [], 0, 0
    fotos_sin_dueno = 0
    for miembros in grupos.values():
        if len(miembros) > GRUPO_MAX:
            gigantes += 1
            continue
        duenos = {dueno[ids[i]] for i in miembros if ids[i] in dueno}
        duenos = {d for d in duenos if d in vivas}
        if not duenos:
            sin_dueno += 1
            fotos_sin_dueno += len(miembros)
            continue
        if len(duenos) > 1:
            # Dos identidades en un mismo grupo: o el grupo esta mal armado, o
            # son dos fichas de la MISMA persona que habria que fusionar. Se
            # listan para mirarlas, nunca se propagan.
            conflictos.append(sorted(duenos))
            continue
        per = duenos.pop()
        for i in miembros:
            if ids[i] in dueno:
                continue
            if (ids[i], per) in rechazos:
                continue
            propuestas.append((ids[i], per))

    print("\nGrupos de mas de una foto: %d" % len(grupos))
    print("  con una sola identidad, propagables: %d fotos" % len(propuestas))
    print("  con dos identidades (no se tocan):   %d grupos" % len(conflictos))
    print("  sin ninguna identidad conocida:      %d grupos, %d fotos"
          % (sin_dueno, fotos_sin_dueno))
    print("  demasiado grandes, descartados:      %d grupos" % gigantes)

    if sin_dueno:
        # Este es el numero que importa para el juego: nombrar UNA foto de cada
        # grupo nombra a todo el grupo.
        print("\n  -> nombrar una foto por grupo resolveria %d fotos con %d respuestas"
              % (fotos_sin_dueno, sin_dueno))

    if conflictos:
        nombres = {p["id"]: p["display_name"]
                   for p in cli.select("people", select="id,display_name")}
        print("\nGrupos con dos identidades (revisar a mano):")
        for par in conflictos:
            print("   " + "  vs  ".join(nombres.get(x, str(x)) for x in par))

    if not a.aplicar:
        print("\nNo se escribio nada. Para aplicarlo:  python faces_agrupar.py --aplicar")
        return

    filas = [{"photo_id": f, "person_id": int(p), "votes": 0, "source": "cluster"}
             for f, p in propuestas]
    n = cli.upsert("photo_people", filas, on_conflict="photo_id,person_id")
    print("\nEscritas: %d identificaciones nuevas" % n)

    # Los grupos se guardan para que una respuesta del juego pueda nombrar al
    # grupo entero. Sin esto solo servian para copiar desde alguien ya
    # identificado, que es la minoria de los casos.
    # "puro" decide si la base propaga el nombre al resto del grupo. Un grupo
    # es impuro cuando tiene dos fotos de una misma carpeta que NO son la
    # misma toma: llegaron por cadena (A se parece a B, B a C) y en el medio
    # se colaron dos personas distintas.
    #
    # Se recalcula en cada corrida. Antes no se escribia nunca, asi que la
    # marca quedaba congelada de una corrida vieja y seguia bloqueando grupos
    # que ya estaban bien armados.
    cl = []
    impuros = 0
    for cid, miembros in enumerate(grupos.values(), 1):
        if len(miembros) > GRUPO_MAX:
            continue
        puro = True
        for x in range(len(miembros)):
            for y in range(x + 1, len(miembros)):
                a_, b_ = miembros[x], miembros[y]
                if carpetas[a_] == carpetas[b_] and float(vecs[a_] @ vecs[b_]) < faces.RAFAGA:
                    puro = False
                    break
            if not puro:
                break
        if not puro:
            impuros += 1
        for i in miembros:
            cl.append({"photo_id": ids[i], "cluster_id": cid, "puro": puro})
    m = cli.upsert("photo_clusters", cl, on_conflict="photo_id")
    print("Grupos guardados: %d fotos en %d grupos  (impuros: %d)"
          % (m, len(grupos), impuros))


if __name__ == "__main__":
    main()
