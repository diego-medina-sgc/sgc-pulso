# -*- coding: utf-8 -*-
"""
Aprende de los "Sí, es": si la cara se parece a varias confirmadas, ya está.

    python caras_si_es.py                   # mide la regla contra las respuestas
    python caras_si_es.py --aplicar
    python caras_si_es.py --votos 3 --corte 0.80

POR QUE

Lo pidió Diego, y es el otro lado de caras_no_es.py: "lo mismo debería
funcionar en los casos positivos. Si Persona H tiene 61 fotos, muchas
confirmadas y muchas por confirmar, para las que quedan pendientes ya debería
aprender que ese es Persona H, tiene mucho para comparar".

Y remató con algo que hay que responder derecho: "ahora entiendo por qué no
pre-aprueba, es porque el sistema no aprende".

Sí pre-aprueba: eventos_aprobar.py está medido en 99,56% sobre las respuestas
que dieron él y John, y está esperando a que se cierren las incoherencias
porque él pidió ese orden. Pero su regla es más pobre que esto. Dice "es la
primera propuesta de esa cara y la persona tiene cinco caras de referencia", y
eso no mira la cara que está en juego: mira cuánto sabe el sistema de la
persona. Dos fotos distintas de dos personas distintas pueden pasar ese filtro
con el mismo puntaje.

LA REGLA DE ACA

Una cara pendiente se aprueba si se parece a VARIAS de las caras que una
persona ya confirmó para esa ficha. No a la más parecida: a varias.

Es la diferencia que importa. Con 61 confirmadas, el máximo de 61 comparaciones
se alcanza por azar con cualquier cara borrosa -es lo que le pasó a María
Soloeta, donde 34 intentos encontraron uno que daba 0,676-. Que tres de sus
caras confirmadas digan lo mismo no se consigue por azar.

SOLO LO QUE CONFIRMO UNA PERSONA

Las caras de referencia salen de photo_people, y ahí hay de todo: lo que alguien
contestó en el juego, lo que aprobó un script, lo que salió del nombre de un
archivo. Para aprender se usan sólo las de source='game', que son las que miró
una persona. Aprender de lo que aprobó un script es cómo se envenena una ficha:
es exactamente la cadena que dejó a Persona I con cuatro caras de un nene.

EL UMBRAL SALE DE LOS DATOS, NO DE MI

Sin --aplicar el script mide: toma las sugerencias que ya tienen respuesta
humana, aplica la regla con varias combinaciones de votos y corte, y dice
cuántas habría aprobado y cuántas mal. Recién con eso se elige.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

import faces
import sb

VOTOS = 3        # cuantas caras confirmadas tienen que coincidir
CORTE = 0.80     # cuanto tiene que parecerse a cada una


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--votos", type=int, default=VOTOS)
    ap.add_argument("--corte", type=float, default=CORTE)
    ap.add_argument("--tope", type=int, default=0)
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
    fila = {(f, p): i for i, (f, p) in enumerate(zip(fotos, personas))}
    de_persona = defaultdict(list)
    for i, p in enumerate(personas):
        de_persona[p].append(i)
    print("Huellas de cara guardadas: %d   fichas con al menos una: %d"
          % (len(fotos), len(de_persona)))

    pp = cli.select("photo_people", select="photo_id,person_id,source")
    # solo lo que miró una persona: ver arriba
    confirmadas = defaultdict(list)
    for r in pp:
        if r.get("source") != "game":
            continue
        i = fila.get((r["photo_id"], int(r["person_id"])))
        if i is not None:
            confirmadas[int(r["person_id"])].append(i)
    print("Fichas con caras confirmadas a mano y con huella: %d   caras: %d"
          % (len(confirmadas), sum(len(v) for v in confirmadas.values())))

    si = {(r["photo_id"], int(r["person_id"])) for r in pp}
    rech = {(r["photo_id"], int(r["person_id"])) for r in
            cli.select("photo_people_rejected", select="photo_id,person_id")}

    def votos_de(i, p, excluir=None):
        """Cuantas caras confirmadas de p se parecen a la cara i."""
        idx = [j for j in confirmadas.get(p, []) if j != i and j != excluir]
        if not idx:
            return 0, 0.0
        s = vecs[idx] @ vecs[i]
        return int((s >= a.corte).sum()), float(s.max())

    # ── medir contra lo que ya contestaron ──────────────────────────────
    print("\nQUE HABRIA DICHO LA REGLA EN LAS RESPUESTAS QUE YA HAY")
    print("   %-18s %8s %8s %9s" % ("votos y corte", "aprueba", "mal", "acierto"))
    for votos in (1, 2, 3, 5):
        for corte in (0.75, 0.80, 0.85):
            ok = mal = 0
            for f, p in list(si) + list(rech):
                i = fila.get((f, p))
                if i is None:
                    continue
                idx = [j for j in confirmadas.get(p, []) if j != i]
                if not idx:
                    continue
                s = vecs[idx] @ vecs[i]
                if int((s >= corte).sum()) < votos:
                    continue
                if (f, p) in si:
                    ok += 1
                else:
                    mal += 1
            if ok + mal:
                print("   %-18s %8d %8d %8.2f%%"
                      % ("%d votos, %.2f" % (votos, corte), ok + mal, mal,
                         100.0 * ok / (ok + mal)))

    # ── las pendientes que pasarían ─────────────────────────────────────
    fs = cli.select("face_suggestions", select="photo_id,person_id,score,bw")
    pasan = []
    for x in fs:
        if x["bw"] is None:
            continue
        p = int(x["person_id"])
        k = (x["photo_id"], p)
        if k in si or k in rech:
            continue
        i = fila.get(k)
        if i is None:
            continue
        n, mejor = votos_de(i, p)
        if n >= a.votos:
            pasan.append((n, mejor, x["photo_id"], p))
    pasan.sort(reverse=True)
    print("\nPendientes que aprobaría con %d votos a %.2f: %d   sobre %d fichas"
          % (a.votos, a.corte, len(pasan), len({x[3] for x in pasan})))
    if pasan:
        g = {int(y["id"]): y["display_name"] for y in
             cli.select("people", select="id,display_name")}
        c = Counter(x[3] for x in pasan)
        for pid, n in c.most_common(10):
            print("   %-34s %d" % (g.get(pid, "?")[:34], n))

    if a.tope:
        pasan = pasan[:a.tope]
        print("Con --tope %d: %d" % (a.tope, len(pasan)))
    if not a.aplicar:
        print("\nEnsayo: no se escribió nada. Agregar --aplicar.")
        return
    filas = [{"photo_id": f, "person_id": p, "votes": 1, "source": "cluster"}
             for _, _, f, p in pasan]
    # La lista exacta queda en un archivo ANTES de escribir.
    #
    # source='cluster' ya lo usa otro camino -1.732 filas-, asi que "borrar por
    # source" se llevaria cosas que no son de esta tanda, y filtrar por hora es
    # fragil. Con la lista, revertir es exacto.
    import time as _t
    reg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_si_es_%s.txt" % _t.strftime("%Y%m%d_%H%M"))
    with open(reg, "w", encoding="utf-8") as fh:
        fh.write("# %d votos a %.2f  -  %s\n"
                 % (a.votos, a.corte, _t.strftime("%Y-%m-%d %H:%M")))
        for n, mejor, f, q in pasan:
            fh.write("%s\t%d\t%d votos\t%.4f\n" % (f, q, n, mejor))
    for i in range(0, len(filas), 200):
        cli.upsert("photo_people", filas[i:i + 200], on_conflict="photo_id,person_id")
    print("\nAprobadas %d." % len(filas))
    print("La lista exacta de lo que se escribio quedo en %s" % os.path.basename(reg))
    print("Para revertir, leyendo esa lista:")
    print("  python caras_si_es_revertir.py %s" % os.path.basename(reg))


if __name__ == "__main__":
    main()
