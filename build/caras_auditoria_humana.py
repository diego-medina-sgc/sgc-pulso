# -*- coding: utf-8 -*-
"""La auditoria de caras mal archivadas, con las respuestas humanas como juez.

    python caras_auditoria_humana.py              # valida, mide y no escribe
    python caras_auditoria_humana.py --aplicar

POR QUE REEMPLAZA A DOS SCRIPTS

caras_incoherentes.py y caras_mal_archivadas.py juzgaban contra las "fotos
oficiales" -planilla de staff, anuario, Valete, fuentes-. Esas fotos tambien
estaban contaminadas, y el 11 y 12/9/2026 dieron vuelta 1.138 "Si" humanos:
incoherentes se equivoco 471 veces contra 0 aciertos, malarchivadas 271 contra
77. Salieron del pulso ese dia (ver pulso.py) hasta tener un juez limpio.

El juez limpio es el que ya se valido dos veces el 13/9 (restituir_si_humanos.py
y si_humanos_juez.py): las caras que una persona confirmo con un "Si" y nadie
contradijo.

LA REGLA (la de caras_mal_archivadas.py, con otro juez)

Una identificacion de P esta mal si la cara se parece al centro humano de otra
persona Q:
    parecido con Q        >= 0,60
    y le gana a P por      >= 0,30
No alcanza con parecerse poco a P -perfil, mala luz, de chico-: otra persona
tiene que reclamarla con fuerza.

QUE SE JUZGA Y QUE NO

  - Solo identificaciones SIN "Si" humano: las de maquina, grupo, nombre de
    archivo. Un "Si" humano no se toca (y sql/disputas.sql tampoco lo deja).
  - Solo si P tiene centro humano (3 caras o mas). Sin eso no hay contra que
    comparar, y comparar contra la foto oficial es lo que fallo.
  - Si P y Q parecen la misma persona anotada dos veces (un nombre contenido
    en el otro), no se saca nada: se informa para fusionar.

LA VALIDACION VIENE CON LA AUDITORIA

Antes de juzgar, la misma regla se corre sobre los "Si" humanos intactos, cada
uno con el centro de su persona calculado SIN esa cara. Cada uno que la regla
marcaria es un "Si" correcto que se sacaria mal. Si pasan del 1% no escribe.

QUE HACE --aplicar

Respaldo primero. Despues el rechazo y el borrado (ver caras_sueltas.py). No se
le pone la cara a Q: esto mide, no confirma. Las sugerencias y referencias las
recalcula el pulso en su proxima vuelta, porque cambian identificaciones y
rechazos.
"""
import argparse
import os
import sys
import time
from collections import Counter, defaultdict

if os.environ.get("FACES_MOTOR", "").lower() != "arcface":
    sys.exit("Falta FACES_MOTOR. En PowerShell: $env:FACES_MOTOR='arcface'")

import numpy as np  # noqa: E402

import faces  # noqa: E402
import sb  # noqa: E402
from retratos_nombres import clave  # noqa: E402

MINIMO = 0.60
MARGEN = 0.30
MIN_CARAS = 3
TOPE_ERROR = 0.01


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cli = sb.SB()

    print("Leyendo respuestas e identificaciones…", flush=True)
    gente = {int(p["id"]): p["display_name"] for p in cli.select("people", select="id,display_name,kind")
             if p["kind"] != "noise"}
    labels = cli.select("photo_labels", select="photo_id,person_id,answer,labeled_by")
    si = {(l["photo_id"], int(l["person_id"])) for l in labels
          if l["answer"] == "yes" and l["labeled_by"]}
    no = {(l["photo_id"], int(l["person_id"])) for l in labels if l["answer"] == "no"}
    rech = {(r["photo_id"], int(r["person_id"])) for r in
            cli.select("photo_people_rejected", select="photo_id,person_id")}
    ident = {(r["photo_id"], int(r["person_id"])): r["source"] for r in
             cli.select("photo_people", select="photo_id,person_id,source")}

    # ── las huellas: la cara de evento propuesta para esa persona, o el retrato
    ev = np.load(faces.npz("_caras_evento.npz"), allow_pickle=True)
    EF, EP = ev["photo_ids"].astype(str), ev["personas"].astype(int)
    EV = ev["vecs"].astype(np.float32)
    EV /= np.maximum(np.linalg.norm(EV, axis=1, keepdims=True), 1e-9)
    de_evento = {}
    for i in range(len(EF)):
        de_evento.setdefault((EF[i], int(EP[i])), i)
    rr = np.load(faces.npz("_huellas_retratos.npz"), allow_pickle=True)
    RF = rr["photo_ids"].astype(str)
    RV = rr["vecs"].astype(np.float32)
    RV /= np.maximum(np.linalg.norm(RV, axis=1, keepdims=True), 1e-9)
    de_retrato = {}
    for i in range(len(RF)):
        de_retrato.setdefault(RF[i], i)

    # LA HUELLA DE LA CARA CONFIRMADA, NO LA DE LA PROPUESTA.
    # Diego, 13/9/2026, sobre un "Si" de Persona D que la validacion
    # marcaba como Persona E: "los veo bien etiquetados, cada uno en su
    # cara". _caras_evento guarda la cara que el reconocimiento PROPUSO para
    # (foto, persona); si la propuesta cayo sobre otra cara de la foto, se
    # juzgaba la cara equivocada. El recuadro de face_suggestions es el que se
    # confirmo, y en _caras_todas esta la huella de cada cara con su recuadro.
    caja = {(s["photo_id"], int(s["person_id"])): (round(float(s["bx"]), 4), round(float(s["by"]), 4))
            for s in cli.select("face_suggestions", select="photo_id,person_id,bx,by,bw")
            if float(s.get("bw") or 0) > 0}
    fotos_caja = {k[0] for k in caja}
    tz = np.load(faces.npz("_caras_todas.npz"), allow_pickle=True)
    TF, TC = tz["photo_ids"].astype(str), tz["cajas"]
    de_caja = {}
    for i in range(len(TF)):
        if TF[i] in fotos_caja:
            de_caja.setdefault((TF[i], round(float(TC[i][0]), 4), round(float(TC[i][1]), 4)), i)
    TV = tz["vecs"]
    print("  huellas por recuadro confirmado: %d" % len(de_caja), flush=True)

    def huella(k):
        c = caja.get(k)
        if c is not None:
            i = de_caja.get((k[0], c[0], c[1]))
            if i is not None:
                v = TV[i].astype(np.float32)
                return v / max(float(np.linalg.norm(v)), 1e-9)
        i = de_evento.get(k)
        if i is not None:
            return EV[i]
        j = de_retrato.get(k[0])
        return RV[j] if j is not None else None

    # ── el juez: centros de caras con "Si" humano intacto
    intactos = [k for k in si if k not in no and k not in rech and k in ident and k[1] in gente]
    por_persona = defaultdict(list)
    for k in intactos:
        v = huella(k)
        if v is not None:
            por_persona[k[1]].append((k, v))
    personas = sorted(p for p, xs in por_persona.items() if len(xs) >= MIN_CARAS)
    suma = {p: np.sum([v for _, v in por_persona[p]], axis=0) for p in personas}
    C = np.stack([suma[p] / np.linalg.norm(suma[p]) for p in personas])
    fila = {p: j for j, p in enumerate(personas)}
    print("  'Si' humanos intactos: %d   personas con centro humano: %d   (%.0f s)"
          % (len(intactos), len(personas), time.time() - t0), flush=True)

    def palabras(x):
        return set(clave(gente.get(x, "")).split())

    def misma_ficha(x, y):
        a1, a2 = palabras(x), palabras(y)
        return bool(a1) and bool(a2) and (a1 <= a2 or a2 <= a1)

    def juzgar(v, p, sin_propia=None):
        """(q, parecido_q, parecido_p) si la regla marca la cara, si no None."""
        s = C @ v
        j = fila[p]
        if sin_propia is not None:
            c = suma[p] - sin_propia
            propia = float(c @ v / max(float(np.linalg.norm(c)), 1e-9))
        else:
            propia = float(s[j])
        s[j] = -2.0
        k = int(np.argmax(s))
        if float(s[k]) >= MINIMO and float(s[k]) - propia >= MARGEN:
            return personas[k], float(s[k]), propia
        return None

    # ── validacion contra los "Si" humanos
    marcados_mal, juzgables = 0, 0
    ejemplos = []
    for p in personas:
        if len(por_persona[p]) - 1 < MIN_CARAS - 1:
            continue
        for k, v in por_persona[p]:
            juzgables += 1
            r = juzgar(v, p, sin_propia=v)
            if r and not misma_ficha(p, r[0]):
                marcados_mal += 1
                if len(ejemplos) < 8:
                    ejemplos.append((k, r))
    tasa = marcados_mal / max(juzgables, 1)
    print("\nVALIDACION: la regla sobre %d 'Si' humanos intactos marcaria %d (%.2f%%)"
          % (juzgables, marcados_mal, 100 * tasa))
    for k, (q, sq, sp) in ejemplos:
        print("   %s  confirmada %s (%.2f) y la regla dice %s (%.2f)"
              % (k[0], gente.get(k[1]), sp, gente.get(q), sq))

    # ── la auditoria
    malas, duplicadas, motivo = [], Counter(), Counter()
    for k, src in ident.items():
        f, p = k
        if k in si or p not in gente:
            continue
        # 'manual' y 'game' las puso una persona aunque no haya un "Si" en
        # photo_labels (el visor reasigna sin pasar por ahi). El primer ensayo
        # queria sacar dos 'manual' de Persona F: eso lo decide quien las
        # puso, no una medicion.
        if src in ("manual", "game"):
            motivo["la puso una persona (manual/game): no se juzga"] += 1
            continue
        if p not in fila:
            motivo["la persona no tiene centro humano"] += 1
            continue
        v = huella(k)
        if v is None:
            motivo["sin huella"] += 1
            continue
        r = juzgar(v, p)
        if not r:
            motivo["bien archivada o sin reclamo claro"] += 1
            continue
        if misma_ficha(p, r[0]):
            duplicadas[tuple(sorted((p, r[0])))] += 1
            motivo["misma persona en dos fichas: se informa"] += 1
            continue
        malas.append((f, p, r[0], r[1], r[2], src))
        motivo["mal archivada"] += 1

    print("\nAUDITORIA de %d identificaciones sin 'Si' humano" % sum(motivo.values()))
    for m, n in motivo.most_common():
        print("  %6d  %s" % (n, m))
    print("  por origen de la etiqueta: %s" % dict(Counter(x[5] for x in malas)))
    for (x, y), n in duplicadas.most_common(15):
        print("  FICHA DUPLICADA  %s = %s  (%d caras)" % (gente.get(x), gente.get(y), n))
    for f, p, q, sq, sp, src in sorted(malas, key=lambda t: t[4] - t[3])[:25]:
        print("  figura %-28s (%.2f, %s)  es %-28s (%.2f)  %s"
              % (gente.get(p, "")[:28], sp, src, gente.get(q, "")[:28], sq, f))
    c = Counter(p for _, p, _, _, _, _ in malas)
    print("  fichas afectadas: %d. Las que mas: %s"
          % (len(c), ", ".join("%s %d" % (gente.get(p), n) for p, n in c.most_common(8))))

    if tasa > TOPE_ERROR:
        print("\nNO SE ESCRIBE: la validacion marca %.2f%% de 'Si' humanos (tope %.0f%%)."
              % (100 * tasa, 100 * TOPE_ERROR))
        return
    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar. (%.0f s)" % (time.time() - t0))
        return
    if not malas:
        print("\nNada que sacar. (%.0f s)" % (time.time() - t0))
        return

    nom = "_auditoria_humana_%s.txt" % time.strftime("%Y%m%d_%H%M")
    with open(nom, "w", encoding="utf-8") as fh:
        fh.write("# photo_id\tsale person_id\tsale nombre\tparecido propio\tes de\tparecido\torigen\n")
        for f, p, q, sq, sp, src in malas:
            fh.write("%s\t%d\t%s\t%.4f\t%s\t%.4f\t%s\n" % (f, p, gente.get(p), sp, gente.get(q), sq, src))
    print("\nrespaldo: %s" % nom)
    filas = [{"photo_id": f, "person_id": p} for f, p, _, _, _, _ in malas]
    for i in range(0, len(filas), 200):
        cli.upsert("photo_people_rejected", filas[i:i + 200], on_conflict="photo_id,person_id")
    for f, p, _, _, _, _ in malas:
        cli.delete("photo_people", photo_id="eq." + f, person_id="eq.%d" % p)
    print("sacadas %d identificaciones mal archivadas (%.0f s)" % (len(malas), time.time() - t0))


if __name__ == "__main__":
    main()
