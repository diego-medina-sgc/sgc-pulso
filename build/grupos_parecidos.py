# -*- coding: utf-8 -*-
"""Que grupos de caras son probablemente la misma persona. El paso 3.

    python grupos_parecidos.py              # calibra y cuenta, no escribe

POR QUE

Diego describio el tercer juego: "Grupos sirve para armar grupos de caras mas
grandes, y el juego las confirma, une grupos o corrige nombres del grupo".

El agrupamiento une caras por encadenamiento y despues saca las que no
acompañan al centro. Eso corta las cadenas que mezclan personas, y a cambio
parte a la misma persona en varios grupos: Juan a los 6 y Juan a los 18 son
dos grupos correctos. Esas no se arreglan agrupando mejor: se arreglan
preguntando.

QUE SE COMPARA

El CENTRO de cada grupo -el promedio de las huellas de sus caras- contra el de
los grupos que ya se sabe de quien son. Un grupo es mucho mas estable que una
cara: una foto de perfil o de noche mueve una cara, no el promedio de cuarenta.

LA VERDAD SALE DE LAS CARAS QUE CONFIRMO UNA PERSONA, NO DE LOS NOMBRES

La primera version calibraba con face_group_names y dio basura: dos grupos de
la MISMA persona se parecian 0,02 de mediana, igual que dos personas
distintas. Se verifico que las huellas estaban bien leidas (el parecido de
cada cara al centro de su grupo, recalculado, difiere 0,0025 del guardado).
El problema eran los nombres: grupos_resolver nombra por FOTO. Si Juan esta
identificado en una foto de acto, un grupo chico de caras de esa foto puede
quedar como "Juan" aunque sean las del de al lado. Medido: 247 personas con
1.433 grupos automaticos, 4,8 caras de promedio, 1.150 de 2 o 3 caras, y solo
14% de sus fotos confirmadas por una persona.

Asi que un grupo "es de P" solo si tiene caras que una persona confirmo como P
CARA POR CARA -un "Si" sobre la sugerencia con su recuadro, o un retrato- y
ninguna confirmada como otro.

EL CORTE NO SE ELIGE: SE MIDE

  misma persona     pares de grupos con caras confirmadas de la misma persona
  distinta persona  pares de grupos con caras confirmadas de personas distintas

Las huellas salen de caras_agrupar_archivo.cargar(), el mismo espacio con el
que se armaron los grupos.
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict

# faces lee el motor al importarse: el default es sface y leeria otras huellas
if os.environ.get("FACES_MOTOR", "").lower() != "arcface":
    sys.exit("Falta FACES_MOTOR. En PowerShell: $env:FACES_MOTOR='arcface'")

import numpy as np  # noqa: E402

import sb  # noqa: E402
from caras_agrupar_archivo import cargar  # noqa: E402

TOPE = 600      # sql/grupos_gigantes.sql: arriba de esto el grupo no se pregunta

# Medidos el 13/9/2026 sobre 1.154 grupos con caras confirmadas: con 0,60 pasan
# 13 de 665.163 pares de personas distintas (0,002%) y el 48% de los de la
# misma persona. El margen sobre la segunda persona evita preguntar cuando hay
# dos candidatos casi iguales (hermanos): con 0,10 quedan 4 dudosos de 4.838.
CORTE = 0.60
MARGEN = 0.10

# Un grupo solo sirve de referencia de P si P esta en la mitad de sus fotos o
# mas. Diego, 28/9/2026: una ficha de staff tenia 547 fotos de chicos. Su
# referencia era un grupo de 584 caras de ~150 chicos donde ella aparecia en
# 11 fotos: una sola cara confirmada alcanzaba para bautizar al grupo entero,
# y el promedio de 584 caras de chicos se parece a cualquier chico. Medido ese
# dia sobre las 331 referencias en uso: 302 con su dueño en el 90-100% de las
# fotos, 20 por debajo del 20% (406 grupos nombrados, 374 de esa ficha) y apenas
# 9 en el medio. Se corta en el hueco.
PRESENCIA_REF = 0.50

# Un grupo nombrado por el sistema cuyo promedio no se parece a las caras
# confirmadas de esa persona no es de ella (sql/caras_no_parecen.sql). Medido
# el 28/9/2026 cara por cara: una confirmada contra el promedio de sus otras
# confirmadas queda debajo de 0,30 el 7,5% de las veces; contra el de otra
# persona, arriba de 0,30 el 0,12%. Por grupo el corte casi no importa: de
# 5.128 grupos medibles, 2.904 quedan debajo de 0,20 y 3.093 de 0,40.
#
# La referencia de la persona suma, ademas de las confirmadas de a una, las
# que confirmo una respuesta "si" de Grupos (photo_people 'grupo' con su
# recuadro en face_suggestions). Sin ellas 215 fichas con 20+ automaticas no
# se podian medir. Medido el 28/9/2026 (build/_medir_por_grupos.py): la
# referencia por Grupos contra la de a una da mediana 0,88 y minimo 0,49 para
# la misma persona, y maximo 0,34 para otra; y alcanza UNA cara: una sola
# confirmada contra el resto de la persona da mediana 0,79, contra otra
# persona nunca mas de 0,37. Las de Grupos se depuran antes (se sacan las que
# no se parecen al promedio de las demas): un grupo contestado mirando 6
# caras puede traer alguna ajena, como mucho el 7% medido.
NO_SE_PARECE = 0.30
MIN_CONFIRMADAS = 1


def clave_cara(foto, bx, by):
    """Una cara: foto y esquina del recuadro, redondeada como la guarda
    caras_agrupar_archivo.py. Se usa para comparar aca; a la base va cruda."""
    return (foto, round(float(bx), 4), round(float(by), 4))


def normal(m):
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def pct(a):
    if not len(a):
        return "sin datos"
    return "min %.2f  p05 %.2f  p25 %.2f  mediana %.2f  p75 %.2f  p95 %.2f  max %.2f" % (
        (a.min(),) + tuple(np.percentile(a, [5, 25, 50, 75, 95])) + (a.max(),))


def verdad_humana(cli, clave):
    """indice de cara -> persona, solo para caras que confirmo una persona.

    Evento: el "Si" humano es sobre (foto, persona) y la sugerencia de esa
    pareja guarda el recuadro: esa es la cara. Retrato: la foto es una sola
    persona, y en el mapa del padron la cara va con recuadro (0, 0).
    """
    si, no = set(), set()
    for l in cli.select("photo_labels", select="photo_id,person_id,answer"):
        k = (l["photo_id"], int(l["person_id"]))
        (si if l["answer"] == "yes" else no).add(k)
    rech = {(r["photo_id"], int(r["person_id"])) for r in
            cli.select("photo_people_rejected", select="photo_id,person_id")}
    buenas = si - no - rech

    cara = {}
    for s in cli.select("face_suggestions", select="photo_id,person_id,bx,by",
                        bw="not.is.null"):
        k = (s["photo_id"], int(s["person_id"]))
        if k not in buenas:
            continue
        i = clave.get((s["photo_id"], round(float(s["bx"]), 4), round(float(s["by"]), 4)))
        if i is not None:
            cara[i] = k[1]
    retratos = 0
    for foto, p in buenas:
        i = clave.get((foto, 0.0, 0.0))
        if i is not None and i not in cara:
            cara[i] = p
            retratos += 1
    return cara, len(buenas), retratos


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    cli = sb.SB()

    print("Leyendo las huellas (el mismo espacio del agrupamiento)…", flush=True)
    fotos, cajas, vecs, desc, quienes, del_padron = cargar()
    print("  %s" % desc)
    clave = {}
    for i, f in enumerate(fotos):
        k = (f, round(float(cajas[i][0]), 4), round(float(cajas[i][1]), 4))
        clave.setdefault(k, i)

    print("Leyendo los grupos…", flush=True)
    meta = {int(r["grupo"]): int(r["caras"]) for r in
            cli.select("face_group_meta", select="grupo,caras")}
    nombres = {int(r["grupo"]): (int(r["person_id"]), r["estado"]) for r in
               cli.select("face_group_names", select="grupo,person_id,estado")}
    gente = {int(p["id"]): p["display_name"] for p in
             cli.select("people", select="id,display_name")}
    miembros = defaultdict(list)
    rep = {}                 # indice de cara -> clave foto|bx|by
    crudo = {}               # indice de cara -> foto, bx, by tal como en la base
    central, mas_central = {}, {}   # grupo -> clave de su cara mas central
    for r in cli.select("face_groups", select="grupo,photo_id,bx,by,centro"):
        g = int(r["grupo"])
        if not (2 <= meta.get(g, 0) <= TOPE):
            continue
        k = clave_cara(r["photo_id"], r["bx"], r["by"])
        cen = float(r["centro"] or 0)
        if cen > mas_central.get(g, -9):
            mas_central[g] = cen
            # crudo, tal como vino de la base: la funcion lo compara por igualdad
            central[g] = (r["photo_id"], r["bx"], r["by"])
        i = clave.get((r["photo_id"], round(float(r["bx"]), 4), round(float(r["by"]), 4)))
        if i is not None:
            miembros[g].append(i)
            rep[i] = k
            crudo[i] = (r["photo_id"], r["bx"], r["by"])

    print("Leyendo las caras que confirmo una persona…", flush=True)
    cara, parejas, retratos = verdad_humana(cli, clave)
    print("  %d parejas foto-persona con 'Si' humano vigente -> %d caras ubicadas (%d de retrato)"
          % (parejas, len(cara), retratos))

    grupos = sorted(g for g, ix in miembros.items() if len(ix) >= 2)
    C = normal(np.stack([vecs[miembros[g]].mean(axis=0) for g in grupos]))
    pos = {g: k for k, g in enumerate(grupos)}

    # de quien es cada grupo, segun sus caras confirmadas
    de_quien, mezclados = {}, 0
    for g in grupos:
        votos = Counter(cara[i] for i in miembros[g] if i in cara)
        if not votos:
            continue
        if len(votos) > 1:
            mezclados += 1
            continue
        de_quien[g] = next(iter(votos))
    print("  grupos con caras confirmadas de UNA persona: %d   con caras confirmadas de dos o mas: %d   (%.0f s)"
          % (len(de_quien), mezclados, time.time() - t0))

    # y que el dueño este en la mitad de las fotos del grupo (PRESENCIA_REF)
    presentes = {(r["photo_id"], int(r["person_id"])) for r in
                 cli.select("photo_people", select="photo_id,person_id")}
    flojos, mezcla = [], set()
    tramos = Counter()
    for g, p in list(de_quien.items()):
        ix = miembros[g]
        con = sum(1 for i in ix if (fotos[i], p) in presentes)
        tramos[min(9, int(10 * con / len(ix)))] += 1
        if con < PRESENCIA_REF * len(ix):
            flojos.append((len(ix), con, p))
            del de_quien[g]
            # tampoco se lo nombra por parecido: es una mezcla, no un grupo
            mezcla.add(g)
    print("  presencia del dueño en su grupo:  " + "  ".join(
        "%d-%d%%: %d" % (t * 10, t * 10 + 10, tramos[t]) for t in range(10)))
    print("  descartados como referencia (su dueño en menos del %d%% de las fotos): %d"
          % (round(PRESENCIA_REF * 100), len(flojos)))
    for tam, con, p in sorted(flojos, reverse=True)[:8]:
        print("     %-28s en %d de %d" % (gente.get(p, "?")[:28], con, tam))

    # los nombres automaticos, contra esa verdad: cuanto le erra grupos_resolver
    auto = [g for g in de_quien if g in nombres]
    if auto:
        bien = sum(1 for g in auto if nombres[g][0] == de_quien[g])
        print("  de los grupos con nombre automatico que tienen caras confirmadas: %d de %d coinciden"
              % (bien, len(auto)))

    # ── calibracion ─────────────────────────────────────────────────────
    sabidos = sorted(de_quien)
    N = C[[pos[g] for g in sabidos]]
    quien = np.array([de_quien[g] for g in sabidos])
    S = N @ N.T
    iu = np.triu_indices(len(sabidos), k=1)
    misma = quien[iu[0]] == quien[iu[1]]
    s_misma = S[iu][misma]
    s_otra = S[iu][~misma]
    print("\nCALIBRACION sobre grupos con caras confirmadas cara por cara")
    print("  misma persona      (%7d pares)  %s" % (len(s_misma), pct(s_misma)))
    print("  personas distintas (%7d pares)  %s" % (len(s_otra), pct(s_otra)))
    print("\n  corte   misma persona que pasa   distintas que pasan")
    for t in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        print("  %.2f       %5.1f%%                  %5d  (%.3f%%)"
              % (t, 100.0 * (s_misma >= t).mean() if len(s_misma) else 0,
                 int((s_otra >= t).sum()), 100.0 * (s_otra >= t).mean() if len(s_otra) else 0))

    a_idx, b_idx = iu[0][~misma], iu[1][~misma]
    print("\n  los pares de personas DISTINTAS mas parecidos (hermanos, fichas duplicadas o un 'Si' mal dado):")
    for k in np.argsort(-S[iu][~misma])[:8]:
        ga, gb = sabidos[a_idx[k]], sabidos[b_idx[k]]
        print("     %.3f  %-28s (%d caras)  vs  %-28s (%d caras)"
              % (S[a_idx[k], b_idx[k]], gente.get(de_quien[ga], "?")[:28], meta[ga],
                 gente.get(de_quien[gb], "?")[:28], meta[gb]))

    # ── cuantas preguntas saldrian ────────────────────────────────────────
    sin_verdad = [g for g in grupos if g not in de_quien and g not in nombres
                  and g not in mezcla]
    U = C[[pos[g] for g in sin_verdad]]
    mejor, segundo, dueno, ref_de = [], [], [], []
    for i in range(0, len(sin_verdad), 2048):
        B = U[i:i + 2048] @ N.T
        for fila in B:
            orden = np.argsort(-fila)
            p1 = quien[orden[0]]
            ref_de.append(sabidos[orden[0]])
            mejor.append(fila[orden[0]])
            segundo.append(next((fila[j] for j in orden[1:60] if quien[j] != p1), -1.0))
            dueno.append(p1)
    mejor, segundo = np.array(mejor), np.array(segundo)
    caras_sv = np.array([meta[g] for g in sin_verdad])
    print("\nGRUPOS SIN NOMBRE NI CARAS CONFIRMADAS (%d) contra el grupo confirmado mas parecido"
          % len(sin_verdad))
    print("  parecido al mejor   %s" % pct(mejor))
    print("\n  corte   grupos que se preguntarian   caras   con duda (otra persona a menos de 0,10)")
    for t in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        m = mejor >= t
        duda = m & (mejor - segundo < 0.10)
        print("  %.2f      %6d                  %7d        %d"
              % (t, int(m.sum()), int(caras_sv[m].sum()), int(duda.sum())))

    orden = np.argsort(-caras_sv * (mejor >= 0.60))[:10]
    print("\n  los mas grandes con parecido >= 0,60:")
    for k in orden:
        if mejor[k] < 0.60:
            continue
        print("     grupo de %4d caras  ->  %-28s %.3f  (segundo %.3f)"
              % (caras_sv[k], gente.get(int(dueno[k]), "?")[:28], mejor[k], segundo[k]))

    # ── grupos nombrados por el sistema que no se parecen a su persona ──
    confirmadas = defaultdict(list)
    for i, p in cara.items():
        confirmadas[p].append(i)
    # las de Grupos: foto 'grupo' de la persona + el recuadro de su cara
    por_grupo_foto = {(r["photo_id"], int(r["person_id"])) for r in
                      cli.select("photo_people", select="photo_id,person_id", source="eq.grupo")}
    de_grupo = defaultdict(list)
    for s in cli.select("face_suggestions", select="photo_id,person_id,bx,by", bw="gt.0"):
        k = (s["photo_id"], int(s["person_id"]))
        if k in por_grupo_foto:
            i = clave.get((s["photo_id"], round(float(s["bx"]), 4), round(float(s["by"]), 4)))
            if i is not None and i not in cara:
                de_grupo[k[1]].append(i)
    def depuradas(ix):
        """Sin las que no se parecen al promedio de las demas."""
        v = normal(vecs[ix])
        s = v @ normal(v.mean(axis=0, keepdims=True))[0]
        return [i for i, x in zip(ix, s) if x >= NO_SE_PARECE]

    for p, ix in de_grupo.items():
        confirmadas[p].extend(depuradas(ix))
    # Los carnets (caras del padron), SOLO para quien no tiene ninguna cara
    # confirmada por una persona: 64 fichas con 20+ automaticas no tenian otra.
    # Medido el 28/9/2026 (build/_medir_por_carnets.py): un carnet contra la
    # referencia humana de su persona da mediana 0,77 y queda debajo de 0,30 el
    # 0,7%; contra otra persona, maximo 0,40. Pero 3 de 1.864 personas tienen
    # los carnets a nombre de otro: por eso no se mezclan con una referencia
    # humana, que manda.
    carnets = defaultdict(list)
    for i in np.flatnonzero(del_padron):
        if int(quienes[i]) > 0 and i not in cara:
            carnets[int(quienes[i])].append(int(i))
    con_carnet = 0
    for p, ix in carnets.items():
        if not confirmadas.get(p):
            confirmadas[p] = depuradas(ix)
            con_carnet += 1
    centro_de = {p: normal(normal(vecs[ix]).mean(axis=0, keepdims=True))[0]
                 for p, ix in confirmadas.items() if len(ix) >= MIN_CONFIRMADAS}
    print("\nReferencias para medir grupos automaticos: %d personas (%d con caras de Grupos, %d solo con carnets)"
          % (len(centro_de), sum(1 for p in de_grupo if p in centro_de), con_carnet))
    no_parecen, grupos_malos, medidos = [], 0, 0
    for g, (p, estado) in nombres.items():
        if estado != "automatico" or p not in centro_de or g not in miembros:
            continue
        ix = [i for i in miembros[g] if i not in cara]
        if not ix:
            continue
        medidos += 1
        s = float(normal(normal(vecs[ix]).mean(axis=0, keepdims=True))[0] @ centro_de[p])
        if s < NO_SE_PARECE:
            grupos_malos += 1
            for i in ix:
                foto, bx, by = crudo[i]
                no_parecen.append({"photo_id": foto, "bx": bx, "by": by,
                                   "person_id": p, "parecido": round(s, 4)})
    print("\nGRUPOS AUTOMATICOS QUE NO SE PARECEN A SU PERSONA (debajo de %.2f): %d de %d medibles, %d caras"
          % (NO_SE_PARECE, grupos_malos, medidos, len(no_parecen)))

    if "--aplicar" not in sys.argv:
        # para poder mirarla antes de aplicar
        with open(os.path.join(os.path.dirname(__file__), "_no_parecen_ensayo.json"), "w",
                  encoding="utf-8") as f:
            json.dump(no_parecen, f)
        print("\nEnsayo: no se escribio nada. Agregar --aplicar. (%.0f s)" % (time.time() - t0))
        return

    # Se anotan las caras (sobreviven a los reagrupamientos) y se deshace lo que
    # el sistema habia puesto encima. El resolver ya no las vuelve a nombrar.
    if no_parecen:
        cli.upsert("caras_no_parecen", no_parecen, on_conflict="photo_id,bx,by,person_id")
    print("  deshecho: %s" % cli.rpc("deshacer_no_parecen", {"p_ensayo": False}))

    # ── escribir los pares ────────────────────────────────────────────────
    #
    # Un par por grupo sin nombre: el grupo confirmado mas parecido, si pasa
    # el corte y le gana a la segunda persona por el margen. Se guarda la cara
    # mas central de cada grupo para que la base pueda saber si el par sigue
    # vigente despues de un reagrupamiento (sql/grupos_parecidos.sql).
    no_son = defaultdict(set)
    for r in cli.select("grupos_no_son", select="photo_id,bx,by,person_id"):
        no_son[clave_cara(r["photo_id"], r["bx"], r["by"])].add(int(r["person_id"]))

    filas = []
    salteados_por_no = 0
    for k, g in enumerate(sin_verdad):
        if mejor[k] < CORTE or mejor[k] - segundo[k] < MARGEN:
            continue
        p = int(dueno[k])
        if any(p in no_son.get(rep[i], ()) for i in miembros[g] if i in rep):
            salteados_por_no += 1
            continue
        ref = ref_de[k]
        filas.append({"grupo": g, "grupo_ref": ref, "person_id": p,
                      "parecido": round(float(mejor[k]), 4),
                      "segundo": round(float(segundo[k]), 4),
                      "caras": meta[g],
                      "rep_photo": central[g][0], "rep_bx": central[g][1],
                      "rep_by": central[g][2],
                      "ref_photo": central[ref][0], "ref_bx": central[ref][1],
                      "ref_by": central[ref][2]})
    print("\nPares a escribir: %d  (corte %.2f, margen %.2f; %d salteados porque alguien ya dijo que no)"
          % (len(filas), CORTE, MARGEN, salteados_por_no))
    # Todos, tambien los de numero negativo. Con "gte.0" los grupos negativos
    # -1.872 hoy, los que nacen de retratos- nunca se limpiaban: el 18/9 habia
    # 761 pares suyos del 14 y 16/9 apuntando a grupos que ya no eran esos.
    cli.delete("grupos_parecidos", grupo="not.is.null")
    cli.upsert("grupos_parecidos", filas, on_conflict="grupo")
    print("Listo. (%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    main()
