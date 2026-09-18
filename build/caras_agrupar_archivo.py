# -*- coding: utf-8 -*-
"""
Agrupa las caras del archivo entre sí, sin nombres. El paso 3 del plan.

    python caras_agrupar_archivo.py                 # mide y no escribe
    python caras_agrupar_archivo.py --corte 0.75
    python caras_agrupar_archivo.py --aplicar

POR QUE

Lo describió Diego y es la idea que cambia todo: "sin importar si lo etiqueto
ahora o en 3 meses, el archivo ya tendría que haber mapeado caras similares y
agruparlas, a la espera de un nombre".

Hoy el trabajo caro -bajar la foto, detectar la cara, medirla- está pegado al
barato -compararla contra el padrón-. Cada vez que mejoramos lo barato volvemos
a hacer lo caro: veinte horas para cambiar una decisión que se toma en
microsegundos.

Separados, el archivo se mide UNA vez y después todo es álgebra. Y si además
las caras ya vienen agrupadas, ponerle el nombre a una es ponérselo a las
cuarenta del grupo, de una y sin comparar nada.

COMO SE AGRUPA

Cada cara contra todas las demás, y las que se parecen más que el corte quedan
en el mismo grupo. Con encadenamiento: si A se parece a B y B a C, los tres van
juntos aunque A y C no se hayan visto nunca. Eso es lo que permite que una
persona de perfil y la misma de frente terminen en el grupo, unidas por una
tercera en tres cuartos.

El encadenamiento también es el riesgo: una sola cara ambigua puede pegar dos
personas. Por eso después de armar cada grupo se lo revisa contra su propio
centro y se sacan los que no lo acompañan. Es barato y corta las cadenas.

EL TAMAÑO DEL PROBLEMA

Con 450.000 caras son 10^11 pares. No se puede armar la matriz entera -serían
terabytes- así que va por bloques: cada bloque de caras contra todas, y sólo se
guardan los pares que pasan el corte. Son unos 20 minutos de cuenta pura, una
vez por noche, sin bajar una sola foto.
"""
import argparse
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

import faces
import sb

CORTE = 0.75        # dos caras de la misma persona
CENTRO = 0.60       # cuánto tiene que acompañar al centro del grupo
BLOQUE = 512        # caras por vuelta; 512 x 450.000 son ~900 MB de cuenta


def memoria(cli, fotos, quienes, confirmado, n, cajas=None):
    """Las respuestas humanas, traducidas a obligaciones y prohibiciones.

    Regla 3 de las que fijo Diego, y no es negociable: "una correccion vale
    tanto como una confirmacion, y para siempre". Si dijo que la cara X no es
    Roberto, y la cara Y si es Roberto, entonces X e Y no pueden volver a
    caer en el mismo grupo nunca. El agrupamiento deja de ser solo geometria.

    DE DONDE SALE UNA RESPUESTA HUMANA

    photo_people tiene 15.570 filas pero solo 3.407 las contesto una persona:
    las de source 'game' y 'manual'. Las otras -face_auto, filename, cluster,
    face, mugshot_filename- las puso el sistema solo.

    La diferencia importa y mucho. El must-link fusiona grupos enteros a
    ciegas, sin mirar si las caras se parecen; es justamente lo que arregla la
    fragmentacion -Juan a los 6 y Juan a los 18 son dos grupos correctos que
    hay que unir-. Pero si la etiqueta esta mal, ese mismo poder sirve para
    pegar dos personas distintas y no hay parecido que lo frene.

    Y esta medido que las etiquetas automaticas se equivocan: 309 grupos
    tienen una ficha con una cara adentro que no es de esa persona. Tomar esas
    como obligacion seria sembrar el mapa entero con esos errores. Por eso el
    must-link sale solo de lo que contesto un humano.

    El cannot-link no tiene ese problema -prohibir de mas cuesta grupos mas
    chicos, no grupos mezclados- asi que toma photo_people_rejected entero.
    """
    humanas = {}
    for r in cli.select("photo_people", select="photo_id,person_id,source"):
        if r["source"] in ("game", "manual"):
            humanas.setdefault(r["photo_id"], set()).add(int(r["person_id"]))
    rechazos = {}
    for r in cli.select("photo_people_rejected", select="photo_id,person_id"):
        rechazos.setdefault(r["photo_id"], set()).add(int(r["person_id"]))

    # Una cara es "de P" si su fila propone a P y un humano confirmo a P en esa
    # foto. Para las caras del padron alcanza la ficha... salvo que no: la
    # ficha es justamente lo que puede estar envenenado. Tambien piden humano.
    de_quien = {}
    prohibido = {}
    for m in range(n):
        q = int(quienes[m])
        f = fotos[m]
        if q >= 0 and q in humanas.get(f, ()):
            de_quien[m] = q
        mal = rechazos.get(f)
        if mal:
            prohibido[m] = set(mal)

    # "Esta cara no es X", POR CARA (grupos_no_son). Desde el 13/9/2026 los
    # "No" del visor, de Confirmar y del captcha se guardan asi y ya no en
    # photo_people_rejected (sql/no_es_por_cara_juegos.sql): sin leerlos aca,
    # el agrupamiento perderia esas prohibiciones y volveria a juntar la cara
    # con la persona.
    if cajas is not None:
        no_es = {}
        for r in cli.select("grupos_no_son", select="photo_id,bx,by,person_id"):
            k = (r["photo_id"], round(float(r["bx"]), 4), round(float(r["by"]), 4))
            no_es.setdefault(k, set()).add(int(r["person_id"]))
        if no_es:
            for m in range(n):
                k = (fotos[m], round(float(cajas[m][0]), 4), round(float(cajas[m][1]), 4))
                if k in no_es:
                    prohibido.setdefault(m, set()).update(no_es[k])

    juntar = {}
    for m, q in de_quien.items():
        juntar.setdefault(q, []).append(m)
    juntar = {q: v for q, v in juntar.items() if len(v) > 1}
    return de_quien, prohibido, juntar


def cargar():
    """Todas las caras medidas, en un solo espacio: el archivo y el padron.

    POR QUE JUNTAS

    Hasta hoy los retratos del padron y las caras de los eventos se agrupaban
    en espacios separados, y eso rompe la promesa central del ciclo: "no me
    preguntes por una cara que ya etiquete". Si la cara de Roberto en su ficha
    vive en un espacio y la cara de Roberto en el acto de 2015 en otro, el
    grupo del acto nace sin nombre aunque el nombre este a un centimetro.

    Juntas, un grupo que toca una sola cara del padron ya sabe como se llama y
    no hay que preguntar nada. Ese es todo el paso 1.

    LAS DOS CLASES DE NOMBRE NO VALEN IGUAL

    La cara del padron trae su persona CONFIRMADA: sale de la ficha, no de un
    parecido. La cara de evento trae la persona que el reconocimiento propuso,
    que es una hipotesis y hay que ir a buscar si alguien la confirmo. Se
    devuelve un booleano por cara para no mezclar las dos cosas: dar por cierta
    una hipotesis es exactamente el error que el ciclo nuevo quiere evitar.

    LAS FOTOS REPETIDAS NO MOLESTAN

    2.567 fotos estan de los dos lados. No se deduplican, y no hace falta: la
    misma cara medida dos veces se junta sola en el primer par que se mire, y
    de paso es el puente que suelda los dos espacios. Como el npz de eventos no
    guarda recuadros, tampoco se podria saber cual de las tres caras de una
    foto es la del padron.
    """
    partes = []

    # -- el archivo ----------------------------------------------------------
    todas = faces.npz("_caras_todas.npz")
    prop = faces.npz("_caras_evento.npz")

    # LOS DOS MAPAS, NO UNO U OTRO.
    #
    # _caras_todas.npz es el mapa bueno: TODAS las caras detectadas, con su
    # recuadro, se parezcan a alguien o no. _caras_evento.npz es el viejo, con
    # solo las que el reconocimiento propuso y sin recuadro.
    #
    # Esto decia "si existe el nuevo, usa el nuevo", que va a ser lo correcto
    # cuando el nuevo cubra el archivo entero. Hoy no: lo escribe la corrida de
    # eventos a medida que avanza, y el 11/9 tenia 2026 y nada mas. La noche que
    # ese archivo aparecio, el agrupamiento habria pasado de 37.572 caras a
    # 11.572, tirando veintitres anios sin que nada avisara.
    #
    # Asi que se cargan los dos y gana el nuevo POR FOTO: de una foto que ya
    # esta en el mapa completo no se toma nada del viejo, porque ahi el viejo no
    # agrega caras, agrega copias sin recuadro de caras que ya estan.
    cubiertas = set()
    if os.path.exists(todas):
        with np.load(todas, allow_pickle=True) as z:
            n = len(z["photo_ids"])
            fotos_t = z["photo_ids"].tolist()
            cubiertas = set(fotos_t)
            partes.append((fotos_t,
                           z["cajas"].astype(np.float32),
                           z["vecs"].astype(np.float32),
                           np.full(n, -1, dtype=int),
                           np.zeros(n, dtype=bool),
                           "%d del archivo (todas las caras detectadas)" % n))
    if os.path.exists(prop):
        with np.load(prop, allow_pickle=True) as z:
            fotos_p = z["photo_ids"].tolist()
            keep = [k for k, f in enumerate(fotos_p) if f not in cubiertas]
            if keep:
                ix = np.array(keep)
                n = len(keep)
                partes.append(([fotos_p[k] for k in keep],
                               np.zeros((n, 4), dtype=np.float32),
                               z["vecs"][ix].astype(np.float32),
                               z["personas"][ix].astype(int),
                               np.zeros(n, dtype=bool),
                               "%d del archivo viejo (sin recuadro)" % n))

    # -- el padron -----------------------------------------------------------
    refs = faces.npz("_referencias.npz")
    if os.path.exists(refs):
        with np.load(refs, allow_pickle=True) as z:
            n = len(z["photo_ids"])
            partes.append((z["photo_ids"].tolist(),
                           np.zeros((n, 4), dtype=np.float32),
                           z["vecs"].astype(np.float32),
                           z["personas"].astype(int),
                           np.ones(n, dtype=bool),
                           "%d del padron (con nombre confirmado)" % n))

    if not partes:
        sys.exit("Todavia no hay caras medidas. Corre faces_eventos.py primero.")

    fotos = [f for p in partes for f in p[0]]
    cajas = np.concatenate([p[1] for p in partes])
    vecs = np.concatenate([p[2] for p in partes])
    quienes = np.concatenate([p[3] for p in partes])
    confirmado = np.concatenate([p[4] for p in partes])
    return fotos, cajas, vecs, " + ".join(p[5] for p in partes), quienes, confirmado


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--corte", type=float, default=CORTE)
    ap.add_argument("--centro", type=float, default=CENTRO)
    a = ap.parse_args()

    fotos, cajas, V, de_donde, quienes, confirmado = cargar()
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    n = len(V)
    print("Caras a agrupar: %d   (%s)" % (n, de_donde))

    # ── el recuadro de cada cara ────────────────────────────────────────
    #
    # Los dos npz guardan (foto, persona, huella) y ninguno guarda donde estaba
    # la cara. Sin recuadro pasan dos cosas y las dos son graves: la pantalla
    # muestra la foto entera en vez de la cara -en un acto de cuarenta personas
    # eso no es una pregunta contestable, que es justo lo que Diego marco con la
    # foto de Persona B- y ademas las caras de una misma foto quedan todas
    # en (0,0,0,0), que para el indice unico son LA MISMA cara: de tres caras de
    # una foto se guardaba una sola. En la primera corrida se perdieron 6.654.
    #
    # El recuadro esta en face_suggestions desde que faces_eventos.py lo empezo
    # a guardar, y la clave es la misma que la de estos npz: (foto, persona).
    cli = sb.SB()
    guardadas = {}
    for r in cli.select("face_suggestions", select="photo_id,person_id,bx,by,bw,bh"):
        if r.get("bw") and r.get("bh"):
            guardadas[(r["photo_id"], int(r["person_id"]))] = (
                r["bx"], r["by"], r["bw"], r["bh"])
    puestos = 0
    for m in range(n):
        q = int(quienes[m])
        if q < 0:
            continue
        c = guardadas.get((fotos[m], q))
        if c:
            cajas[m] = c
            puestos += 1
    print("Caras con recuadro conocido: %d de %d" % (puestos, n))

    de_quien, prohibido, juntar = memoria(cli, fotos, quienes, confirmado, n, cajas)
    print("Memoria: %d caras con nombre puesto por un humano, %d con algun 'no es'"
          % (len(de_quien), len(prohibido)))

    # ── union-find con memoria ──────────────────────────────────────────
    #
    # Un union-find comun solo sabe juntar. El cannot-link es lo contrario -dos
    # caras que tienen PROHIBIDO juntarse- y eso no entra en la estructura: hay
    # que preguntarlo antes de cada fusion. Cada raiz lleva entonces dos
    # conjuntos, los nombres que tiene adentro y los que tiene prohibidos, y la
    # fusion se hace solo si ninguno de los dos lados contradice al otro.
    #
    # POR QUE LOS PARES VAN ORDENADOS
    #
    # Con prohibiciones el resultado depende del orden: la primera fusion que
    # llega se queda con el lugar y la que venia despues puede quedar vetada por
    # una prohibicion que la primera trajo. Entonces que llegue primero la mejor
    # evidencia, no la que estaba mas arriba en el archivo. Ordenar 374.000
    # pares cuesta menos de un segundo y saca el azar del medio.
    padre = np.arange(n)

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    nombres_de = defaultdict(set)
    vetos_de = defaultdict(set)
    for m, q in de_quien.items():
        nombres_de[m].add(q)
    for m, qs in prohibido.items():
        vetos_de[m] |= qs

    def fundir(i, j):
        ri, rj = raiz(i), raiz(j)
        if ri == rj:
            return True
        if nombres_de[ri] & vetos_de[rj] or nombres_de[rj] & vetos_de[ri]:
            return False
        padre[ri] = rj
        nombres_de[rj] |= nombres_de[ri]
        vetos_de[rj] |= vetos_de[ri]
        nombres_de.pop(ri, None)
        vetos_de.pop(ri, None)
        return True

    # LOS PARES VAN EN NUMPY, NO EN TUPLAS DE PYTHON.
    #
    # La primera version los juntaba como (float, int, int) en una lista. Con
    # 48.000 caras eran 374.000 pares y no se notaba. El mapa completo del
    # archivo va por 243.097 caras y los pares crecen al cuadrado: una tupla de
    # Python pesa unos 72 bytes contra 12 de tres numeros en un array, asi que
    # lo que en numpy son 400 MB en tuplas son dos gigas y medio.
    #
    # Importa porque esto corre a las 02:00 sin nadie mirando, en la misma
    # maquina donde el rehacer de eventos ya esta usando memoria. Quedarse sin
    # RAM a esa hora es otra corrida que aparece muerta a la manana.
    t0 = time.time()
    trozos_s, trozos_i, trozos_j = [], [], []
    cuantos = 0
    for i0 in range(0, n, BLOQUE):
        S = V[i0:i0 + BLOQUE] @ V.T
        # el triangulo de arriba y nada mas: el par (i,j) ya se miro cuando
        # toco j. Se arma de una para todo el bloque en vez de fila por fila.
        filas, cols = np.nonzero(S >= a.corte)
        reales = filas + i0
        quedan = cols > reales
        filas, cols, reales = filas[quedan], cols[quedan], reales[quedan]
        if len(filas):
            trozos_s.append(S[filas, cols].astype(np.float32))
            trozos_i.append(reales.astype(np.int32))
            trozos_j.append(cols.astype(np.int32))
            cuantos += len(filas)
        if (i0 // BLOQUE) % 20 == 0 and i0:
            hechas = i0 + BLOQUE
            falta = (time.time() - t0) / hechas * (n - hechas)
            print("   %d/%d caras   pares hallados: %d   faltan ~%d min"
                  % (hechas, n, cuantos, falta / 60))
    ps = np.concatenate(trozos_s) if trozos_s else np.zeros(0, dtype=np.float32)
    pi = np.concatenate(trozos_i) if trozos_i else np.zeros(0, dtype=np.int32)
    pj = np.concatenate(trozos_j) if trozos_j else np.zeros(0, dtype=np.int32)
    del trozos_s, trozos_i, trozos_j
    print("Pares por encima de %.2f: %d   (%.0f s, %.0f MB)"
          % (a.corte, len(ps), time.time() - t0,
             (ps.nbytes + pi.nbytes + pj.nbytes) / 1e6))

    # argsort sobre el puntaje y despues se recorre por indice. Sin .tolist():
    # convertir 33 millones de enteros a objetos de Python vuelve a costar el
    # gigabyte que se acaba de ahorrar.
    orden = np.argsort(-ps)
    pi, pj = pi[orden], pj[orden]
    del ps, orden
    vetadas = 0
    for k in range(len(pi)):
        if not fundir(int(pi[k]), int(pj[k])):
            vetadas += 1
    print("fusiones frenadas por un 'no es': %d" % vetadas)

    # ── lo que el humano mando juntar ───────────────────────────────────
    #
    # Esto es lo unico que une sin mirar el parecido, y es a proposito: la cara
    # de primer grado y la de ultimo ano no se parecen, pero es la misma
    # persona y el que lo sabe lo dijo. Va DESPUES de los pares para que las
    # prohibiciones ya esten adentro de cada grupo cuando se intente.
    unidos = frenados = 0
    for q, miembros in juntar.items():
        for otro in miembros[1:]:
            if fundir(miembros[0], otro):
                unidos += 1
            else:
                frenados += 1
    print("caras unidas porque un humano las llamo igual: %d   (frenadas: %d)"
          % (unidos, frenados))

    grupos = defaultdict(list)
    for i in range(n):
        grupos[raiz(i)].append(i)

    # ── cortar las cadenas ──────────────────────────────────────────────
    #
    # Una sola cara ambigua puede pegar dos personas: A se parece a B, B a C, y
    # A con C no tienen nada que ver. Se mide cada grupo contra su propio centro
    # y se sacan los que no lo acompañan; esos quedan solos, que es la verdad
    # -no sabemos con quién van- y no ensucian a nadie.
    sueltos = 0
    limpios = {}
    for r, miembros in grupos.items():
        if len(miembros) < 2:
            limpios[r] = miembros
            continue
        centro = V[miembros].mean(axis=0)
        centro /= np.linalg.norm(centro)
        s = V[miembros] @ centro
        quedan = [m for m, x in zip(miembros, s) if x >= a.centro]
        fuera = [m for m, x in zip(miembros, s) if x < a.centro]
        sueltos += len(fuera)
        limpios[r] = quedan
        for m in fuera:
            limpios[-1 - m] = [m]
    grupos = {r: m for r, m in limpios.items() if m}

    # ── partir los gigantes ─────────────────────────────────────────────
    #
    # salud() avisaba 7 grupos de mas de 600 caras (grupo_gigante), el mayor de
    # 5.559: cadenas de muchas personas que ningun juego puede preguntar
    # (sql/grupos_gigantes.sql). Bajar el corte para todos rompe los grupos
    # sanos; se sube SOLO adentro del gigante, de a 0,05, hasta que cada pedazo
    # entra en el tope. Partir no puede violar un "no es" (solo separa) y lo
    # que queda afuera de todo pedazo vuelve suelto, como en el corte de
    # cadenas de arriba.
    TOPE = 600
    # UN GIGANTE COMPACTO NO SE PARTE (sql/grupos_compactos.sql). El de Persona C,
    # 1.846 caras de entregas de diplomas, tiene su 5% de caras mas flojas a
    # 0,866 del centro; las cadenas andan en 0,57-0,65. Si ese 5% llega a 0,80
    # es una persona y la base lo deja preguntar: partirlo solo haria pedazos
    # de la misma persona.
    COMPACTO = 0.80
    gigantes = []
    compactos = 0
    for r, m in grupos.items():
        if len(m) <= TOPE:
            continue
        c = V[m].mean(axis=0)
        c /= np.linalg.norm(c)
        if float(np.percentile(V[m] @ c, 5)) >= COMPACTO:
            compactos += 1
        else:
            gigantes.append(r)
    if compactos:
        print("grupos de mas de %d caras que no se parten por compactos: %d" % (TOPE, compactos))
    partidos = 0
    for r in gigantes:
        pendientes = [(grupos.pop(r), a.corte + 0.05)]
        while pendientes:
            miembros, corte = pendientes.pop()
            if len(miembros) <= TOPE or corte > 0.95:
                grupos[-10_000_000 - partidos] = miembros
                partidos += 1
                continue
            M = V[miembros]
            pad = list(range(len(miembros)))

            def r2(x):
                while pad[x] != x:
                    pad[x] = pad[pad[x]]
                    x = pad[x]
                return x
            for i0 in range(0, len(miembros), BLOQUE):
                S = M[i0:i0 + BLOQUE] @ M.T
                fi, co = np.nonzero(S >= corte)
                for x, y in zip(fi + i0, co):
                    if y > x:
                        rx, ry = r2(int(x)), r2(int(y))
                        if rx != ry:
                            pad[rx] = ry
            trozos = defaultdict(list)
            for k in range(len(miembros)):
                trozos[r2(k)].append(miembros[k])
            for t in trozos.values():
                pendientes.append((t, corte + 0.05))
    if gigantes:
        print("grupos de mas de %d caras partidos subiendo el corte adentro: %d -> %d pedazos"
              % (TOPE, len(gigantes), partidos))

    tam = Counter(len(m) for m in grupos.values())
    juntos = sum(len(m) for m in grupos.values() if len(m) > 1)
    print("\nGrupos: %d   caras agrupadas: %d   caras solas: %d"
          % (sum(1 for m in grupos.values() if len(m) > 1), juntos, tam[1]))
    print("sacadas del grupo por no acompañar al centro: %d" % sueltos)
    print("\ntamaño de los grupos:")
    for k in sorted(tam):
        if k > 1:
            print("   %3d caras  ->  %d grupos" % (k, tam[k]))
        if k > 20:
            break

    # ── contra lo que ya sabemos ────────────────────────────────────────
    #
    # La prueba de si el agrupamiento sirve: los grupos que contienen caras ya
    # identificadas, ¿tienen un solo nombre adentro o varios? Un grupo con dos
    # nombres confirmados esta mal armado, y se ve sin mirar una sola foto.
    #
    # CUIDADO CON COMO SE MIDE. La primera version preguntaba "que personas hay
    # identificadas en la foto de esta cara" y eso no es lo mismo: en una foto
    # de acto con cinco personas identificadas, cualquier cara de esa foto
    # arrastra los cinco nombres. Daba 170 grupos "mezclados" que no lo estaban.
    #
    # El nombre de UNA cara sale del cache: cada entrada es (foto, persona
    # propuesta). Si ese par ademas esta confirmado en photo_people, entonces
    # esa cara ES esa persona, y ahi si se puede juzgar.
    confirmadas = {(r["photo_id"], int(r["person_id"])) for r in
                   cli.select("photo_people", select="photo_id,person_id")}
    gente = {int(x["id"]): x["display_name"] for x in
             cli.select("people", select="id,display_name")}

    # El nombre de UNA cara, no el de su foto.
    #
    # CUIDADO CON COMO SE MIDE. La primera version preguntaba "que personas hay
    # identificadas en la foto de esta cara" y eso no es lo mismo: en una foto
    # de acto con cinco personas identificadas, cualquier cara de esa foto
    # arrastra los cinco nombres. Daba 170 grupos "mezclados" que no lo estaban.
    #
    # La cara del padron ya viene con nombre. La de evento trae una propuesta,
    # y solo cuenta si ese par (foto, persona) esta confirmado en photo_people.
    etiqueta, del_padron = {}, set()
    for m in range(n):
        q = int(quienes[m])
        if q < 0:
            continue
        if confirmado[m]:
            etiqueta[m] = q
            del_padron.add(m)
        elif (fotos[m], q) in confirmadas:
            etiqueta[m] = q
    print("")
    print("caras con nombre, una por cara: %d   (%d del padron, %d confirmadas a mano)"
          % (len(etiqueta), len(del_padron), len(etiqueta) - len(del_padron)))

    puros = mezclados = sin = 0
    gratis = ganadas = 0
    ejemplos = []
    for r, miembros in grupos.items():
        if len(miembros) < 2:
            continue
        nombres = {etiqueta[m] for m in miembros if m in etiqueta}
        if not nombres:
            sin += 1
            continue
        if len(nombres) == 1:
            puros += 1
            # El premio del paso 1: grupos que se nombran solos porque tocan
            # una cara del padron, sin que nadie haya contestado nada.
            if all(m in del_padron for m in miembros if m in etiqueta):
                gratis += 1
                ganadas += sum(1 for m in miembros if m not in etiqueta)
        else:
            mezclados += 1
            if len(ejemplos) < 6:
                ejemplos.append((len(miembros),
                                 [gente.get(x, "?") for x in list(nombres)[:4]]))
    print("GRUPOS QUE TOCAN CARAS CON NOMBRE")
    print("   con un solo nombre adentro:  %d" % puros)
    print("   con dos o mas nombres:       %d   <- mal armados" % mezclados)
    print("   sin ningun nombre todavia:   %d   <- los que esperan" % sin)
    if puros + mezclados:
        print("   -> %.1f%% de los grupos juzgables estan bien"
              % (100.0 * puros / (puros + mezclados)))
    print("")
    print("LO QUE APORTA UNIR LOS DOS MAPAS")
    print("   grupos nombrados solo por el padron: %d" % gratis)
    print("   caras que quedan nombradas sin preguntar nada: %d" % ganadas)
    for cuantas, quien in ejemplos:
        print("      grupo de %d caras: %s" % (cuantas, ", ".join(quien)))

    if not a.aplicar:
        print("\nEnsayo: no se escribió nada. Agregar --aplicar.")
        return

    # ── los grupos a la base ────────────────────────────────────────────
    #
    # Va sólo el número de grupo y el recuadro, no la huella. La app no
    # necesita comparar nada -esa cuenta ya está hecha acá- y las huellas son
    # 900 MB contra 10 MB de esto.
    #
    # Se manda tambien "centro": cuánto acompaña cada cara al centro de su
    # grupo. Es lo que ordena el captcha, que muestra las MENOS parecidas. Sin
    # ese número habría que mandar seis caras al azar, y seis al azar no
    # prueban nada: si el grupo tiene un centro ya lo sabíamos.
    #
    # SE REEMPLAZA TODO, Y LOS NOMBRES EN CURSO SE PIERDEN
    #
    # Los números de grupo salen del union-find y cambian de una corrida a la
    # otra: el grupo 814 de hoy no es el 814 de mañana. Así que face_group_names
    # -que guarda "a este grupo alguien le puso este nombre"- se vacía junto con
    # los grupos. No se pierde trabajo: lo que el captcha confirmó ya está en
    # photo_people, y lo que rechazó en photo_people_rejected, que son las dos
    # cosas que el agrupamiento lee como obligación y prohibición. Lo único que
    # se pierde es un nombre propuesto que nadie llegó a verificar.
    filas = []
    for r, miembros in grupos.items():
        if len(miembros) < 2:
            continue
        centro = V[miembros].mean(axis=0)
        centro /= np.linalg.norm(centro)
        sim = V[miembros] @ centro
        for m, x in zip(miembros, sim):
            filas.append({"photo_id": fotos[m],
                          "bx": round(float(cajas[m][0]), 4),
                          "by": round(float(cajas[m][1]), 4),
                          "bw": round(float(cajas[m][2]), 4),
                          "bh": round(float(cajas[m][3]), 4),
                          "grupo": int(r), "centro": round(float(x), 4)})

    # Una cara es su foto y su recuadro. Las del npz de eventos no traen
    # recuadro -se guardaban sin él- y todas quedarían en (0,0,0,0), que para
    # el índice único es la misma cara: la segunda pisaría a la primera y el
    # grupo se quedaría con una sola. Hasta que el mapa completo traiga los
    # recuadros, se manda una por foto y se avisa cuántas se dejaron afuera.
    vistas, limpias, chocadas = set(), [], 0
    for f in filas:
        k = (f["photo_id"], f["bx"], f["by"])
        if k in vistas:
            chocadas += 1
            continue
        vistas.add(k)
        limpias.append(f)
    if chocadas:
        print("\nCaras que comparten foto y recuadro (el npz de eventos no lo "
              "guarda): %d, se manda una" % chocadas)

    print("\nEscribiendo %d caras en %d grupos…"
          % (len(limpias), len({f["grupo"] for f in limpias})))
    # TRUNCATE y no DELETE, y TODOS los nombres, tambien los de grupos con
    # numero negativo (los pedazos de gigantes): sql/vaciar_face_groups.sql.
    cli.rpc("vaciar_face_groups")
    cli.upsert("face_groups", limpias, on_conflict="photo_id,bx,by")

    # De que sede y de que seccion es cada grupo, para los filtros de la
    # pantalla. Va aca y no como un paso aparte del nocturno porque sin esto
    # los grupos quedan escritos pero invisibles: la cola lee face_group_meta,
    # asi que un refresco olvidado deja la pantalla diciendo que no hay nada.
    # Si esto falla, los grupos quedan escritos y la pantalla no los ve: lee
    # face_group_meta para elegir el proximo. El 12/9 fallo por un "delete
    # sin WHERE" y dejo el juego caido con el costado diciendo 2.955.
    # Se avisa fuerte en vez de morir con un traceback al final de una corrida
    # de veinte minutos.
    try:
        n = cli.rpc("refrescar_face_group_meta")
        print("Sede y seccion calculadas para %s grupos." % n)
    except Exception as e:
        print("")
        print("ERROR: los grupos se escribieron pero refrescar_face_group_meta")
        print("fallo: %s" % str(e)[:160])
        print("La pantalla de Grupos va a decir que no hay nada hasta que corra.")
        raise
    print("Listo. La pantalla de Etiquetar ya tiene de dónde sacar el próximo grupo.")


if __name__ == "__main__":
    main()
