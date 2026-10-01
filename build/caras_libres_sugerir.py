# -*- coding: utf-8 -*-
"""Las caras libres, otra vez contra lo que ya se sabe de cada persona.

    python caras_libres_sugerir.py                  # ensayo: cuenta y no escribe
    python caras_libres_sugerir.py --ensayo         # lo mismo, dicho explicito
    python caras_libres_sugerir.py --aplicar        # escribe sugerencias (origen 'libres')
    python caras_libres_sugerir.py --revertir       # ensayo del borrado de lo suyo
    python caras_libres_sugerir.py --revertir --aplicar

POR QUE

Una cara "libre" es una cara del archivo que no esta en ningun grupo (39% del
archivo) o que esta en un grupo que nadie nombro. Esas caras recibieron su
sugerencia UNA sola vez, cuando faces_eventos.py proceso su anio, con las
referencias que habia ese dia. Despues las referencias crecieron -cada "Si" del
juego, cada grupo nombrado- y nadie las volvio a comparar. Medido el 29/9/2026
sobre una camada de P6: a una alumna le faltaban ~43 fotos asi, y a su camada
~1.195.

Este paso las vuelve a comparar en cada vuelta del pulso contra el PROMEDIO de
las caras que una persona confirmo de cada ficha, y deja SUGERENCIAS, no
identificaciones: la decision sigue siendo de quien contesta, o de una regla
automatica medida aparte (caras_si_es.py).

QUE ES "CONFIRMADA POR UNA PERSONA"

  - un "Si" humano vigente sobre (foto, persona) con su recuadro, o un retrato
    (grupos_parecidos.verdad_humana: la misma verdad que usa todo el motor)
  - las caras de un grupo que una persona nombro en Grupos (photo_people
    'grupo' con su recuadro), DEPURADAS: si la ficha tiene "Si" de a una, se
    quedan solo las que se parecen al promedio de esas; si no, al promedio de
    las de grupo. Un grupo contestado mirando seis caras puede traer alguna
    ajena (hasta 7%, grupos_parecidos.NO_SE_PARECE).

Los carnets NO entran: esto es para lo que confirmo una persona.

EL CORTE Y EL MARGEN SE MIDIERON (29/9/2026, 1.963 fichas con referencia humana)

Cara confirmada de a una contra el promedio de cada ficha, sacando de la suya
TODAS las caras de la misma carpeta (si no, la cara se compara con sus propias
tomas y el error da de menos). Las que hoy estan sueltas, 1.706 caras:

    corte  margen   asignadas  mal   %mal   recupera
    0,50   0,10       1.216     7   0,58%    70,9%
    0,55   0,10         782     6   0,77%    45,5%
    0,60   0,10         251     6   2,39%    14,4%

Y la otra mitad del error, la que no sale de ahi: una cara de alguien que NO
tiene referencia se parece igual a alguien que si. Simulado sacando a su
persona de la galeria (18.779 caras, 1.726 sueltas):

    corte  margen   todas    sueltas
    0,50   0,10     2,48%     2,09%
    0,55   0,10     1,58%     1,22%
    0,60   0,10     1,20%     0,64%

0,55 con margen 0,10 queda debajo de la tolerancia de Diego (1,6%) en las dos
mediciones y recupera el triple que 0,60.

Remedido el 30/9/2026 con la referencia de ESTE paso y MIN_REFS = 3
(build/_medir_caras_libres.py): con referencia 15 mal de 15.909 (0,09%),
sueltas 3 de 749 (0,40%); sin referencia, 0,81% de las caras (sueltas 1,11%).

Y una señal sobre las sugerencias REALES, que no es error: las que caen en un
año a mas de 2 del rango de años de las fotos identificadas de ese alumno.
Da 1,54% a 0,55 y no sube al bajar el corte (1,35% a 0,65): si el corte dejara
pasar caras ajenas en cantidad, subiria. OJO: first_seen/last_seen de people no
sirven para esto, salen del anio del carnet. El margen es contra la SEGUNDA
persona mas parecida: evita proponer cuando hay dos candidatos casi iguales
(hermanos, fichas duplicadas).

LA MARCA: origen = 'libres'

Desde el 1/10/2026 face_suggestions tiene columna origen (sql/
face_suggestions_origen.sql): todo lo de este paso se escribe con
origen = 'libres', y --revertir borra exactamente eso (lo que nadie contesto
todavia; lo que ya esta en photo_labels o photo_people es de quien contesto).
Hasta el 30/9 la marca era rank = 0; base paso esas filas a origen 'libres'.

rank ya no marca nada: es solo orden. Se sigue escribiendo 0 (RANK, abajo)
porque approve_face_suggestions aprueba solo con rank = 1 y score alto: si
este paso escribiera 1, sus sugerencias entrarian a esa aprobacion sin la
medicion de caras_si_es.py.

FOTOS QUE YA NO ESTAN

Las huellas (.npz) guardan caras de fotos que despues se borraron o purgaron
de photos. La clave foranea face_suggestions -> photos rechaza esas filas
(23503), y el 1/10/2026 tiraba el paso entero. Antes de escribir se leen, en
lotes, cuales photo_id existen, y las demas se descartan y se cuentan.

QUE RESPETA

  - rechazos (photo_people_rejected) y "No" humanos (photo_labels)
  - caras_no_parecen y grupos_no_son de esa persona sobre esa cara
  - cara_de_otro: una cara que ya es de otra persona no se le propone a nadie
  - lo confirmado: si la persona ya esta en la foto, no se propone
  - lo sugerido: si la cara ya tiene una sugerencia viva, o la persona ya
    tiene una en la foto (la clave es foto+persona), no se pisa
  - fichas mezcladas: si mas de MEZCLA de sus referencias no se parece al
    resto, el promedio no es de nadie y no se propone a esa ficha

Corre despues del resolver: reagrupar borra los nombres automaticos, y "grupo
sin nombre" solo significa algo cuando el resolver ya volvio a nombrar.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict

# faces lee el motor al importarse: el default es sface y leeria otras huellas
if os.environ.get("FACES_MOTOR", "").lower() != "arcface":
    sys.exit("Falta FACES_MOTOR. En PowerShell: $env:FACES_MOTOR='arcface'")

import numpy as np  # noqa: E402

import sb  # noqa: E402
from caras_agrupar_archivo import cargar  # noqa: E402
from grupos_parecidos import NO_SE_PARECE, normal, verdad_humana  # noqa: E402

# Medidos el 29/9/2026, ver el docstring: 0,77% mal entre personas con
# referencia y 1,22-1,58% para caras de alguien sin referencia.
CORTE = 0.55
MARGEN = 0.10

# Una ficha con mas de este tanto de referencias lejos del resto (debajo de
# NO_SE_PARECE contra el promedio de las demas) esta mezclada: su promedio no
# es de nadie. Es el freno de caras_sueltas.py. Con menos de MIN_MEZCLA
# referencias no se puede juzgar y la ficha se usa tal cual.
MEZCLA = 0.25
MIN_MEZCLA = 4

# A una ficha con menos caras de referencia que esto no se le propone nada: su
# promedio es flojo y atrae caras ajenas. Sigue en la galeria, compitiendo
# para el margen. Medido el 29/9/2026 (build/_medir_caras_libres.py, corte
# 0,55 / 0,10): cuando la ficha que gana tiene 1 o 2 referencias, 11 de 34
# propuestas estan mal (32%); con 3 a 5, 0 de 209; y de los 320 errores con
# gente sin referencia, 183 caen en fichas de 1 o 2.
MIN_REFS = 3

# La marca de lo que escribe este paso (ver LA MARCA arriba), y el rank que
# lleva: solo orden, 0 para quedar fuera de approve_face_suggestions (rank = 1).
ORIGEN = "libres"
RANK = 0

# Dos recuadros son la misma cara si la esquina difiere menos que esto: es la
# tolerancia de la base (grupos_resolver, sugerencia_respeta_no_es).
MISMA_CAJA = 0.005

BLOQUE = 8192
LOTE = 500
# ids por pedido al preguntar que fotos existen (id de Drive ~33 caracteres:
# 150 entran comodos en la URL)
LOTE_IDS = 150
FUENTES_CARNET = ("mugshot_filename", "filename", "filename_face")


def ahora():
    return time.strftime("%H:%M:%S")


def di(txt):
    print("%s  %s" % (ahora(), txt), flush=True)


class Cajas:
    """Recuadros por foto, para preguntar "hay algo sobre esta cara"."""

    def __init__(self):
        self.d = defaultdict(list)

    def poner(self, foto, bx, by, dato=None):
        if bx is None or by is None:
            return
        self.d[foto].append((float(bx), float(by), dato))

    def hay(self, foto, bx, by, dato=None):
        for x, y, d in self.d.get(foto, ()):
            if abs(x - bx) < MISMA_CAJA and abs(y - by) < MISMA_CAJA and \
                    (dato is None or d == dato):
                return True
        return False

    def datos(self, foto, bx, by):
        return {d for x, y, d in self.d.get(foto, ())
                if abs(x - bx) < MISMA_CAJA and abs(y - by) < MISMA_CAJA}


def leer(cli):
    """Todo lo que hace falta, leido una vez."""
    t0 = time.time()
    di("Leyendo huellas (el espacio del agrupamiento)…")
    fotos, cajas, vecs, desc, quienes, del_padron = cargar()
    vecs = normal(vecs.astype(np.float32))
    di("  %s" % desc)
    clave = {}
    for i, f in enumerate(fotos):
        clave.setdefault((f, round(float(cajas[i][0]), 4), round(float(cajas[i][1]), 4)), i)

    di("Leyendo grupos y nombres…")
    grupo_de = {}
    for r in cli.select("face_groups", select="grupo,photo_id,bx,by"):
        i = clave.get((r["photo_id"], round(float(r["bx"]), 4), round(float(r["by"]), 4)))
        if i is not None:
            grupo_de[i] = int(r["grupo"])
    con_nombre = {int(r["grupo"]) for r in cli.select("face_group_names", select="grupo")}

    di("Leyendo lo que confirmo una persona…")
    cara, parejas, retratos = verdad_humana(cli, clave)
    di("  %d parejas con 'Si' humano -> %d caras ubicadas (%d de retrato)"
       % (parejas, len(cara), retratos))

    pp = cli.select("photo_people", select="photo_id,person_id,source")
    presentes = defaultdict(set)
    por_grupo = set()
    for r in pp:
        presentes[r["photo_id"]].add(int(r["person_id"]))
        if r.get("source") == "grupo":
            por_grupo.add((r["photo_id"], int(r["person_id"])))

    di("Leyendo sugerencias, rechazos y 'no es'…")
    sug = cli.select("face_suggestions", select="photo_id,person_id,bx,by,bw,rank")
    rech = {(r["photo_id"], int(r["person_id"])) for r in
            cli.select("photo_people_rejected", select="photo_id,person_id")}
    etiquetas = cli.select("photo_labels", select="photo_id,person_id,answer")
    no_humano = {(r["photo_id"], int(r["person_id"])) for r in etiquetas
                 if r.get("answer") != "yes"}
    si_humano = defaultdict(set)
    for r in etiquetas:
        if r.get("answer") == "yes":
            si_humano[r["photo_id"]].add(int(r["person_id"]))
    ya_sug = set()               # (foto, persona) con sugerencia: la clave de la tabla
    viva = Cajas()               # caras con una sugerencia sin rechazar
    de_otro = Cajas()            # caras identificadas: dato = persona
    de_grupo = defaultdict(list)
    for s in sug:
        k = (s["photo_id"], int(s["person_id"]))
        ya_sug.add(k)
        if not (s.get("bw") or 0) > 0:
            continue
        if k not in rech:
            viva.poner(k[0], s["bx"], s["by"], k[1])
        if k[1] in presentes.get(k[0], ()):
            de_otro.poner(k[0], s["bx"], s["by"], k[1])
        if k in por_grupo:
            i = clave.get((k[0], round(float(s["bx"]), 4), round(float(s["by"]), 4)))
            if i is not None and i not in cara:
                de_grupo[k[1]].append(i)

    no_parece = Cajas()
    for r in cli.select("caras_no_parecen", select="photo_id,bx,by,person_id",
                        order="photo_id.asc,bx.asc,by.asc,person_id.asc"):
        no_parece.poner(r["photo_id"], r["bx"], r["by"], int(r["person_id"]))
    for r in cli.select("grupos_no_son", select="photo_id,bx,by,person_id"):
        no_parece.poner(r["photo_id"], r["bx"], r["by"], int(r["person_id"]))

    gente = {int(r["id"]): r for r in cli.select("people", select="id,kind,year_code,first_seen,last_seen")}

    # cara_de_otro, primera mitad: en una carpeta de carnets, una foto que no
    # es grupal y ya es de otro por su nombre de archivo o por un "Si" humano
    carnet = {r["id"] for r in cli.select("folders", select="id,is_mugshot")
              if r.get("is_mugshot")}
    duenos_carnet = defaultdict(set)
    fotos_arch = {fotos[i] for i in range(len(fotos)) if not del_padron[i]}
    ids = sorted(carnet)
    for a in range(0, len(ids), 100):
        for r in cli.select("photos", select="id,folder_id,is_group",
                            folder_id="in.(%s)" % ",".join(ids[a:a + 100])):
            if r.get("is_group") or r["id"] not in fotos_arch:
                continue
            duenos_carnet[r["id"]] = set()
    for r in pp:
        f = r["photo_id"]
        if f in duenos_carnet and (r.get("source") in FUENTES_CARNET or
                                   int(r["person_id"]) in si_humano.get(f, ())):
            duenos_carnet[f].add(int(r["person_id"]))

    di("  leido en %.0f s" % (time.time() - t0))
    return dict(fotos=fotos, cajas=cajas, vecs=vecs, del_padron=del_padron,
                clave=clave, grupo_de=grupo_de, con_nombre=con_nombre, cara=cara,
                presentes=presentes, de_grupo=de_grupo, ya_sug=ya_sug, viva=viva,
                de_otro=de_otro, rech=rech, no_humano=no_humano,
                no_parece=no_parece, gente=gente, duenos_carnet=duenos_carnet,
                sug=sug)


def referencias(c, excluir=()):
    """persona -> indices de sus caras confirmadas por una persona.

    excluir: personas a dejar afuera (para medir)."""
    vecs, cara = c["vecs"], c["cara"]
    una = defaultdict(list)
    for i, p in cara.items():
        una[p].append(i)
    refs = {}
    for p in set(una) | set(c["de_grupo"]):
        if p in excluir or (c["gente"].get(p) or {}).get("kind") in (None, "noise"):
            continue
        ix = list(una.get(p, []))
        g = sorted(set(c["de_grupo"].get(p, [])) - set(ix))
        if g:
            base = ix if ix else g
            centro = normal(vecs[base].mean(axis=0, keepdims=True))[0]
            s = vecs[g] @ centro
            ix += [i for i, x in zip(g, s) if x >= NO_SE_PARECE]
        if ix:
            refs[p] = sorted(set(ix))
    return refs


def mezcladas(c, refs):
    """Fichas cuyo promedio no es de nadie (ver MEZCLA)."""
    vecs = c["vecs"]
    out = {}
    for p, ix in refs.items():
        if len(ix) < MIN_MEZCLA:
            continue
        M = vecs[ix]
        suma = M.sum(axis=0)
        otros = normal(suma[None, :] - M)
        s = (M * otros).sum(axis=1)
        lejos = int((s < NO_SE_PARECE).sum())
        if lejos > MEZCLA * len(ix):
            out[p] = (lejos, len(ix))
    return out


def centros(c, refs):
    personas = sorted(refs)
    C = normal(np.stack([c["vecs"][refs[p]].mean(axis=0) for p in personas]))
    return personas, C


def libres(c):
    """Indices de las caras del archivo sueltas o en grupos sin nombre."""
    cajas, del_padron, grupo_de, con_nombre = (c["cajas"], c["del_padron"],
                                               c["grupo_de"], c["con_nombre"])
    ref = set(c["cara"])
    out, sueltas = [], 0
    for i in np.flatnonzero(~del_padron & (cajas[:, 2] > 0)):
        i = int(i)
        if i in ref:
            continue
        g = grupo_de.get(i)
        if g is None:
            sueltas += 1
            out.append(i)
        elif g not in con_nombre:
            out.append(i)
    return out, sueltas


def mejor_y_segundo(c, idx, C):
    """Para cada cara: indice de la persona mas parecida, su parecido y el de
    la segunda. Un solo producto grande, en bloques."""
    V = c["vecs"]
    n = len(idx)
    arg = np.empty(n, np.int64)
    b1 = np.empty(n, np.float32)
    b2 = np.empty(n, np.float32)
    idx = np.asarray(idx)
    for a in range(0, n, BLOQUE):
        S = V[idx[a:a + BLOQUE]] @ C.T
        o = np.argpartition(-S, 1, axis=1)[:, :2]
        r = np.arange(len(S))
        s0, s1 = S[r, o[:, 0]], S[r, o[:, 1]]
        primero = s0 >= s1
        arg[a:a + BLOQUE] = np.where(primero, o[:, 0], o[:, 1])
        b1[a:a + BLOQUE] = np.maximum(s0, s1)
        b2[a:a + BLOQUE] = np.minimum(s0, s1)
    return arg, b1, b2


def proponer(c, corte=CORTE, margen=MARGEN, min_refs=MIN_REFS, log=True):
    """Las sugerencias que escribiria, y por que no escribe las demas."""
    refs = referencias(c)
    malas = mezcladas(c, refs)
    for p in malas:
        refs.pop(p, None)
    personas, C = centros(c, refs)
    idx, sueltas = libres(c)
    if log:
        di("Referencias humanas: %d fichas (%d mezcladas, afuera)" % (len(refs), len(malas)))
        di("Caras libres: %d  (%d sueltas, %d en grupos sin nombre)"
           % (len(idx), sueltas, len(idx) - sueltas))
    arg, b1, b2 = mejor_y_segundo(c, idx, C)

    fotos, cajas = c["fotos"], c["cajas"]
    motivo = Counter()
    mejor = {}                    # (foto, persona) -> fila; una por foto y persona
    for k, i in enumerate(idx):
        if b1[k] < corte:
            motivo["debajo del corte"] += 1
            continue
        if b1[k] - b2[k] < margen:
            motivo["sin margen sobre la segunda"] += 1
            continue
        p = personas[int(arg[k])]
        if len(refs[p]) < min_refs:
            motivo["la ficha que gana tiene pocas referencias"] += 1
            continue
        f = fotos[i]
        bx, by = round(float(cajas[i][0]), 4), round(float(cajas[i][1]), 4)
        if c["de_otro"].hay(f, bx, by):
            motivo["la cara ya es de alguien"] += 1
            continue
        d = c["duenos_carnet"].get(f)
        if d and (d - {p}):
            motivo["carnet de otra persona"] += 1
            continue
        if p in c["presentes"].get(f, ()):
            motivo["la persona ya esta en la foto"] += 1
            continue
        if (f, p) in c["rech"] or (f, p) in c["no_humano"]:
            motivo["rechazada o 'No' humano"] += 1
            continue
        if c["no_parece"].hay(f, bx, by, p):
            motivo["caras_no_parecen / grupos_no_son"] += 1
            continue
        if (f, p) in c["ya_sug"]:
            motivo["ya tiene sugerencia en la foto"] += 1
            continue
        if c["viva"].hay(f, bx, by):
            motivo["la cara ya tiene otra sugerencia"] += 1
            continue
        fila = {"photo_id": f, "person_id": int(p), "score": round(float(b1[k]), 4),
                "rank": RANK, "origen": ORIGEN, "bx": bx, "by": by,
                "bw": round(float(cajas[i][2]), 4), "bh": round(float(cajas[i][3]), 4)}
        viejo = mejor.get((f, p))
        if viejo is None or fila["score"] > viejo["score"]:
            if viejo is not None:
                motivo["otra cara de la foto se le parece mas"] += 1
            mejor[(f, p)] = fila
        else:
            motivo["otra cara de la foto se le parece mas"] += 1
    filas = sorted(mejor.values(), key=lambda x: -x["score"])
    return filas, motivo, dict(refs=refs, malas=malas, libres=len(idx), sueltas=sueltas)


def fotos_que_existen(cli, ids):
    """De estos photo_id, los que estan en photos. Una lectura en lotes, no
    fila por fila. Si un lote falla, se corta: sin saber que existe no se
    escribe (el pulso lo ve como error, no se tapa)."""
    ids = sorted(set(ids))
    hay = set()
    for a in range(0, len(ids), LOTE_IDS):
        trozo = ids[a:a + LOTE_IDS]
        for r in cli.select("photos", select="id", id="in.(%s)" % ",".join(trozo)):
            hay.add(r["id"])
        try:
            if (a // LOTE_IDS) % 50 == 0:
                di("  fotos: %d de %d consultadas" % (min(a + LOTE_IDS, len(ids)), len(ids)))
        except Exception:
            pass
    return hay


def es_fk_fotos(err):
    """La fila apunta a una foto que ya no esta (se borro entre la lectura y
    la escritura): no es una fila rota, es una descartada."""
    return "23503" in err and "photos" in err


def insertar(cli, filas):
    """INSERT que ignora lo que ya existe: nunca pisa una sugerencia ajena.

    sb.upsert hace merge, y una sugerencia que aparecio entre la lectura y la
    escritura (otro paso, una respuesta) perderia su recuadro. Una fila rota no
    tira el lote: se reintenta de a una y se cuentan las que fallan. Las que
    fallan porque la foto ya no esta en photos se cuentan aparte (sin_foto)."""
    ok, rotas, sin_foto = 0, [], 0
    url = "%s/rest/v1/face_suggestions?on_conflict=photo_id,person_id" % cli.url

    def mandar(trozo):
        req = urllib.request.Request(
            url, data=json.dumps(trozo, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=cli._headers({"Prefer": "resolution=ignore-duplicates,return=minimal"}))
        cli._pedir(req)

    for a in range(0, len(filas), LOTE):
        trozo = filas[a:a + LOTE]
        try:
            mandar(trozo)
            ok += len(trozo)
        except Exception as e:
            di("  lote %d fallo (%s): de a una" % (a // LOTE, str(e)[:120]))
            for f in trozo:
                try:
                    mandar([f])
                    ok += 1
                except Exception as e2:
                    if es_fk_fotos(str(e2)):
                        sin_foto += 1
                    else:
                        rotas.append((f, str(e2)[:200]))
        try:
            if (a // LOTE) % 20 == 0:
                di("  %d de %d" % (min(a + LOTE, len(filas)), len(filas)))
        except Exception:
            pass
    return ok, rotas, sin_foto


def revertir(cli, aplicar):
    """Borra lo que puso este paso y nadie contesto. Respaldo antes."""
    filas = cli.select("face_suggestions",
                       select="photo_id,person_id,score,rank,origen,bx,by,bw,bh",
                       origen="eq.%s" % ORIGEN)
    contestadas = {(r["photo_id"], int(r["person_id"])) for r in
                   cli.select("photo_labels", select="photo_id,person_id")}
    contestadas |= {(r["photo_id"], int(r["person_id"])) for r in
                    cli.select("photo_people", select="photo_id,person_id")}
    borrar = [r for r in filas if (r["photo_id"], int(r["person_id"])) not in contestadas]
    di("Con origen '%s': %d. Contestadas o ya en la foto (quedan): %d. "
       "Sin contestar (se borrarian): %d"
       % (ORIGEN, len(filas), len(filas) - len(borrar), len(borrar)))
    if not aplicar:
        di("Ensayo: no se borro nada. Agregar --aplicar.")
        return 0
    nom = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_caras_libres_revertir_%s.txt" % time.strftime("%Y%m%d_%H%M"))
    with open(nom, "w", encoding="utf-8") as fh:
        for r in borrar:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    di("Respaldo: %s" % os.path.basename(nom))
    por_persona = defaultdict(list)
    for r in borrar:
        por_persona[int(r["person_id"])].append(r["photo_id"])
    fallos = 0
    for p, ids in por_persona.items():
        for a in range(0, len(ids), 100):
            try:
                cli.delete("face_suggestions", person_id="eq.%d" % p, origen="eq.%s" % ORIGEN,
                           photo_id="in.(%s)" % ",".join(ids[a:a + 100]))
            except Exception as e:
                fallos += 1
                di("  no se pudo borrar un lote de %d: %s" % (p, str(e)[:120]))
    di("Borradas las de %d personas; %d lotes fallaron" % (len(por_persona), fallos))
    return 1 if fallos else 0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--ensayo", action="store_true",
                    help="no escribe (es lo que hace sin --aplicar)")
    ap.add_argument("--revertir", action="store_true")
    ap.add_argument("--persona", type=int, action="append", default=[],
                    help="en el ensayo, detallar esta ficha (se puede repetir)")
    ap.add_argument("--camada", action="append", default=[],
                    help="en el ensayo, detallar este year_code")
    a = ap.parse_args()
    if a.ensayo and a.aplicar:
        sys.exit("--ensayo y --aplicar no van juntos")
    cli = sb.SB()

    if a.revertir:
        sys.exit(revertir(cli, a.aplicar))

    t0 = time.time()
    c = leer(cli)
    filas, motivo, info = proponer(c)

    # las huellas traen fotos que ya no estan en photos: la clave foranea las
    # rechaza. Se filtran antes de escribir (tambien en el ensayo, para contar)
    di("Mirando que fotos siguen en photos...")
    hay = fotos_que_existen(cli, [f["photo_id"] for f in filas])
    n_antes = len(filas)
    filas = [f for f in filas if f["photo_id"] in hay]
    descartadas = n_antes - len(filas)
    if descartadas:
        motivo["la foto ya no esta en photos"] += descartadas

    di("")
    di("SUGERENCIAS NUEVAS (corte %.2f, margen %.2f): %d  a %d personas"
       % (CORTE, MARGEN, len(filas), len({f["person_id"] for f in filas})))
    for m, n in motivo.most_common():
        di("   no se propone, %-40s %d" % (m + ":", n))
    if info["malas"]:
        di("   fichas mezcladas dejadas afuera: %d (las 5 mas grandes: %s)"
           % (len(info["malas"]), ", ".join(
               "%d (%d de %d lejos)" % (p, x[0], x[1]) for p, x in
               sorted(info["malas"].items(), key=lambda y: -y[1][1])[:5])))
    tramos = Counter(min(9, int(f["score"] * 10)) for f in filas)
    di("   por parecido: " + "  ".join("%.1f-%.1f: %d" % (t / 10, t / 10 + 0.1, tramos[t])
                                        for t in sorted(tramos)))
    por = Counter(f["person_id"] for f in filas)
    di("   personas con mas: " + ", ".join("%d: %d" % x for x in por.most_common(10)))
    for p in a.persona:
        di("   ficha %d: %d sugerencias nuevas (tiene %d caras de referencia humana)"
           % (p, por.get(p, 0), len(info["refs"].get(p, []))))
    for cam in a.camada:
        ids = {p for p, r in c["gente"].items() if r.get("year_code") == cam}
        n = sum(por.get(p, 0) for p in ids)
        di("   camada %s: %d fichas, %d con referencia humana, %d sugerencias nuevas a %d de ellas"
           % (cam, len(ids), len(ids & set(info["refs"])), n,
              len([p for p in ids if por.get(p)])))

    if not a.aplicar:
        di("Ensayo: no se escribio nada. Agregar --aplicar. (%.0f s)" % (time.time() - t0))
        return
    ok, rotas, sin_foto = insertar(cli, filas)
    di("Escritas %d sugerencias con origen '%s'; %d descartadas antes por foto que ya "
       "no esta, %d mas al escribir; %d filas rotas"
       % (ok, ORIGEN, descartadas, sin_foto, len(rotas)))
    for f, e in rotas[:10]:
        di("   rota %s / %d: %s" % (f["photo_id"], f["person_id"], e))
    # un error no se tapa: si algo no entro, el pulso lo ve como fallo
    if rotas:
        sys.exit(1)
    di("Listo. (%.0f s)" % (time.time() - t0))


if __name__ == "__main__":
    main()
