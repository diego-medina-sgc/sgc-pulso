# -*- coding: utf-8 -*-
"""
Reconoce personas en las fotos de evento. Este es el objetivo del proyecto:
que buscar un nombre traiga las fotos de actos, viajes y partidos donde esa
persona aparece. Los mugshots, el juego y los anuarios son la fuente de caras
conocidas que alimenta esto.

Dos diferencias con los retratos, que cambian el diseño:

  VARIAS CARAS  Un retrato tiene una; una foto de acto tiene cinco o diez. Se
                guarda el recuadro de cada una, porque sin saber cual es cual
                la sugerencia no se puede ni mostrar ni verificar.

  EL AÑO ACOTA  Alguien solo puede estar en una foto de un año en que estuvo
                en el colegio. Es un filtro que sale gratis y descarta la
                mayoria de las confusiones antes de mirar la cara.

    python faces_eventos.py --anio 2021 --ensayo    # mide, no escribe
    python faces_eventos.py --anio 2021             # escribe
"""
import argparse
import os
import sys
import threading
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import faces
import sb
from faces_sugerir import todas_las_referencias
from index_drive import Drive, SA_PATH

# Cuantas fotos se procesan a la vez.
#
# Cada hilo baja su foto y le mide las caras de punta a punta, asi que el
# numero util depende de cuanto se espera a la red. Con las librerias en un
# solo hilo adentro (ver faces.CV_HILOS) sobra CPU mientras se baja, y conviene
# mas de un hilo por nucleo.
#
# Medido sobre 600 fotos de 2009 en una notebook de 4 nucleos, con un hilo
# adentro:  6 hilos 171 s,  10 hilos 152 s,  14 hilos 155 s. Con la
# configuracion vieja -4 hilos adentro y 6 afuera- eran 0,327 s por foto; con
# esta son 0,242 s: 26% menos, y con la misma salida exacta (297 caras, 37
# fotos con alguien, 64 personas en la misma muestra de 200).
HILOS = int(os.environ.get("FACES_HILOS", "10"))
# En una foto de grupo las caras chicas del fondo no tienen pixeles para
# reconocerlas; medido antes, con 80 px hay ~1,4 caras utiles por foto.
LADO_MIN = 80
# Mas exigente que en los retratos: una etiqueta equivocada en la foto de un
# acto la ve la familia, y molesta mas que una que falta.
UMBRAL = faces.umbral("eventos")

# Abajo de esto solo se propone a quien el sistema conoce bien.
#
# Lo destapo Diego con una pregunta que no tenia respuesta: "¿Es Maria
# Soloeta?" sobre la cara de un señor pelado con bigote. Y su diagnostico fue
# el correcto: Persona G es una mujer, con pelo, con una foto clara y
# nombrada, y con 34 caras de referencia que son todas ella. No es que el
# sistema la confundio.
#
# Lo que pasa es que el sistema NO TIENE UN "no se" PARA SI MISMO. Compara cada
# cara contra las 46.000 referencias de las 3.000 personas del padron, se queda
# con la mas parecida y la propone si pasa el umbral. Con 46.000 vectores,
# cualquier cara borrosa esta a 0,65-0,70 de alguien. Esa cara de 108 px en una
# fiesta de noche no se parecia a nadie, y Maria fue la que quedo menos lejos.
#
# Medido sobre las 2.541 respuestas que ya dieron Diego y John:
#
#     0.60 a 0.65    86 casos    87,2% de acierto
#     0.65 a 0.70   107          88,8%
#     0.70 a 0.75   155          97,4%
#     0.75 a 0.90   671          96-98%
#     0.97 a 1.00  1311          99,5%
#
# Los errores viven abajo de 0,70. Y ahi dentro, lo que separa el acierto del
# error es cuantas caras de referencia tiene la persona propuesta:
#
#     0 a 5 caras    77 casos    70,1%   (23 de los 61 errores medidos)
#     6 o mas       113 casos   100,0%
#
# Tiene sentido: con dos caras guardadas el sistema no sabe como es esa persona,
# sabe como salio en dos fotos. Asi que abajo de 0,70 se pregunta solo por
# quien tiene seis caras o mas. Saca 21.265 preguntas de las 86.128 pendientes.
#
# Esto no arregla el fondo -Maria tiene 34 caras y su caso igual pasa el filtro-
# y el fondo es lo que Diego describio: agrupar las caras entre si y ponerle
# nombre al grupo, en vez de ir cara por cara contra el padron. Con un grupo,
# una cara mala no entra porque no se parece a TODAS las de la persona, y puede
# quedar sin nombre, que es la respuesta verdadera.
CONOCIDA = 0.70
MIN_CARAS = 6

# Donde queda la huella de cada cara que se propuso.
#
# Hasta ahora se calculaba, se comparaba y se tiraba: quedaban el nombre y el
# recuadro. Y sin la huella hay tres cosas que no se pueden hacer, y las tres
# las pidio Diego:
#
#   1. que el "No es" se aprenda. Hoy el rechazo se guarda como (esa foto, esa
#      persona) y nada mas: "si le digo que no es Persona G, que no me
#      vuelva a preguntar, porque ese ya no es Persona G". Para eso hay que
#      poder decir que la cara de la foto 823 y la de la 998 son el mismo
#      senor, y eso se ve comparando las huellas.
#
#   2. el margen. Saber si la segunda persona mas parecida quedo a 0,67 cuando
#      la primera quedo a 0,676, que es la senal de que no es ninguna.
#
#   3. agrupar antes de nombrar, que es como lo hace Apple Photos.
#
# Va a un .npz local y no a la base a proposito: son 512 numeros por cara y
# unas 120.000 caras, o sea cientos de megas que la app no necesita. El trabajo
# que los usa corre aca.
#
# Se guardan solo las caras que se propusieron, no todas las detectadas. Las que
# no se propusieron no estan en ninguna cola, asi que no hay nada que aprender
# sobre ellas todavia; guardar el millon entero son gigas y es para cuando se
# haga el agrupamiento de verdad.
CACHE_CARAS = faces.npz("_caras_evento.npz")

# Y el otro cache: TODAS las caras detectadas, propuestas o no.
#
# Es lo que falta para que el sistema aprenda como lo describio Diego:
# "nosotros identificamos una, dos, tres veces; el sistema automaticamente
# tiene que aprender y buscar la misma cara de esa persona en toda la
# biblioteca, todo lo que sea similar 0.80 agrupa con esa etiqueta de nombre,
# las dudas van a confirmar y las que sabe que no son las evita".
#
# Todo eso se puede hacer con lo que ya hay, menos el "buscar en toda la
# biblioteca". Hoy una cara que no llego al umbral contra NINGUNA referencia se
# descarta y no queda registro de que existio. Asi que cuando alguien confirma a
# Gabriel por tercera vez, lo unico que se puede reconsiderar son las caras que
# alguna vez se parecieron a alguien; las suyas que dieron 0,55 contra las
# referencias viejas ya no estan en ninguna parte y habria que volver a bajar
# 362.000 fotos para encontrarlas.
#
# Guardarlas no cuesta reconocimiento: la huella de cada cara YA se calcula
# -es la parte cara- y despues se tira. Cuesta disco: unas 450.000 caras a 512
# numeros de 16 bits son ~450 MB.
#
# Se guarda el recuadro junto con la huella porque sin el no se puede armar la
# pregunta: confirmar dibuja el cuadrado sobre la foto.
CACHE_TODAS = faces.npz("_caras_todas.npz")


def guardar_todas(nuevas):
    """Suma al cache de todas las caras. Igual que guardar_caras, con recuadro.

    La clave es (foto, recuadro) y no (foto, persona): aca una cara no tiene
    nombre todavia, y de eso se trata.
    """
    if not nuevas:
        return 0
    import os as _os
    fotos = np.array([x[0] for x in nuevas])
    cajas = np.array([x[1] for x in nuevas], dtype=np.float32)
    vecs = np.array([x[2] for x in nuevas], dtype=np.float16)
    if _os.path.exists(CACHE_TODAS):
        with np.load(CACHE_TODAS, allow_pickle=True) as z:
            claves = {(f, tuple(np.round(c, 3))) for f, c in
                      zip(fotos.tolist(), cajas.tolist())}
            keep = [i for i, (f, c) in enumerate(zip(z["photo_ids"].tolist(),
                                                     z["cajas"].tolist()))
                    if (f, tuple(np.round(c, 3))) not in claves]
            fotos = np.concatenate([z["photo_ids"][keep], fotos])
            cajas = np.concatenate([z["cajas"][keep].astype(np.float32), cajas])
            vecs = np.concatenate([z["vecs"][keep].astype(np.float16), vecs])
    tmp = CACHE_TODAS + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, photo_ids=fotos, cajas=cajas, vecs=vecs)
    _os.replace(tmp, CACHE_TODAS)
    return len(fotos)


def guardar_caras(nuevas):
    """Suma las huellas nuevas al cache, sin perder lo que ya habia.

    Se escribe a un temporal y despues se renombra: si se corta la luz a mitad
    de un archivo de 200 MB, el viejo sigue estando. Es el mismo cuidado que en
    fotos_duplicadas.py, donde un corte entre dos escrituras dejaba un grupo de
    copias sin ninguna primaria.
    """
    if not nuevas:
        return 0
    import os as _os
    fotos = np.array([x[0] for x in nuevas])
    personas = np.array([x[1] for x in nuevas], dtype=np.int64)
    vecs = np.array([x[2] for x in nuevas], dtype=np.float16)
    if _os.path.exists(CACHE_CARAS):
        # el with cierra el archivo: en Windows, un NpzFile abierto no deja
        # renombrar encima y os.replace falla con "Acceso denegado"
        with np.load(CACHE_CARAS, allow_pickle=True) as z:
            # (foto, persona) es la clave de una sugerencia: si el anio se
            # repasa, la huella nueva reemplaza a la vieja y no se duplica
            viejas = set(zip(fotos.tolist(), personas.tolist()))
            keep = [i for i, k in enumerate(zip(z["photo_ids"].tolist(),
                                                z["personas"].tolist()))
                    if k not in viejas]
            fotos = np.concatenate([z["photo_ids"][keep], fotos])
            personas = np.concatenate([z["personas"][keep].astype(np.int64),
                                       personas])
            vecs = np.concatenate([z["vecs"][keep].astype(np.float16), vecs])
    tmp = CACHE_CARAS + ".tmp"
    # el archivo se abre a mano porque savez_compressed le pega ".npz" al nombre
    # si no lo tiene, y el temporal terminaba en ".npz.tmp.npz"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, photo_ids=fotos, personas=personas, vecs=vecs)
    _os.replace(tmp, CACHE_CARAS)
    return len(fotos)


def sin_pisar(cli, filas, cuenta):
    """Saca las propuestas de (foto, persona) que ya existen.

    El upsert por (foto, persona) actualiza el recuadro. Si una persona corrigio
    en el visor cual cara es -"esta cara no es", "+ ¿Quién es?"- o un paso le
    busco el recuadro (recuadros_faltantes.py), repasar el album la devolvia a
    la cara que el reconocedor habia elegido. Tampoco se propone a quien ya
    esta identificado en la foto: ponerle recuadro es trabajo de
    recuadros_faltantes.py, que esta validado para eso.
    """
    if not filas:
        return filas
    ids = sorted({f["photo_id"] for f in filas})
    ya = set()
    for i in range(0, len(ids), 100):
        lote = "in.(%s)" % ",".join(ids[i:i + 100])
        for tabla in ("face_suggestions", "photo_people"):
            for r in cli.select(tabla, select="photo_id,person_id", photo_id=lote):
                ya.add((r["photo_id"], int(r["person_id"])))
    quedan = [f for f in filas if (f["photo_id"], int(f["person_id"])) not in ya]
    cuenta["ya_estaban"] += len(filas) - len(quedan)
    return quedan


def anotar_revisadas(cli, fol, conteo, por_carpeta, medidas, fallidas):
    """Anota los albumes que pasaron ENTEROS por el detector.

    Ver sql/caras_revisadas.sql. Uno cortado a la mitad no se anota y queda
    pendiente. Las fotos que no se pudieron bajar cuentan como vistas: si no,
    una foto sin miniatura haria repasar su album cada hora para siempre.
    """
    import datetime as _dt
    cuando = _dt.datetime.now(_dt.timezone.utc).isoformat()
    filas = [{"folder_id": k, "fotos": conteo.get(k, 0), "procesadas": medidas[k],
              "errores": fallidas[k], "revisado_at": cuando}
             for k in fol if medidas[k] + fallidas[k] == por_carpeta[k]]
    try:
        cli.upsert("carpetas_caras_revisadas", filas, on_conflict="folder_id")
        print("Albumes anotados como revisados: %d de %d" % (len(filas), len(fol)))
    except Exception as e:
        print("No se pudieron anotar los albumes revisados: %s" % str(e)[:120])


# Margen sobre los años de cursada, porque first_seen/last_seen salen de datos
# incompletos y cortar justo dejaria afuera casos reales.
MARGEN = 1


def avisar(cli, anio, total, hechas, con_gente, personas, estado, mensaje=None,
           callar=False):
    """Deja el avance en la tabla corridas, para verlo desde la app.

    callar es para --carpeta. La fila de corridas es la del AÑO, y
    eventos_todos.py la lee para saltear los años que ya cerraron: una corrida
    de una carpeta sola que la sobreescribe deja el año marcado como hecho con
    el total de esa carpeta. Me lo hice encima al probar la Fiesta de Fin de
    Año: la fila de 2025 quedo diciendo "total 1031" cuando el año tiene
    treinta mil fotos de evento.

    Envuelto en try/except a proposito y sin reintentos: informar el avance no
    puede voltear la corrida. Si la tabla no existe todavia, o si se corta la
    red un segundo, la corrida sigue y el proximo aviso se pone al dia.

    Ademas es lo que delata que se murio: si "actualizado" quedo viejo, no
    esta trabajando. Esta corrida se murio tres veces y el log seguia diciendo
    "--- 2010 ---" como si estuviera procesando.
    """
    if callar:
        return
    try:
        cli.upsert("corridas", [{
            "tarea": "eventos", "etiqueta": str(anio), "total": total,
            "hechas": hechas, "con_gente": con_gente, "encontradas": personas,
            "estado": estado, "mensaje": mensaje,
            "actualizado_at": "now()"}], on_conflict="tarea,etiqueta")
    except Exception as e:
        print("  (no se pudo informar el avance: %s)" % str(e)[:100])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--anio", type=int, required=True)
    ap.add_argument("--muestra", type=int, default=0, help="procesar solo N fotos")
    ap.add_argument("--carpeta", default=None,
                    help="procesar solo esta carpeta de Drive, del mismo ano "
                         "(o varias, separadas por coma)")
    ap.add_argument("--umbral", type=float, default=UMBRAL)
    ap.add_argument("--ensayo", action="store_true", help="medir sin escribir")
    a = ap.parse_args()

    cli = sb.SB()
    vecs, personas, anios_ref = todas_las_referencias(cli)

    # Rango de años en que cada persona estuvo en el colegio.
    #
    # Sale de first_seen y last_seen, que para 1.595 personas son la cursada
    # que declaran las planillas del colegio -"NS desde"/"NS hasta" en los
    # exalumnos, el nivel que hoy cursan en los alumnos actuales- y para el
    # resto, los años de las fotos que ya tienen identificadas.
    #
    # Esa segunda mitad queda corta: quien tiene un solo mugshot de 2021 pasa
    # por haber estado un año cuando estuvo catorce, y era inelegible en 2022.
    # Por eso el margen y por eso conviene que la planilla los cubra.
    #
    # No se usa enrollments. Esa tabla son prospectos, no alumnos: de 314
    # chicos que hoy estan cursando y figuran ahi, a 130 les da la cursada
    # terminada hace años. Y como el filtro era excluyente -si figuraba, solo
    # se lo buscaba en esos años- a esos 130 los descartaba sin mirarles la
    # cara.
    gente = {p["id"]: p for p in cli.select("people", select="id,display_name,first_seen,last_seen,kind")}
    # cuantas caras de referencia tiene cada uno: es lo que decide si se le
    # cree una propuesta de parecido bajo
    caras_de = Counter(int(r["person_id"]) for r in
                       cli.select("photo_people", select="person_id"))
    print("Personas con %d+ caras de referencia: %d"
          % (MIN_CARAS, sum(1 for v in caras_de.values() if v >= MIN_CARAS)))
    print("Personas con cursada acotada: %d de %d"
          % (sum(1 for g in gente.values() if g.get("first_seen") or g.get("last_seen")),
             len(gente)))

    def estuvo(pid):
        g = gente.get(pid)
        if g is None:
            return False
        # El staff no tiene cursada, y aplicarle una lo borraba del archivo
        # reciente.
        #
        # Lo encontro Diego con una foto de la Fiesta de Fin de Ano 2025:
        # "¿por que no se reconocio a John en esta foto?". Persona P
        # tiene 19 caras de referencia y fotos identificadas de 2012 a 2023,
        # pero first_seen y last_seen le decian 2021 y 2021. Con el margen de
        # un ano eso lo hacia elegible entre 2020 y 2022 y nada mas: en 2025 no
        # entraba al set de referencias, asi que no es que la cara no se
        # parecio, es que su cara no estaba entre las comparadas. Cero
        # propuestas suyas en las 1.031 fotos de esa carpeta.
        #
        # No era un caso: de los 546 del staff, 123 quedaban afuera de 2025
        # por lo mismo. first_seen/last_seen se derivan de las fotos que a cada
        # uno le toco tener identificadas, y eso no dice desde ni hasta cuando
        # trabaja en el colegio.
        #
        # El codigo ya sabia esto en otro lugar -las referencias de la planilla
        # de staff entran con con_anio=False, "su foto oficial vale para
        # cualquiera"- y este filtro se lo habia olvidado.
        #
        # Los alumnos si se filtran: un exalumno de 1998 no tiene por que
        # aparecer en un acto de 2025, y medido, ninguno de los 1.169 que el
        # filtro deja afuera de 2025 tiene una camada que diga lo contrario.
        if g.get("kind") == "staff":
            # Los 33 del staff que tienen camada fueron alumnos del colegio, y
            # de esos años hay fotos suyas de chico. Su cara de referencia es
            # la del adulto, asi que proponerla sobre una foto de 1995 es el
            # mismo error de Persona I al revés. Es la regla que ya usan
            # las colas -event_faces_pending filtra f.year > p.cohort- y aca
            # evita que la sugerencia llegue a existir.
            return g.get("cohort") is None or a.anio > g["cohort"]
        return ((g.get("first_seen") is None or g["first_seen"] - MARGEN <= a.anio)
                and (g.get("last_seen") is None or g["last_seen"] + MARGEN >= a.anio))

    elegible = np.array([estuvo(int(p)) for p in personas])
    print("Referencias: %d   utilizables en %d: %d"
          % (len(vecs), a.anio, int(elegible.sum())))
    if not elegible.any():
        sys.exit("Ninguna persona conocida estuvo en el colegio ese año.")
    vecs_ok = vecs[elegible]
    pers_ok = personas[elegible]
    anios_ok = anios_ref[elegible]

    # fotos de evento de ese año
    conteo = {f["id"]: int(f.get("photo_count") or 0)
              for f in cli.select("folders", select="id,year,is_mugshot,noise,photo_count")
              if not f.get("is_mugshot") and f.get("year") == a.anio and f.get("noise") is None}
    fol = list(conteo)
    # Una carpeta sola, o varias. Sirve para volver a pasar un album despues de
    # cambiar algo -el filtro de cursada, el umbral- sin repetir el ano entero,
    # que son horas; y es lo que usa caras_revisar.py para los albumes que no
    # pasaron enteros por el detector (sql/caras_revisadas.sql).
    if a.carpeta:
        pedidas = {x.strip() for x in a.carpeta.split(",") if x.strip()}
        fol = [x for x in fol if x in pedidas]
        if not fol:
            sys.exit("Esa carpeta no es una carpeta de evento de %d." % a.anio)
    fotos = []
    for i in range(0, len(fol), 40):
        fotos += cli.select("photos", select="id,folder_id,is_primary,source",
                            folder_id="in.(%s)" % ",".join(fol[i:i + 40]))
    # Solo Drive, y esta vez a proposito: Zenfolio topea en 400 px porque el
    # AccessMask de la cuenta protege los tamaños grandes (el sufijo -3 y los
    # que siguen devuelven un placeholder de 3 KB). Medido sobre 30 fotos de
    # acto de Quilmes 2021 a 400 px: CERO caras de 44 px o mas. Sus 15.183
    # fotos primarias de evento no se pueden reconocer con este acceso, y
    # bajarlas seria gastar para no encontrar a nadie. Si algun dia la cuenta
    # habilita los originales, esto se reabre.
    fotos = [f for f in fotos if f.get("is_primary") and f.get("source") == "drive"]
    carpeta_de = {f["id"]: f["folder_id"] for f in fotos}
    fotos = sorted(carpeta_de)
    por_carpeta = Counter(carpeta_de.values())
    medidas, fallidas = Counter(), Counter()
    if a.muestra:
        paso = max(len(fotos) // a.muestra, 1)
        fotos = fotos[::paso][:a.muestra]
    print("Fotos de evento de %d a procesar: %d\n" % (a.anio, len(fotos)))
    if not a.ensayo:
        avisar(cli, a.anio, len(fotos), 0, 0, 0, "corriendo",
               callar=bool(a.carpeta))

    d = Drive(SA_PATH)
    lock = threading.Lock()
    local = threading.local()
    cuenta = Counter()
    brecha = []
    huellas = []
    todas = []

    def mis_caras():
        if not hasattr(local, "c"):
            local.c = faces.Caras()
            local.c_lado = None
        return local.c

    def procesar(pid):
        try:
            link = d.get("files/" + pid, fields="thumbnailLink").get("thumbnailLink")
            if not link:
                # Sin miniatura es una foto que no se pudo bajar, y se cuenta
                # como tal. Salia sin contarse: su album nunca cerraba
                # (anotar_revisadas) y el paso "caras" lo repasaba cada hora
                # (13/9/2026, 5 albumes de 1 a 41 fotos).
                with lock:
                    cuenta["error"] += 1
                    fallidas[carpeta_de[pid]] += 1
                return []
            with urllib.request.urlopen(link.replace("=s220", "=s1600"), timeout=60) as r:
                data = r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            with lock:
                cuenta["error"] += 1
                fallidas[carpeta_de[pid]] += 1
            return []

        c = mis_caras()
        img = c.leer(data)
        if img is None:
            with lock:
                cuenta["error"] += 1
                fallidas[carpeta_de[pid]] += 1
            return []
        alto, ancho = img.shape[:2]

        filas, vistos = [], set()
        caras = [x for x in c.detectar(img) if min(x[2], x[3]) >= LADO_MIN]
        for cara in caras:
            h = c.huella(img, cara)
            if h is None:
                continue
            # se guarda ANTES de decidir si se propone a alguien: el punto es
            # justamente quedarse con las que hoy no se parecen a nadie
            x0, y0, w0, h0 = [float(v) for v in cara[:4]]
            with lock:
                todas.append((pid, (round(x0 / ancho, 4), round(y0 / alto, 4),
                                    round(w0 / ancho, 4), round(h0 / alto, 4)), h))
            s = vecs_ok @ h
            k = int(np.argmax(s))
            if s[k] < a.umbral:
                continue
            p = int(pers_ok[k])
            # parecido bajo y una ficha que el sistema apenas conoce: ver
            # CONOCIDA arriba
            if s[k] < CONOCIDA and caras_de.get(p, 0) < MIN_CARAS:
                with lock:
                    cuenta["poco_conocida"] += 1
                continue
            # la misma persona no puede estar dos veces en la misma foto: si
            # dos caras apuntan al mismo, gana la mas parecida
            if p in vistos:
                continue
            vistos.add(p)
            x, y, w, hh = [float(v) for v in cara[:4]]
            filas.append({"photo_id": pid, "person_id": p,
                          "score": round(float(s[k]), 4), "rank": len(filas) + 1,
                          "bx": round(x / ancho, 4), "by": round(y / alto, 4),
                          "bw": round(w / ancho, 4), "bh": round(hh / alto, 4)})
            with lock:
                huellas.append((pid, p, h))
            with lock:
                # el staff no tiene año de referencia: se marca -1 y contarlo
                # como distancia daria 2022 años
                ar = int(anios_ok[k])
                brecha.append(abs(ar - a.anio) if ar > 0 else None)

        with lock:
            cuenta["fotos"] += 1
            medidas[carpeta_de[pid]] += 1
            cuenta["caras"] += len(caras)
            if filas:
                cuenta["con_gente"] += 1
                cuenta["personas"] += len(filas)
            if cuenta["fotos"] % 100 == 0:
                print("  %d/%d   fotos con alguien: %d   personas: %d"
                      % (cuenta["fotos"], len(fotos), cuenta["con_gente"], cuenta["personas"]))
            # a la pantalla cada 100 fotos y a la base cada 1.000: son unos
            # 30 segundos de trabajo, suficiente para ver que avanza y sin
            # escribirle a la base cada tres segundos por una barra
            if cuenta["fotos"] % 1000 == 0 and not a.ensayo:
                avisar(cli, a.anio, len(fotos), cuenta["fotos"],
                       cuenta["con_gente"], cuenta["personas"], "corriendo",
                       callar=bool(a.carpeta))
        return filas

    acumulado = []
    TANDA = 400
    try:
        for i in range(0, len(fotos), TANDA):
            d._refresh()
            with ThreadPoolExecutor(max_workers=HILOS) as ex:
                for filas in ex.map(procesar, fotos[i:i + TANDA]):
                    acumulado += filas
            if acumulado and not a.ensayo:
                acumulado = sin_pisar(cli, acumulado, cuenta)
                cli.upsert("face_suggestions", acumulado, on_conflict="photo_id,person_id")
                acumulado = []
    except KeyboardInterrupt:
        print("\nInterrumpido.")
        if acumulado and not a.ensayo:
            acumulado = sin_pisar(cli, acumulado, cuenta)
            cli.upsert("face_suggestions", acumulado, on_conflict="photo_id,person_id")
        # afuera del if: una corrida cortada con el buffer justo vacio tambien
        # tiene que quedar marcada como cortada
        if not a.ensayo:
            avisar(cli, a.anio, len(fotos), cuenta["fotos"],
                   cuenta["con_gente"], cuenta["personas"], "cortada",
                   "la cortaron a mano", callar=bool(a.carpeta))

    if not a.ensayo:
        avisar(cli, a.anio, len(fotos), cuenta["fotos"],
               cuenta["con_gente"], cuenta["personas"], "listo",
               callar=bool(a.carpeta))
        # El cache se escribe una vez al final y no por tanda: son 512 numeros
        # por cara y reescribir el archivo entero cada 400 fotos seria mas
        # tiempo de disco que de reconocimiento.
        try:
            n = guardar_caras(huellas)
            print("Huellas de cara guardadas en el cache: %d nuevas, %d en total"
                  % (len(huellas), n))
            m = guardar_todas(todas)
            print("Caras detectadas guardadas (propuestas o no): %d nuevas, "
                  "%d en total" % (len(todas), m))
            # Recien con las caras guardadas: anotar antes dejaria un album
            # "revisado" sin sus caras, que es el agujero que esto tapa.
            if not a.muestra:
                anotar_revisadas(cli, fol, conteo, por_carpeta, medidas, fallidas)
        except Exception as e:
            # que no se pueda guardar el cache no puede voltear la corrida: lo
            # que importa -las propuestas- ya esta escrito en la base
            print("No se pudo guardar el cache de caras: %s" % str(e)[:120])
    n = max(cuenta["fotos"], 1)
    print("\n--- %d ---" % a.anio)
    print("Fotos procesadas:        %d" % cuenta["fotos"])
    print("Caras grandes halladas:  %d  (%.1f por foto)" % (cuenta["caras"], cuenta["caras"] / n))
    print("Fotos con alguien:       %d  (%.0f%%)" % (cuenta["con_gente"], 100.0 * cuenta["con_gente"] / n))
    print("Personas reconocidas:    %d  (%.2f por foto)" % (cuenta["personas"], cuenta["personas"] / n))
    print("Errores de descarga:     %d" % cuenta["error"])
    if cuenta["ya_estaban"]:
        print("Propuestas que ya existian y no se pisaron: %d" % cuenta["ya_estaban"])
    if cuenta["poco_conocida"]:
        print("Descartadas por parecido bajo sobre una ficha con menos de %d caras: %d"
              % (MIN_CARAS, cuenta["poco_conocida"]))
    if brecha:
        b = Counter(x for x in brecha if x is not None)
        sin_anio = sum(1 for x in brecha if x is None)
        print("Distancia en años entre la foto y la cara de referencia:")
        for k in sorted(b):
            print("   %d año(s): %d" % (k, b[k]))
        if sin_anio:
            print("   sin año (staff): %d" % sin_anio)
    if a.ensayo:
        print("\nEnsayo: no se escribio nada.")
    else:
        # cuanto costaria el archivo entero a este ritmo
        print("\nA este ritmo, las 362.287 fotos de evento darian ~%d filas"
              % int(cuenta["personas"] / n * 362287))


if __name__ == "__main__":
    main()
