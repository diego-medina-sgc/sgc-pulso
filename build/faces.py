# -*- coding: utf-8 -*-
"""
Detección y comparación de caras. Módulo compartido; no se corre solo.

Por qué OpenCV y no insightface, que es lo habitual: acá corre sobre Python
3.14 en Windows, e insightface necesita compilar. OpenCV 5 trae el detector
(YuNet) y el reconocedor (SFace) adentro, así que la única dependencia es
opencv-python-headless. Menos piezas, menos cosas que se rompan en la
máquina de otro.

SFace es menos preciso que ArcFace, pero la diferencia importa poco acá: la
salida no etiqueta a nadie sola, entra al juego como sugerencia para que una
persona confirme.

Los modelos no están en el repo (sface.onnx pesa 37 MB). Se bajan con:
    python faces.py --bajar-modelos
"""
import io
import os
import threading
import urllib.request
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
PESOS = {
    "yunet.onnx": ZOO + "/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "sface.onnx": ZOO + "/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

# El reconocedor se elige con la variable de entorno FACES_MOTOR, asi los
# scripts que ya existen no cambian:
#     set FACES_MOTOR=arcface
#
# sface    el que trae OpenCV. 128 dimensiones. Es el mas flojo de los dos; se
#          eligio porque viene incluido y no hay que instalar nada.
# arcface  w600k_r50 de InsightFace (buffalo_l). 512 dimensiones. Es el
#          estandar de facto. No hace falta el paquete insightface, que en
#          Windows habria que compilar: alcanza con el .onnx y onnxruntime,
#          que si tiene wheel para Python 3.14.
#
# Las huellas de los dos NO son comparables entre si: distinta dimension y
# distinta escala de puntajes. Cambiar de motor obliga a recalcular todos los
# _*.npz y a recalibrar el umbral con faces_calibrar.py.
MOTOR = os.environ.get("FACES_MOTOR", "sface").lower()

# Cuantos hilos usa cada libreria POR ADENTRO.
#
# Las dos arrancan con tantos hilos como nucleos, y quien las llama ya trae su
# propio paralelismo: faces_eventos.py procesa seis fotos a la vez. En una
# notebook de cuatro nucleos eso da seis detectores de cuatro hilos peleando
# por cuatro nucleos, y el trabajo se va en cambiar de contexto en vez de en
# medir caras: medido, la corrida usaba 2,46 nucleos de 4.
#
# Con uno adentro y el paralelismo afuera, cada hilo hace su foto de punta a
# punta. Se dejan como variable de entorno para poder volver atras y para
# poder medir las dos configuraciones sin tocar el codigo.
CV_HILOS = int(os.environ.get("FACES_CV_HILOS", "1"))
ORT_HILOS = int(os.environ.get("FACES_ORT_HILOS", "1"))
if CV_HILOS > 0:
    cv2.setNumThreads(CV_HILOS)

ARCFACE = os.path.join(MODELS, "arcface_w600k_r50.onnx")
ARCFACE_ZIP = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

# onnxruntime SI es thread-safe para inferencia, al reves que los modelos de
# OpenCV: una sola sesion para todos los hilos en vez de una por hilo, que a
# 174 MB cada una serian mas de 1 GB con 6 hilos.
_sesion = None
_sesion_lock = threading.Lock()


# Cada motor tiene su escala de puntajes: 0,50 con arcface es el punto donde
# deja de equivocarse, y con sface son 75 falsos positivos sobre 9.175 pares.
# Por eso los umbrales viven aca y no sueltos en cada script: cambiar de motor
# y olvidarse de moverlos es un error que no avisa.
#
# Medido con faces_calibrar.py sobre 141 pares de la misma persona y 9.175 de
# personas distintas. Entre parentesis, que porcentaje de los pares de la
# misma persona queda por encima; los seis valores dan cero falsos positivos.
UMBRALES = {
    "sface": {
        "sugerir": 0.65,    # (84%)
        "agrupar": 0.80,    # (17%)
        "eventos": 0.70,    # (67%)
    },
    "arcface": {
        "sugerir": 0.50,    # (97%)
        "agrupar": 0.63,    # (72%)
        "eventos": 0.60,    # (87%)
    },
}

# Dos fotos de una misma carpeta son, por regla, personas distintas: en una
# sesion de retratos cada chico pasa una vez. La excepcion es la rafaga, o sea
# las dos o tres tomas seguidas del mismo chico, que existen y son muchas.
#
# Se pueden distinguir sin riesgo porque no hay zona gris entre una cosa y la
# otra. Medido sobre los mismos pares del set de calibracion:
#
#                          personas distintas    misma persona, mismo dia
#   sface                    maximo 0,643            minimo 0,927
#   arcface                  maximo 0,473            minimo 0,929
#
# 0,90 cae en el medio del hueco para los dos motores. Por debajo, la regla de
# "una vez por carpeta" sigue valiendo tal cual.
RAFAGA = 0.90


def umbral(cual):
    """Umbral de decision para el motor activo.

    Los valores no estan pegados al maximo medido entre personas distintas
    (0,643 en sface, 0,473 en arcface) sino bastante arriba: el set de
    calibracion tiene 9.175 pares y el agrupamiento real compara millones, asi
    que la cola es mas larga de lo que se llego a ver.
    """
    return UMBRALES[MOTOR][cual]


def npz(nombre):
    """Ruta de un cache de huellas, con el sufijo del motor que lo escribio.

    Las huellas de sface son de 128 dimensiones y las de arcface de 512:
    mezclarlas en un mismo archivo no da un error, da resultados sin sentido.
    Con el sufijo los dos juegos conviven, y volver atras es cambiar una
    variable de entorno en vez de recalcular 11.647 caras.

    El motor sface no lleva sufijo, asi que los archivos que ya existen siguen
    valiendo tal cual estan.
    """
    base, ext = os.path.splitext(nombre)
    if MOTOR != "sface":
        base += "_" + MOTOR
    return os.path.join(HERE, base + ext)


# DE QUE SE PUEDE APRENDER (5/10/2026)
#
# Decision de Diego: lo que nombra el sistema no puede ser referencia. Una
# respuesta equivocada que se vuelve referencia atrae mas caras ajenas, que
# tambien se nombran solas, y el error se multiplica sin que nadie conteste
# nada: 28 de 42 reagrupamientos pasaron sin una sola respuesta humana (2/10).
#
#   humanas  'game' y 'manual' (los juegos y la ficha) y 'grupo' (un "Si" en
#            Grupos: lo contesto una persona mirando las caras del grupo)
#   carnets  el nombre sale del archivo del retrato. Solo cuentan para quien
#            no tiene NINGUNA cara humana de referencia: 3 de 1.864 personas
#            tenian carnets a nombre de otro (28/9), y lo humano manda.
#   fuentes  las que anoto Diego (Valete, anuarios, staff, carpetas de
#            _referencias_fuente_*): existen para darle cara a quien no tiene
#            fotos, y valen igual que los carnets, solo para quien no tiene
#            ninguna cara humana (faces_sugerir.todas_las_referencias).
#
# Afuera todo lo de maquina: auto_grupo, face_auto, face, cluster, dedupe,
# filename_face y cualquier source nuevo que no este en estas dos listas.
# Medido el 5/10/2026 en build/_medir_referencias_humanas.py: personas con
# referencia (fotos + fuentes) 4.824 -> 4.782; 44 se quedan sin ninguna (41
# con identificacion humana en fotos sin cara ubicable). Leave-one-out sobre
# 40.782 caras humanas, a 0,50: error 21,3% -> 0,62%, captura 78,4% -> 98,2%;
# casi todo por tomar la persona de la pareja (foto, persona) y no de la foto
# (con eso solo: 0,55% / 99,1%).
FUENTES_HUMANAS = ("game", "manual", "grupo")
FUENTES_CARNET = ("filename", "mugshot_filename")


# UNA CARA EN DISPUTA NO ES REFERENCIA (15/10/2026)
#
# caras_disputadas guarda las parejas (foto, persona) donde una respuesta humana
# choca con otra evidencia y alguien tiene que mirarla (el "si" de un carnet
# que se parece a carnets ajenos y no a las demas caras de la ficha). El "si"
# no se borra -el juez es quien carga- pero mientras la disputa este abierta
# (resuelta_at nula) esa cara no puede ser referencia ni verdad humana: si
# estaba mal, atrae caras ajenas y el error se multiplica (regla 10).
#
# La regla vive aca: disputas_abiertas() las lee una vez y cada lugar que arma
# "lo que confirmo una persona" las saca con sin_disputas() o pasandolas a
# filtrar_referencias(). Si la tabla no se puede leer se avisa y se sigue
# como antes: no se corta un paso por esto.


def disputas_abiertas(cli):
    """{(photo_id, person_id)} de caras_disputadas con resuelta_at nula."""
    try:
        filas = cli.select("caras_disputadas", select="photo_id,person_id",
                           resuelta_at="is.null")
    except Exception as e:
        print("  (no se pudo leer caras_disputadas: %s; las disputas no se "
              "descuentan)" % str(e)[:120], flush=True)
        return set()
    return {(r["photo_id"], int(r["person_id"])) for r in filas
            if r.get("photo_id") and r.get("person_id") is not None}


def sin_disputas(pares, disputadas, que=""):
    """Las parejas (foto, persona) que no estan en disputa. Avisa cuantas saca."""
    if not disputadas:
        return pares
    fuera = [k for k in pares if (k[0], int(k[1])) in disputadas]
    if not fuera:
        return pares
    if que:
        print("  %s: %d caras en disputa abierta no cuentan como confirmadas"
              % (que, len(fuera)), flush=True)
    tipo = type(pares)
    quedan = [k for k in pares if (k[0], int(k[1])) not in disputadas]
    return tipo(quedan) if tipo in (set, frozenset) else quedan


def filtrar_referencias(fotos, personas, vigentes, disputadas=frozenset()):
    """Que caras de un .npz de referencias valen hoy, y de quien son.

    fotos, personas: por cara, la foto y la persona con que se archivo.
    vigentes: {(foto, persona): set de sources} de photo_people HOY, solo de
              personas vivas.

    Devuelve (indices, persona de cada uno). Vale la pareja (foto, persona)
    con que se archivo la cara, no "alguien de esa foto": hasta el 5/10/2026
    cargar_refs() tomaba la persona por FOTO, y en una foto de acto con varias
    identificadas le daba a la cara de uno el nombre de otro. Si la pareja ya
    no esta (una fusion, un "no es"), la cara no vale hasta que
    faces_referencias.py la vuelva a archivar con la persona de hoy.
    """
    cand = []
    for i, (f, p) in enumerate(zip(fotos, personas)):
        p = int(p)
        # disputadas: (foto, persona) en disputa abierta (disputas_abiertas)
        if (f, p) in disputadas:
            continue
        ss = vigentes.get((f, p))
        if not ss:
            continue
        if ss & set(FUENTES_HUMANAS):
            cand.append((i, p, True))
        elif ss & set(FUENTES_CARNET):
            cand.append((i, p, False))
    con_humana = {p for _, p, h in cand if h}
    out = [(i, p) for i, p, h in cand if h or p not in con_humana]
    return [i for i, _ in out], [p for _, p in out]


def fotos_vivas(cli, fotos, que):
    """Mascara por cara: True si su foto sigue en photos (13/10/2026).

    Las huellas guardan caras de fotos que despues se borraron en Drive
    (sync_drive.py las saca de photos). Para los pasos donde una cara de mas
    cambia el resultado -el agrupamiento las usa de puente entre dos personas,
    el mapa del visor las muestra- se dejan afuera al cargar, no al escribir.
    Mismo criterio que sb.SB.fotos_que_existen, que es el que se usa en todos
    lados. que: el nombre del .npz, para el aviso."""
    fotos = [str(f) for f in fotos]
    hay = cli.fotos_que_existen(fotos, avisar=que)
    m = np.fromiter((f in hay for f in fotos), dtype=bool, count=len(fotos))
    fuera = int(len(fotos) - m.sum())
    print("  %s: %d caras de %d fotos que ya no estan en photos, afuera"
          % (que, fuera, len(set(fotos) - hay)), flush=True)
    return m


# LAS CARAS DE FUENTE DE UNA FICHA FUSIONADA (14/10/2026)
#
# merge_person() mueve photo_people, photo_labels, face_suggestions y los
# rechazos de la ficha que se va a la que queda, y apaga la que se va (kind
# noise). Las caras de fuente -staff, anuarios, Valete, carpetas anotadas
# (_referencias_fuente_*)- no viven en la base: estan en los .npz con el id de
# la ficha con que se archivaron. Hasta aca, al cargarlas se tiraba toda cara de
# una ficha noise, asi que una fusion PERDIA sus referencias de fuente en vez de
# pasarlas a la que queda (fichas 7616, 7622 y 7618: cada una con su cara de
# una carpeta de staff, que se perdia al fusionarla).
#
# Ahora la cara pasa a la ficha que queda, siguiendo la cadena de fusiones,
# pero con un control (regla 10: nada entra como referencia sin chequear
# coherencia): pasa si se parece al menos umbral("sugerir") -0,50 con arcface,
# 97% de los pares de la misma persona y ningun par de personas distintas en la
# calibracion- a alguna cara de referencia que la que queda ya tenga, o si la
# que queda no tiene ninguna. Una fusion equivocada no mete una cara ajena.
# Medido el 13-14/10/2026 en las tres de arriba: 0,885, 0,968 y 0,963 contra la
# foto de staff de la ficha que queda; control con 20 staff al azar, maximo
# 0,176.


def fusiones_destino(cli):
    """{id que se fue: id que quedo}, siguiendo cadenas (a->b, b->c: a->c).

    Si la tabla no se puede leer, devuelve {} y avisa: se carga como antes
    (las caras de fichas fusionadas quedan afuera), no se corta el paso."""
    try:
        filas = cli.select("fusiones", select="se_va_id,queda_id")
    except Exception as e:
        print("  (no se pudo leer fusiones: %s; las caras de fichas fusionadas "
              "quedan afuera)" % str(e)[:120], flush=True)
        return {}
    directo = {}
    for r in filas:
        if r.get("se_va_id") is not None and r.get("queda_id") is not None \
                and int(r["se_va_id"]) != int(r["queda_id"]):
            directo[int(r["se_va_id"])] = int(r["queda_id"])
    final = {}
    for va in directo:
        q, vistos = directo[va], {va}
        while q in directo and q not in vistos:
            vistos.add(q)
            q = directo[q]
        final[va] = q
    return final


def _rutas_referencias():
    """Todos los .npz de caras de referencia del motor activo."""
    import glob
    import re
    rutas = [npz(x) for x in ("_referencias.npz", "_referencias_staff.npz",
                              "_referencias_anuario.npz", "_referencias_anuario_pdf.npz",
                              "_referencias_valete.npz")]
    for r in sorted(glob.glob(npz(os.path.join(HERE, "_referencias_fuente_*.npz")))):
        # con sface el patron no lleva sufijo y tomaria tambien los de arcface
        # (el mismo criterio que faces_sugerir.todas_las_referencias)
        if MOTOR == "sface" and re.search(r"_[a-z]+\.npz$", r):
            continue
        rutas.append(r)
    return [r for r in rutas if os.path.exists(r)]


class Fusiones:
    """Pasa las caras de fichas fusionadas a la ficha que quedo (ver arriba).

        fus = faces.Fusiones(cli)
        personas, pasa = fus.pasar(personas, vecs, "staff")
        # personas: con el id de la que quedo; pasa[i] False = no se usa

    Las referencias de las fichas que quedan se leen recien la primera vez que
    aparece una cara fusionada, de todos los .npz de referencias, una vez."""

    def __init__(self, cli, destino=None, refs=None):
        self.destino = fusiones_destino(cli) if destino is None else dict(destino)
        self._refs = refs           # {persona: matriz normalizada}; para tests
        self.pasadas = 0
        self.frenadas = 0

    def _cargar_refs(self):
        quedan = set(self.destino.values())
        de = {}
        for ruta in _rutas_referencias():
            with np.load(ruta, allow_pickle=True) as z:
                pers = z["personas"].astype(int)
                idx = [i for i, p in enumerate(pers) if int(p) in quedan]
                if not idx:
                    continue
                V = z["vecs"][idx].astype(np.float32)
            V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
            for k, i in enumerate(idx):
                de.setdefault(int(pers[i]), []).append(V[k])
        self._refs = {q: np.stack(v) for q, v in de.items()}

    def pasar(self, personas, vecs, que=""):
        """(personas con el id de destino, mascara de las que se usan)."""
        pers = np.asarray(personas).astype(int).copy()
        pasa = np.ones(len(pers), dtype=bool)
        if not self.destino:
            return pers, pasa
        idx = [i for i, p in enumerate(pers) if int(p) in self.destino]
        if not idx:
            return pers, pasa
        if self._refs is None:
            self._cargar_refs()
        corte = umbral("sugerir")
        pasadas = frenadas = 0
        for i in idx:
            q = self.destino[int(pers[i])]
            v = np.asarray(vecs[i], dtype=np.float32)
            v = v / max(float(np.linalg.norm(v)), 1e-9)
            R = self._refs.get(q)
            if R is None or not len(R) or float((R @ v).max()) >= corte:
                pers[i] = q
                pasadas += 1
            else:
                pasa[i] = False
                frenadas += 1
        self.pasadas += pasadas
        self.frenadas += frenadas
        print("  %s: %d caras de fichas fusionadas pasan a la que quedo, %d no "
              "(no se parecen a sus referencias, corte %.2f)"
              % (que or "referencias", pasadas, frenadas, corte), flush=True)
        return pers, pasa


def sesion_arcface():
    global _sesion
    if _sesion is None:
        with _sesion_lock:
            if _sesion is None:
                import onnxruntime
                if not os.path.exists(ARCFACE):
                    raise SystemExit(
                        "Falta %s.\nCorré primero:  python faces.py --bajar-modelos"
                        % ARCFACE)
                so = onnxruntime.SessionOptions()
                so.log_severity_level = 3
                # la sesion es unica y compartida por todos los hilos: si cada
                # llamada abre su propio abanico de hilos adentro, se pisan
                if ORT_HILOS > 0:
                    so.intra_op_num_threads = ORT_HILOS
                    so.inter_op_num_threads = ORT_HILOS
                _sesion = onnxruntime.InferenceSession(
                    ARCFACE, so, providers=["CPUExecutionProvider"])
    return _sesion

# Por debajo de esto YuNet devuelve basura: manos, nucas, el fondo.
CONF_MIN = 0.75
# Una cara más chica que esto no tiene píxeles suficientes para reconocerla,
# y en una foto de acto escolar hay decenas asi al fondo.
LADO_MIN = 44
# El piso de las fotos de EVENTO, medido sobre la miniatura de 1600 px que bajan
# faces_eventos.py y caras_huellas_evento.py: lo que queda abajo no se guarda en
# _caras_todas, y entonces tampoco se agrupa, ni tiene recuadro, ni se propone.
# Vive aca y no en cada script para que el que guarda y el que verifica midan
# con la misma vara.
#
# Era 80. Tabla del 15/10/2026 (dejando afuera las referencias de la misma
# carpeta, contra 46.213 referencias): las 89 caras REALES confirmadas por una
# persona de 60 a 79 px dieron 0 % de error con referencia (46,4 % de acierto a
# 0,60; 69,0 % a 0,55 con margen 0,10) y 2,2 % sin referencia, igual que las de
# 80 a 119 px (0 % / 1,0-1,3 %). Abajo de 60 hay solo 15 caras reales (44 a 59
# px): no alcanza para bajar mas. Muestra de 3.131 fotos del archivo: con 60 px
# se guardan 1,46 veces las caras de hoy (1,41-1,50).
LADO_EVENTOS = 60
# El piso del RECUADRO (21/10/2026, decision de Diego): las caras de
# LADO_RECUADRO a LADO_EVENTOS-1 px, medidas igual que LADO_EVENTOS (lado menor
# de la caja de YuNet sobre la miniatura de 1600), llevan recuadro en el visor y
# se pueden nombrar, pero NO tienen huella: no van a _caras_todas, ni a grupos,
# ni a sugerencias, ni son referencia. Van a CACHE_CHICAS y de ahi a
# caras_mapa.chicas (caras_mapa.py).
#
# Medido el 21/10/2026, deteccion a 800 como hoy:
#   asado de camada 2004 (31 fotos):   >=60: 44 caras   40-59: 106   <40: 169
#   muestra de 299 fotos del archivo:  por foto >=60: 1,90  40-59: 1,01  <40: 1,85;
#                                      el 28 % de las fotos gana al menos una
#   falsos positivos contados a ojo (120 recortes por tramo): 3-5 % en 40-59
# Y la huella a ese tamaño, achicando 247 caras confirmadas por una persona (con
# referencia propia fuera de su sesion) hasta que midan N px y comparando
# contra 46.208 referencias: a 0,60 acierto/error 68,8 %/0,6 % a 60 px,
# 66,0 %/0,6 % a 50 y 64,4 %/0,6 % a 40. Es el mejor caso (achicar no
# desenfoca): por eso esta tanda no les da huella.
LADO_RECUADRO = 40
# Donde quedan los recuadros chicos: photo_ids, cajas (fracciones, 4 decimales,
# como _caras_todas) y lados (px en la miniatura de 1600). Sin huella.
CACHE_CHICAS = "_caras_chicas.npz"


def repartir(caras, piso_huella=None, piso_recuadro=None):
    """(grandes, chicas) de una lista de caras de YuNet.

    grandes: lado >= LADO_EVENTOS (huella, _caras_todas, grupos, sugerencias).
    chicas:  LADO_RECUADRO <= lado < LADO_EVENTOS (solo recuadro).
    Lo de abajo de LADO_RECUADRO se descarta."""
    ph = LADO_EVENTOS if piso_huella is None else piso_huella
    pr = LADO_RECUADRO if piso_recuadro is None else piso_recuadro
    grandes, chicas = [], []
    for c in caras:
        lado = float(min(c[2], c[3]))
        if lado >= ph:
            grandes.append(c)
        elif lado >= pr:
            chicas.append(c)
    return grandes, chicas


def caja_chica(caja, ancho, alto):
    """True si un recuadro guardado (fracciones bx, by, bw, bh) mide menos de
    LADO_EVENTOS sobre una imagen de ancho x alto (la miniatura de 1600).

    Un recuadro con 4 decimales sobre 1600 px se corre a lo sumo 0,08 px: se
    da medio pixel de margen para que una cara de 60 justos no pase por chica."""
    lado = min(float(caja[2]) * ancho, float(caja[3]) * alto)
    return lado + 0.5 < LADO_EVENTOS


def cajas_chicas(fotos=None):
    """{(photo_id, bx, by)} de CACHE_CHICAS, redondeado a 4 como el resto.

    Para los que ubican una cara confirmada por su recuadro: un recuadro chico
    no tiene huella, y no hay que ir a buscarle una prestada. Si el archivo no
    esta, conjunto vacio (todavia no corrio el detector con este piso)."""
    ruta = npz(CACHE_CHICAS)
    if not os.path.exists(ruta):
        return set()
    with np.load(ruta, allow_pickle=True) as z:
        F = z["photo_ids"].astype(str)
        C = z["cajas"].astype(np.float32)
    if fotos is not None:
        fotos = set(fotos)
    out = set()
    for i in range(len(F)):
        if fotos is None or F[i] in fotos:
            out.add((F[i], round(float(C[i][0]), 4), round(float(C[i][1]), 4)))
    return out


# A qué tamaño se detecta. YuNet fue entrenado a 320 px: sobre una foto de
# 1920 deja de ver caras que encuentra sin problema al achicarla. 800 anda
# bien tanto en retratos como en fotos de grupo.
LADO_DETECCION = 800


def bajar_modelos():
    os.makedirs(MODELS, exist_ok=True)
    for nombre, url in PESOS.items():
        dest = os.path.join(MODELS, nombre)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print("  ya está: %s" % nombre)
            continue
        print("  bajando %s ..." % nombre)
        urllib.request.urlretrieve(url, dest)
        print("    %s  %.1f MB" % (nombre, os.path.getsize(dest) / 1e6))

    # ArcFace viene dentro del zip de buffalo_l, que trae cinco modelos; solo
    # se guarda el reconocedor. Son 289 MB de descarga para quedarse con 174.
    if os.path.exists(ARCFACE) and os.path.getsize(ARCFACE) > 1000:
        print("  ya está: %s" % os.path.basename(ARCFACE))
        return
    print("  bajando buffalo_l.zip (289 MB) para sacarle ArcFace ...")
    data = urllib.request.urlopen(ARCFACE_ZIP, timeout=600).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    for n in z.infolist():
        if n.filename.endswith("w600k_r50.onnx"):
            with open(ARCFACE, "wb") as fh:
                fh.write(z.read(n))
            print("    %s  %.1f MB" % (os.path.basename(ARCFACE),
                                       os.path.getsize(ARCFACE) / 1e6))
            return
    print("    no se encontro w600k_r50.onnx dentro del zip")


class Caras:
    def __init__(self, conf=CONF_MIN, motor=None):
        for nombre in PESOS:
            p = os.path.join(MODELS, nombre)
            if not os.path.exists(p):
                raise SystemExit(
                    "Falta %s.\nCorré primero:  python faces.py --bajar-modelos" % p)
        self.det = cv2.FaceDetectorYN.create(
            os.path.join(MODELS, "yunet.onnx"), "", (320, 320), conf, 0.3, 5000)
        # SFace se carga siempre, aunque el motor sea ArcFace: su alignCrop es
        # el que recorta la cara a 112x112 usando los 5 puntos, que es
        # exactamente la plantilla que ArcFace espera a la entrada.
        self.rec = cv2.FaceRecognizerSF.create(
            os.path.join(MODELS, "sface.onnx"), "")
        self.motor = (motor or MOTOR).lower()
        if self.motor not in ("sface", "arcface"):
            raise SystemExit("Motor desconocido: %s (sface o arcface)" % self.motor)
        if self.motor == "arcface":
            sesion_arcface()      # que falle aca y no adentro de un hilo

    def leer(self, data):
        """Bytes de un JPEG a matriz BGR. None si no es una imagen válida."""
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def detectar(self, img, lado_min=None):
        """Caras de una imagen, de la más grande a la más chica.

        lado_min baja el piso de tamaño para un uso puntual. El piso de la casa
        -44 px- es el que hace falta para RECONOCER una cara: con menos no hay
        pixeles y la huella es ruido. Contarlas es otra cosa, y ahi el piso
        estorba: en la foto de K5 -treinta chicos en las gradas- el detector ve
        las treinta caras, miden entre 32 y 43 px, y el filtro las descarta
        todas. La foto pasaba por retrato de una persona.

        Se ordenan por tamaño porque en un retrato la cara del sujeto es la
        más grande, y en una foto de grupo las grandes son las de adelante,
        que son las que se pueden reconocer.

        Se detecta sobre una copia reducida. YuNet fue entrenado a 320 px y en
        imágenes muy grandes deja de reconocer caras que ve perfectamente al
        achicarlas: en las fotos de staff de 1920x1920 encontraba cero, y
        reducidas a 800 encontraba la cara en todas. Las coordenadas se
        devuelven en la escala original, así que el recorte para la huella
        sigue saliendo de la imagen en calidad completa.
        """
        if img is None:
            return []
        h, w = img.shape[:2]
        esc = float(LADO_DETECCION) / max(h, w)
        if esc < 1.0:
            chica = cv2.resize(img, (max(int(w * esc), 1), max(int(h * esc), 1)))
        else:
            chica, esc = img, 1.0

        ch, cw = chica.shape[:2]
        self.det.setInputSize((cw, ch))
        _, caras = self.det.detect(chica)
        if caras is None:
            return []

        if esc != 1.0:
            caras = caras.copy()
            caras[:, :14] /= esc      # caja y los 5 puntos; la confianza no se toca

        out = [f for f in caras
               if min(f[2], f[3]) >= (LADO_MIN if lado_min is None else lado_min)]
        out.sort(key=lambda f: f[2] * f[3], reverse=True)
        return out

    def huella(self, img, cara):
        """Vector normalizado de una cara. Normalizado para que comparar sea
        un producto punto y no haya que dividir en cada comparación."""
        try:
            recorte = self.rec.alignCrop(img, cara)
            if self.motor == "arcface":
                # 112x112, RGB, escalado a [-1, 1]: lo que espera w600k_r50
                blob = cv2.dnn.blobFromImage(
                    recorte, 1.0 / 127.5, (112, 112),
                    (127.5, 127.5, 127.5), swapRB=True)
                ses = sesion_arcface()
                v = ses.run(None, {ses.get_inputs()[0].name: blob})[0]
                v = np.asarray(v).flatten().astype(np.float32)
            else:
                v = self.rec.feature(recorte).flatten().astype(np.float32)
        except cv2.error:
            return None
        n = np.linalg.norm(v)
        return None if n == 0 else v / n

    def huellas(self, data, max_caras=None, lado_min=None):
        """De bytes de imagen a [(huella, caja, confianza)]."""
        img = self.leer(data)
        if img is None:
            return []
        out = []
        for cara in self.detectar(img, lado_min)[:max_caras]:
            h = self.huella(img, cara)
            if h is not None:
                out.append((h, cara[:4].astype(int).tolist(), float(cara[14])))
        return out


def parecido(a, b):
    """Coseno entre huellas ya normalizadas: -1 a 1, más alto es más parecido."""
    return float(np.dot(a, b))


if __name__ == "__main__":
    import sys
    if "--bajar-modelos" in sys.argv:
        bajar_modelos()
    else:
        print(__doc__)
