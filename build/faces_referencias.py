# -*- coding: utf-8 -*-
"""
Arma las caras de referencia: una huella por cada retrato ya identificado.

Es contra esto que después se comparan las caras de las fotos de evento. Sin
referencia no hay reconocimiento posible, y hoy sólo 960 personas tienen una.

Se guarda UNA HUELLA POR RETRATO, no un promedio por persona. La calibración
mostró que la misma persona baja de 0,94 a 0,72 de parecido de un año a
otro: promediar el mugshot de 1er grado con el de 5to da una cara que no es
ninguna de las dos. Cada huella conserva su año y la comparación después se
acota a los años cercanos.

    python faces_referencias.py              # todas
    python faces_referencias.py --limite 50  # prueba corta

Escribe _referencias.npz. Es reanudable: si se corta, la próxima corrida
sigue desde donde estaba.

UNA CARA POR PAREJA (foto, persona), NO POR FOTO (3/11/2026). El cache y el
.npz iban por photo_id: de una foto con tres "Si" en Grupos se archivaba una
cara y las otras dos no entraban nunca (la foto ya estaba "procesada").
Medido el 3/11 contra el .npz del pulso: 80.688 caras, una por foto; 26.968
parejas humanas con recuadro afuera por eso. El formato del archivo no
cambia (listas paralelas photo_ids/personas/vecs; photo_ids ahora se repite)
y no hay que reconstruirlo: cada entrada vieja ya es una pareja.

Las que faltan con recuadro se buscan primero en _caras_todas (la misma
miniatura de 1600, el mismo detector y el mismo recuadro a 4 decimales, como
recuadros_faltantes.py): no se baja nada. Medido el 3/11: 26.881 de 26.968.

LAS CARAS DE EPIGRAFE ('epigrafe_cara') se miden sobre la foto grande, no
sobre la de 1600 (decision de Diego del 3/11/2026): una bajada por foto, del
tamaño que deja sus caras sobre el piso (ver FUENTE_EPIGRAFE). El piso
(faces.LADO_EVENTOS) es el mismo; el resto de los origenes, igual que antes.
"""
import argparse
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import faces
import sb
import zenfolio
from index_drive import Drive, SA_PATH

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = faces.npz("_referencias.npz")
HUELLAS = faces.npz("_huellas_retratos.npz")
HILOS = 6          # más que esto y Drive empieza a devolver 403 por rate limit


def cajas_guardadas(cli):
    """El recuadro de cada (foto, persona), en fracciones de la imagen.

    Es lo que convierte "esta persona esta en esta foto" en "esta persona es
    ESTA cara de esta foto". Sin eso no hay forma de saber cual de las cuarenta
    caras de un acto es la que alguien confirmo.
    """
    d = {}
    for r in cli.select("face_suggestions",
                        select="photo_id,person_id,bx,by,bw,bh"):
        if r.get("bw") and r.get("bh"):
            d[(r["photo_id"], int(r["person_id"]))] = (
                float(r["bx"]), float(r["by"]), float(r["bw"]), float(r["bh"]))
    return d


def solape(a, b):
    """Cuanto se pisan dos recuadros (x, y, w, h), de 0 a 1."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / float(aw * ah + bw * bh - inter)


# Cuanto se tienen que pisar el recuadro guardado y el detectado para aceptar
# que son la misma cara. Es el mismo numero que usa caras_huellas_evento.py:
# mismo detector sobre la misma imagen, los recuadros dan casi identicos.
IOU_MIN = 0.5
TODAS = faces.npz("_caras_todas.npz")


def cara_de(c, data, caja, hecho, lock, varias=False):
    """La cara de LA PERSONA en esta foto. None si no se puede saber cual es.

    ESTE ERA EL AGUJERO POR DONDE ENTRABA TODO

    Antes esto era una linea: huellas(data, max_caras=1), la cara mas grande,
    con el comentario "en un retrato la cara del sujeto es la mas grande". El
    comentario es cierto y la funcion no: catalogo() no trae solo retratos,
    trae TODAS las identificaciones confirmadas, y la mayoria son fotos de
    acto. En una foto de cuarenta personas la cara mas grande es la del que
    quedo mas cerca de la camara.

    Asi que el juego preguntaba "¿esta Roberto en esta foto?", Diego decia que
    si -y tenia razon, Roberto estaba-, y el sistema archivaba como cara de
    Roberto la del senor de adelante. Esa cara pasaba a ser referencia, atraia
    mas caras que no eran de Roberto, que tambien se confirmaban porque Roberto
    tambien estaba ahi. Roberto termino con 185 referencias de las que 109 no
    eran el, y su nombre aparecia en toda la biblioteca.

    LA REGLA AHORA

    Una sola cara en la foto: es esa, no hay ambiguedad.
    Varias caras y hay recuadro guardado: la que se pisa con el recuadro.
    Varias caras y no hay recuadro: NINGUNA.

    varias=True (la foto tiene mas de una persona de referencia, 3/11/2026):
    sin recuadro no hay cara, aunque se detecte una sola, y con recuadro la
    cara tiene que pisarlo. Antes daba igual porque se archivaba una cara por
    foto; ahora cada persona se llevaria la misma.

    El ultimo caso es el importante. Perder una referencia cuesta una cara;
    inventarla cuesta una ficha entera y todo lo que esa ficha despues atrae.
    """
    img = c.leer(data)
    if img is None:
        return None
    alto, ancho = img.shape[:2]
    # UN RECUADRO CHICO NO ES REFERENCIA (21/10/2026, decision de Diego). Las
    # caras de 40 a 59 px tienen recuadro y se pueden nombrar, pero no huella:
    # se mira el recuadro guardado sobre esta misma miniatura de 1600, antes de
    # detectar. Sin esto, si la foto tiene una sola cara de 44 px o mas, se
    # archivaba ESA (la de otro) como cara de quien nombraron en la chica.
    if caja is not None and faces.caja_chica(caja, ancho, alto):
        with lock:
            hecho["recuadro_chico"] += 1
        return None
    caras = c.detectar(img)
    if not caras:
        return None
    if len(caras) == 1 and not varias:
        return c.huella(img, caras[0])
    if caja is None:
        with lock:
            hecho["varias_sin_caja"] += 1
        return None
    objetivo = (caja[0] * ancho, caja[1] * alto, caja[2] * ancho, caja[3] * alto)
    mejor, cuanto = None, 0.0
    for cara in caras:
        s = solape(objetivo, [float(v) for v in cara[:4]])
        if s > cuanto:
            mejor, cuanto = cara, s
    if mejor is None or cuanto < IOU_MIN:
        with lock:
            hecho["caja_sin_cara"] += 1
        return None
    return c.huella(img, mejor)


# LA CARA QUE NOMBRA UN EPIGRAFE SE MIDE SOBRE LA FOTO GRANDE (3/11/2026,
# decision de Diego: "que en estos casos no las achique"). Las fotos de grupo
# con epigrafe son de 40 a 60 personas: en la miniatura de 1600 sus caras miden
# 25-43 px y el piso (faces.LADO_EVENTOS) las tiraba todas, aunque la
# identificacion venga con posicion (epigrafe_caras.py). Para estos pares, y
# solo para estos, la foto se baja del tamaño que hace falta para que la cara
# llegue al piso, y el piso se aplica sobre ESA imagen. El piso no cambia.
FUENTE_EPIGRAFE = "epigrafe_cara"
# Cuanto por encima del piso se pide: Drive entrega el tamaño pedido en el lado
# largo y la caja guardada sale de otra deteccion. Medido el 3/11/2026 con
# _medir_epigrafe_original.py en la nota 11 (7016 px, 38 caras con ficha):
#   miniatura 1600: 0,42 MB, caras 25-43 px, 0 de 38 llegan al piso
#   pedida a 4401:  1,64 MB, caras 69-117 px, 38 de 38 con huella
#   original:       6,27 MB, caras 110-187 px, 38 de 38 con huella
# La misma cara a 4401 y en el original se parece 0,942-0,988 (mediana 0,966);
# contra su ficha (10 con referencias humanas) 0,352-0,678 y ninguna se parece
# mas a otra persona de la foto (la otra mas parecida, 0,241 como mucho).
AIRE_EPIGRAFE = 1.15
# El recorte alrededor de la caja guardada, en lados de caja por costado: la
# deteccion se hace ahi y no en la foto entera (en 7.000 px reducida a
# faces.LADO_DETECCION una cara de 111 px mide 13 y YuNet no la ve).
RECORTE_EPIGRAFE = 1.0


def lado_para_epigrafe(cajas_foto, ancho, alto):
    """Lado largo a pedirle a Drive para que las cajas que pueden llegar al
    piso en el original (ancho x alto, orientado) lleguen. (lado, cajas que
    llegan). lado None: ninguna llega ni en el original, no se baja nada."""
    largo = float(max(ancho, alto))
    sirven, pedir = [], 0.0
    for caja in cajas_foto:
        lado_orig = min(caja[2] * ancho, caja[3] * alto)
        if lado_orig + 0.5 < faces.LADO_EVENTOS:
            continue
        sirven.append(caja)
        pedir = max(pedir, faces.LADO_EVENTOS * AIRE_EPIGRAFE * largo / lado_orig)
    if not sirven:
        return None, []
    return int(min(largo, max(1600.0, np.ceil(pedir)))), sirven


def _original(d, pid):
    if time.time() > d.expires:
        d._refresh()
    return urllib.request.Request(
        "https://www.googleapis.com/drive/v3/files/%s?alt=media&supportsAllDrives=true"
        % pid, headers={"Authorization": "Bearer " + d.creds.token})


def _bajar(req):
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), len(data)


def bajar_epigrafe(d, pid, cajas_foto):
    """(imagen, bytes bajados, lado pedido) de una foto de Drive del tamaño
    justo para sus caras de epigrafe (lado_para_epigrafe). (None, 0, None) si
    ninguna llega al piso ni en el original.

    Si hace falta el lado entero, el original (alt=media: =sNNNN con NNNN >= el
    lado largo devuelve el mismo archivo, medido). Si Drive dio las medidas
    sin rotar y la miniatura deja abajo del piso una caja que en el original
    llega, se baja el original: una segunda bajada solo en ese caso."""
    m = d.get("files/" + pid,
              fields="thumbnailLink,imageMediaMetadata(width,height,rotation)")
    md = m.get("imageMediaMetadata") or {}
    ancho, alto = int(md.get("width") or 0), int(md.get("height") or 0)
    if not ancho or not alto:
        raise ValueError("sin medidas en Drive")
    if int(md.get("rotation") or 0) in (1, 3, 90, 270):
        ancho, alto = alto, ancho
    lado, sirven = lado_para_epigrafe(cajas_foto, ancho, alto)
    if lado is None:
        return None, 0, None
    largo = max(ancho, alto)
    if lado < largo and m.get("thumbnailLink"):
        img, n = _bajar(urllib.request.Request(
            m["thumbnailLink"].replace("=s220", "=s%d" % lado)))
        if img is not None:
            h, w = img.shape[:2]
            if not any(faces.caja_chica(cj, w, h) for cj in sirven):
                return img, n, lado
        img2, n2 = _bajar(_original(d, pid))
        return img2, n + n2, largo
    img, n = _bajar(_original(d, pid))
    return img, n, largo


def cara_en_caja(c, img, caja):
    """La huella de la cara que pisa la caja guardada (fracciones), detectando
    sobre un recorte alrededor de la caja (RECORTE_EPIGRAFE). None si no hay
    una cara con IOU_MIN. El piso se mira antes, sobre esta misma imagen."""
    alto, ancho = img.shape[:2]
    x, y, w, h = caja[0] * ancho, caja[1] * alto, caja[2] * ancho, caja[3] * alto
    mx, my = w * RECORTE_EPIGRAFE, h * RECORTE_EPIGRAFE
    x0, y0 = max(0, int(x - mx)), max(0, int(y - my))
    x1, y1 = min(ancho, int(np.ceil(x + w + mx))), min(alto, int(np.ceil(y + h + my)))
    recorte = np.ascontiguousarray(img[y0:y1, x0:x1])
    objetivo = (x - x0, y - y0, w, h)
    mejor, cuanto = None, 0.0
    for cara in c.detectar(recorte):
        s = solape(objetivo, [float(v) for v in cara[:4]])
        if s > cuanto:
            mejor, cuanto = cara, s
    if mejor is None or cuanto < IOU_MIN:
        return None
    return c.huella(recorte, mejor)


def caras_de_epigrafe(c, img, pares, cajas, hecho, lock):
    """[(fila, huella)] de los pares 'epigrafe_cara' de UNA foto, bajada una
    vez (bajar_epigrafe). Piso de faces.LADO_EVENTOS sobre esta imagen."""
    alto, ancho = img.shape[:2]
    out = []
    for fila in pares:
        caja = cajas.get((fila[0], int(fila[1])))
        if caja is None:
            with lock:
                hecho["varias_sin_caja"] += 1
            continue
        if faces.caja_chica(caja, ancho, alto):
            with lock:
                hecho["recuadro_chico"] += 1
            continue
        v = cara_en_caja(c, img, caja)
        if v is None:
            with lock:
                hecho["caja_sin_cara"] += 1
            continue
        out.append((fila, v))
    return out


def catalogo(cli):
    """(photo_id, person_id, año) de cada retrato con identidad confirmada."""
    print("Leyendo identificaciones confirmadas…")
    pp = cli.select("photo_people",
                    select="photo_id,person_id,source,"
                           "photos(folder_id,source,url_host,url_core,external_realm)")
    fol = cli.select("folders", select="id,year,campus")
    anio = {f["id"]: f.get("year") for f in fol}
    campus = {f["id"]: f.get("campus") for f in fol}

    # Zenfolio tambien: sus huellas salen del cache de faces_huellas.py, y si
    # falta alguna se baja con zenfolio.py. Quedaba afuera de aca desde antes
    # de que ese modulo existiera, y el efecto era que Quilmes nunca podia
    # reciclar sus propias identificaciones como caras de referencia.
    from collections import Counter
    # SOLO LO QUE CONFIRMO UNA PERSONA (5/10/2026). Lo que nombra el sistema
    # -auto_grupo, face_auto, cluster, filename_face...- no es referencia: si
    # esta mal, atrae mas caras ajenas que tambien se nombran solas. Aca se
    # archivan las humanas y TODOS los carnets; cual carnet vale (solo el de
    # quien no tiene ninguna cara humana) se decide al guardar, con las caras
    # ya ubicadas: una identificacion humana en una foto de acto sin recuadro
    # no da cara, y esa persona tiene que poder quedarse con su carnet.
    sirve = set(faces.FUENTES_HUMANAS) | set(faces.FUENTES_CARNET)
    todo = []
    for r in pp:
        if (r.get("source") or "") not in sirve:
            continue
        ph = r.get("photos") or {}
        if ph.get("source") not in ("drive", "zenfolio"):
            continue
        fid = ph.get("folder_id")
        todo.append((r["photo_id"], r["person_id"], anio.get(fid), campus.get(fid),
                     dict(ph, _fuente=r.get("source"))))

    # La marca de agua de Zenfolio cae encima de la cara y la arruina como
    # referencia. Medido sobre los 622 retratos que estan en las dos fuentes,
    # la misma foto limpia y marcada se parecen 0,857 de mediana, y 73 de esos
    # 622 quedan por debajo de 0,50, que es el umbral con el que se aprueba: el
    # motor no reconoce que son la misma persona.
    #
    # Asi que una foto marcada solo se usa cuando esa persona no tiene ninguna
    # limpia. Es mejor que nada, pero peor que cualquier otra cosa.
    limpias = {t[1] for t in todo if t[4].get("source") == "drive"}
    out = [t for t in todo
           if t[4].get("source") != "zenfolio" or t[1] not in limpias]
    tirados = len(todo) - len(out)
    print("  identificaciones: %d   utilizables: %d  (%s)"
          % (len(pp), len(out), ", ".join("%s %d" % (k, v) for k, v in
                                          sorted(Counter(t[4].get("source")
                                                         for t in out).items()))))
    if tirados:
        print("  con marca de agua descartadas por tener una limpia: %d" % tirados)
    return out


def de_caras_todas(ruta, pend, cajas):
    """{(foto, persona): (vec, persona, anio, campus)} de las pendientes con
    recuadro cuya cara ya esta medida en _caras_todas (misma foto, mismo
    recuadro a 4 decimales: es la misma deteccion sobre la misma miniatura).
    np.load una vez; los vectores se leen solo si hay algo que buscar."""
    quiero = {}
    for f in pend:
        cj = cajas.get((f[0], int(f[1])))
        if cj is not None:
            quiero[(f[0], round(cj[0], 4), round(cj[1], 4))] = f
    if not quiero or not os.path.exists(ruta):
        return {}
    fotos = {k[0] for k in quiero}
    with np.load(ruta, allow_pickle=True) as z:
        F = z["photo_ids"].astype(str)
        C = z["cajas"].astype(np.float32)
        donde = {}
        for i in np.flatnonzero(np.isin(F, list(fotos))):
            k = (F[i], round(float(C[i][0]), 4), round(float(C[i][1]), 4))
            if k in quiero:
                donde[k] = int(i)
        if not donde:
            return {}
        orden = sorted(donde.values())
        V = z["vecs"][orden]
    fila = {i: n for n, i in enumerate(orden)}
    out = {}
    for k, i in donde.items():
        f = quiero[k]
        out[(f[0], int(f[1]))] = (V[fila[i]], int(f[1]), f[2], f[3])
    return out


def coherentes(nuevas, ya, por_foto):
    """Las caras nuevas de fotos con varias personas que se parecen mas a su
    ficha que a las otras personas de la foto (regla 10: nada entra como
    referencia sin mirar las otras caras de la ficha). Sin otra referencia de
    su ficha, o sin otras de las demas, pasa: no hay con que comparar. No hay
    corte: es "a quien se parece mas", no "cuanto".

    Medido el 3/11/2026 sobre las 26.820 recuperables con referencias: su
    ficha gana en 26.800 (99,9%); el control, las que ya estaban, 18.207 de
    18.209."""
    if not nuevas:
        return {}
    multi = {k for k in nuevas if len(por_foto.get(k[0], ())) > 1}
    if not multi:
        return dict(nuevas)
    quienes = {p for k in multi for p in por_foto[k[0]]}
    de = {}
    for (f, p), v in list(ya.items()) + list(nuevas.items()):
        if p in quienes:
            de.setdefault(p, []).append((f, v[0]))
    mats = {}
    for p, lista in de.items():
        R = np.asarray([x for _g, x in lista], dtype=np.float32)
        R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-9)
        mats[p] = (np.asarray([g for g, _x in lista]), R)

    def mejor(p, f, v):
        if p not in mats:
            return None
        g, R = mats[p]
        m = g != f
        if not m.any():
            return None
        return float(np.max(R[m] @ v))

    out, fuera = {}, 0
    for k, val in nuevas.items():
        if k in multi:
            v = np.asarray(val[0], dtype=np.float32)
            v /= max(float(np.linalg.norm(v)), 1e-9)
            propia = mejor(k[1], k[0], v)
            otras = [mejor(q, k[0], v) for q in por_foto[k[0]] if q != k[1]]
            otras = [x for x in otras if x is not None]
            if propia is not None and otras and max(otras) > propia:
                fuera += 1
                continue
        out[k] = val
    if fuera:
        print("Caras nuevas que se parecen mas a otra persona de su foto, no son "
              "referencia: %d" % fuera)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--ensayo", action="store_true",
                    help="decir que quedaria, sin bajar fotos ni escribir el .npz")
    ap.add_argument("--rehacer-con-caja", action="store_true",
                    help="recalcular las referencias que salieron de una foto "
                         "de evento: son las que se archivaron tomando la cara "
                         "mas grande de la foto en vez de la de la persona")
    ap.add_argument("--npz-origen", help="(solo con --ensayo) leer otro _referencias "
                                         "en vez del de build/, p. ej. el del pulso")
    ap.add_argument("--caras-todas", help="(solo con --ensayo) otro _caras_todas")
    a = ap.parse_args()
    if (a.npz_origen or a.caras_todas) and not a.ensayo:
        ap.error("--npz-origen y --caras-todas son solo para --ensayo")
    origen = a.npz_origen or DEST
    todas = a.caras_todas or TODAS

    cli = sb.SB()
    filas = catalogo(cli)
    # Una cara en disputa abierta no se archiva como referencia ni cuenta como
    # "caras humanas" de su ficha (faces.disputas_abiertas, 15/10/2026)
    disputadas = faces.disputas_abiertas(cli)
    if disputadas:
        antes = len(filas)
        filas = [f for f in filas if (f[0], int(f[1])) not in disputadas]
        print("Caras en disputa abierta que no son referencia: %d" % (antes - len(filas)))
    if a.limite:
        filas = filas[:a.limite]

    # Reanudar: no volver a bajar lo que ya se proceso. La clave es la pareja
    # (foto, persona): ver el docstring. Un .npz de antes del 3/11/2026 se lee
    # igual, cada entrada suya ya es una pareja.
    ya = {}
    if os.path.exists(origen):
        z = np.load(origen, allow_pickle=True)
        # cada acceso z["campo"] descomprime el array entero: se leen una vez
        ids, vecs = z["photo_ids"], z["vecs"]
        pers, anios, camp = z["personas"], z["anios"], z["campus"]
        for i, pid in enumerate(ids):
            ya[(str(pid), int(pers[i]))] = (vecs[i], int(pers[i]), anios[i], str(camp[i]))
        print("Ya procesados de antes: %d" % len(ya))

    # LA PODA. De una cara de referencia no se sale.
    #
    # Este diccionario se sembraba con el npz anterior y no se limpiaba nunca:
    # lo unico que hacia era no volver a bajar lo ya bajado. El efecto es que
    # una cara que entro mal se queda para siempre, aunque despues alguien
    # conteste "No es" o la saque una auditoria.
    #
    # Se vio la noche del 11/9/2026: caras_mal_archivadas.py saco 481 caras
    # ajenas de photo_people a las 18:58 y las referencias las seguian teniendo
    # a las 21:30. Rehacer el archivo tampoco las habria sacado, porque salian
    # de aca y no de la base. Persona V tenia 226 caras de referencia
    # contra 135 identificaciones vivas: la diferencia era pura acumulacion.
    #
    # Ahora sobrevive solo lo que sigue confirmado HOY, y con la misma persona:
    # si la foto se reasigno, la huella vieja quedo colgada de un nombre que ya
    # no es el suyo.
    vigentes = {(f[0], int(f[1])) for f in filas}
    antes = len(ya)
    ya = {k: v for k, v in ya.items() if k in vigentes}
    if antes != len(ya):
        print("Podadas: %d caras que ya no estan confirmadas" % (antes - len(ya)))

    cajas = cajas_guardadas(cli)
    print("Recuadros guardados: %d" % len(cajas))

    # Rehacer las que salieron de una foto de evento.
    #
    # La poda saca lo que ya no esta confirmado, pero no arregla lo que sigue
    # confirmado y se calculo mal. Esas son las que tienen recuadro guardado:
    # tener recuadro significa que la cara salio de una foto de acto, y hasta
    # hoy de esas fotos se archivaba la cara mas grande. Se las tira del cache
    # para que se vuelvan a bajar y ahora si se recorte la cara que indica el
    # recuadro.
    #
    # No se hace siempre porque cuesta bajar una imagen por cada una. Los
    # retratos de ficha no hace falta tocarlos: ahi la cara mas grande es la
    # de la persona, que es lo que decia el comentario original.
    if getattr(a, "rehacer_con_caja", False):
        antes = len(ya)
        ya = {k: v for k, v in ya.items() if k not in cajas}
        print("A rehacer por venir de una foto de evento: %d" % (antes - len(ya)))

    pend = [f for f in filas if (f[0], int(f[1])) not in ya]
    # Cuantas personas de referencia tiene cada foto: con mas de una, la cara
    # mas grande no es de nadie en particular (cara_de, varias=True).
    por_foto = {}
    for f in filas:
        por_foto.setdefault(f[0], set()).add(int(f[1]))
    # Un carnet de quien ya tiene una cara humana archivada no se va a guardar
    # (ver abajo): no se procesa. Sin esto se volveria a intentar en cada
    # corrida, y los de Zenfolio abririan sesion cada vez.
    con_humana = {int(f[1]) for f in filas
                  if (f[0], int(f[1])) in ya
                  and f[4].get("_fuente") in faces.FUENTES_HUMANAS}
    antes = len(pend)
    pend = [f for f in pend if f[4].get("_fuente") in faces.FUENTES_HUMANAS
            or int(f[1]) not in con_humana]
    if antes != len(pend):
        print("Carnets salteados porque su persona ya tiene caras humanas: %d" % (antes - len(pend)))
    # Las nombradas sobre un recuadro chico (faces.CACHE_CHICAS) no se bajan:
    # no van a ser referencia (ver cara_de). Sin esto se bajarian en cada
    # corrida para volver a descartarlas.
    chicas = faces.cajas_chicas({f[0] for f in pend})
    if chicas:
        def es_chica(f):
            # las de epigrafe se miden sobre la foto grande (bajar_epigrafe):
            # chica en la miniatura de 1600 no quiere decir nada para ellas
            if f[4].get("_fuente") == FUENTE_EPIGRAFE:
                return False
            cj = cajas.get((f[0], int(f[1])))
            return cj is not None and (f[0], round(cj[0], 4), round(cj[1], 4)) in chicas
        antes = len(pend)
        pend = [f for f in pend if not es_chica(f)]
        if antes != len(pend):
            print("Nombradas sobre un recuadro chico (%d-%d px), no son referencia: %d"
                  % (faces.LADO_RECUADRO, faces.LADO_EVENTOS - 1, antes - len(pend)))
    print("A procesar ahora: %d" % len(pend))

    # Las que tienen recuadro, primero de _caras_todas: misma miniatura, mismo
    # detector, mismo recuadro. No se baja nada (ver el docstring).
    nuevas = de_caras_todas(todas, pend, cajas)
    if nuevas:
        pend = [f for f in pend if (f[0], int(f[1])) not in nuevas]
        print("Desde _caras_todas (sin bajar nada): %d; quedan para bajar: %d"
              % (len(nuevas), len(pend)))
    # Las de epigrafe van aparte: una bajada por foto, del tamaño que haga
    # falta (FUENTE_EPIGRAFE). Las de Zenfolio siguen por el camino de siempre.
    epi = {}
    for f in pend:
        if f[4].get("_fuente") == FUENTE_EPIGRAFE and f[4].get("source") == "drive":
            epi.setdefault(f[0], []).append(f)
    if epi:
        pend = [f for f in pend if not (f[4].get("_fuente") == FUENTE_EPIGRAFE
                                        and f[4].get("source") == "drive")]
        print("Caras de epigrafe a medir sobre la foto grande: %d en %d fotos"
              % (sum(len(v) for v in epi.values()), len(epi)))
    if a.ensayo:
        # lo que quedaria guardado con lo del cache y lo de _caras_todas: no
        # baja ninguna foto ni escribe
        prueba = dict(ya)
        prueba.update(coherentes(nuevas, prueba, por_foto))
        fuentes = {}
        for f in filas:
            fuentes.setdefault((f[0], int(f[1])), set()).add(f[4].get("_fuente") or "")
        ids = list(prueba)
        ok, quien = faces.filtrar_referencias([k[0] for k in ids],
                                              [prueba[k][1] for k in ids], fuentes,
                                              disputadas)
        print("Ensayo: con el cache y _caras_todas quedarian %d caras de %d personas "
              "(antes %d; %d carnets de gente con caras humanas afuera; %d por bajar). "
              "No se escribio nada."
              % (len(ok), len(set(quien)), len(ya), len(ids) - len(ok),
                 len(pend) + sum(len(v) for v in epi.values())))
        return

    # Las huellas de los retratos ya estan calculadas: para esas no hay que
    # bajar nada, ni de Drive ni de Zenfolio. Reconstruir las referencias pasa
    # a costar segundos, que es lo que permite rehacerlas cada vez que el
    # juego confirma nombres nuevos.
    cache = {}
    if os.path.exists(HUELLAS):
        z = np.load(HUELLAS, allow_pickle=True)
        ids_c, vecs_c = z["photo_ids"], z["vecs"]
        cache = {str(pid): vecs_c[i] for i, pid in enumerate(ids_c)}
        print("Huellas en cache: %d" % len(cache))

    # Zenfolio solo se abre si queda algo suyo fuera del cache: pide login.
    zf = None
    realms = {f[4].get("external_realm") for f in pend
              if f[4].get("source") == "zenfolio" and f[0] not in cache}
    realms.discard(None)
    if realms:
        print("Abriendo Zenfolio (%d realms)…" % len(realms))
        zf = zenfolio.Zenfolio()
        zf.abrir(realms)

    d = Drive(SA_PATH)
    lock = threading.Lock()
    hecho = {"n": 0, "sin_cara": 0, "error": 0, "de_cache": 0,
             "varias_sin_caja": 0, "caja_sin_cara": 0, "recuadro_chico": 0,
             "epigrafe_fotos": 0, "epigrafe_bajado": 0}

    # Los modelos de OpenCV NO son thread-safe: compartir un detector entre
    # hilos corrompe su estado y tira "buf.shape() == m.shape()". Cada hilo
    # carga el suyo; son ~40 MB cada uno, y el detector es lo que aprovecha
    # los varios núcleos.
    local = threading.local()

    def mis_caras():
        if not hasattr(local, "c"):
            local.c = faces.Caras()
        return local.c

    def procesar(fila):
        pid, persona, anio, camp, meta = fila
        caja = cajas.get((pid, int(persona)))
        varias = len(por_foto.get(pid, ())) > 1
        # El cache de retratos guarda la cara mas grande de cada foto, que es
        # la buena en una ficha y cualquiera en un acto. Si hay recuadro, esta
        # foto es una foto de evento: se ignora el cache y se va a buscar esa
        # cara y no otra. Con varias personas de referencia en la foto,
        # tampoco: la cara mas grande seria de todas ellas.
        if caja is None and varias:
            with lock:
                hecho["varias_sin_caja"] += 1
            return None
        if caja is None:
            v = cache.get(pid)
            if v is not None:
                with lock:
                    hecho["n"] += 1
                    hecho["de_cache"] += 1
                return (pid, v, persona, anio, camp)
        try:
            if meta.get("source") == "zenfolio":
                data = zf.bajar(meta.get("url_host"), meta.get("url_core")) if zf else None
                if not data:
                    raise ValueError("imagen protegida o sin sesion")
            else:
                link = d.get("files/" + pid, fields="thumbnailLink").get("thumbnailLink")
                if not link:
                    return None
                # =s1600: la miniatura por defecto es de 220 px y una cara ahí
                # no tiene píxeles suficientes para reconocerla
                with urllib.request.urlopen(link.replace("=s220", "=s1600"), timeout=60) as r:
                    data = r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError, ValueError):
            with lock:
                hecho["error"] += 1
            return None

        v = cara_de(mis_caras(), data, caja, hecho, lock, varias=varias)
        with lock:
            hecho["n"] += 1
            if hecho["n"] % 100 == 0:
                print("  %d/%d" % (hecho["n"], len(pend)))
            if v is None:
                hecho["sin_cara"] += 1
        return None if v is None else (pid, v, persona, anio, camp)

    def procesar_epigrafe(pares):
        """Las caras de epigrafe de UNA foto, con una sola bajada."""
        pid = pares[0][0]
        try:
            img, bajado, _lado = bajar_epigrafe(
                d, pid, [cajas[(f[0], int(f[1]))] for f in pares
                         if (f[0], int(f[1])) in cajas])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError, ValueError) as e:
            with lock:
                hecho["error"] += len(pares)
            print("  epigrafe %s: no se pudo bajar (%s)" % (pid, e))
            return []
        if img is None and not bajado:
            # ninguna llega al piso ni en el original: no se baja nada
            with lock:
                hecho["recuadro_chico"] += len(pares)
            return []
        try:
            if img is None:
                raise ValueError("no es una imagen")
            r = caras_de_epigrafe(mis_caras(), img, pares, cajas, hecho, lock)
        except Exception as e:  # una foto rota no tira el lote
            with lock:
                hecho["error"] += len(pares)
            print("  epigrafe %s: %s" % (pid, e))
            return []
        with lock:
            hecho["n"] += len(pares)
            hecho["epigrafe_bajado"] += bajado
            hecho["epigrafe_fotos"] += 1
            hecho["sin_cara"] += len(pares) - len(r)
        return [(f[0], v, f[1], f[2], f[3]) for f, v in r]

    # De a tandas, refrescando el token entre una y otra desde el hilo
    # principal: google-auth tampoco es thread-safe, y un refresh disparado
    # por dos hilos a la vez deja la credencial en un estado raro.
    resultados = []
    TANDA = 400
    try:
        for i in range(0, len(pend), TANDA):
            d._refresh()
            with ThreadPoolExecutor(max_workers=HILOS) as ex:
                for r in ex.map(procesar, pend[i:i + TANDA]):
                    if r:
                        resultados.append(r)
        # una tarea por foto: las 38 caras de un epigrafe, una sola bajada
        grupos = list(epi.values())
        for i in range(0, len(grupos), TANDA):
            d._refresh()
            with ThreadPoolExecutor(max_workers=HILOS) as ex:
                for rs in ex.map(procesar_epigrafe, grupos[i:i + TANDA]):
                    resultados.extend(rs)
    except KeyboardInterrupt:
        print("\nInterrumpido: se guarda lo que haya.")

    for pid, vec, persona, anio, camp in resultados:
        nuevas[(pid, int(persona))] = (vec, int(persona), anio, camp)
    ya.update(coherentes(nuevas, ya, por_foto))

    # Los carnets de quien ya tiene caras humanas se van (faces.FUENTES_CARNET).
    # Lo leen sin pasar por cargar_refs() el agrupamiento, grupos_parecidos,
    # recuadros, fichas y carnets: la regla tiene que estar en el archivo.
    fuentes = {}
    for f in filas:
        fuentes.setdefault((f[0], int(f[1])), set()).add(f[4].get("_fuente") or "")
    ids = list(ya)
    ok, _ = faces.filtrar_referencias([k[0] for k in ids], [ya[k][1] for k in ids],
                                      fuentes, disputadas)
    if len(ok) != len(ids):
        print("Carnets que no se guardan porque su persona tiene caras humanas: %d"
              % (len(ids) - len(ok)))
    ya = {ids[i]: ya[ids[i]] for i in ok}

    if not ya:
        sys.exit("No se pudo armar ninguna referencia.")

    ids = list(ya)
    # A un temporal y despues se reemplaza: un paso cortado por TOPE_H a mitad
    # de escritura dejaba un .npz roto que el workflow guardaba igual (Trampas -
    # motor, 30/10/2026). El temporal termina en .tmp.npz: no entra al estado.
    tmp = DEST[:-len(".npz")] + ".tmp.npz"
    np.savez(tmp,
             photo_ids=np.array([k[0] for k in ids]),
             vecs=np.array([ya[k][0] for k in ids], dtype=np.float32),
             personas=np.array([ya[k][1] for k in ids]),
             anios=np.array([ya[k][2] if ya[k][2] is not None else -1 for k in ids]),
             campus=np.array([ya[k][3] if ya[k][3] else "" for k in ids]))
    os.replace(tmp, DEST)

    personas = {ya[k][1] for k in ids}
    print("\nReferencias: %d caras de %d personas (%d fotos)"
          % (len(ids), len(personas), len({k[0] for k in ids})))
    print("Desde el cache de huellas: %d (sin bajar nada)" % hecho["de_cache"])
    print("Sin cara detectada: %d   errores de descarga: %d"
          % (hecho["sin_cara"], hecho["error"]))
    # Las que NO se archivaron a proposito. Conviene mirarlas: si este numero
    # es grande, hay muchas identificaciones confirmadas sobre fotos de grupo
    # sin recuadro, y esas son justo las que antes envenenaban la ficha.
    print("Descartadas por no poder saber cual es la cara: %d sin recuadro, "
          "%d con recuadro que no coincide"
          % (hecho["varias_sin_caja"], hecho["caja_sin_cara"]))
    if hecho["recuadro_chico"]:
        print("Descartadas por recuadro chico (menos de %d px, sin huella): %d"
              % (faces.LADO_EVENTOS, hecho["recuadro_chico"]))
    if hecho["epigrafe_fotos"]:
        print("Fotos de epigrafe bajadas en grande: %d (%.1f MB)"
              % (hecho["epigrafe_fotos"], hecho["epigrafe_bajado"] / 1e6))
    print("Escrito: %s  (%.1f MB)" % (DEST, os.path.getsize(DEST) / 1e6))


if __name__ == "__main__":
    main()
