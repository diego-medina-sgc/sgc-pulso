# -*- coding: utf-8 -*-
"""
Saca caras y nombres de las fotos de curso de un anuario en PDF.

A diferencia de los anuarios viejos, que son fotos de una pagina y hay que
transcribir a mano, el PDF trae el texto adentro. Y las fotos de curso listan
los nombres por fila:

    Top Row (L To R): Persona B; Persona C; ...
    Second Row: Garcia Persona D; ...

Eso alcanza para mapear cara -> nombre sin escribir un nombre a mano: se
detectan las caras, se las agrupa en filas por su altura, se ordena cada fila
de izquierda a derecha, y se emparejan con la lista.

    python anuario_pdf.py --pdf x.pdf --pagina 141     # una, para mirarla
    python anuario_pdf.py --pdf x.pdf --todas          # las que tengan filas

EL CONTROL QUE HACE QUE ESTO SEA CONFIABLE

Si el detector encuentra 43 caras y el texto tiene 44 nombres, no hay forma de
saber cual falta, y emparejar corrido etiqueta mal a todos los que siguen. Por
eso la comparacion se hace FILA POR FILA: si alguna no coincide, la pagina
entera se marca para revisar y no se escribe nada.

Ademas se deja una imagen numerada al lado del json, igual que con los
anuarios de papel, para poder verificar a ojo.
"""
import argparse
import os
import re
import sys
import unicodedata

import cv2
import numpy as np

import faces

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "_anuarios_pdf")
# Correcciones al pie de foto: el anuario tiene erratas de puntuacion que
# pegan dos personas en una o parten una en dos. Ver el encabezado del archivo.
CORRECCIONES = os.path.join(HERE, "_anuario_correcciones.json")

# Una pagina A4 a 300 dpi son 2.500 px de ancho; con 70 chicos en la foto cada
# cara queda en 30 px y el detector, que reduce a 800, no ve nada. Se detecta
# sobre la pagina grande y sin reducir.
LADO_DET = 1600
CONF = 0.6
# Una cara mas chica que esto en la pagina renderizada no da para reconocer.
LADO_MIN = 18

# "stading" y "standig" no son error de tipeo mio: asi salen en el anuario.
# Sin contemplarlos, esa fila entera no se detecta y sus veinte nombres se
# pierden en silencio, que es peor que un descuadre ruidoso.
FILA = re.compile(
    r"((?:top|second|third|fourth|fifth|standing|stading|standig|front|back"
    r"|bottom|middle)\s+row"
    r"|seated|sentados)\s*(?:\(l\s*to\s*r\))?\s*:", re.I)


def texto_pagina(pdf, n):
    import logging
    logging.disable(logging.CRITICAL)
    import pypdf
    return pypdf.PdfReader(pdf).pages[n - 1].extract_text() or ""


def render(pdf, n, dpi=300):
    import pymupdf
    doc = pymupdf.open(pdf)
    pix = doc[n - 1].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)


def detectar(img):
    """Caras de la pagina, sin el recorte a 800 px de faces.detectar()."""
    h, w = img.shape[:2]
    esc = float(LADO_DET) / max(h, w)
    chica = cv2.resize(img, (int(w * esc), int(h * esc))) if esc < 1 else img
    det = cv2.FaceDetectorYN.create(
        os.path.join(faces.MODELS, "yunet.onnx"), "",
        (chica.shape[1], chica.shape[0]), CONF, 0.3, 5000)
    det.setInputSize((chica.shape[1], chica.shape[0]))
    _, cs = det.detect(chica)
    if cs is None:
        return []
    if esc < 1:
        cs = cs.copy()
        cs[:, :14] /= esc
    return [f for f in cs if min(f[2], f[3]) >= LADO_MIN]


def enderezar(img):
    """La orientacion en la que el detector encuentra mas caras.

    Las paginas apaisadas del anuario vienen giradas 90 grados, y de costado
    el detector no ve una sola cara. Probar las tres y quedarse con la mejor
    sale mas barato que confiar en la rotacion declarada del PDF, que no
    siempre esta.
    """
    mejor, mejor_n, mejor_rot = img, len(detectar(img)), "0"
    for rot, nombre in ((cv2.ROTATE_90_CLOCKWISE, "90 horario"),
                        (cv2.ROTATE_90_COUNTERCLOCKWISE, "90 antihorario")):
        g = cv2.rotate(img, rot)
        n = len(detectar(g))
        if n > mejor_n:
            mejor, mejor_n, mejor_rot = g, n, nombre
    return mejor, mejor_n, mejor_rot


def nombre_persona(txt):
    """'Persona B' -> 'Persona E'.

    Devuelve (nombre, sospechoso). Sospechoso es el que no trae coma en una
    lista donde todos los demas la traen: casi siempre es un punto y coma mal
    puesto que parte una persona en dos. En la pagina 141, "Apellido Apellido,
    Nombre; Apellido; Nombre" son dos personas y no tres, pero contadas como
    tres el total cuadraba igual con las caras y el error pasaba de largo.
    """
    t = " ".join(txt.split()).strip(" ;")
    # el punto cierra la lista; lo que sigue es el titulo de la pagina o el
    # nombre de la casa, que se colaron al extraer el texto
    if "." in t:
        t = t.split(".")[0].strip()
    if not t or t.upper() in ("NN", "N N"):
        return None, False
    if "," in t:
        ape, _, nom = t.partition(",")
        ape, nom = ape.strip(), nom.strip()
        if ape and nom:
            return "%s %s" % (nom, ape), False
    return (t or None), True


def correcciones(pagina):
    import io
    import json
    if not os.path.exists(CORRECCIONES):
        return {}
    with io.open(CORRECCIONES, encoding="utf-8") as fh:
        d = json.load(fh)
    return d.get(str(pagina), {})


def filas_del_texto(t, pagina=None):
    """[(etiqueta, [nombres], [sospechosos])] en el orden en que aparecen."""
    partes = FILA.split(t)
    out = []
    for i in range(1, len(partes), 2):
        etiqueta = " ".join(partes[i].split())
        cuerpo = partes[i + 1]
        # el cuerpo termina donde arranca otra cosa que no es la lista
        cuerpo = cuerpo.split("\n\n")[0]
        # Los None se quedan: son los "NN" del anuario, chicos que la foto
        # tiene pero el pie no nombra. Sacarlos correria un lugar a todos los
        # que siguen y quedarian etiquetados con el nombre del de al lado.
        pares = [nombre_persona(x) for x in cuerpo.replace("\n", " ").split(";")]
        while pares and pares[-1][0] is None:
            pares.pop()            # la basura del final del parrafo, esa si sobra
        nombres = [p[0] for p in pares]
        raros = [p[0] for p in pares if p[1]]
        if any(nombres):
            out.append((etiqueta, nombres, raros))

    # lo corregido a mano pisa lo que salio del PDF
    if pagina is not None:
        arreglos = correcciones(pagina)
        if arreglos:
            out = [(et, arreglos[et], []) if et in arreglos else (et, n, r)
                   for et, n, r in out]
    return out


def agrupar_en_filas(caras, tamanos, vueltas=6):
    """Parte las caras en filas del tamaño que dice el texto.

    Cortar por altura no alcanza. En una foto de curso las filas se solapan:
    los de las puntas quedan mas bajos que los del centro de la fila de
    atras, porque la gente se acomoda en arco y el piso no es plano. Ordenando
    por altura y cortando por la cuenta, el de la punta de una fila se cuela
    en la otra y a partir de ahi todos los nombres quedan corridos.

    Asi que cada fila se trata como lo que es, una linea con pendiente: se
    arranca del corte por altura, se ajusta una recta a cada fila con los
    puntos que le tocaron, y se reasignan las caras a la recta que les quede
    mas cerca, respetando los tamaños. Un par de vueltas y se estabiliza.

    Solo tiene sentido si el total coincide, y eso lo verifica quien llama.
    """
    if not caras or not tamanos:
        return []
    cx = np.array([c[0] + c[2] / 2.0 for c in caras])
    cy = np.array([c[1] + c[3] / 2.0 for c in caras])

    orden = sorted(range(len(caras)), key=lambda i: cy[i])
    filas, ini = [], 0
    for n in tamanos:
        filas.append(orden[ini:ini + n])
        ini += n

    for _ in range(vueltas):
        # una recta por fila; con menos de tres puntos, la altura promedio
        rectas = []
        for idx in filas:
            if len(idx) >= 3:
                a, b = np.polyfit(cx[idx], cy[idx], 1)
            else:
                a, b = 0.0, float(np.mean(cy[idx])) if idx else 0.0
            rectas.append((a, b))

        # cada cara a la recta mas cercana, por orden de que tan clara es la
        # eleccion: primero las que no tienen duda, y las dudosas se acomodan
        # en el cupo que queda
        cand = []
        for i in range(len(caras)):
            d = [abs(cy[i] - (a * cx[i] + b)) for a, b in rectas]
            o = sorted(range(len(rectas)), key=lambda k: d[k])
            margen = d[o[1]] - d[o[0]] if len(o) > 1 else 1e9
            cand.append((-margen, i, o, d))
        cand.sort()

        nuevas = [[] for _ in tamanos]
        for _, i, o, d in cand:
            for k in o:
                if len(nuevas[k]) < tamanos[k]:
                    nuevas[k].append(i)
                    break
        if nuevas == filas:
            break
        filas = nuevas

    return [sorted(idx, key=lambda i: cx[i]) for idx in filas]


def numerar(img, caras, orden):
    out = img.copy()
    for n, i in enumerate(orden, 1):
        x, y, w, h = [int(v) for v in caras[i][:4]]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cv2.putText(out, str(n), (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 200, 255), 2, cv2.LINE_AA)
    return out


def tira(img, caras, idx, nombres, alto=190):
    """Una fila recortada y puesta en linea, con el nombre debajo de cada cara.

    Revisar el emparejamiento sobre la pagina entera no se puede: son ochenta
    caras de treinta pixeles. Recortada fila por fila y con el nombre debajo,
    un desfasaje se ve de una: aparece la cara de un varon con nombre de mujer,
    o la ultima queda sin nadie.
    """
    celdas = []
    for k, i in enumerate(idx):
        x, y, w, h = [int(v) for v in caras[i][:4]]
        m = int(max(w, h) * 0.45)
        rec = img[max(y - m, 0):min(y + h + m, img.shape[0]),
                  max(x - m, 0):min(x + w + m, img.shape[1])]
        if rec.size == 0:
            rec = np.full((10, 10, 3), 220, np.uint8)
        lado = alto - 52
        rec = cv2.resize(rec, (lado, lado))
        cel = np.full((alto, lado, 3), 250, np.uint8)
        cel[:lado] = rec
        txt = nombres[k] if k < len(nombres) else "?"
        cv2.putText(cel, str(k + 1), (3, lado + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 90, 200), 2, cv2.LINE_AA)
        for j, parte in enumerate((txt or "SIN NOMBRE").split()[:2]):
            cv2.putText(cel, parte[:13], (22, lado + 16 + j * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        celdas.append(cel)
    if not celdas:
        return None
    return np.hstack(celdas)


def procesar(pdf, n, escribir):
    t = texto_pagina(pdf, n)
    filas_txt = filas_del_texto(t, n)
    if not filas_txt:
        return None
    img, _, rot = enderezar(render(pdf, n))
    caras = detectar(img)
    tamanos = [len(f) for _, f, _ in filas_txt]
    raros = [x for _, _, r in filas_txt for x in r]
    # Un nombre sin coma en una lista que usa coma es casi siempre un
    # separador mal puesto, y eso corre a todos los que siguen. La pagina
    # queda para mirar aunque los numeros cierren.
    cuadra = len(caras) == sum(tamanos) and not raros

    # Las filas se arman siempre, aunque falten o sobren caras: la tira es
    # justamente la herramienta para revisar las paginas que NO cuadran, y sin
    # armarlas no habria nada que mirar. Si sobran caras van al final en una
    # fila aparte; si faltan, la ultima fila queda corta y se ve dónde.
    tam_uso = list(tamanos)
    sobran = len(caras) - sum(tamanos)
    if sobran > 0:
        tam_uso.append(sobran)
    elif sobran < 0:
        # recortar de atras para adelante hasta que entre
        falta = -sobran
        for k in range(len(tam_uso) - 1, -1, -1):
            quita = min(falta, tam_uso[k])
            tam_uso[k] -= quita
            falta -= quita
            if not falta:
                break
    filas_cara = agrupar_en_filas(caras, tam_uso)

    print("  p%-4d rot %-14s caras %3d   texto %3d   %s"
          % (n, rot, len(caras), sum(tamanos),
             "OK" if cuadra else ("REVISAR: separador" if raros and
                                  len(caras) == sum(tamanos) else "REVISAR: no cuadra")))
    for et, nombres, r in filas_txt:
        extra = ""
        if any(x is None for x in nombres):
            extra += "  (%d sin nombre)" % sum(1 for x in nombres if x is None)
        if r:
            extra += "  <- sin coma: " + ", ".join(x for x in r if x)
        print("      %-14s %2d nombres%s" % (et[:14], len(nombres), extra))

    os.makedirs(DEST, exist_ok=True)
    orden = [i for f in filas_cara for i in f] if filas_cara else \
            sorted(range(len(caras)), key=lambda i: (caras[i][1], caras[i][0]))
    cv2.imwrite(os.path.join(DEST, "p%03d_num.jpg" % n), numerar(img, caras, orden),
                [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    # una tira por fila, que es lo unico revisable a ojo
    if filas_cara:
        tiras = []
        etiquetadas = list(filas_txt)
        if len(filas_cara) > len(etiquetadas):
            etiquetadas = etiquetadas + [("SOBRAN (sin nombre en el pie)", [], [])]
        for (et, nombres, _), idx in zip(etiquetadas, filas_cara):
            t = tira(img, caras, idx, nombres)
            if t is None:
                continue
            cab = np.full((26, t.shape[1], 3), 255, np.uint8)
            aviso = "" if len(idx) == len(nombres) else "   %+d" % (len(idx) - len(nombres))
            cv2.putText(cab, "%s  (%d caras / %d nombres)%s" % (et, len(idx), len(nombres), aviso),
                        (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 40, 40), 2, cv2.LINE_AA)
            tiras.append(np.vstack([cab, t]))
        if tiras:
            ancho = max(t.shape[1] for t in tiras)
            tiras = [np.hstack([t, np.full((t.shape[0], ancho - t.shape[1], 3), 255, np.uint8)])
                     if t.shape[1] < ancho else t for t in tiras]
            cv2.imwrite(os.path.join(DEST, "p%03d_filas.jpg" % n), np.vstack(tiras),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not cuadra or not escribir:
        return cuadra

    c = faces.Caras()
    vecs, nombres_ok = [], []
    plano = [x for _, f, _ in filas_txt for x in f]
    for k, i in enumerate(orden):
        if plano[k] is None:
            continue                  # un NN: la cara esta, el nombre no
        h = c.huella(img, caras[i])
        if h is None:
            continue
        vecs.append(h)
        nombres_ok.append(plano[k])
    np.savez(os.path.join(DEST, "p%03d.npz" % n),
             vecs=np.array(vecs, dtype=np.float32),
             nombres=np.array(nombres_ok), pagina=n)
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pagina", type=int, default=None)
    ap.add_argument("--todas", action="store_true")
    ap.add_argument("--escribir", action="store_true")
    a = ap.parse_args()

    if a.pagina:
        procesar(a.pdf, a.pagina, a.escribir)
        return

    import logging
    logging.disable(logging.CRITICAL)
    import pypdf
    r = pypdf.PdfReader(a.pdf)
    paginas = []
    for i in range(len(r.pages)):
        try:
            if FILA.search(r.pages[i].extract_text() or ""):
                paginas.append(i + 1)
        except Exception:
            pass
    print("Paginas con listas por fila: %d\n" % len(paginas))
    ok = 0
    for n in paginas:
        if procesar(a.pdf, n, a.escribir):
            ok += 1
    print("\nCuadran: %d de %d" % (ok, len(paginas)))


if __name__ == "__main__":
    main()
