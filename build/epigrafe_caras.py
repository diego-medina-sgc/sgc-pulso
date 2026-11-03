# -*- coding: utf-8 -*-
"""
Une cada nombre de un epigrafe con su cara, por posicion, y (con --aplicar)
escribe esa cara: la cara que el epigrafe nombra CON POSICION vale como
carnet (decision del 3/11/2026, CONTRATO.md 3 'epigrafe_cara').

El epigrafe (carpeta_notas) lista los nombres fila por fila, de arriba hacia
abajo y de izquierda a derecha (epigrafes.nombres_de). Es el mismo problema
que las fotos de curso de los anuarios en PDF, y se resuelve con el mismo
codigo, sin escribir otro emparejador:

  - anuario_pdf.detectar: YuNet sobre la foto reducida a 1600 px, sin el piso
    de 44 px de la casa (en una foto de 57 personas las caras son chicas).
  - anuario_pdf.agrupar_en_filas: filas del tamaño que dice el texto, con una
    recta por fila (la gente se acomoda en arco).
  - EL CONTROL: se compara fila por fila. Si el total de caras no coincide con
    el de nombres, o alguna fila no cuadra, no se propone nada.

QUE SE ESCRIBE (CONTRATO.md 2 y 3). Para cada nombre que resuelve en UNA
ficha (las mismas reglas que epigrafes.py), solo si la foto entera cuadra y
ninguna cara quedo forzada en su fila, y -si la ficha ya tiene caras humanas
de referencia- solo si la cara asignada es la mas parecida de la foto a esas
referencias:
  - photo_people (foto, persona, source = 'epigrafe_cara'): insert, o update
    si el par ya tenia la fila 'epigrafe' de nivel foto. Un par con cualquier
    otro source no se toca (humano: manda ese; de maquina: se anota).
  - face_suggestions (foto, persona, recuadro en fraccion, score 1, rank 0,
    origen 'epigrafe'). rank 0: approve_face_suggestions no la mira.
Las dos filas van juntas, en la foto y en sus copias por dedupe_key. Lo
escrito se cuenta contra lo mandado: los triggers descartan en silencio
(photo_people_guard, face_suggestions_guard, sugerencia_respeta_no_es). Si una
mitad no entro, la otra se vuelve atras en la misma corrida (salud():
epigrafe_cara_descalzada).

    python epigrafe_caras.py --nota <id carpeta_notas>            # ensayo
    python epigrafe_caras.py --nota <id> --verificar              # + tabla de parecidos
    python epigrafe_caras.py --nota <id> --aplicar                # escribe
    python epigrafe_caras.py --pulso [--aplicar]                  # notas nuevas o cambiadas

--pulso: las notas con foto cuyo md5 no esta en el cursor (corridas, tarea
'epigrafes', etiqueta 'caras'). Cada nota se procesa una vez por contenido:
cuadre o no, su md5 queda anotado y no se vuelve a bajar la foto hasta que
el .txt cambie. Una nota que falla no se anota (se reintenta) y el paso sale
con error.

--refs <ruta>: otro _referencias_<motor>.npz (para ensayar en la notebook con
el del pulso; ver Trampas - motor: el de build/ es viejo).

Deja en build/_epigrafe_caras/:
    nota<id>_num.jpg     la foto con las caras numeradas en orden de lectura
    nota<id>_filas.jpg   una tira por fila, con el nombre debajo de cada cara
    nota<id>.json        el emparejamiento propuesto (orden, fila, pos, caja)
y cada corrida su respaldo en build/_epigrafe_caras_<fecha>.txt.

Revertir (CONTRATO.md 3):
    delete from face_suggestions fs where fs.origen = 'epigrafe' and not exists
      (select 1 from photo_labels l where l.photo_id = fs.photo_id
          and l.person_id = fs.person_id and l.answer = 'yes');
    update photo_people set source = 'epigrafe' where source = 'epigrafe_cara';
"""
import argparse
import datetime as _dt
import json
import os
import sys
import urllib.parse
import urllib.request

import cv2
import numpy as np

import anuario_pdf
import epigrafes
import sb

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "_epigrafe_caras")
SOURCE = "epigrafe_cara"
ORIGEN = "epigrafe"
CURSOR = {"tarea": "epigrafes", "etiqueta": "caras"}

# Una fila cuadra si le tocaron tantas caras como nombres Y ninguna cara quedo
# forzada: agrupar_en_filas SIEMPRE llena los cupos, asi que una cara puede
# caer en una fila por falta de lugar en otra. Forzada = la recta de otra fila
# le queda mas cerca que la propia. No hay umbral: una distancia absoluta a la
# recta no sirve, porque una fila mixta (parados en las puntas, sentados en el
# medio) no es una recta. Medido en la nota 11 (3/11/2026): la fila 4 llega a
# 0,89 altos de cara de su recta y el reparto es el correcto.
# Lo que si se informa es el margen (otra - propia, en altos de cara): un margen
# chico es una cara en el borde entre dos filas, que hay que mirar.

# Lo que pone una persona manda: un par con estos source no se toca. Los de
# carnet tambien (CONTRATO.md 3: "si es game/manual/grupo/filename/
# mugshot_filename manda ese").
QUE_MANDAN = ("game", "manual", "grupo", "filename", "mugshot_filename")
# Cuanto se tienen que pisar dos recuadros para ser la misma cara: el mismo
# numero que faces_referencias.IOU_MIN y caras_huellas_evento.py.
IOU_MISMA_CARA = 0.5
# sugerencia_respeta_no_es descarta un recuadro a menos de esto de un "no es"
# (sql, abs(bx) y abs(by) < 0,005): se mira antes para no mandar lo que no entra.
NO_ES_CERCA = 0.005


def bajar_original(drive_id):
    import requests
    from sync_drive import Drive
    d = Drive(os.path.join(HERE, "service-account.json"))
    r = requests.get(
        "https://www.googleapis.com/drive/v3/files/%s?alt=media&supportsAllDrives=true"
        % drive_id, headers={"Authorization": "Bearer " + d._token()}, timeout=120)
    r.raise_for_status()
    return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


def distancias_a_la_recta(caras, filas):
    """{cara: (distancia a su recta, distancia a la recta mas cercana de otra
    fila)}, en altos de cara. Lo que agrupar_en_filas no informa: si repartio
    por cupo una cara que estaba sobre otra recta."""
    cx = np.array([c[0] + c[2] / 2.0 for c in caras])
    cy = np.array([c[1] + c[3] / 2.0 for c in caras])
    alto = float(np.median([c[3] for c in caras]))
    rectas = []
    for idx in filas:
        if len(idx) >= 3:
            a, b = np.polyfit(cx[idx], cy[idx], 1)
        else:
            a, b = 0.0, float(np.mean(cy[idx])) if idx else 0.0
        rectas.append((a, b))
    out = {}
    for k, idx in enumerate(filas):
        for i in idx:
            d = [abs(cy[i] - (a * cx[i] + b)) / alto for a, b in rectas]
            otra = min([d[j] for j in range(len(d)) if j != k] or [1e9])
            out[i] = (d[k], otra)
    return out


def solape(a, b):
    """Cuanto se pisan dos recuadros (x, y, w, h), de 0 a 1."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter)


def resolver(gente, nota):
    """{orden: person_id} de los nombres que resuelven en UNA ficha, con las
    mismas reglas que epigrafes.main (sin acortados, sin dudosas)."""
    camada = None
    m = epigrafes.CAMADA_EN_NOMBRE.search(nota["nombre"] or "") or \
        epigrafes.CAMADA_EN_NOMBRE.search(nota["texto"] or "")
    if m:
        camada = int(m.group(1))
    camada_foto = None
    if not camada:
        camada_foto = epigrafes.camada_de_la_foto(gente, nota["texto"], camada)[0]
    out = {}
    for (n, _f, _p, _c, es_staff, ini, ape, pila) in epigrafes.nombres_de(nota["texto"]):
        cands, etq = epigrafes.candidatas(gente, ini, ape, es_staff, camada,
                                          camada_foto, pila=pila)
        if len(cands) == 1 and "acortado" not in etq:
            out[n] = int(cands[0]["id"])
    return out


# ---------------------------------------------------------------- emparejar

def emparejar(cli, nota, a):
    """Baja la foto, detecta, arma las filas y deja las imagenes de control.
    Devuelve un dict con foto, img, caras, cuadra y la propuesta (lista por
    nombre, en orden de lectura) o None si la nota no tiene foto."""
    foto = epigrafes.foto_de(cli, nota)
    if not foto:
        print("La nota %s no tiene foto: se espera a que se indexe." % nota["id"])
        return None

    nombres = epigrafes.nombres_de(nota["texto"])
    filas_txt, vistas = [], {}
    for t in nombres:
        if t[1] not in vistas:
            vistas[t[1]] = len(filas_txt)
            filas_txt.append((t[1], []))
        filas_txt[vistas[t[1]]][1].append(t)
    tamanos = [len(f) for _, f in filas_txt]

    img = bajar_original(foto)
    if img is None:
        raise RuntimeError("No se pudo leer la foto %s" % foto)
    anuario_pdf.CONF, anuario_pdf.LADO_MIN, anuario_pdf.LADO_DET = \
        a.conf, a.lado_min, a.lado_det
    caras = anuario_pdf.detectar(img)

    print("Foto %s  %dx%d" % (foto, img.shape[1], img.shape[0]))
    print("Caras detectadas: %d   nombres: %d   por fila: %s"
          % (len(caras), sum(tamanos), "/".join(map(str, tamanos))))
    if caras:
        lados = sorted(min(c[2], c[3]) for c in caras)
        confs = sorted(float(c[14]) for c in caras)
        print("  lado (px, original): min %.0f  mediana %.0f  max %.0f"
              % (lados[0], lados[len(lados) // 2], lados[-1]))
        print("  confianza:           min %.2f  mediana %.2f  max %.2f"
              % (confs[0], confs[len(confs) // 2], confs[-1]))

    # Mismo armado que anuario_pdf.procesar: las filas se arman aunque no
    # cuadre, para poder mirar donde falla.
    tam_uso = list(tamanos)
    sobran = len(caras) - sum(tamanos)
    if sobran > 0:
        tam_uso.append(sobran)
    elif sobran < 0:
        falta = -sobran
        for k in range(len(tam_uso) - 1, -1, -1):
            quita = min(falta, tam_uso[k])
            tam_uso[k] -= quita
            falta -= quita
            if not falta:
                break
    filas_cara = anuario_pdf.agrupar_en_filas(caras, tam_uso) if caras else []
    dist = distancias_a_la_recta(caras, filas_cara) if caras else {}

    cuadra = bool(caras) and len(caras) == sum(tamanos)
    for k, (et, ns) in enumerate(filas_txt):
        idx = filas_cara[k] if k < len(filas_cara) else []
        forzadas = [i for i in idx if dist[i][1] <= dist[i][0]]
        ok = len(idx) == len(ns) and not forzadas
        cuadra = cuadra and ok
        margen = min([dist[i][1] - dist[i][0] for i in idx] or [0])
        print("  %-8s caras %2d  nombres %2d  max a su recta %.2f  margen min %.2f%s"
              % (et, len(idx), len(ns), max([dist[i][0] for i in idx] or [0]), margen,
                 "" if ok else "  <- NO CUADRA" +
                 (" (%d forzadas)" % len(forzadas) if forzadas else "")))
    if len(filas_cara) > len(filas_txt):
        print("  sobran   %d caras sin nombre" % len(filas_cara[-1]))
    print("CUADRA" if cuadra else "NO CUADRA: no se propone nada")

    os.makedirs(DEST, exist_ok=True)
    pre = os.path.join(DEST, "nota%s" % nota["id"])
    if caras:
        orden = [i for f in filas_cara for i in f]
        cv2.imwrite(pre + "_num.jpg", anuario_pdf.numerar(img, caras, orden),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        tiras = []
        etiq = filas_txt + ([("SOBRAN", [])] if len(filas_cara) > len(filas_txt) else [])
        for (et, ns), idx in zip(etiq, filas_cara):
            t = anuario_pdf.tira(img, caras, idx, [x[3] for x in ns])
            if t is None:
                continue
            cab = np.full((26, t.shape[1], 3), 255, np.uint8)
            cv2.putText(cab, "%s  (%d caras / %d nombres)" % (et, len(idx), len(ns)),
                        (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 40, 40), 2, cv2.LINE_AA)
            tiras.append(np.vstack([cab, t]))
        if tiras:
            ancho = max(t.shape[1] for t in tiras)
            tiras = [np.hstack([t, np.full((t.shape[0], ancho - t.shape[1], 3), 255, np.uint8)])
                     if t.shape[1] < ancho else t for t in tiras]
            cv2.imwrite(pre + "_filas.jpg", np.vstack(tiras), [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    prop = []
    if cuadra:
        for (et, ns), idx in zip(filas_txt, filas_cara):
            for t, i in zip(ns, idx):
                x, y, w, h = [int(v) for v in caras[i][:4]]
                prop.append({"orden": t[0], "fila": et, "pos": t[2], "crudo": t[3],
                             "cara": int(i), "caja_px": [x, y, w, h],
                             "caja_rel": [x / img.shape[1], y / img.shape[0],
                                          w / img.shape[1], h / img.shape[0]],
                             "conf": round(float(caras[i][14]), 3),
                             "dist_recta": round(float(dist[i][0]), 2),
                             "margen_fila": round(float(dist[i][1] - dist[i][0]), 2)})
    with open(pre + ".json", "w", encoding="utf-8") as fh:
        json.dump({"nota": nota["id"], "foto": foto, "cuadra": cuadra,
                   "ancho": img.shape[1], "alto": img.shape[0],
                   "por_fila": tamanos, "caras": len(caras), "propuesta": prop},
                  fh, ensure_ascii=False, indent=1)
    if caras:
        print("Imagenes: %s_num.jpg  %s_filas.jpg" % (pre, pre))
    return {"foto": foto, "img": img, "caras": caras, "cuadra": cuadra, "prop": prop}


# ---------------------------------------------------------------- comparar

class Referencias:
    """Las caras de referencia, cargadas una vez por corrida (np.load una sola
    vez: faces_piloto.cargar_refs_humanas). humanas = personas con al menos
    una cara humana de referencia (game/manual/grupo)."""

    def __init__(self, cli, ruta=None):
        import faces_piloto
        if ruta:
            faces_piloto.REFS = ruta
        vecs, personas, _a, fotos, humanas = faces_piloto.cargar_refs_humanas(cli)
        v = vecs.astype(np.float32)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        self.vecs, self.personas = v, np.asarray(personas)
        self.fotos = np.asarray([str(f) for f in fotos])
        self.humanas = set(int(p) for p in humanas)

    def de(self, pid, sin_fotos=()):
        """Matriz de referencias de pid, sin las que salen de sin_fotos (la foto
        misma y sus copias: no puede ser juez de si misma)."""
        m = self.personas == pid
        if sin_fotos:
            m &= ~np.isin(self.fotos, list(sin_fotos))
        return self.vecs[m]


def parecidos(emp, quien, refs, sin_fotos):
    """{orden: (asignada, mejor, 2a, la asignada es la mejor)} para cada nombre
    cuya ficha tiene referencias. Al reves de "a quien se parece esta cara":
    de TODAS las caras de la foto, cual se parece mas a la ficha. Si el
    reparto esta corrido un lugar, gana la cara de al lado."""
    import faces
    c = faces.Caras()
    todas = []
    for cc in emp["caras"]:
        v = c.huella(emp["img"], cc)
        if v is not None:
            v = np.asarray(v, dtype=np.float32)
            v /= max(float(np.linalg.norm(v)), 1e-9)
        todas.append(v)
    caja = {t["orden"]: t for t in emp["prop"]}
    out = {}
    for orden, pid in sorted(quien.items()):
        t = caja.get(orden)
        if t is None:
            continue
        R = refs.de(pid, sin_fotos)
        if not len(R):
            continue
        s = sorted(((-1.0 if v is None else float(np.max(R @ v)), i)
                    for i, v in enumerate(todas)), reverse=True)
        mia = t["cara"]
        propia = dict((i, x) for x, i in s)[mia]
        out[orden] = (propia, s[0][0], s[1][0] if len(s) > 1 else -1.0, s[0][1] == mia)
    return out


# ---------------------------------------------------------------- planear

def _en(valores):
    return "in.(%s)" % ",".join(str(v) for v in valores)


def leer_estado(cli, fotos, pids):
    """Lo que hay hoy en la base para esas fotos y esas personas."""
    pp = cli.select("photo_people", select="photo_id,person_id,source",
                    photo_id=_en(fotos)) or []
    fs = cli.select("face_suggestions",
                    select="photo_id,person_id,bx,by,bw,bh,score,rank,origen",
                    photo_id=_en(fotos)) or []
    rech = cli.select("photo_people_rejected", select="photo_id,person_id",
                      photo_id=_en(fotos), person_id=_en(pids)) or []
    si = cli.select("photo_labels", select="photo_id,person_id,answer",
                    photo_id=_en(fotos), person_id=_en(pids),
                    answer="eq.yes") or []
    no_son = cli.select("grupos_no_son", select="photo_id,person_id,bx,by",
                        photo_id=_en(fotos), person_id=_en(pids)) or []
    return {
        "pp": {(r["photo_id"], int(r["person_id"])): r.get("source") or "" for r in pp},
        "fs": {(r["photo_id"], int(r["person_id"])): r for r in fs},
        "rech": {(r["photo_id"], int(r["person_id"])) for r in rech},
        "si": {(r["photo_id"], int(r["person_id"])) for r in si},
        "no_son": [(r["photo_id"], int(r["person_id"]), float(r["bx"] or 0), float(r["by"] or 0))
                   for r in no_son],
    }


def copias_de(cli, foto):
    """Las otras fotos con la misma dedupe_key (la misma imagen)."""
    f = (cli.select("photos", select="id,dedupe_key", id="eq." + foto) or [{}])[0]
    if not f.get("dedupe_key"):
        return []
    otras = cli.select("photos", select="id", dedupe_key="eq." + f["dedupe_key"]) or []
    return sorted(o["id"] for o in otras if o["id"] != foto)


def dudosas_pendientes(cli):
    """Los nombres de epigrafe que esperan respuesta en padron_dudosas.

    UNA DUDOSA SIN RESPONDER NO SE ESCRIBE (3/11/2026). epigrafes.py manda a
    dudosas, con su candidata, lo que se resolvio despues de la tanda
    autorizada o lo que tenia dos fichas; si despues el padron cambia (una
    fusion, una ficha a ruido) el nombre resuelve solo, pero la pregunta sigue
    abierta. Medido en la nota 2: dos nombres asi entraban sin que nadie los
    contestara."""
    try:
        filas = cli.select("padron_dudosas", select="nombre,resuelto", hoja="eq.epigrafe") or []
    except Exception as e:                                        # noqa: BLE001
        raise RuntimeError("no se pudo leer padron_dudosas: %s" % str(e)[:120])
    return {r["nombre"] for r in filas if not r.get("resuelto")}


def planear(cli, nota, emp, gente, refs, dudosas=frozenset()):
    """Lista de filas a escribir (una por nombre y foto) y lista de lo que no
    se escribe, con su motivo. No escribe."""
    quien = resolver(gente, nota)
    nombre_de = {int(p["id"]): p.get("display_name") for p in gente}
    foto = emp["foto"]
    fotos = [foto] + copias_de(cli, foto)
    plan, fuera = [], []
    caja = {t["orden"]: t for t in emp["prop"]}

    todos = epigrafes.nombres_de(nota["texto"])
    for t in todos:
        if t[0] not in quien:
            fuera.append((t[0], t[3], None, foto, "no resuelve en una sola ficha"))
    if not emp["cuadra"]:
        for orden, pid in sorted(quien.items()):
            fuera.append((orden, caja.get(orden, {}).get("crudo") or "", pid, foto,
                          "la foto no cuadra"))
        return plan, fuera, fotos

    # dos nombres que dan la misma ficha: alguno esta mal, ninguno se escribe
    veces = {}
    for orden, pid in quien.items():
        veces[pid] = veces.get(pid, 0) + 1
    par = parecidos(emp, quien, refs, set(fotos)) if refs is not None else {}
    est = leer_estado(cli, fotos, sorted(set(quien.values())))

    # recuadros confirmados por una persona de OTRA ficha en estas fotos
    ajenos = [(k[0], k[1], (float(r["bx"]), float(r["by"]), float(r["bw"]), float(r["bh"])))
              for k, r in est["fs"].items()
              if r.get("bw") and est["pp"].get(k) in QUE_MANDAN]

    for orden, pid in sorted(quien.items()):
        t = caja[orden]
        motivo = None
        if veces[pid] > 1:
            motivo = "la misma ficha sale %d veces en el epigrafe" % veces[pid]
        elif t["crudo"] in dudosas:
            motivo = "esta en padron_dudosas sin responder"
        elif refs is None:
            motivo = "sin referencias cargadas: no se puede verificar"
        elif pid in refs.humanas:
            p = par.get(orden)
            if p is None:
                motivo = "tiene caras humanas pero ninguna fuera de esta foto"
            elif not p[3]:
                motivo = "gana otra cara de la foto (asignada %.3f, mejor %.3f)" % (p[0], p[1])
        if motivo:
            fuera.append((orden, t["crudo"], pid, foto, motivo))
            continue
        bx, by, bw, bh = [round(float(v), 4) for v in t["caja_rel"]]
        for f in fotos:
            k = (f, pid)
            antes = est["pp"].get(k)
            m2 = None
            if antes and antes not in ("epigrafe", SOURCE):
                m2 = ("ya la nombro una persona (%s): manda esa" % antes
                      if antes in QUE_MANDAN else
                      "ya esta de maquina (%s): no se pisa" % antes)
            elif k in est["rech"]:
                m2 = "hay un rechazo humano de ese par"
            elif k in est["si"] and antes != SOURCE:
                m2 = "hay un Si humano de ese par"
            elif any(r[0] == f and r[1] == pid and abs(r[2] - bx) < NO_ES_CERCA
                     and abs(r[3] - by) < NO_ES_CERCA for r in est["no_son"]):
                m2 = "hay un 'no es' de esa persona en ese recuadro"
            else:
                otra = [r for r in ajenos if r[0] == f and r[1] != pid
                        and solape(r[2], (bx, by, bw, bh)) >= IOU_MISMA_CARA]
                if otra:
                    m2 = "ese recuadro ya lo confirmo una persona para la ficha %s" % otra[0][1]
            if m2:
                fuera.append((orden, t["crudo"], pid, f, m2))
                continue
            fs_antes = est["fs"].get(k)
            ya = (antes == SOURCE and fs_antes and fs_antes.get("origen") == ORIGEN)
            plan.append({"orden": orden, "crudo": t["crudo"], "fila": t["fila"],
                         "pos": t["pos"], "person_id": pid,
                         "nombre": nombre_de.get(pid) or "", "photo_id": f,
                         "copia": f != foto,
                         "accion": ("ya esta" if ya else
                                    "insertar" if not antes else
                                    "actualizar" if antes == "epigrafe" else "completar"),
                         "source_antes": antes,
                         "fs_antes": fs_antes, "caja": [bx, by, bw, bh],
                         "parecido": par.get(orden)})
    return plan, fuera, fotos


# ---------------------------------------------------------------- escribir

def _post_ignorar(cli, tabla, filas, on_conflict):
    """INSERT ... ON CONFLICT DO NOTHING: lo que ya esta (por ejemplo la copia
    que acaba de escribir photo_people_copias) no se pisa."""
    q = "?" + urllib.parse.urlencode({"on_conflict": on_conflict})
    req = urllib.request.Request(
        "%s/rest/v1/%s%s" % (cli.url, tabla, q),
        data=json.dumps(filas, ensure_ascii=False).encode("utf-8"), method="POST",
        headers=cli._headers({"Prefer": "resolution=ignore-duplicates,return=minimal"}))
    cli._pedir(req)


def escribir(cli, plan, fotos):
    """Escribe el plan y devuelve (pares completos, pares vueltos atras)."""
    hacer = [r for r in plan if r["accion"] != "ya esta"]
    pids = sorted({r["person_id"] for r in plan})
    if not hacer:
        return {(r["photo_id"], r["person_id"]) for r in plan}, []
    # 1. photo_people. Primero las fotos originales: el trigger
    # photo_people_copias copia el insert a las copias (on conflict do nothing).
    for copia in (False, True):
        ins = [{"photo_id": r["photo_id"], "person_id": r["person_id"], "source": SOURCE}
               for r in hacer if r["accion"] == "insertar" and r["copia"] == copia]
        if ins:
            _post_ignorar(cli, "photo_people", ins, "photo_id,person_id")
    for r in hacer:
        if r["accion"] == "actualizar":
            cli.update("photo_people", {"source": SOURCE},
                       photo_id="eq." + r["photo_id"], person_id="eq.%d" % r["person_id"],
                       source="eq.epigrafe")
    pp = cli.select("photo_people", select="photo_id,person_id,source",
                    photo_id=_en(fotos), person_id=_en(pids),
                    source="eq." + SOURCE) or []
    con_pp = {(x["photo_id"], int(x["person_id"])) for x in pp}

    # 2. face_suggestions, solo de los pares que si quedaron en photo_people.
    # merge: una sugerencia de maquina del par se pisa con esta (CONTRATO.md 3).
    fs = [{"photo_id": r["photo_id"], "person_id": r["person_id"],
           "bx": r["caja"][0], "by": r["caja"][1], "bw": r["caja"][2], "bh": r["caja"][3],
           "score": 1, "rank": 0, "origen": ORIGEN}
          for r in plan if (r["photo_id"], r["person_id"]) in con_pp]
    if fs:
        cli.upsert("face_suggestions", fs, on_conflict="photo_id,person_id")
    q = cli.select("face_suggestions", select="photo_id,person_id",
                   photo_id=_en(fotos), person_id=_en(pids),
                   origen="eq." + ORIGEN) or []
    con_fs = {(x["photo_id"], int(x["person_id"])) for x in q}

    # 3. Una mitad sola no se deja (salud: epigrafe_cara_descalzada). Tambien
    # las copias que el trigger escribio y el plan no tenia.
    previo = {(r["photo_id"], r["person_id"]): r["source_antes"] for r in plan}
    atras = []
    for k in sorted(con_pp - con_fs):
        antes = previo.get(k)
        if antes == "epigrafe":
            cli.update("photo_people", {"source": "epigrafe"}, photo_id="eq." + k[0],
                       person_id="eq.%d" % k[1], source="eq." + SOURCE)
        elif not antes:
            # la escribio esta misma corrida (o su trigger) segundos antes
            cli.delete("photo_people", photo_id="eq." + k[0], person_id="eq.%d" % k[1],
                       source="eq." + SOURCE)
        atras.append(k)
    return con_pp & con_fs, atras


# ---------------------------------------------------------------- cursor

def leer_cursor(cli):
    filas = cli.select("corridas", select="mensaje", tarea="eq." + CURSOR["tarea"],
                       etiqueta="eq." + CURSOR["etiqueta"]) or []
    try:
        return json.loads(filas[0]["mensaje"] or "{}") if filas else {}
    except ValueError:
        return {}


def anotar_cursor(cli, cursor, hechas, escritas):
    """estado listo y no corriendo: salud() marca trabada una corrida que dice
    'corriendo' y no se mueve en dos horas (como fotos_duplicadas.py)."""
    cli.upsert("corridas", [dict(CURSOR, estado="listo", total=len(cursor), hechas=hechas,
                                 encontradas=escritas, con_gente=0,
                                 mensaje=json.dumps(cursor, ensure_ascii=False),
                                 actualizado_at=_dt.datetime.now(_dt.timezone.utc).isoformat())],
               on_conflict="tarea,etiqueta")


# ---------------------------------------------------------------- main

def mostrar(nota, plan, fuera):
    print("\nNota %s: %d filas de cada tabla para escribir, %d nombres o pares afuera"
          % (nota["id"], len([r for r in plan if r["accion"] != "ya esta"]), len(fuera)))
    print("  %-4s %-26s %-9s %-4s %-10s %-11s %-12s %s"
          % ("ord", "en el epigrafe", "fila", "pos", "persona", "accion", "antes", "parecido"))
    for r in plan:
        p = r["parecido"]
        print("  %-4d %-26s %-9s %-4s %-10s %-11s %-12s %s%s"
              % (r["orden"], r["crudo"][:26], r["fila"], r["pos"], r["person_id"],
                 r["accion"], r["source_antes"] or "-",
                 "%.3f (mejor %.3f, 2a %.3f)" % p[:3] if p else "sin referencias",
                 "  [copia %s]" % r["photo_id"] if r["copia"] else ""))
    for o, crudo, pid, f, motivo in fuera:
        print("  %-4s %-26s afuera: %s%s" % (o, (crudo or "")[:26], motivo,
                                             "" if pid is None else "  (ficha %s)" % pid))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--nota", type=int, help="id de carpeta_notas")
    ap.add_argument("--pulso", action="store_true",
                    help="las notas nuevas o cambiadas (cursor en corridas)")
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--igual-con-el-pulso", action="store_true",
                    help="escribir aunque el pulso este en un paso que frena (lo usa el pulso)")
    ap.add_argument("--refs", help="otro _referencias_<motor>.npz (solo para ensayar)")
    ap.add_argument("--conf", type=float, default=anuario_pdf.CONF)
    ap.add_argument("--lado-min", type=int, default=anuario_pdf.LADO_MIN)
    ap.add_argument("--lado-det", type=int, default=anuario_pdf.LADO_DET)
    ap.add_argument("--verificar", action="store_true",
                    help="(se mantiene por compatibilidad: la verificacion corre siempre)")
    a = ap.parse_args()
    if not a.nota and not a.pulso:
        ap.error("--nota o --pulso")
    import faces
    if faces.MOTOR != "arcface":
        # las referencias de otro motor dan parecidos sin sentido (12/9/2026)
        sys.exit("FACES_MOTOR tiene que ser arcface (es %s)" % faces.MOTOR)
    if a.refs and a.aplicar:
        ap.error("--refs es solo para ensayar: aplicar usa las referencias del motor")

    cli = sb.SB()
    if a.aplicar:
        frenan, _otros = sb.pasos_del_pulso(cli)
        if frenan and not a.igual_con_el_pulso:
            print("NO escribo: el pulso esta en %s. Probar de nuevo despues." % ", ".join(frenan))
            sys.exit(2)

    cursor = leer_cursor(cli) if a.pulso else {}
    if a.pulso:
        # notas_de_la_base no trae md5: se lee aparte
        md5 = {int(r["id"]): r.get("md5") for r in
               (cli.select("carpeta_notas", select="id,md5", ignorada="is.false") or [])}
        notas = [n for n in epigrafes.notas_de_la_base(cli)
                 if n.get("photo_id") and cursor.get(str(n["id"])) != md5.get(int(n["id"]))]
        print("Notas con foto nuevas o cambiadas: %d" % len(notas))
    else:
        notas = cli.select("carpeta_notas", select="id,folder_id,photo_id,nombre,texto,md5",
                           id="eq.%d" % a.nota) or []
        if not notas:
            sys.exit("No existe la nota %d" % a.nota)
        md5 = {int(notas[0]["id"]): notas[0].get("md5")}
    if not notas:
        print("No hay nada que hacer.")
        return

    gente = cli.select("people", select="id,display_name,kind,cohort,campus,house,aliases")
    refs = Referencias(cli, a.refs)
    dudosas = dudosas_pendientes(cli)
    print("Referencias: %d caras, %d personas con caras humanas"
          % (len(refs.vecs), len(refs.humanas)))

    sello = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    respaldo = os.path.join(HERE, "_epigrafe_caras_%s.txt" % sello)
    fh = open(respaldo, "w", encoding="utf-8")
    fh.write("# epigrafe_caras.py %s  %s\n" % (sello, "APLICADO" if a.aplicar else "ensayo"))
    fh.write("# revertir: ver el docstring de epigrafe_caras.py (CONTRATO.md 3)\n")
    fh.write("# nota\torden\tfila\tpos\tpersona\tphoto_id\taccion\tsource_antes\t"
             "caja\tface_suggestions_antes\n")

    hechas = escritas = 0
    fallidas = []
    for nota in notas:
        print("\n" + "=" * 78 + "\nNota %s  %s" % (nota["id"], nota.get("nombre") or ""))
        try:
            emp = emparejar(cli, nota, a)
            if emp is None:
                continue
            plan, fuera, fotos = planear(cli, nota, emp, gente, refs, dudosas)
            mostrar(nota, plan, fuera)
            for r in plan:
                fh.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"
                         % (nota["id"], r["orden"], r["fila"], r["pos"], r["person_id"],
                            r["photo_id"], r["accion"], r["source_antes"] or "",
                            json.dumps(r["caja"]),
                            json.dumps(r["fs_antes"], default=str) if r["fs_antes"] else ""))
            for o, crudo, pid, f, motivo in fuera:
                fh.write("%s\t%s\t\t\t%s\t%s\tafuera: %s\t\t\t\n"
                         % (nota["id"], o, pid if pid is not None else "", f, motivo))
            fh.flush()
            if a.aplicar and plan:
                ok, atras = escribir(cli, plan, fotos)
                mandadas = len(plan)
                print("Escritas y verificadas (las dos filas): %d de %d" % (len(ok), mandadas))
                for k in atras:
                    print("   VUELTA ATRAS (entro una mitad sola): foto %s ficha %s" % k)
                    fh.write("%s\tvuelta atras\t\t\t%s\t%s\n" % (nota["id"], k[1], k[0]))
                no = {(r["photo_id"], r["person_id"]) for r in plan} - ok
                for k in sorted(no):
                    print("   NO entro: foto %s ficha %s" % k)
                escritas += len(ok)
                if no or atras:
                    raise RuntimeError("%d pares no entraron enteros" % len(no))
            hechas += 1
            if a.pulso and a.aplicar:
                cursor[str(nota["id"])] = md5.get(int(nota["id"]))
                anotar_cursor(cli, cursor, hechas, escritas)
        except Exception as e:                                    # noqa: BLE001
            # una nota rota no tira las demas, pero no se anota: se reintenta
            fallidas.append(nota["id"])
            print("ERROR en la nota %s: %s" % (nota["id"], str(e)[:300]))
            fh.write("%s\tERROR\t%s\n" % (nota["id"], str(e)[:300]))
    fh.close()
    print("\nRespaldo: %s" % respaldo)
    if not a.aplicar:
        print("Ensayo: no se escribio nada. Agregar --aplicar.")
    try:
        sb.salida_pulso({"notas": len(notas), "hechas": hechas, "escritas": escritas,
                         "fallidas": fallidas})
    except Exception:                                             # noqa: BLE001
        pass
    if fallidas:
        sys.exit("Notas con error: %s" % ", ".join(map(str, fallidas)))


if __name__ == "__main__":
    main()
