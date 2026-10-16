# -*- coding: utf-8 -*-
"""Lo comun de "Que se ve": el modelo de imagen, las miniaturas y los vectores.

  - CLIP ViT-B/32, solo la parte de imagen, fp32, de Xenova/clip-vit-base-patch32
    en una revision FIJA y verificada por sha256 (elegido y medido 11-13/10/2026
    contra SigLIP 2, RAM++ y Florence-2: el mejor acierto por segundo de CPU).
  - Miniatura s256 de Drive por thumbnailLink (cuenta de servicio, gratis) y,
    para Zenfolio, la Edge Function `thumb` con la llave de servicio (de a 24,
    CONTRATO.md seccion 1).
  - Los vectores (512-d, int8 con una escala por foto) en trozos
    _escenas_NNN.npz de TROZO fotos, solo agregar: un trozo lleno no se vuelve
    a escribir. Son el estado del paso escenas y viajan cifrados aparte de las
    huellas de caras (estado.py --conjunto escenas). NUNCA *_arcface.npz.

Ningun nombre de persona en este archivo: se publica en el repo publico.
"""
import glob
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

MODELO_REPO = "Xenova/clip-vit-base-patch32"
MODELO_REV = "d15189d7028b43f1d3e65039190477f6af591c2a"
MODELO_ARCHIVO = "onnx/vision_model.onnx"
MODELO_SHA256 = "fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40"
MODELO_BYTES = 351685709
CARPETA_MODELO = os.path.join(HERE, "modelos_escenas")
RUTA_MODELO = os.path.join(CARPETA_MODELO, "clip_b32_vision_%s.onnx" % MODELO_REV[:8])

LADO = 224
MEDIA = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
DESVIO = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

# Trozos de vectores. 20.000 fotos x 512 bytes = ~10 MB por trozo, debajo de
# los 45 MB por objeto del bucket (estado.TROZO) sin partirlo.
TROZO = 20000
PATRON_TROZO = "_escenas_[0-9][0-9][0-9].npz"
DIM = 512

# Miniatura que se pide a Drive: s256 alcanza para 224 (medido 11/10/2026:
# s512 no cambia el acierto y pesa 3 veces mas).
DRIVE_LADO = 256
# La Edge Function thumb solo da 400 (y guarda la copia en thumbs/zf/).
ZF_LADO = 400
ZF_LOTE = 24


# ------------------------------------------------------------------ modelo

def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ruta_modelo():
    """El .onnx verificado. ESCENAS_MODELO apunta a una copia local (notebook)."""
    r = os.environ.get("ESCENAS_MODELO") or RUTA_MODELO
    return r


def bajar_modelo():
    """Baja el modelo de la revision fija y controla el sha256. Idempotente."""
    if os.path.exists(RUTA_MODELO) and os.path.getsize(RUTA_MODELO) == MODELO_BYTES \
            and sha256(RUTA_MODELO) == MODELO_SHA256:
        print("modelo de escenas: ya estaba y coincide")
        return RUTA_MODELO
    os.makedirs(CARPETA_MODELO, exist_ok=True)
    url = "https://huggingface.co/%s/resolve/%s/%s" % (MODELO_REPO, MODELO_REV, MODELO_ARCHIVO)
    tmp = RUTA_MODELO + ".bajando"
    for intento in range(4):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            break
        except (urllib.error.URLError, TimeoutError, OSError):
            if intento == 3:
                raise
            time.sleep(10 * (intento + 1))
    h = sha256(tmp)
    if h != MODELO_SHA256:
        os.remove(tmp)
        raise RuntimeError("el modelo bajado no coincide: sha256 %s" % h)
    os.replace(tmp, RUTA_MODELO)
    print("modelo de escenas: bajado y verificado (%d MB)" % (MODELO_BYTES // 1e6))
    return RUTA_MODELO


class Clip:
    def __init__(self, ruta=None, hilos=None):
        import onnxruntime as ort
        ruta = ruta or ruta_modelo()
        if not os.path.exists(ruta):
            raise RuntimeError("falta el modelo de escenas (%s): escenas_procesar.py --bajar-modelo"
                               % ruta)
        o = ort.SessionOptions()
        o.intra_op_num_threads = hilos or (os.cpu_count() or 4)
        o.inter_op_num_threads = 1
        self.s = ort.InferenceSession(ruta, o, providers=["CPUExecutionProvider"])

    @staticmethod
    def preparar(img):
        """BGR de OpenCV -> (3, 224, 224) como CLIPProcessor: lado corto a
        224 (bicubica), recorte central, RGB, media y desvio de CLIP."""
        h, w = img.shape[:2]
        k = LADO / min(h, w)
        img = cv2.resize(img, (max(LADO, round(w * k)), max(LADO, round(h * k))),
                         interpolation=cv2.INTER_CUBIC)
        h, w = img.shape[:2]
        y, x = (h - LADO) // 2, (w - LADO) // 2
        img = img[y:y + LADO, x:x + LADO, ::-1].astype(np.float32) / 255.0
        img = (img - MEDIA) / DESVIO
        return img.transpose(2, 0, 1).astype(np.float32)

    def vectores(self, lote):
        """lista de (3,224,224) -> (n, 512) normalizados."""
        if not lote:
            return np.zeros((0, DIM), np.float32)
        v = self.s.run(None, {"pixel_values": np.stack(lote)})[0].astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def decodificar(datos):
    """bytes -> imagen BGR, o None si no es una imagen legible."""
    if not datos:
        return None
    img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.shape[0] < 16 or img.shape[1] < 16:
        return None
    return img


# --------------------------------------------------------------- miniaturas

class Miniaturas:
    """Trae los bytes de la miniatura de una foto. Cada metodo devuelve
    (bytes, None) o (None, "motivo"): una foto rota no tira nada."""

    def __init__(self, cli, drive=None):
        self.cli = cli
        self._drive = drive
        self._lock = threading.Lock()

    def drive(self):
        with self._lock:
            if self._drive is None:
                from index_drive import Drive, SA_PATH
                self._drive = Drive(SA_PATH)
            return self._drive

    def de_drive(self, pid):
        try:
            link = self.drive().get("files/" + pid, fields="thumbnailLink").get("thumbnailLink")
        except Exception as e:
            return None, "drive: %s" % str(e)[:80]
        if not link:
            return None, "sin miniatura en drive"
        if "=s" in link:
            link = link.rsplit("=s", 1)[0] + "=s%d" % DRIVE_LADO
        return bajar_url(link)

    def urls_zenfolio(self, ids):
        """{id: url firmada} de la Edge Function thumb, de a ZF_LOTE, volviendo
        a pedir las que no vinieron (la funcion trae 24 por llamada)."""
        out, faltan = {}, list(ids)
        for _ in range(4):
            if not faltan:
                break
            pedir = faltan[:]
            faltan = []
            for i in range(0, len(pedir), ZF_LOTE):
                lote = pedir[i:i + ZF_LOTE]
                try:
                    r = pedir_thumb(self.cli, lote)
                except Exception as e:
                    print("  thumb: %s" % str(e)[:120], flush=True)
                    r = {}
                for pid in lote:
                    u = r.get(pid)
                    if u:
                        out[pid] = u
                    else:
                        faltan.append(pid)
        return out


def pedir_thumb(cli, ids):
    req = urllib.request.Request(
        "%s/functions/v1/thumb" % cli.url,
        data=json.dumps({"ids": ids, "size": ZF_LADO}).encode("utf-8"), method="POST",
        headers={"apikey": cli.key, "Authorization": "Bearer " + cli.key,
                 "Content-Type": "application/json"})
    for intento in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and intento < 3:
                time.sleep(2 ** intento * 2)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if intento < 3:
                time.sleep(2 ** intento * 2)
                continue
            raise


def bajar_url(url):
    for intento in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read(), None
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and intento < 2:
                time.sleep(2 ** intento)
                continue
            return None, "http %s" % e.code
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if intento < 2:
                time.sleep(2 ** intento)
                continue
            return None, "red: %s" % str(e)[:60]
    return None, "no se pudo bajar"


HILOS_BAJADA = 16
LOTE_MODELO = 32


def pasar(filas, mini, clip, hasta=None):
    """Baja, decodifica y pasa por CLIP una lista de fotos.

    filas: dicts con photo_id y source ('drive' o 'zenfolio').
    hasta: time.time() limite; lo que no llego a bajarse queda sin tocar.
    Devuelve (ids, vecs (n,512), errores {id: motivo}). Una foto que no se
    puede bajar o leer va a errores y el resto sigue."""
    from concurrent.futures import ThreadPoolExecutor
    errores = {}
    zf = [f["photo_id"] for f in filas if f.get("source") == "zenfolio"]
    urls = mini.urls_zenfolio(zf) if zf else {}

    def traer(f):
        pid = f["photo_id"]
        if hasta and time.time() > hasta:
            return pid, None, None          # sin tocar: queda para la vuelta siguiente
        try:
            if f.get("source") == "zenfolio":
                u = urls.get(pid)
                datos, err = bajar_url(u) if u else (None, "thumb no dio url")
            else:
                datos, err = mini.de_drive(pid)
            if datos is None:
                return pid, None, err or "no se pudo bajar"
            img = decodificar(datos)
            if img is None:
                return pid, None, "no se pudo leer la imagen"
            return pid, Clip.preparar(img), None
        except Exception as e:                # una foto rota no tira el lote
            return pid, None, "error: %s" % str(e)[:80]

    ids, vecs, lote, lote_ids = [], [], [], []

    def correr_lote():
        # Un error del MODELO no es una foto rota: sube y voltea el paso. Si se
        # anotara por foto, un modelo que no carga marcaria miles de fotos como
        # "hechas con error" en la base.
        v = clip.vectores(lote)
        ids.extend(lote_ids)
        vecs.append(v)
        lote.clear()
        lote_ids.clear()

    with ThreadPoolExecutor(HILOS_BAJADA) as ex:
        for pid, x, err in ex.map(traer, filas):
            if x is None:
                if err:
                    errores[pid] = err
                continue
            lote.append(x)
            lote_ids.append(pid)
            if len(lote) >= LOTE_MODELO:
                correr_lote()
    if lote:
        correr_lote()
    v = np.concatenate(vecs) if vecs else np.zeros((0, DIM), np.float32)
    return ids, v, errores


# --------------------------------------------------------- vectores (trozos)

def cuantizar(v):
    """float (n,512) normalizados -> int8 y escala por fila."""
    m = np.abs(v).max(axis=1)
    m[m == 0] = 1.0
    esc = (m / 127.0).astype(np.float32)
    q = np.clip(np.rint(v / esc[:, None]), -127, 127).astype(np.int8)
    return q, esc


def descuantizar(q, esc):
    v = q.astype(np.float32) * esc[:, None]
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


class Vectores:
    """Los vectores de todas las fotos ya pasadas por CLIP, en trozos de
    TROZO fotos. Se cargan una vez (np.load fuera de cualquier loop); agregar
    solo toca el ultimo trozo; un trozo lleno no se reescribe nunca."""

    def __init__(self, carpeta=HERE):
        self.carpeta = carpeta
        self.trozos = []          # [(ruta, ids list, q, esc)]
        self.indice = {}          # id -> (trozo, fila)
        for ruta in sorted(glob.glob(os.path.join(carpeta, PATRON_TROZO))):
            with np.load(ruta, allow_pickle=False) as z:
                ids = [x.decode("ascii") if isinstance(x, bytes) else str(x) for x in z["ids"]]
                q, esc = z["q"], z["esc"]
            k = len(self.trozos)
            self.trozos.append([ruta, ids, q, esc])
            for i, pid in enumerate(ids):
                self.indice[pid] = (k, i)
        self._sucios = set()

    def __len__(self):
        return len(self.indice)

    def __contains__(self, pid):
        return pid in self.indice

    def nombre_trozo(self, k):
        return os.path.join(self.carpeta, "_escenas_%03d.npz" % k)

    def agregar(self, ids, vecs):
        """Agrega fotos nuevas (las que ya estan se ignoran). No escribe a
        disco: guardar() lo hace, por tandas."""
        nuevos = [(pid, i) for i, pid in enumerate(ids) if pid not in self.indice]
        if not nuevos:
            return 0
        q, esc = cuantizar(np.asarray(vecs, np.float32)[[i for _, i in nuevos]])
        j = 0
        while j < len(nuevos):
            if not self.trozos or len(self.trozos[-1][1]) >= TROZO:
                k = len(self.trozos)
                self.trozos.append([self.nombre_trozo(k), [], np.zeros((0, DIM), np.int8),
                                    np.zeros(0, np.float32)])
            k = len(self.trozos) - 1
            t = self.trozos[k]
            cabe = TROZO - len(t[1])
            tramo = nuevos[j:j + cabe]
            n0 = len(t[1])
            t[1] = t[1] + [pid for pid, _ in tramo]
            t[2] = np.concatenate([t[2], q[j:j + len(tramo)]])
            t[3] = np.concatenate([t[3], esc[j:j + len(tramo)]])
            for r, (pid, _) in enumerate(tramo):
                self.indice[pid] = (k, n0 + r)
            self._sucios.add(k)
            j += len(tramo)
        return len(nuevos)

    def guardar(self):
        """Escribe los trozos que cambiaron (atomico: tmp y replace)."""
        for k in sorted(self._sucios):
            ruta, ids, q, esc = self.trozos[k]
            tmp = ruta + ".tmp.npz"
            # ids en ASCII ("S"): en unicode numpy usa 4 bytes por letra y los
            # ids pesaban un cuarto del trozo
            np.savez(tmp, ids=np.array(ids, dtype="S"), q=q, esc=esc)
            os.replace(tmp, ruta)
        n = len(self._sucios)
        self._sucios = set()
        return n

    def de(self, ids):
        """(ids presentes, (n,512) float32 normalizados)."""
        hay = [pid for pid in ids if pid in self.indice]
        if not hay:
            return [], np.zeros((0, DIM), np.float32)
        q = np.empty((len(hay), DIM), np.int8)
        esc = np.empty(len(hay), np.float32)
        for r, pid in enumerate(hay):
            k, i = self.indice[pid]
            q[r] = self.trozos[k][2][i]
            esc[r] = self.trozos[k][3][i]
        return hay, descuantizar(q, esc)
