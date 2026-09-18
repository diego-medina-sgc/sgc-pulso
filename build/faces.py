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
