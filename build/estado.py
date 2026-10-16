# -*- coding: utf-8 -*-
"""El estado del pulso entre vueltas: las huellas de caras, cifradas.

    python estado.py cifrar  <carpeta>   # build/*_arcface.npz -> <carpeta>/*.enc
    python estado.py descifrar <carpeta> # <carpeta>/*.enc -> build/
    python estado.py subir   <carpeta>   # lo cifrado que cambio -> bucket privado
    python estado.py bajar   <carpeta>   # bucket privado -> <carpeta>
    python estado.py cambio  <carpeta>   # sale con 1 si build/ difiere de lo cifrado

    python estado.py --conjunto escenas <orden> <carpeta>
        lo mismo para los vectores de "Que se ve" (_escenas_NNN.npz y la capa),
        que viajan APARTE de las huellas de caras; ademas:
    python estado.py --conjunto escenas traer <carpeta>       # baja del bucket solo
        lo que difiere de la cache, descifra y deja la marca _escenas_estado.ok
    python estado.py --conjunto escenas clave-cache <carpeta> # clave para la cache

POR QUE (18/9/2026)

El pulso pasa de la notebook de Diego a GitHub Actions, en un repo publico:
"necesitamos que el pulso corra en un servidor", sin pagar. Un runner de
Actions arranca vacio cada vez, y el pulso necesita ~790 MB de huellas de
caras (los .npz) que armo en vueltas anteriores. Recalcularlas lleva dias.

Son datos biometricos de chicos. En un repo publico:
  - la cache de Actions la puede restaurar un pull request de cualquiera;
  - los artefactos los baja cualquier usuario de GitHub.
Asi que nada sale del runner sin cifrar: AES-256-GCM con la clave en el secret
ESTADO_CLAVE, que los pull requests de terceros no reciben. GCM ademas
autentica: un archivo cambiado por otro no se descifra, y np.load nunca ve
un .npz que no haya escrito el propio pulso.

Donde vive:
  1. la cache de Actions (rapida, sin costo, se borra si no se usa 7 dias);
  2. un bucket PRIVADO de Supabase, "pulso-estado" (el respaldo durable).
     Subir no cuesta egress; bajar solo cuando la cache se perdio.
El bucket parte cada archivo en trozos de 45 MB: el limite por archivo del
proyecto es 50 MB.
"""
import base64
import glob
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import sb

HERE = os.path.dirname(os.path.abspath(__file__))
BUCKET = "pulso-estado"
TROZO = 45 * 1024 * 1024
MANIFIESTO = "manifiesto.json"

# Lo que el pulso necesita de una vuelta a la otra. Afuera: respaldos viejos
# y la calibracion, que no usa ningun paso.
AFUERA = (".ANTES", "_calib_")


# LOS VECTORES DE ESCENAS VAN APARTE (14/10/2026)
#
# "Que se ve" guarda un vector CLIP por foto primaria (~250 MB al final). No
# son huellas de caras, pero son del archivo: tambien viajan cifrados. Van en
# su propio conjunto -otro patron, otra carpeta, otra cache y otro prefijo en
# el bucket- para que un problema de uno no toque al otro: el estado de caras
# sigue siendo exactamente lo que era (*_arcface.npz) y nunca incluye estos.
CONJUNTOS = {
    "caras": {"patrones": ["*_arcface.npz"], "prefijo": "", "marca": None},
    "escenas": {"patrones": ["_escenas_[0-9][0-9][0-9].npz", "_escenas_capa.npz"],
                "prefijo": "escenas/", "marca": "_escenas_estado.ok"},
}
CONJUNTO = "caras"
# cuantos trozos de escenas cambian la clave de la cache (ver clave_cache)
TROZOS_POR_CLAVE = 5


def conj():
    return CONJUNTOS[CONJUNTO]


def PRE():
    return conj()["prefijo"]


def ruta_marca():
    m = conj()["marca"]
    return os.path.join(HERE, m) if m else None


def exigir_marca():
    """Escenas: no se cifra ni se sube nada si el estado no se trajo bien en
    esta vuelta. Si no, unos trozos nuevos (empezados de cero) pisarian en el
    bucket a los buenos."""
    m = ruta_marca()
    if m and not os.path.exists(m):
        sys.exit("No se trajo el estado de %s en esta vuelta (falta %s): no se guarda."
                 % (CONJUNTO, os.path.basename(m)))


def archivos_estado():
    if CONJUNTO != "caras":
        out = set()
        for pat in conj()["patrones"]:
            out |= {os.path.basename(x) for x in glob.glob(os.path.join(HERE, pat))}
        return sorted(n for n in out if not n.endswith(".tmp.npz"))
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, "*_arcface.npz"))):
        n = os.path.basename(p)
        if any(x in n for x in AFUERA):
            continue
        out.append(n)
    marca = os.path.join(HERE, "_nocturno.motor")
    if os.path.exists(marca):
        out.append("_nocturno.motor")
    return out


def clave():
    sb.load_env()
    k = os.environ.get("ESTADO_CLAVE", "")
    if not k:
        sys.exit("Falta ESTADO_CLAVE (32 bytes en base64).")
    b = base64.b64decode(k)
    if len(b) != 32:
        sys.exit("ESTADO_CLAVE tiene que ser de 32 bytes.")
    return AESGCM(b)


def sha(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as fh:
        for bloque in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_manifiesto(carpeta):
    p = os.path.join(carpeta, MANIFIESTO)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def cifrar(carpeta):
    exigir_marca()
    os.makedirs(carpeta, exist_ok=True)
    viejo = leer_manifiesto(carpeta)
    aes = clave()
    nuevo, hechos = {}, 0
    for n in archivos_estado():
        ruta = os.path.join(HERE, n)
        h = sha(ruta)
        # la fecha tambien viaja: pulso.py decide si un paso cambio por tamaño y
        # fecha de sus .npz (archivos()), y un archivo descifrado hoy parece
        # nuevo aunque sea el mismo. Sin esto, en cada vuelta corrian todos.
        nuevo[n] = {"sha256": h, "bytes": os.path.getsize(ruta),
                    "mtime": int(os.path.getmtime(ruta))}
        destino = os.path.join(carpeta, n + ".enc")
        if viejo.get(n, {}).get("sha256") == h and os.path.exists(destino):
            continue                       # el .enc sirve; el manifiesto se reescribe igual
        nonce = os.urandom(12)
        with open(ruta, "rb") as fh:
            datos = fh.read()
        # el nombre va como dato asociado: un .enc no se puede hacer pasar por otro
        with open(destino, "wb") as fh:
            fh.write(nonce + aes.encrypt(nonce, datos, n.encode("utf-8")))
        hechos += 1
    for n in set(viejo) - set(nuevo):
        try:
            os.remove(os.path.join(carpeta, n + ".enc"))
        except OSError:
            pass
    with open(os.path.join(carpeta, MANIFIESTO), "w", encoding="utf-8") as fh:
        json.dump(nuevo, fh, indent=1, sort_keys=True)
    print("cifrados: %d de %d archivos" % (hechos, len(nuevo)))


def descifrar(carpeta):
    man = leer_manifiesto(carpeta)
    if not man:
        sys.exit("No hay estado en %s: no se puede correr el pulso sin huellas." % carpeta)
    aes = clave()
    for n, info in man.items():
        with open(os.path.join(carpeta, n + ".enc"), "rb") as fh:
            crudo = fh.read()
        datos = aes.decrypt(crudo[:12], crudo[12:], n.encode("utf-8"))
        if hashlib.sha256(datos).hexdigest() != info["sha256"]:
            sys.exit("%s no coincide con el manifiesto" % n)
        ruta = os.path.join(HERE, n)
        with open(ruta, "wb") as fh:
            fh.write(datos)
        if info.get("mtime"):
            os.utime(ruta, (info["mtime"], info["mtime"]))
    print("descifrados: %d archivos" % len(man))


def cambio(carpeta):
    """Si hay que volver a cifrar y guardar: compara CONTENIDO, no fecha.

    Un .npz reescrito igual no se vuelve a guardar (serian ~790 MB de cache
    por nada), asi que su fecha nueva se pierde y la vuelta siguiente lo
    descifra con la del manifiesto. Hasta el 5/10/2026 eso disparaba otra vez
    todo lo que lo lee, porque pulso.py firmaba los .npz por tamaño y fecha.
    El arreglo esta del lado de pulso.py (archivos() / mismo_npz): la firma
    lleva el sha256 y se compara por contenido, asi que la fecha que restaura
    descifrar() ya no importa. Aca no hace falta comparar fechas.
    """
    man = leer_manifiesto(carpeta)
    actuales = archivos_estado()
    if set(man) != set(actuales):
        return True
    return any(sha(os.path.join(HERE, n)) != man[n]["sha256"] for n in actuales)


def traer(carpeta):
    """Escenas: lo de la cache (si hubo) mas lo que el bucket tiene distinto,
    descifrado en build/, y la marca. Sin manifiesto en el bucket es la
    primera vez: marca de estado nuevo y nada que descifrar. Cualquier otro
    error sale sin marca: el paso no corre y no se guarda nada."""
    cli = sb.SB()
    os.makedirs(carpeta, exist_ok=True)
    local = leer_manifiesto(carpeta)
    try:
        remoto = json.loads(_pedir(cli, "GET", "object/%s/%s%s" % (BUCKET, PRE(), MANIFIESTO)))
    except urllib.error.HTTPError as e:
        # Storage contesta 400 (not_found) o 404 si el objeto no existe
        if e.code in (400, 404):
            remoto = None
        else:
            raise
    if remoto is None:
        if local:
            sys.exit("la cache tiene estado de %s y el bucket no: no se sigue" % CONJUNTO)
        with open(ruta_marca(), "w") as fh:
            fh.write("nuevo")
        print("estado de %s: el bucket no tiene ninguno (primera vez)" % CONJUNTO)
        return
    bajados = 0
    for n, info in remoto.items():
        destino = os.path.join(carpeta, n + ".enc")
        if local.get(n, {}).get("sha256") == info["sha256"] and os.path.exists(destino):
            continue
        with open(destino + ".bajando", "wb") as fh:
            for i in range(int(info.get("partes", 1))):
                fh.write(_pedir(cli, "GET", "object/%s/%s%s.enc.%03d" % (BUCKET, PRE(), n, i)))
        os.replace(destino + ".bajando", destino)
        bajados += 1
    with open(os.path.join(carpeta, MANIFIESTO), "w", encoding="utf-8") as fh:
        json.dump(remoto, fh, indent=1, sort_keys=True)
    descifrar(carpeta)
    with open(ruta_marca(), "w") as fh:
        fh.write("traido")
    print("estado de %s: %d de %d archivos del bucket (el resto, de la cache)"
          % (CONJUNTO, bajados, len(remoto)))


def clave_cache(carpeta):
    """Clave de la cache de Actions de escenas. Cambia cada TROZOS_POR_CLAVE
    trozos y una vez por dia: asi no se guarda una copia entera (hasta ~250
    MB) en cada vuelta, y lo que cambio en el dia (el trozo abierto) se baja
    del bucket con traer()."""
    import datetime as dt
    man = leer_manifiesto(carpeta)
    trozos = sum(1 for n in man if n[len("_escenas_"):-len(".npz")].isdigit())
    return "escenas-v1-%s-g%03d" % (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d"),
                                    trozos // TROZOS_POR_CLAVE)


# ------------------------------------------------------------------ bucket

def _pedir(cli, metodo, ruta, datos=None, tipo="application/octet-stream"):
    h = {"apikey": cli.key, "Authorization": "Bearer " + cli.key}
    if datos is not None:
        h["Content-Type"] = tipo
        h["x-upsert"] = "true"
    req = urllib.request.Request("%s/storage/v1/%s" % (cli.url, ruta),
                                 data=datos, method=metodo, headers=h)
    # Reintentos: el 18/9 un trozo de 45 MB volvio con 504 Gateway Timeout y
    # tiro abajo la corrida entera, que en la anterior habia bajado todo bien.
    # Un 404 no se reintenta: es "no existe" (el primer respaldo, por ejemplo).
    for intento in range(6):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 or intento == 5:
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if intento == 5:
                raise
        time.sleep(min(2 ** intento * 3, 60))


def subir(carpeta):
    cli = sb.SB()
    man = leer_manifiesto(carpeta)
    try:
        remoto = json.loads(_pedir(cli, "GET", "object/%s/%s%s" % (BUCKET, PRE(), MANIFIESTO)))
    except urllib.error.HTTPError:
        remoto = {}
    if CONJUNTO != "caras":
        exigir_marca()
        # nunca achicar lo guardado: si aca faltan archivos que el bucket
        # tiene, el estado no se trajo bien y subir los sacaria del manifiesto
        perdidos = sorted(set(remoto) - set(man))
        if perdidos:
            sys.exit("el bucket tiene %d archivos que aca no estan (%s...): no se sube"
                     % (len(perdidos), perdidos[0]))
    subidos = 0
    for n, info in man.items():
        if remoto.get(n, {}).get("sha256") == info["sha256"]:
            continue
        with open(os.path.join(carpeta, n + ".enc"), "rb") as fh:
            crudo = fh.read()
        partes = max(1, -(-len(crudo) // TROZO))
        for i in range(partes):
            _pedir(cli, "POST", "object/%s/%s%s.enc.%03d" % (BUCKET, PRE(), n, i),
                   crudo[i * TROZO:(i + 1) * TROZO])
        info["partes"] = partes
        subidos += 1
    for n, info in man.items():
        if "partes" not in info:
            info["partes"] = remoto.get(n, {}).get("partes", 1)
    _pedir(cli, "POST", "object/%s/%s%s" % (BUCKET, PRE(), MANIFIESTO),
           json.dumps(man).encode("utf-8"), "application/json")
    print("respaldo: %d archivos subidos al bucket privado" % subidos)


def bajar(carpeta):
    cli = sb.SB()
    os.makedirs(carpeta, exist_ok=True)
    man = json.loads(_pedir(cli, "GET", "object/%s/%s%s" % (BUCKET, PRE(), MANIFIESTO)))
    for n, info in man.items():
        with open(os.path.join(carpeta, n + ".enc"), "wb") as fh:
            for i in range(int(info.get("partes", 1))):
                fh.write(_pedir(cli, "GET", "object/%s/%s%s.enc.%03d" % (BUCKET, PRE(), n, i)))
    with open(os.path.join(carpeta, MANIFIESTO), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    print("bajados del respaldo: %d archivos" % len(man))


def main():
    global CONJUNTO
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "--conjunto":
        if args[1] not in CONJUNTOS:
            sys.exit("conjunto desconocido: %s" % args[1])
        CONJUNTO = args[1]
        args = args[2:]
    if len(args) != 2:
        sys.exit(__doc__)
    orden, carpeta = args
    if orden in ("traer", "clave-cache") and CONJUNTO == "caras":
        sys.exit("%s es solo para --conjunto escenas" % orden)
    if orden == "traer":
        traer(carpeta)
        return
    if orden == "clave-cache":
        print(clave_cache(carpeta))
        return
    if orden == "cifrar":
        cifrar(carpeta)
    elif orden == "descifrar":
        descifrar(carpeta)
    elif orden == "subir":
        subir(carpeta)
    elif orden == "bajar":
        bajar(carpeta)
    elif orden == "cambio":
        sys.exit(1 if cambio(carpeta) else 0)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
