# -*- coding: utf-8 -*-
"""El estado del pulso entre vueltas: las huellas de caras, cifradas.

    python estado.py cifrar  <carpeta>   # build/*_arcface.npz -> <carpeta>/*.enc
    python estado.py descifrar <carpeta> # <carpeta>/*.enc -> build/
    python estado.py subir   <carpeta>   # lo cifrado que cambio -> bucket privado
    python estado.py bajar   <carpeta>   # bucket privado -> <carpeta>
    python estado.py cambio  <carpeta>   # sale con 1 si build/ difiere de lo cifrado

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


def archivos_estado():
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
    man = leer_manifiesto(carpeta)
    actuales = archivos_estado()
    if set(man) != set(actuales):
        return True
    return any(sha(os.path.join(HERE, n)) != man[n]["sha256"] for n in actuales)


# ------------------------------------------------------------------ bucket

def _pedir(cli, metodo, ruta, datos=None, tipo="application/octet-stream"):
    h = {"apikey": cli.key, "Authorization": "Bearer " + cli.key}
    if datos is not None:
        h["Content-Type"] = tipo
        h["x-upsert"] = "true"
    req = urllib.request.Request("%s/storage/v1/%s" % (cli.url, ruta),
                                 data=datos, method=metodo, headers=h)
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def subir(carpeta):
    cli = sb.SB()
    man = leer_manifiesto(carpeta)
    try:
        remoto = json.loads(_pedir(cli, "GET", "object/%s/%s" % (BUCKET, MANIFIESTO)))
    except urllib.error.HTTPError:
        remoto = {}
    subidos = 0
    for n, info in man.items():
        if remoto.get(n, {}).get("sha256") == info["sha256"]:
            continue
        with open(os.path.join(carpeta, n + ".enc"), "rb") as fh:
            crudo = fh.read()
        partes = max(1, -(-len(crudo) // TROZO))
        for i in range(partes):
            _pedir(cli, "POST", "object/%s/%s.enc.%03d" % (BUCKET, n, i),
                   crudo[i * TROZO:(i + 1) * TROZO])
        info["partes"] = partes
        subidos += 1
    for n, info in man.items():
        if "partes" not in info:
            info["partes"] = remoto.get(n, {}).get("partes", 1)
    _pedir(cli, "POST", "object/%s/%s" % (BUCKET, MANIFIESTO),
           json.dumps(man).encode("utf-8"), "application/json")
    print("respaldo: %d archivos subidos al bucket privado" % subidos)


def bajar(carpeta):
    cli = sb.SB()
    os.makedirs(carpeta, exist_ok=True)
    man = json.loads(_pedir(cli, "GET", "object/%s/%s" % (BUCKET, MANIFIESTO)))
    for n, info in man.items():
        with open(os.path.join(carpeta, n + ".enc"), "wb") as fh:
            for i in range(int(info.get("partes", 1))):
                fh.write(_pedir(cli, "GET", "object/%s/%s.enc.%03d" % (BUCKET, n, i)))
    with open(os.path.join(carpeta, MANIFIESTO), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    print("bajados del respaldo: %d archivos" % len(man))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    orden, carpeta = sys.argv[1], sys.argv[2]
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
