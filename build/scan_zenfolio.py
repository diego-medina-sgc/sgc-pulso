# -*- coding: utf-8 -*-
"""
Relevamiento de la cuenta de Zenfolio de SGC (campus Quilmes).

La cuenta es del colegio (stgeorgescollege.zenfolio.com), está activa y cubre
desde 2017, el mismo rango que la carpeta de Drive de Quilmes. La pregunta que
este script contesta es si son el mismo material y qué metadata aporta
Zenfolio que a Drive le falta: en Drive el 92,3% de las carpetas de Quilmes no
tiene fecha en el nombre.

Dos modos de acceso, porque no ven lo mismo:

  GALERIA  La contraseña de la galería abre un "keyring" y muestra lo que está
           publicado. Alcanza para mirar, pero la galería pública no
           necesariamente incluye todo lo que hay en la biblioteca.

  DUEÑO    Las credenciales de la cuenta dan acceso al contenido completo,
           incluido lo no publicado. Usa challenge-response, así que la
           contraseña nunca viaja por la red.

Ninguna credencial va en el código ni en el repo: todo sale de build/.env,
que está en .gitignore.

    build/.env
        ZENFOLIO_PASSWORD=...              # contraseña de la galería
        ZENFOLIO_USER=...                  # usuario de la cuenta (modo dueño)
        ZENFOLIO_ACCOUNT_PASSWORD=...

Uso:
    python scan_zenfolio.py                  # usa lo que encuentre en .env
    python scan_zenfolio.py --sample 3       # además, fotos de 3 galerías
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = "https://api.zenfolio.com/api/1.8/zfapi.asmx"
# X-Zenfolio-User-Agent identifica a la app ante la API. El User-Agent HTTP
# es otra cosa: Zenfolio esta detras de Cloudflare, que bloquea con error 1010
# los agentes de Python y Deno. Sin uno de navegador, todo devuelve 403.
UA = "SGC-PhotoLibrary/1.0"
HTTP_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
LOGIN = "stgeorgescollege"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))


def load_env():
    """Lee build/.env.

    utf-8-sig y no utf-8: PowerShell 5.1 escribe los archivos con BOM, y con
    utf-8 a secas la primera clave quedaria como '﻿ZENFOLIO_PASSWORD' y
    nunca matchearia. Se sacan tambien las comillas, porque es habitual
    pegarlas al copiar una contraseña.
    """
    p = os.path.join(HERE, ".env")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) > 1 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            os.environ.setdefault(k.strip(), v)


class Zen:
    def __init__(self):
        self.keyring = ""
        self.token = None

    def rpc(self, method, params):
        body = json.dumps({"method": method, "params": params, "id": 1}).encode()
        h = {"Content-Type": "application/json",
             "X-Zenfolio-User-Agent": UA, "User-Agent": HTTP_UA}
        if self.keyring:
            h["X-Zenfolio-Keyring"] = self.keyring
        if self.token:
            h["X-Zenfolio-Token"] = self.token
        req = urllib.request.Request(API, data=body, method="POST", headers=h)
        for i in range(4):
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    out = json.loads(r.read().decode("utf-8"))
                if out.get("error"):
                    raise RuntimeError("%s: %s" % (method, out["error"]))
                return out["result"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503) and i < 3:
                    time.sleep(2 ** i)
                    continue
                raise RuntimeError("%s -> HTTP %s %s"
                                   % (method, e.code, e.read().decode("utf-8", "replace")[:200]))

    def unlock(self, realm_id, password):
        """Abre el realm protegido y guarda el keyring para las llamadas siguientes."""
        self.keyring = self.rpc("KeyringAddKeyPlain", [self.keyring, realm_id, password])
        return self.keyring

    def login(self, user, password):
        """Autenticación como dueño, con challenge-response.

        La contraseña no viaja: el servidor manda un desafío y una sal, y se
        devuelve SHA256(desafío + SHA256(sal + contraseña)). Zenfolio expone
        también AuthenticatePlain, que sí manda la contraseña en el cuerpo del
        pedido; no se usa porque no hace falta.
        """
        def b(v):
            # sobre JSON-RPC los byte[] de .NET llegan como lista de enteros,
            # a veces con signo; base64 solo aparece en el transporte SOAP
            if isinstance(v, str):
                return base64.b64decode(v)
            return bytes(x & 0xFF for x in v)

        ch = self.rpc("GetChallenge", [user])
        salt = b(ch["PasswordSalt"])
        chal = b(ch["Challenge"])
        pw_hash = hashlib.sha256(salt + password.encode("utf-8")).digest()
        proof = hashlib.sha256(chal + pw_hash).digest()
        # se devuelven como lista de enteros, en el mismo formato en que
        # llegaron: mandarlos en base64 hace fallar Authenticate
        self.token = self.rpc("Authenticate", [list(chal), list(proof)])
        return self.token


def walk(node, path, depth, grupos, galerias):
    for e in (node.get("Elements") or []):
        t = e.get("$type", "")
        titulo = e.get("Title") or "(sin titulo)"
        if "Group" in t:
            grupos.append({"title": titulo, "depth": depth, "path": path + "/" + titulo})
            walk(e, path + "/" + titulo, depth + 1, grupos, galerias)
        else:
            def dt(v):
                return (v or {}).get("Value") if isinstance(v, dict) else v
            galerias.append({
                "id": e.get("Id"), "title": titulo, "depth": depth,
                "path": path + "/" + titulo,
                "photos": e.get("PhotoCount"),
                "created": dt(e.get("CreatedOn")),
                "modified": dt(e.get("ModifiedOn")),
                "url": e.get("PageUrl"),
            })


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="cuántas galerías abrir para inspeccionar sus fotos")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(DATA, exist_ok=True)

    pwd = os.environ.get("ZENFOLIO_PASSWORD")
    user = os.environ.get("ZENFOLIO_USER")
    acct = os.environ.get("ZENFOLIO_ACCOUNT_PASSWORD")
    if not pwd and not (user and acct):
        sys.exit("Falta configurar build/.env. Alguna de las dos:\n"
                 "  ZENFOLIO_PASSWORD=...  (contraseña de la galería)\n"
                 "  ZENFOLIO_USER=... y ZENFOLIO_ACCOUNT_PASSWORD=...  (cuenta, ve todo)")

    z = Zen()
    perfil = z.rpc("LoadPublicProfile", [LOGIN])
    root = perfil.get("RootGroup") or {}
    realm = ((root.get("AccessDescriptor") or {}).get("RealmId"))
    print("Cuenta: %s  (%s)" % (perfil.get("DisplayName"), LOGIN))
    print("Grupo raiz: %s   realm=%s" % (root.get("Title"), realm))

    # el modo dueño ve todo, así que si hay credenciales de cuenta van primero
    if user and acct:
        z.login(user, acct)
        print("Autenticado como dueño: se ve el contenido completo.\n")
    elif realm:
        z.unlock(realm, pwd)
        print("Acceso por contraseña de galería: se ve solo lo publicado.\n")

    root = z.rpc("LoadGroupHierarchy", [LOGIN])
    grupos, galerias = [], []
    walk(root, "", 1, grupos, galerias)

    total = sum(g["photos"] or 0 for g in galerias)
    print("Grupos:   %d" % len(grupos))
    print("Galerias: %d" % len(galerias))
    print("Fotos:    %d\n" % total)

    print("GRUPOS DE NIVEL 1")
    for g in [x for x in grupos if x["depth"] == 1]:
        print("   %s" % g["title"][:64])

    print("\nGALERIAS (primeras 25, por fecha)")
    for g in sorted(galerias, key=lambda x: x["created"] or "")[:25]:
        print("   %-50s %5s fotos  %s"
              % (g["title"][:50], g["photos"], (g["created"] or "")[:10]))

    dest = os.path.join(DATA, "zenfolio_scan.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump({"grupos": grupos, "galerias": galerias}, fh, ensure_ascii=False)
    print("\nEscrito: %s" % dest)

    # Una galería abierta dice qué metadata trae cada foto, que es lo que
    # define si Zenfolio sirve como fuente de fechas y títulos.
    for g in galerias[:a.sample]:
        try:
            ps = z.rpc("LoadPhotoSet", [g["id"], "Full", True])
        except RuntimeError as e:
            print("\n  (no se pudo abrir %s: %s)" % (g["title"][:40], e))
            continue
        fotos = ps.get("Photos") or []
        print("\n=== %s  (%d fotos) ===" % (g["title"][:50], len(fotos)))
        for f in fotos[:3]:
            def dt(v):
                return (v or {}).get("Value") if isinstance(v, dict) else v
            print("   %-30s taken=%s  keywords=%s"
                  % ((f.get("FileName") or "")[:30],
                     (dt(f.get("TakenOn")) or "")[:19],
                     (f.get("Keywords") or [])[:4]))


if __name__ == "__main__":
    main()
