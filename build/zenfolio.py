# -*- coding: utf-8 -*-
"""
Descarga de imagenes de Zenfolio. Modulo compartido; no se corre solo.

Las imagenes de Zenfolio no se pueden pedir con una URL a secas: exigen dos
cookies de sesion (`zf_keyring` y `zf_pat`). El keyring se arma abriendo cada
"realm" con la contraseña de galeria, y es acumulativo: se le van sumando
realms y sirve para todos.

Un detalle que costo caro en la Edge Function: los realm id son enteros de 19
digitos. En JavaScript Number("1099698941514015364") da ...400 y el realm deja
de existir. Python maneja enteros grandes sin perder precision, asi que aca no
hay que hacer nada especial — pero conviene tenerlo presente si esto se
reescribe en otro lenguaje.

Y solo funcionan los tamaños chicos: el AccessMask de la cuenta protege los
originales. El sufijo -2 son 400 px, el -1 unos 200.
"""
import os
import urllib.request

from scan_zenfolio import Zen, load_env, HTTP_UA


class Zenfolio:
    def __init__(self):
        load_env()
        self.user = os.environ.get("ZENFOLIO_USER")
        self.pw = os.environ.get("ZENFOLIO_ACCOUNT_PASSWORD")
        self.galeria = os.environ.get("ZENFOLIO_PASSWORD")
        if not (self.user and self.pw and self.galeria):
            raise SystemExit(
                "Faltan credenciales de Zenfolio en build/.env:\n"
                "  ZENFOLIO_USER, ZENFOLIO_ACCOUNT_PASSWORD, ZENFOLIO_PASSWORD")
        self.z = Zen()
        self.abiertos = set()
        self.z.login(self.user, self.pw)

    def abrir(self, realms):
        """Suma al keyring los realms que falten."""
        for r in realms:
            if not r or r in self.abiertos:
                continue
            try:
                # el realm va como entero: Python no lo redondea
                self.z.keyring = self.z.rpc(
                    "KeyringAddKeyPlain", [self.z.keyring, int(r), self.galeria])
                self.abiertos.add(r)
            except Exception as e:
                print("  no se pudo abrir el realm %s: %s" % (r, str(e)[:80]))

    @property
    def cookie(self):
        return "zf_keyring=%s; zf_pat=%s" % (self.z.keyring, self.z.token)

    def bajar(self, url_host, url_core, tam=2):
        """Bytes de la imagen. tam 2 = 400 px, que es lo que la cuenta permite."""
        url = "https://%s%s-%d.jpg" % (url_host, url_core, tam)
        req = urllib.request.Request(
            url, headers={"Cookie": self.cookie, "User-Agent": HTTP_UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        # el placeholder de "protegido" es un PNG chico: si vino eso, no sirve
        if len(data) < 5000:
            return None
        return data
