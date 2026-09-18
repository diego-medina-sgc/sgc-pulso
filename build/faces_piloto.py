# -*- coding: utf-8 -*-
"""
Mide si el reconocimiento sirve para reducir la cola de retratos sin nombre.

Dos preguntas distintas, que se confunden fácil:

  ACIERTA   Cuando propone un nombre, ¿es el correcto? Se mide dejando afuera
            cada retrato conocido y viendo si el resto del padrón lo reconoce.
            Es la única forma honesta: hay respuesta correcta para comparar.

  ALCANZA   ¿A cuántos de los 10.487 pendientes les propone algo? Un sistema
            que acierta el 100% pero contesta en el 2% de los casos no sirve.

    python faces_piloto.py --muestra 300
"""
import argparse
import os
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import faces
import sb
from index_drive import Drive, SA_PATH

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = faces.npz("_referencias.npz")
HILOS = 6
UMBRALES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def cargar_refs(cli):
    """Huellas del archivo, identidades de la base.

    La huella es lo caro de calcular y no cambia nunca: se cachea. A quién
    pertenece cada foto sí cambia — se fusionan duplicados, se corrigen
    identificaciones erradas — así que eso se relee siempre. Si se confiara
    en el person_id guardado, después de una fusión el padrón mediría contra
    personas que ya no existen.
    """
    if not os.path.exists(REFS):
        sys.exit("Faltan las referencias. Corré antes:  python faces_referencias.py")
    z = np.load(REFS, allow_pickle=True)
    fotos = [str(x) for x in z["photo_ids"]]

    actual = {r["photo_id"]: r["person_id"]
              for r in cli.select("photo_people", select="photo_id,person_id")}
    validas = {p["id"] for p in cli.select("people", select="id,kind")
               if p.get("kind") != "noise"}

    keep = [i for i, f in enumerate(fotos)
            if actual.get(f) in validas]
    perdidas = len(fotos) - len(keep)
    if perdidas:
        print("  descartadas %d referencias (identidad borrada o marcada como ruido)"
              % perdidas)
    return (z["vecs"][keep].astype(np.float32),
            np.array([actual[fotos[i]] for i in keep], dtype=int),
            z["anios"][keep].astype(int),
            np.array([fotos[i] for i in keep]))


def acierta(vecs, personas):
    """Deja afuera cada retrato y ve si el resto lo reconoce bien.

    Sólo cuenta a quien tiene otro retrato en el padrón: si alguien aparece
    una sola vez, sacarlo hace que la respuesta correcta no exista y medirlo
    sería mentirse.
    """
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -2.0)          # nunca compararse consigo mismo
    veces = {}
    for p in personas:
        veces[p] = veces.get(p, 0) + 1
    evaluables = [i for i in range(len(personas)) if veces[personas[i]] >= 2]

    print("\nACIERTA  (sobre %d retratos de gente con más de una foto)" % len(evaluables))
    print("  umbral   propone      de esos, bien     mal")
    for u in UMBRALES:
        prop = bien = mal = 0
        for i in evaluables:
            j = int(np.argmax(sim[i]))
            if sim[i][j] < u:
                continue
            prop += 1
            if personas[j] == personas[i]:
                bien += 1
            else:
                mal += 1
        pct_prop = 100.0 * prop / max(len(evaluables), 1)
        pct_bien = 100.0 * bien / max(prop, 1)
        print("   %.2f    %4d (%2.0f%%)    %4d (%5.1f%%)    %3d"
              % (u, prop, pct_prop, bien, pct_bien, mal))


def pendientes(cli, limite):
    """Retratos en carpetas de mugshots que todavía no tienen identidad."""
    print("\nBuscando retratos pendientes…")
    fol = {f["id"]: f for f in cli.select("folders", select="id,year,campus,is_mugshot")
           if f.get("is_mugshot")}
    ya = {r["photo_id"] for r in cli.select("photo_people", select="photo_id")}

    ids = list(fol)
    fotos = []
    # de a 40 carpetas por pedido: 437 ids en una sola URL la hacen explotar
    for i in range(0, len(ids), 40):
        trozo = ids[i:i + 40]
        fotos += cli.select("photos", select="id,folder_id,source",
                            folder_id="in.(%s)" % ",".join(trozo))
    pend = [(f["id"], fol[f["folder_id"]].get("year"), fol[f["folder_id"]].get("campus"))
            for f in fotos
            if f["id"] not in ya and f.get("source") == "drive"]
    print("  retratos en total: %d   pendientes desde Drive: %d" % (len(fotos), len(pend)))

    # muestreo estable: siempre la misma muestra entre corridas
    pend.sort(key=lambda t: t[0])
    paso = max(len(pend) // limite, 1)
    return pend[::paso][:limite]


def alcanza(muestra, vecs, personas, anios):
    d = Drive(SA_PATH)
    lock = threading.Lock()
    local = threading.local()
    hecho = {"n": 0, "sin_cara": 0, "error": 0}

    def mis_caras():
        if not hasattr(local, "c"):
            local.c = faces.Caras()
        return local.c

    def procesar(t):
        pid, anio, camp = t
        try:
            link = d.get("files/" + pid, fields="thumbnailLink").get("thumbnailLink")
            if not link:
                return None
            with urllib.request.urlopen(link.replace("=s220", "=s1600"), timeout=60) as r:
                data = r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
            with lock:
                hecho["error"] += 1
            return None
        hs = mis_caras().huellas(data, max_caras=1)
        with lock:
            hecho["n"] += 1
            if hecho["n"] % 50 == 0:
                print("  %d/%d" % (hecho["n"], len(muestra)))
            if not hs:
                hecho["sin_cara"] += 1
        return None if not hs else (hs[0][0], anio)

    res = []
    TANDA = 400
    for i in range(0, len(muestra), TANDA):
        d._refresh()
        with ThreadPoolExecutor(max_workers=HILOS) as ex:
            for r in ex.map(procesar, muestra[i:i + TANDA]):
                if r:
                    res.append(r)

    print("\nCon cara detectada: %d de %d   sin cara: %d   error: %d"
          % (len(res), len(muestra), hecho["sin_cara"], hecho["error"]))
    if not res:
        return

    mejores, mejores_cerca = [], []
    for vec, anio in res:
        s = vecs @ vec
        mejores.append(float(s.max()))
        # sólo referencias de años cercanos: la calibración mostró que la
        # misma persona pierde mucho parecido de un año a otro
        if anio:
            cerca = np.abs(anios - int(anio)) <= 1
            mejores_cerca.append(float(s[cerca].max()) if cerca.any() else -1.0)

    print("\nALCANZA  (a cuántos pendientes les propone algo)")
    print("  umbral   contra todo el padrón     sólo años cercanos")
    for u in UMBRALES:
        a = sum(1 for x in mejores if x >= u)
        b = sum(1 for x in mejores_cerca if x >= u)
        print("   %.2f     %4d/%-4d (%2.0f%%)          %4d/%-4d (%2.0f%%)"
              % (u, a, len(mejores), 100.0 * a / len(mejores),
                 b, len(mejores_cerca), 100.0 * b / max(len(mejores_cerca), 1)))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--muestra", type=int, default=300)
    a = ap.parse_args()

    cli = sb.SB()
    vecs, personas, anios, _ = cargar_refs(cli)
    print("Referencias: %d caras de %d personas" % (len(vecs), len(set(personas))))

    acierta(vecs, personas)

    muestra = pendientes(cli, a.muestra)
    print("Muestra a procesar: %d" % len(muestra))
    alcanza(muestra, vecs, personas, anios)


if __name__ == "__main__":
    main()
