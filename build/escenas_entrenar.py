# -*- coding: utf-8 -*-
"""La capa lineal de "Que se ve": de vector CLIP a puntaje por escena.

    python escenas_entrenar.py --medir --tmp <carpeta fuera del repo>
        Solo lectura de la base. Elige la muestra, baja las miniaturas (en
        memoria: no guarda imagenes), saca los vectores a <carpeta>, entrena,
        mide dejando la carpeta afuera e imprime la tabla. No escribe nada en
        build/ ni en la base.

El pulso usa las mismas funciones (escenas_procesar.py): la primera vuelta de
cada VERSION arma la muestra, entrena y guarda _escenas_capa.npz en su estado
cifrado.

COMO ES (14/10/2026)

  - Una regresion logistica por escena (uno contra el resto, numpy), sobre el
    vector CLIP normalizado. Las escenas no se excluyen entre si: una foto de
    rugby puede ser tambien una foto grupal.
  - Ejemplos: lo que dice la carpeta (escenas_vocab.etiquetas_carpeta) y, por
    foto, photos.is_group. A lo sumo POR_CARPETA_MAX fotos de una carpeta: se
    cuentan sesiones, no fotos.
  - Se entrena una escena solo si tiene MIN_CARPETAS carpetas y MIN_FOTOS
    fotos de ejemplo. Las demas van por cero ejemplos (coseno contra su frase,
    softmax entre las 57 con la temperatura de CLIP).
  - Se valida en K_TANDAS tandas por CARPETA: ninguna foto se mide con un
    modelo que vio su carpeta.
  - NO entrena con escenas_respuestas: escena_medida (la base) mide con esas
    mismas respuestas para publicar, y entrenar con ellas inflaria el 95%.
    Si algun dia entran, tiene que ser con una parte apartada que la base no
    use para medir (pedido a base).
"""
import argparse
import datetime as dt
import json
import os
import random
import sys
import time

import numpy as np

import escenas_vocab as V

# Version de la capa: es parte del nombre del modelo que se carga en la base
# (vision_modelos). Cambiarla = reentrenar y repuntuar todas las fotos con los
# vectores guardados (sin volver a bajar imagenes), y las escenas publicadas se
# apagan hasta medirlas de nuevo (escena_pasa exige el mismo modelo). Por eso
# no se reentrena solo con cada respuesta: se cambia aca, con un commit.
VERSION = "v1"

# Muestra (medida 14/10/2026, ver la tabla en el informe de esa fecha)
POR_ESCENA = 400          # fotos de ejemplo por escena, como mucho
POR_CARPETA_MIN = 3
POR_CARPETA_MAX = 40
FONDO = 3000              # fotos al azar del archivo: negativos y cobertura real
SEMILLA = 20261014

# Una escena se entrena si tiene al menos esto; si no, cero ejemplos.
MIN_CARPETAS = 8
MIN_FOTOS = 80

K_TANDAS = 5
ESCALA = 10.0             # el vector normalizado por 10: la logistica converge
PASOS = 400
APRENDIZAJE = 0.05
REGULARIZACION = 1e-4
TEMPERATURA_CERO = 100.0  # la de CLIP (logit_scale)


def nombre_modelo():
    from escenas_clip import MODELO_REV
    return "clip-vitb32-%s+lineal-%s" % (MODELO_REV[:8], VERSION)


# ------------------------------------------------------------------ muestra

def leer_carpetas(cli):
    return {f["id"]: f for f in cli.select(
        "folders", select="id,name,title,display_title,event_types,activities,sports,"
                          "section,is_mugshot,noise")}


def leer_primarias(cli):
    return cli.select("photos", select="id,folder_id,source,is_group,thumb_url", is_primary="eq.true")


def muestra(carpetas, fotos, por_escena=POR_ESCENA, fondo=FONDO, semilla=SEMILLA):
    """[(photo_id, source, folder_id, set(etiquetas), es_fondo)] ordenada."""
    rnd = random.Random(semilla)
    et_carpeta = {}
    for fid, f in carpetas.items():
        if f.get("noise"):
            continue
        et_carpeta[fid] = V.etiquetas_carpeta(f)

    def etiquetas(p):
        e = set(et_carpeta.get(p["folder_id"], ()))
        if p.get("is_group"):
            e.add("foto_grupal")
        return e

    validas = [p for p in fotos if p.get("source") in ("drive", "zenfolio")
               and p.get("folder_id") in et_carpeta]
    por_carpeta = {}
    for p in validas:
        por_carpeta.setdefault(p["folder_id"], []).append(p)
    for lst in por_carpeta.values():
        lst.sort(key=lambda p: p["id"])

    elegidas = {}
    for clave in V.CLAVES:
        if clave == "foto_grupal":
            cand = [p for p in validas if p.get("is_group")]
            grupos = {}
            for p in sorted(cand, key=lambda p: p["id"]):
                grupos.setdefault(p["folder_id"], []).append(p)
        else:
            grupos = {fid: por_carpeta[fid] for fid in sorted(por_carpeta)
                      if clave in et_carpeta.get(fid, ())}
        if not grupos:
            continue
        k = max(POR_CARPETA_MIN, min(POR_CARPETA_MAX, -(-por_escena // len(grupos))))
        orden = sorted(grupos)
        rnd.shuffle(orden)
        n = 0
        for fid in orden:
            lst = grupos[fid][:]
            rnd.shuffle(lst)
            for p in lst[:k]:
                elegidas[p["id"]] = p
                n += 1
            if n >= por_escena:
                break
    todas = sorted(validas, key=lambda p: p["id"])
    de_fondo = set(p["id"] for p in rnd.sample(todas, min(fondo, len(todas))))
    for p in todas:
        if p["id"] in de_fondo:
            elegidas[p["id"]] = p
    return [(p["id"], p["source"], p["folder_id"], etiquetas(p), p["id"] in de_fondo)
            for p in sorted(elegidas.values(), key=lambda p: p["id"])]


def matriz_etiquetas(m):
    y = np.zeros((len(m), len(V.CLAVES)), np.float32)
    col = {c: i for i, c in enumerate(V.CLAVES)}
    for r, (_, _, _, et, _) in enumerate(m):
        for e in et:
            y[r, col[e]] = 1.0
    return y


# ----------------------------------------------------------------- entrenar

def _sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def entrenar(X, Y, cols):
    """Logistica uno contra el resto para las columnas cols, con peso
    balanceado (positivos y negativos pesan lo mismo). Adam, lote completo,
    determinista. Devuelve (W (512,k), b (k,))."""
    X = X * ESCALA
    Y = Y[:, cols]
    n, d = X.shape
    k = Y.shape[1]
    pos = Y.sum(0)
    wp = np.where(pos > 0, 0.5 * n / np.maximum(pos, 1), 0)
    wn = 0.5 * n / np.maximum(n - pos, 1)
    peso = Y * wp + (1 - Y) * wn
    W = np.zeros((d, k), np.float32)
    b = np.zeros(k, np.float32)
    mW, vW, mb, vb = 0 * W, 0 * W, 0 * b, 0 * b
    for t in range(1, PASOS + 1):
        p = _sigmoide(X @ W + b)
        g = (p - Y) * peso / n
        gW = X.T @ g + REGULARIZACION * W
        gb = g.sum(0)
        mW = 0.9 * mW + 0.1 * gW
        vW = 0.999 * vW + 0.001 * gW * gW
        mb = 0.9 * mb + 0.1 * gb
        vb = 0.999 * vb + 0.001 * gb * gb
        c1, c2 = 1 - 0.9 ** t, 1 - 0.999 ** t
        W -= APRENDIZAJE * (mW / c1) / (np.sqrt(vW / c2) + 1e-8)
        b -= APRENDIZAJE * (mb / c1) / (np.sqrt(vb / c2) + 1e-8)
    return W.astype(np.float32), b.astype(np.float32)


def entrenables(m):
    """Las claves con ejemplos suficientes: {clave: (carpetas, fotos)}."""
    out = {}
    for c in V.CLAVES:
        fotos = [x for x in m if c in x[3]]
        carp = {x[2] for x in fotos}
        out[c] = (len(carp), len(fotos))
    return out


def puntuar(X, capa):
    """(n, 57) puntajes 0..1 en el orden de V.CLAVES."""
    T = capa["frases"]
    z = TEMPERATURA_CERO * (X @ T.T)
    z -= z.max(1, keepdims=True)
    cero = np.exp(z)
    cero /= cero.sum(1, keepdims=True)
    out = cero.astype(np.float32)
    cols = [V.CLAVES.index(c) for c in capa["entrenadas"]]
    if cols:
        out[:, cols] = _sigmoide((X * ESCALA) @ capa["W"] + capa["b"])
    return out


def armar_capa(X, Y, cuentas, filas=None):
    """filas: la medicion (medir). Con ella, una escena con ejemplos va por la
    capa solo si su AUC dejando la carpeta afuera no es peor que la de cero
    ejemplos (14/10/2026: cocina, futbol y laboratorio median mejor sin
    entrenar)."""
    cols_c = [c for c in V.CLAVES if cuentas[c][0] >= MIN_CARPETAS and cuentas[c][1] >= MIN_FOTOS]
    if filas:
        f = {r["clave"]: r for r in filas}
        cols_c = [c for c in cols_c if f[c]["auc"] is None or f[c]["auc_cero"] is None
                  or f[c]["auc"] >= f[c]["auc_cero"]]
    cols = [V.CLAVES.index(c) for c in cols_c]
    W, b = entrenar(X, Y, cols)
    return {"W": W, "b": b, "entrenadas": cols_c, "frases": V.vectores_frases(),
            "modelo": nombre_modelo(), "claves": list(V.CLAVES)}


def guardar_capa(capa, ruta, medida=None):
    tmp = ruta + ".tmp.npz"
    np.savez(tmp, W=capa["W"], b=capa["b"], entrenadas=np.array(capa["entrenadas"]),
             frases=capa["frases"], modelo=np.array(capa["modelo"]),
             claves=np.array(capa["claves"]),
             medida=np.array(json.dumps(medida or {}, ensure_ascii=False)),
             fecha=np.array(dt.datetime.now(dt.timezone.utc).isoformat()))
    os.replace(tmp, ruta)


def leer_capa(ruta):
    if not os.path.exists(ruta):
        return None
    with np.load(ruta, allow_pickle=False) as z:
        capa = {"W": z["W"], "b": z["b"], "entrenadas": [str(x) for x in z["entrenadas"]],
                "frases": z["frases"], "modelo": str(z["modelo"]),
                "claves": [str(x) for x in z["claves"]]}
    if capa["claves"] != list(V.CLAVES):
        return None                      # vocabulario distinto: hay que reentrenar
    return capa


# -------------------------------------------------------------------- medir

CORTES = (0.5, 0.7, 0.9)


def medir(X, Y, m, cuentas):
    """Puntajes fuera de tanda (la carpeta nunca en entrenamiento y medicion a
    la vez) y la tabla por escena."""
    carpetas = sorted({x[2] for x in m})
    rnd = random.Random(SEMILLA)
    rnd.shuffle(carpetas)
    tanda = {c: i % K_TANDAS for i, c in enumerate(carpetas)}
    t = np.array([tanda[x[2]] for x in m])
    fondo = np.array([x[4] for x in m])
    P = np.zeros_like(Y)
    for k in range(K_TANDAS):
        tr, te = t != k, t == k
        cu = {c: (0, 0) for c in V.CLAVES}
        for c in V.CLAVES:
            sel = [i for i in np.where(tr)[0] if c in m[i][3]]
            cu[c] = (len({m[i][2] for i in sel}), len(sel))
        capa = armar_capa(X[tr], Y[tr], cu)
        # las que no se entrenaron en esta tanda pero si en la capa final
        # quedan con cero ejemplos en la medicion: se marca en la tabla
        P[te] = puntuar(X[te], capa)
    capa_cero = {"W": np.zeros((X.shape[1], 0), np.float32), "b": np.zeros(0, np.float32),
                 "entrenadas": [], "frases": V.vectores_frases()}
    P0 = puntuar(X, capa_cero)
    filas = []
    for j, c in enumerate(V.CLAVES):
        y = Y[:, j] > 0
        npos = int(y.sum())
        fila = {"clave": c, "grupo": V.VOCAB[j][1], "carpetas": cuentas[c][0], "fotos": npos,
                "entrena": cuentas[c][0] >= MIN_CARPETAS and cuentas[c][1] >= MIN_FOTOS}
        fila["auc"] = auc(P[:, j], y) if 0 < npos < len(y) else None
        fila["auc_cero"] = auc(P0[:, j], y) if 0 < npos < len(y) else None
        for u in CORTES:
            dice = P[:, j] >= u
            fila["prec_%g" % u] = float((dice & y).sum() / dice.sum()) if dice.sum() else None
            fila["rec_%g" % u] = float((dice & y).sum() / npos) if npos else None
            # cobertura real: fraccion de las fotos AL AZAR del archivo que pasan
            fila["arch_%g" % u] = float((dice & fondo).sum() / max(fondo.sum(), 1))
        filas.append(fila)
    return filas, P


def auc(s, y):
    """Area bajo la curva ROC por rangos (Mann-Whitney)."""
    orden = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[orden] = np.arange(1, len(s) + 1)
    npos = y.sum()
    nneg = len(y) - npos
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def tabla(filas):
    def f(x, pct=True):
        if x is None:
            return "   -"
        return "%4.0f" % (100 * x) if pct else "%.3f" % x
    out = ["| escena | grupo | carpetas | fotos | entrena | AUC lineal | AUC cero | prec@.5 | cob@.5 | "
           "prec@.7 | cob@.7 | prec@.9 | cob@.9 | archivo@.5 | archivo@.9 |",
           "|" + "---|" * 15]
    for r in filas:
        out.append("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["clave"], r["grupo"], r["carpetas"], r["fotos"],
            "si" if r["entrena"] else "**no: cero ejemplos**",
            f(r["auc"], False), f(r["auc_cero"], False),
            f(r["prec_0.5"]), f(r["rec_0.5"]), f(r["prec_0.7"]), f(r["rec_0.7"]),
            f(r["prec_0.9"]), f(r["rec_0.9"]),
            "%.1f%%" % (100 * r["arch_0.5"]), "%.1f%%" % (100 * r["arch_0.9"])))
    return "\n".join(out)


# --------------------------------------------------------------------- main

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--medir", action="store_true")
    ap.add_argument("--tmp", required=True, help="carpeta temporal FUERA del repo")
    ap.add_argument("--por-escena", type=int, default=POR_ESCENA)
    ap.add_argument("--fondo", type=int, default=FONDO)
    a = ap.parse_args()
    tmp = os.path.abspath(a.tmp)
    if tmp.startswith(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        sys.exit("--tmp tiene que estar fuera del repo")
    os.makedirs(tmp, exist_ok=True)

    import sb
    import escenas_clip as C
    cli = sb.SB()
    ruta_m = os.path.join(tmp, "muestra.json")
    if os.path.exists(ruta_m):
        with open(ruta_m, encoding="utf-8") as fh:
            m = [(a_, b_, c_, set(d_), e_) for a_, b_, c_, d_, e_ in json.load(fh)]
    else:
        t0 = time.time()
        carpetas = leer_carpetas(cli)
        fotos = leer_primarias(cli)
        # --medir es solo lectura: una foto de Zenfolio que no esta guardada
        # en thumbs/zf/ no se le pide a thumb, porque la funcion la guardaria y
        # escribiria photos.thumb_url (paso el 14/10/2026 con 372 fotos).
        fotos = [p for p in fotos if p.get("source") != "zenfolio"
                 or (p.get("thumb_url") or "").startswith("zf/")]
        m = muestra(carpetas, fotos, a.por_escena, a.fondo)
        with open(ruta_m, "w", encoding="utf-8") as fh:
            json.dump([(x[0], x[1], x[2], sorted(x[3]), x[4]) for x in m], fh)
        print("muestra: %d fotos (%d al azar) de %d primarias, en %.0f s"
              % (len(m), sum(x[4] for x in m), len(fotos), time.time() - t0), flush=True)

    vec = C.Vectores(tmp)
    clip = C.Clip()
    mini = C.Miniaturas(cli)
    falta = [{"photo_id": x[0], "source": x[1]} for x in m if x[0] not in vec]
    errores = {}
    ruta_e = os.path.join(tmp, "errores.json")
    if os.path.exists(ruta_e):
        with open(ruta_e, encoding="utf-8") as fh:
            errores = json.load(fh)
    falta = [f for f in falta if f["photo_id"] not in errores]
    print("vectores: %d ya, faltan %d" % (len(vec), len(falta)), flush=True)
    t0 = time.time()
    for i in range(0, len(falta), 500):
        ids, v, err = C.pasar(falta[i:i + 500], mini, clip)
        vec.agregar(ids, v)
        vec.guardar()
        errores.update(err)
        with open(ruta_e, "w", encoding="utf-8") as fh:
            json.dump(errores, fh)
        try:
            hechas = i + len(falta[i:i + 500])
            print("  %d de %d (%.0f ms/foto, %d errores)"
                  % (hechas, len(falta), 1000 * (time.time() - t0) / hechas, len(errores)), flush=True)
        except Exception:
            pass

    ids, X = vec.de([x[0] for x in m])
    pos = {pid: i for i, pid in enumerate(ids)}
    m = [x for x in m if x[0] in pos]
    X = X[[pos[x[0]] for x in m]]
    Y = matriz_etiquetas(m)
    cuentas = entrenables(m)
    print("con vector: %d fotos, %d sin (errores)" % (len(m), len(errores)), flush=True)
    filas, _ = medir(X, Y, m, cuentas)
    print(tabla(filas))
    capa = armar_capa(X, Y, cuentas, filas)
    print("van por cero ejemplos aunque tienen ejemplos (miden mejor sin entrenar): %s"
          % ", ".join(c for c in V.CLAVES if cuentas[c][0] >= MIN_CARPETAS
                      and cuentas[c][1] >= MIN_FOTOS and c not in capa["entrenadas"]))
    guardar_capa(capa, os.path.join(tmp, "_escenas_capa.npz"), {"filas": filas})
    with open(os.path.join(tmp, "medida.json"), "w", encoding="utf-8") as fh:
        json.dump(filas, fh, indent=1)
    print("capa: %d escenas entrenadas, %d por cero ejemplos (%s)"
          % (len(capa["entrenadas"]), len(V.CLAVES) - len(capa["entrenadas"]), capa["modelo"]))


if __name__ == "__main__":
    main()
