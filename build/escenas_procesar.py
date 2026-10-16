# -*- coding: utf-8 -*-
"""El paso "escenas" del pulso: que se ve en cada foto primaria.

    python escenas_procesar.py                 # ensayo: procesa y dice que cargaria
    python escenas_procesar.py --aplicar       # (en el pulso) carga en la base
    python escenas_procesar.py --bajar-modelo  # baja y verifica el modelo CLIP

QUE HACE (fase 1.2 del plan de vision, 14/10/2026)

  1. Si no hay capa lineal de esta VERSION (escenas_entrenar.py), la arma: la
     muestra de entrenamiento sale de las etiquetas de carpeta, sus vectores
     quedan guardados y se entrena. Si no termina en el tope, sigue la vuelta
     siguiente desde donde quedo.
  2. Pide a la base las primarias que este modelo no paso
     (escenas_pendientes, de a 1.000 por id), trae cada miniatura (Drive s256,
     Zenfolio por la Edge Function thumb), saca el vector CLIP y lo guarda en
     _escenas_NNN.npz. Una foto que ya tiene vector no se vuelve a bajar: un
     modelo nuevo repuntua con lo guardado.
  3. Carga las 3 escenas de mayor puntaje (>= PISO) con foto_escenas_cargar.
     Una foto que no se pudo bajar o leer va con "error" (queda hecha con su
     motivo; --reintentar las vuelve a pedir). El corte y la publicacion son
     de la base (escena_pasa: >= 60 respuestas y >= 95%).
  4. Al final, si cargo algo, escenas_recontar() (la regla del album).

TOPES: TOPE_MIN minutos o TOPE_FOTOS fotos por vuelta, lo primero que llegue.
Guarda los vectores y carga en la base por tandas de 1.000: si la vuelta se
corta, lo hecho queda (la trampa de la corrida que perdio todo por guardar al
final).

ESTADO: los .npz de escenas viajan cifrados APARTE de las huellas de caras
(estado.py --conjunto escenas, carpeta, cache y prefijo del bucket propios).
Con --aplicar no corre si el estado no se pudo traer (falta MARCA_ESTADO): si
arrancara de cero, la vuelta siguiente pisaria en el bucket los trozos buenos
con trozos nuevos.

Ningun nombre de persona en este archivo: se publica en el repo publico.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import escenas_clip as C
import escenas_entrenar as E
import escenas_vocab as V

HERE = os.path.dirname(os.path.abspath(__file__))
RUTA_CAPA = os.path.join(HERE, "_escenas_capa.npz")
# La escribe estado.py (conjunto escenas) al traer el estado, o al confirmar
# que el bucket no tiene ninguno (primera vez).
MARCA_ESTADO = os.path.join(HERE, "_escenas_estado.ok")

# ~20 min o 15.000 fotos por vuelta (pedido del plan, 13/10/2026). El pulso
# corre cada hora y este paso va ultimo: no puede comerse la vuelta.
TOPE_MIN = 20
TOPE_FOTOS = 15000
PAGINA = 1000
# Escenas con puntaje menor no se mandan (ni ocupan uno de los 3 lugares).
# Provisorio hasta tener respuestas del juego; medido en el ensayo del
# 14/10/2026 (cuantas fotos se quedan con menos de 3 escenas a este piso).
PISO = 0.05
TOP = 3
# Si en una pagina falla mas que esto (y al menos ERRORES_MIN fotos), no son
# fotos rotas: es Drive, thumb o la red. Se corta el paso SIN cargar esos
# errores (si no, quedarian miles de fotos "hechas con error" en la base). En
# la medicion del 14/10/2026 fallaron 2 de cada 3.000 (0,07%).
ERRORES_MAX = 0.5
ERRORES_MIN = 50


def demasiados_errores(n_err, n):
    return n_err >= ERRORES_MIN and n_err > ERRORES_MAX * max(n, 1)


def top3(P):
    """(n,57) -> [[[clave, puntaje], ...], ...] las TOP de mayor puntaje >= PISO."""
    out = []
    orden = np.argsort(-P, axis=1)[:, :TOP]
    for r in range(P.shape[0]):
        e = []
        for j in orden[r]:
            s = float(P[r, j])
            if s >= PISO:
                e.append([V.CLAVES[j], round(min(max(s, 0.0), 1.0), 4)])
        out.append(e)
    return out


def pendientes(cli, modelo, desde, reintentar, ensayo):
    """Una pagina de escenas_pendientes. En ensayo, si la funcion todavia no
    existe en la base (preparada por base, sin aplicar), lee photos con el
    mismo orden y filtro de primarias: no hay escenas_hechas que restar."""
    try:
        return cli.rpc("escenas_pendientes", {"p_modelo": modelo, "p_desde": desde,
                                              "lim": PAGINA, "p_reintentar": reintentar}) or []
    except RuntimeError as e:
        if not ensayo or "escenas_pendientes" not in str(e):
            raise
        filas = cli._pedir(_req(cli, "photos?select=id,source,folder_id,thumb_url"
                                     "&is_primary=eq.true&id=gt.%s&order=id&limit=%d"
                                     % (_q(desde), PAGINA)))
        return [{"photo_id": f["id"], "source": f["source"], "folder_id": f["folder_id"],
                 "zf_path": f["thumb_url"] if (f.get("thumb_url") or "").startswith("zf/") else None}
                for f in filas or []]


def _q(s):
    import urllib.parse
    return urllib.parse.quote(s or "", safe="")


def _req(cli, ruta):
    import urllib.request
    return urllib.request.Request("%s/rest/v1/%s" % (cli.url, ruta), headers=cli._headers())


def armar_capa(cli, vec, mini, clip, hasta, aplicar, informe):
    """La capa de esta VERSION. Devuelve la capa o None si no termino."""
    t0 = time.time()
    carpetas = E.leer_carpetas(cli)
    fotos = E.leer_primarias(cli)
    m = E.muestra(carpetas, fotos)
    falta = [{"photo_id": x[0], "source": x[1]} for x in m if x[0] not in vec]
    print("capa %s: muestra de %d fotos, faltan %d vectores (%.0f s leyendo la base)"
          % (E.nombre_modelo(), len(m), len(falta), time.time() - t0), flush=True)
    errores = 0
    for i in range(0, len(falta), PAGINA):
        if time.time() > hasta:
            break
        ids, v, err = C.pasar(falta[i:i + PAGINA], mini, clip, hasta)
        vec.agregar(ids, v)
        errores += len(err)
        if aplicar:
            vec.guardar()
        try:
            print("  capa: %d de %d vectores (%d no se pudieron)"
                  % (min(i + PAGINA, len(falta)), len(falta), errores), flush=True)
        except Exception:
            pass
    sin = [x for x in m if x[0] not in vec]
    # las que no se pueden bajar no traban la capa: se entrena sin ellas, salvo
    # que se haya cortado por tiempo (quedan muchas sin tocar)
    if time.time() > hasta and len(sin) > errores:
        informe["capa"] = "a medio armar: faltan %d vectores" % len(sin)
        return None
    ids, X = vec.de([x[0] for x in m])
    pos = {pid: i for i, pid in enumerate(ids)}
    m = [x for x in m if x[0] in pos]
    X = X[[pos[x[0]] for x in m]]
    Y = E.matriz_etiquetas(m)
    cuentas = E.entrenables(m)
    filas, _ = E.medir(X, Y, m, cuentas)       # carpeta afuera: segundos
    capa = E.armar_capa(X, Y, cuentas, filas)
    print(E.tabla(filas), flush=True)
    if aplicar:
        E.guardar_capa(capa, RUTA_CAPA, {"fotos": len(m), "filas": filas})
    informe["capa"] = "entrenada: %d escenas con ejemplos, %d por cero ejemplos, %d fotos" % (
        len(capa["entrenadas"]), len(V.CLAVES) - len(capa["entrenadas"]), len(m))
    print("  " + informe["capa"], flush=True)
    return capa


class Perezoso:
    def __init__(self, fabrica):
        self.fabrica, self.obj = fabrica, None

    def vectores(self, lote):
        if self.obj is None:
            self.obj = self.fabrica()
        return self.obj.vectores(lote)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--bajar-modelo", action="store_true")
    ap.add_argument("--reintentar", action="store_true",
                    help="volver a pedir las que dieron error")
    ap.add_argument("--tope-min", type=float, default=TOPE_MIN)
    ap.add_argument("--tope-fotos", type=int, default=TOPE_FOTOS)
    ap.add_argument("--desde", default="", help="(ensayo) arrancar despues de este id")
    ap.add_argument("--capa", help="(ensayo) capa armada en otro lado")
    ap.add_argument("--tmp", help="(ensayo) carpeta FUERA del repo para los vectores")
    ap.add_argument("--ejemplo", help="(ensayo) guardar aca el JSON que se cargaria")
    a = ap.parse_args()

    if a.bajar_modelo:
        C.bajar_modelo()
        return
    if a.aplicar and (a.capa or a.tmp or a.desde):
        sys.exit("--capa, --tmp y --desde son solo para el ensayo")
    if a.aplicar and not os.path.exists(MARCA_ESTADO):
        # visible como falla del paso, no tapado: sin el estado no se escribe
        sys.exit("No se trajo el estado de escenas (falta %s): no corre para no pisar "
                 "los vectores guardados." % os.path.basename(MARCA_ESTADO))

    import sb
    t0 = time.time()
    hasta = t0 + a.tope_min * 60
    cli = sb.SB()
    modelo = E.nombre_modelo()
    carpeta_vec = HERE if a.aplicar else (a.tmp or None)
    vec = C.Vectores(carpeta_vec) if carpeta_vec else C.Vectores(os.path.join(HERE, "_no_existe"))
    if not a.aplicar and a.tmp and os.path.abspath(a.tmp).startswith(os.path.dirname(HERE)):
        sys.exit("--tmp tiene que estar fuera del repo")
    print("%s: %d vectores guardados en %d trozos; modelo %s"
          % ("aplicar" if a.aplicar else "ENSAYO", len(vec), len(vec.trozos), modelo), flush=True)

    informe = {"modelo": modelo}
    # el modelo (350 MB) se abre recien si hay algo que bajar: sin nada
    # pendiente el paso termina en segundos
    clip = Perezoso(C.Clip)
    mini = C.Miniaturas(cli)
    capa = E.leer_capa(a.capa or RUTA_CAPA)
    if capa is None or capa["modelo"] != modelo:
        capa = armar_capa(cli, vec, mini, clip, hasta, a.aplicar, informe)
        if capa is None:
            print("=== capa sin terminar: sigue la vuelta siguiente (%s) ===" % informe["capa"])
            return

    desde, hechas, bajadas, reusadas, errores, cargadas = a.desde, 0, 0, 0, 0, 0
    filas_carga, ejemplo, t_bajar = 0, [], 0.0
    sin_foto, salteadas_zf = 0, 0
    # El tope de fotos cuenta las que hay que BAJAR: repuntuar con el vector
    # guardado (una VERSION nueva) no baja nada y solo lo frena el tiempo.
    while time.time() < hasta and bajadas + errores < a.tope_fotos:
        filas = pendientes(cli, modelo, desde, a.reintentar, not a.aplicar)
        if not filas:
            break
        desde = filas[-1]["photo_id"]
        nuevas = [f for f in filas if f["photo_id"] not in vec]
        if not a.aplicar:
            # En ensayo no se le pide a thumb una foto de Zenfolio que no esta
            # guardada: la funcion la guardaria en el bucket y escribiria
            # photos.thumb_url. Las guardadas (zf_path) solo se firman.
            antes = len(nuevas)
            nuevas = [f for f in nuevas if f.get("source") != "zenfolio" or f.get("zf_path")]
            filas = [f for f in filas if f.get("source") != "zenfolio" or f.get("zf_path")
                     or f["photo_id"] in vec]
            salteadas_zf += antes - len(nuevas)
        # las que no entran en el tope quedan sin tocar: no se cargan y
        # escenas_pendientes las vuelve a dar en la vuelta siguiente
        nuevas = nuevas[:max(a.tope_fotos - bajadas - errores, 0)]
        tb = time.time()
        ids, v, err = C.pasar(nuevas, mini, clip, hasta) if nuevas else ([], None, {})
        t_bajar += time.time() - tb
        if demasiados_errores(len(err), len(nuevas)):
            if ids and (a.aplicar or a.tmp):
                vec.agregar(ids, v)
                vec.guardar()
            muestra_err = sorted(set(err.values()))[:3]
            print("=== %d de %d fotos fallaron en una pagina (%s): no es una foto rota, "
                  "se corta sin cargar ===" % (len(err), len(nuevas), "; ".join(muestra_err)))
            sys.exit(1)
        if ids:
            vec.agregar(ids, v)
            if a.aplicar or a.tmp:
                vec.guardar()                   # antes de cargar: el vector no se pierde
        listos, X = vec.de([f["photo_id"] for f in filas])
        P = E.puntuar(X, capa) if listos else np.zeros((0, len(V.CLAVES)))
        carga = [{"id": pid, "e": e} for pid, e in zip(listos, top3(P))]
        carga += [{"id": pid, "error": motivo} for pid, motivo in err.items()]
        bajadas += len(ids)
        reusadas += len(listos) - len(ids)
        errores += len(err)
        hechas += len(carga)
        if len(ejemplo) < 6:
            ejemplo += carga[:3] + [c for c in carga if "error" in c][:1]
        if carga and a.aplicar:
            r = cli.rpc("foto_escenas_cargar", {"p_modelo": modelo, "p_filas": carga}) or {}
            cargadas += int(r.get("fotos") or 0)
            filas_carga += int(r.get("filas") or 0)
            sin_foto += int(r.get("sin_foto") or 0)
            if r.get("desconocidas"):
                print("  claves que la base no conoce: %s" % r["desconocidas"], flush=True)
        try:
            print("  %d fotos (%d bajadas, %d con vector guardado, %d errores), %.0f ms/foto bajando"
                  % (hechas, bajadas, reusadas, errores, 1000 * t_bajar / max(bajadas + errores, 1)),
                  flush=True)
        except Exception:
            pass
        # lo que quedo sin tocar por el tope de tiempo no se cargo: vuelve
        # en la vuelta siguiente (escenas_pendientes lo sigue dando)
        if time.time() > hasta:
            break

    if a.aplicar and cargadas:
        try:
            r = cli.rpc("escenas_recontar", {}) or {}
            print("recontar: %s" % json.dumps(r), flush=True)
        except Exception as e:
            # no se tapa: el paso falla y salud() avisa escenas_sin_recontar
            print("escenas_recontar fallo: %s" % str(e)[:200], flush=True)
            sys.exit(1)

    if a.ejemplo:
        with open(a.ejemplo, "w", encoding="utf-8") as fh:
            json.dump(ejemplo, fh, ensure_ascii=False, indent=1)
    if salteadas_zf:
        print("ensayo: %d de Zenfolio sin guardar no se pidieron a thumb" % salteadas_zf)
    mb = sum(os.path.getsize(t[0]) for t in vec.trozos if os.path.exists(t[0])) / 1e6
    print("=== %s: %d fotos en %.1f min (%d bajadas, %d reusadas, %d errores); cargadas %d "
          "(%d filas, %d sin foto); vectores %d en %d trozos, %.1f MB ==="
          % ("listo" if a.aplicar else "ensayo, no se cargo nada", hechas, (time.time() - t0) / 60,
             bajadas, reusadas, errores, cargadas, filas_carga, sin_foto, len(vec),
             len(vec.trozos), mb))


if __name__ == "__main__":
    main()
