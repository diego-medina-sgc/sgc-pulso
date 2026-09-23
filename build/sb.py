# -*- coding: utf-8 -*-
"""
Cliente mínimo de Supabase para los scripts de build.

Existía la misma lógica copiada en load_folders.py y scan_zenfolio.py; esto
la junta. Usa la service_role key, que saltea RLS: nunca va al front, sale de
build/.env, que está en .gitignore.

    SUPABASE_URL=https://gfctvsxgulpftiytaxrn.supabase.co
    SUPABASE_SERVICE_KEY=eyJ...
"""
import atexit
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URL = "https://gfctvsxgulpftiytaxrn.supabase.co"
PAGINA = 1000          # tope por defecto de PostgREST

# LO QUE BAJA DE LA BASE SE PAGA (14/9/2026)
#
# El plan gratis incluye 5 GB de egress por mes y la organizacion llego a
# 7,87 GB en diez dias: Supabase avisa que desde el 14/10 restringe el proyecto
# (las llamadas dan 402). Casi todo eran estos scripts: cada paso del pulso
# vuelve a leer tablas enteras (photo_people, face_suggestions, face_groups,
# photos) y urllib no pide compresion, asi que el JSON viajaba crudo.
#
# Pidiendo gzip, una pagina de face_suggestions pasa de 115 KB a 35 KB (-69%).
# Y cada script cuenta lo que bajo: el pulso lo anota por paso (SB_EGRESS) para
# ver de donde sale el consumo sin abrir el panel de Supabase.
BYTES_RECIBIDOS = 0


def _informar_egress():
    if not BYTES_RECIBIDOS:
        return
    destino = os.environ.get("SB_EGRESS")
    if destino:
        try:
            with open(destino, "w") as fh:
                fh.write(str(BYTES_RECIBIDOS))
        except OSError:
            pass
    print("  (bajado de la base: %.1f MB)" % (BYTES_RECIBIDOS / 1e6), flush=True)


atexit.register(_informar_egress)


def load_env():
    """Lee build/.env.

    utf-8-sig porque PowerShell 5.1 escribe con BOM y con utf-8 a secas la
    primera clave quedaría con un caracter invisible adelante y no matchearía.
    Saca comillas, que es habitual pegarlas al copiar.
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


# La clave primaria de cada tabla. Es lo que permite paginar por clave.
#
# Va escrito y no consultado a la base: preguntarlo costaria un viaje por cada
# select, y ademas asi el mapa vive en el repo, donde se ve en un diff si
# alguien cambia una clave. Sale de pg_constraint, no de la memoria de nadie.
CLAVES = {
    "caras_dudosas_vistas":  ("person_id",),
    "caras_fuente_mala":     ("person_id", "fuente"),
    "corridas":              ("id",),
    "enrollments":           ("id",),
    "face_group_meta":       ("grupo",),
    "face_group_names":      ("grupo",),
    "face_groups":           ("id",),
    "face_suggestions":      ("photo_id", "person_id"),
    "folders":               ("id",),
    "fuentes":               ("id",),
    "grupos_no_son":         ("id",),
    "grupos_parecidos":      ("grupo",),
    # las fuentes del panel (sql/fuentes.sql, 14/9/2026)
    "padron_fuentes":        ("clave",),
    "fuentes_fotos":         ("id",),
    # la cola de correcciones del padron que se escriben en el sheet (23/9/2026)
    "padron_cambios":        ("id",),
    "persona_grupos":        ("person_id", "grupo"),
    "padron_dudosas":        ("nombre",),
    "fusiones":              ("id",),
    "people":                ("id",),
    # Nacio el 12/9/2026 y nacio sin clave aca, asi que cada lectura avisaba
    # "se pagina por desplazamiento y puede saltear filas". Con 4.761 filas
    # entra en una pagina sola y nunca llego a doler, pero es la tabla que
    # decide a quien se le deja de preguntar en confirmar: una fila salteada
    # es alguien que vuelve a la cola sin motivo, o que sale de ella sin
    # haberlo ganado.
    "persona_referencias":   ("person_id",),
    "photo_clusters":        ("photo_id",),
    "photo_labels":          ("id",),
    "photo_people":          ("photo_id", "person_id"),
    "photo_people_rejected": ("photo_id", "person_id"),
    "photos":                ("id",),
    "settings":              ("key",),
    "sync_state":            ("id",),
}


def columnas_pedidas(sel):
    """Los nombres que devuelve un select de PostgREST.

    Hay que saltear las comas que estan adentro de un recurso embebido:
    "photo_id,photos(folder_id,source)" pide dos cosas, no cuatro. Y una
    columna puede venir renombrada como "alias:columna", en cuyo caso el que
    vuelve en el diccionario es el alias.
    """
    partes, nivel, actual = [], 0, ""
    for ch in sel:
        if ch == "(":
            nivel += 1
        elif ch == ")":
            nivel -= 1
        if ch == "," and nivel == 0:
            partes.append(actual)
            actual = ""
        else:
            actual += ch
    partes.append(actual)
    fuera = []
    for c in partes:
        c = c.split("(")[0].strip()
        if ":" in c:
            c = c.split(":", 1)[0].strip()
        if c:
            fuera.append(c)
    return fuera


def valor_filtro(v):
    """Un valor para un filtro de PostgREST.

    Los textos van entre comillas: adentro de un or=(...) una coma o un
    parentesis sin comillas partirian la expresion en dos condiciones.
    """
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    return '"%s"' % str(v).replace("\\", "\\\\").replace('"', '\\"')


def filtro_siguiente(clave, ultimo):
    """"Dame lo que viene despues de esta fila", en sintaxis de PostgREST.

    Con una columna es simple. Con dos hay que comparar el PAR ordenado, y
    PostgREST no compara tuplas, asi que se escribe a mano lo mismo que haria
    (a, b) > (X, Y):

        a > X   o   (a = X  y  b > Y)

    Se devuelve como un parametro "and" u "or" y no como "columna=gt.valor"
    para no pisarle a un llamador su propio filtro sobre esa misma columna:
    los parametros son un diccionario y se indexan por nombre.
    """
    if len(clave) == 1:
        k = clave[0]
        return "and", "(%s.gt.%s)" % (k, valor_filtro(ultimo[k]))
    a, b = clave[0], clave[1]
    return "or", "(%s.gt.%s,and(%s.eq.%s,%s.gt.%s))" % (
        a, valor_filtro(ultimo[a]), a, valor_filtro(ultimo[a]),
        b, valor_filtro(ultimo[b]))


class SB:
    def __init__(self):
        load_env()
        self.url = os.environ.get("SUPABASE_URL", DEFAULT_URL).rstrip("/")
        self.key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not self.key:
            sys.exit("Falta SUPABASE_SERVICE_KEY en build/.env.\n"
                     "Supabase > Project Settings > API Keys > service_role.")

    def _headers(self, extra=None):
        h = {"apikey": self.key, "Authorization": "Bearer " + self.key,
             "Content-Type": "application/json", "Accept-Encoding": "gzip"}
        if extra:
            h.update(extra)
        return h

    def _pedir(self, req):
        global BYTES_RECIBIDOS
        for intento in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    cuerpo = r.read()
                    BYTES_RECIBIDOS += len(cuerpo)
                    if r.headers.get("Content-Encoding") == "gzip":
                        cuerpo = gzip.decompress(cuerpo)
                    return json.loads(cuerpo) if cuerpo else None
            except urllib.error.HTTPError as e:
                # EL CUERPO DEL ERROR TAMBIEN VIENE GZIPEADO.
                #
                # Arriba se descomprime la respuesta buena y aca no, asi que
                # cada error de PostgREST salia como un chorro de simbolos:
                #
                #   RuntimeError: HTTP 400: \ufffd  \P\ufffdNA\ufffd\ufffdf...
                #
                # Y el codigo -23514, 23502, 22008- es lo unico que dice que
                # paso. Sin eso hay que adivinar, o peor: un "if '23514' in
                # str(e)" nunca acierta y el programa toma el camino
                # equivocado creyendo que fue otra cosa.
                crudo = e.read()
                if e.headers.get("Content-Encoding") == "gzip":
                    try:
                        crudo = gzip.decompress(crudo)
                    except Exception:
                        pass
                # 2000 y no 300: PostgREST manda {code, details, hint,
                # message} en ese orden, y "message" -que es donde dice QUE
                # check se violo- viene al final. Con 300 se cortaba justo ahi
                # y quedaba el volcado de la fila sin el nombre de la regla.
                detalle = crudo.decode("utf-8", "replace")[:2000]
                # El 401 entra aca a proposito. Una key equivocada falla
                # siempre, asi que reintentar no la arregla ni la tapa: se
                # agotan los intentos y el error sale igual. Pero el gateway
                # de Supabase devuelve "Invalid API key" por unos segundos
                # cuando se reinicia, y ese 401 se cura solo. La noche del
                # 11/9 ese blip corto el nocturno a los 48 min, despues de
                # que la misma key hubiera funcionado 15 min antes.
                if e.code in (401, 429, 500, 502, 503, 504) and intento < 4:
                    time.sleep(2 ** intento)
                    continue
                raise RuntimeError("HTTP %s: %s" % (e.code, detalle))
            except (urllib.error.URLError, TimeoutError, OSError):
                if intento < 4:
                    time.sleep(2 ** intento)
                    continue
                raise

    def select(self, tabla, **params):
        """Trae todas las filas, paginando POR CLAVE.

        POR QUE NO POR DESPLAZAMIENTO

        Esto pedia "las filas 400.000 a 401.000" con el header Range y sin
        ORDER BY. PostgREST no garantiza ningun orden, asi que mientras otra
        corrida escribe en esa tabla el orden puede cambiar entre una pagina y
        la siguiente: una fila se corre hacia atras y sale DOS VECES, o se
        corre hacia adelante y no sale NUNCA.

        Las dos cosas ya pasaron y costaron caro:

        - Los duplicados mataron la corrida nocturna tres noches seguidas. Dos
          filas con la misma clave en un mismo lote de upsert dan 21000 y
          Postgres rechaza el lote entero.
        - Los duplicados tambien quemaron 19 anios del rehacer de eventos en
          doce minutos, sin procesar una foto.

        Y lo peor no hizo ruido nunca: las filas SALTEADAS. No rompen nada, no
        aparecen en ningun log, y hacen que una foto no se procese jamas.

        COMO ES AHORA

        "Dame las 1.000 siguientes a esta clave". Ordenado por la clave
        primaria, que es unica: no hay empates, asi que ninguna fila puede
        colarse entre dos paginas ni quedarse afuera. Lo que se inserte
        mientras tanto aparecera o no segun donde caiga, pero cada fila que
        exista todo el tiempo sale exactamente una vez.

        De paso es mas rapido: cada pagina es una busqueda por indice, en vez
        de un barrido que lee y descarta 400.000 filas antes de empezar.
        """
        clave = CLAVES.get(tabla)

        # Un llamador que pide su propio orden sabe lo que quiere -y hoy no hay
        # ninguno-. Una tabla sin clave conocida es un olvido: se avisa fuerte y
        # se sigue por el camino viejo, porque cortar una corrida de quince
        # horas por una tabla que falta en un diccionario es peor.
        if not clave or "order" in params or "limit" in params:
            if not clave:
                print("  (OJO: %s no esta en sb.CLAVES; se pagina por "
                      "desplazamiento y puede saltear filas)" % tabla)
            return self._select_por_desplazamiento(tabla, params)

        # La clave tiene que volver en cada fila para saber por donde seguir. Si
        # el llamador no la pidio se agrega, y despues se saca de las filas: el
        # que pidio tres columnas espera tres columnas, y hay codigo que
        # reinserta lo que leyo.
        params = dict(params)
        sel = params.get("select")
        agregadas = []
        if sel and sel != "*":
            tiene = columnas_pedidas(sel)
            agregadas = [k for k in clave if k not in tiene]
            if agregadas:
                params["select"] = sel + "," + ",".join(agregadas)

        params["order"] = ",".join("%s.asc" % k for k in clave)
        params["limit"] = str(PAGINA)

        out, ultimo = [], None
        while True:
            p = dict(params)
            if ultimo is not None:
                nombre, expr = filtro_siguiente(clave, ultimo)
                p[nombre] = expr
            q = urllib.parse.urlencode(p, safe='*.,()"')
            req = urllib.request.Request(
                "%s/rest/v1/%s?%s" % (self.url, tabla, q),
                headers=self._headers())
            lote = self._pedir(req) or []
            if not lote:
                break
            # El cursor se copia ANTES de limpiar las filas. Si no, cuando la
            # clave es una columna que el llamador no pidio -y por eso se
            # agrego sola- la vuelta siguiente se queda sin por donde seguir:
            # la ultima fila ya no la tiene. Daba KeyError en la pagina dos.
            ultimo = dict((k, lote[-1].get(k)) for k in clave)
            if agregadas:
                for f in lote:
                    for k in agregadas:
                        f.pop(k, None)
            out.extend(lote)
            if len(lote) < PAGINA:
                break
        return out

    def _select_por_desplazamiento(self, tabla, params):
        """El paginado viejo, por Range. Queda para los dos casos que la clave
        no cubre: un llamador con su propio order o limit, y una tabla que no
        esta en CLAVES. Puede repetir y saltear filas; no usarlo por gusto."""
        out = []
        desde = 0
        while True:
            q = urllib.parse.urlencode(params, safe='*.,()"')
            req = urllib.request.Request(
                "%s/rest/v1/%s?%s" % (self.url, tabla, q),
                headers=self._headers({"Range-Unit": "items",
                                       "Range": "%d-%d" % (desde, desde + PAGINA - 1)}))
            lote = self._pedir(req) or []
            out.extend(lote)
            if len(lote) < PAGINA:
                return out
            desde += PAGINA

    def rpc(self, nombre, payload=None):
        req = urllib.request.Request(
            "%s/rest/v1/rpc/%s" % (self.url, nombre),
            data=json.dumps(payload or {}).encode("utf-8"),
            method="POST", headers=self._headers())
        return self._pedir(req)

    def delete(self, tabla, **filtros):
        """DELETE de las filas que cumplen el filtro.

        Exige filtro por la misma razon que update: sin el, PostgREST borra la
        tabla entera y no hay como deshacerlo.
        """
        if not filtros:
            raise ValueError("delete sin filtro: eso vaciaria la tabla")
        q = urllib.parse.urlencode(filtros, safe="*.,()")
        req = urllib.request.Request(
            "%s/rest/v1/%s?%s" % (self.url, tabla, q),
            method="DELETE", headers=self._headers({"Prefer": "return=minimal"}))
        self._pedir(req)

    def update(self, tabla, campos, **filtros):
        """PATCH sobre las filas que cumplen el filtro. Devuelve None.

        No es lo mismo que un upsert de esos campos. PostgREST traduce el
        upsert a INSERT ... ON CONFLICT, y Postgres valida los NOT NULL al
        armar la tupla del INSERT, antes de enterarse de que habia conflicto:
        mandar {"id": ..., "is_group": true} sobre una fila que existe falla
        igual, pidiendo las columnas que no vienen en el payload. Para tocar
        un campo de filas que ya estan, va PATCH.

            cli.update("photos", {"is_group": True}, id="in.(a,b,c)")
        """
        if not filtros:
            raise ValueError("update sin filtro: eso tocaria la tabla entera")
        q = urllib.parse.urlencode(filtros, safe="*.,()")
        req = urllib.request.Request(
            "%s/rest/v1/%s?%s" % (self.url, tabla, q),
            data=json.dumps(campos, ensure_ascii=False).encode("utf-8"),
            method="PATCH", headers=self._headers({"Prefer": "return=minimal"}))
        self._pedir(req)

    def upsert(self, tabla, filas, on_conflict=None, lote=500):
        """Inserta o actualiza de a lotes; devuelve cuántas filas mandó.

        SIN FILAS REPETIDAS DENTRO DE UN MISMO LOTE

        Postgres no deja que un ON CONFLICT toque la misma fila dos veces en un
        comando: falla entero con 21000, "no rows proposed for insertion within
        the same command have duplicate constrained values". No actualiza la
        primera y descarta la segunda: rechaza el lote completo.

        Tumbó la corrida nocturna tres noches seguidas, en el paso de
        sugerencias. La causa esta más abajo de lo que parecía: select() pagina
        con desplazamientos y sin orden garantizado, así que mientras otra
        corrida escribe, una fila se corre entre página y página y vuelve a
        salir. La lista de pendientes traía la misma foto dos veces.

        Se limpia acá y no en cada script porque el que sabe cuál es la clave es
        este método: se la están pasando en on_conflict. Gana la última, que es
        la más nueva.
        """
        if not filas:
            return 0
        if on_conflict:
            claves = [c.strip() for c in on_conflict.split(",")]
            vistas = {}
            for f in filas:
                vistas[tuple(f.get(c) for c in claves)] = f
            if len(vistas) != len(filas):
                print("  (se descartaron %d filas repetidas dentro del lote de %s)"
                      % (len(filas) - len(vistas), tabla))
                filas = list(vistas.values())
        params = {}
        if on_conflict:
            params["on_conflict"] = on_conflict
        q = ("?" + urllib.parse.urlencode(params)) if params else ""
        pref = "return=minimal"
        if on_conflict:
            pref = "resolution=merge-duplicates," + pref
        n = 0
        for i in range(0, len(filas), lote):
            trozo = filas[i:i + lote]
            req = urllib.request.Request(
                "%s/rest/v1/%s%s" % (self.url, tabla, q),
                data=json.dumps(trozo, ensure_ascii=False).encode("utf-8"),
                method="POST", headers=self._headers({"Prefer": pref}))
            self._pedir(req)
            n += len(trozo)
        return n
