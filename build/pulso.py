# -*- coding: utf-8 -*-
"""El pulso: el procesamiento que reemplaza al nocturno.

    python pulso.py                  # una vuelta: corre lo que cambio
    python pulso.py --ensayo         # dice que correria y por que, sin correr
    python pulso.py --paso sugerencias   # fuerza ese paso aunque no haya cambios

POR QUE

El nocturno corria todo una vez por noche, en cadena. Murio cuatro noches
seguidas por cuatro causas distintas -un NOT NULL, un corte SSL con Google,
una clave rechazada, un lote con duplicados- y cada mañana Diego esperaba un
arreglo que no habia pasado. Dicho por el: "me levanto a la mañana, espere al
nocturno al pedo y nunca se arreglan las cosas".

El problema no era ninguna de las cuatro causas. Era el diseño:

  - una sola oportunidad por dia: si fallaba, la proxima era en 24 horas
  - todo o nada: corria cada paso aunque no hubiera cambiado nada
  - mudo: lo que pasaba quedaba en un log de la notebook

COMO ES AHORA

Corre cada hora. Para cada paso:

  1. Mira sus ENTRADAS: contadores de la base (pulso_firmas(), sql/pulso.sql)
     y los archivos de huellas que lee (tamaño y fecha).
  2. Si son las mismas que la ultima vez que termino bien, no corre.
     Lo que Diego confirmo a las 10 esta en referencias y sugerencias a las 11.
  3. Si el mismo script ya esta corriendo -a mano, o la corrida de eventos-
     espera a la proxima vuelta en vez de pisarlo.
  4. Ningun paso corre si alguien contesto algo en los ultimos 40 minutos.
     Primero fue solo para los que renumeran los grupos: cambiarle el numero
     de grupo a quien esta nombrando grupos le cambia el grupo en la mano.
     Desde el 14/9/2026 es para todos (ver QUIETO_MIN).
  5. Si falla, anota el final del error y la vuelta siguiente reintenta. Tres
     fallos seguidos ya no son mala suerte: reintenta cada 6 horas y salud()
     lo marca como error.

Cada paso queda en corridas (tarea 'pulso'), y al final de la vuelta corre
salud() y anota cada chequeo (tarea 'salud'). La app y el tablero leen esa
tabla: lo roto se ve sin abrir un log.

No usa nada de Windows. En un servidor Linux la tarea es una linea de cron:

    0 * * * *  cd /ruta/build && FACES_MOTOR=arcface .venv/bin/python -u pulso.py >> _pulso.log 2>&1

LOS PASOS

Son los mismos del nocturno, en el mismo orden, y el orden importa: cada uno
usa lo que dejo el anterior. Los porques de cada uno estaban en nocturno.ps1,
retirado el 13/9/2026: estan en su historia de git (git log -p -- build/nocturno.ps1).
"""
import argparse
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time

# Antes de importar faces: faces.MOTOR se lee al importar, y su default es
# sface. Olvidar esto borro 1.615 filas del padron el 12/9/2026.
MOTOR = "arcface"
os.environ["FACES_MOTOR"] = MOTOR

import sb  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "_pulso")
LOCK = os.path.join(HERE, "_pulso.lock")
FRENO = os.path.join(HERE, "_PULSO_FRENO.txt")
LOCK_NOCTURNO = os.path.join(HERE, "_nocturno.lock")
MARCA_MOTOR = os.path.join(HERE, "_nocturno.motor")

# Sin respuestas humanas hace esto, el pulso puede usar la base.
#
# EL PULSO NO LE PELEA LA BASE A QUIEN ESTA JUGANDO (14/9/2026)
#
# Diego, jugando: "fue super lento cuando escribir el nombre y carga los de la
# lista, lo seleccionas y tarda en etiquetar". Y John vio el contador de
# Etiquetar saltar de 42.847 a 1.974. Medido esa noche mientras corria el paso
# sugerencias: avance() 14 s, search_people 0,76 s, confirmar_candidatas 7,7 s;
# con la vuelta ya terminada, 0,9 s, 0,03 s y 0,8 s. La base es una sola, del
# plan gratis, y los pasos leen y reescriben tablas enteras.
#
# Hasta aca esperaban solo los pasos que renumeran grupos. Ahora esperan todos,
# y tambien salud() y calentar(): lo que se atrasa es a lo sumo lo que dura la
# sesion de juego, y la vuelta siguiente lo recupera porque la firma no cambio.
QUIETO_MIN = 40
SALUD_MAX_H = 3           # salud() corre igual si la ultima tiene mas que esto
# UN PASO NO PUEDE ESPERAR PARA SIEMPRE (14/9/2026)
#
# Con John y Diego jugando a toda hora, "alguien contesto hace menos de 40 min"
# fue verdad en casi todas las vueltas: parecidos se salteo 23 veces seguidas
# desde el 13/9, resolver 18, carnets 21. Los grupos se rearmaron el 14/9 a las
# 2 y los pares del juego Grupos quedaron apuntando a numeros viejos: el menu
# decia 42 y el juego no tenia ninguno (sql/grupos_pares_una_regla.sql).
#
# La misma regla que salud(): si un paso tiene cambios y lleva mas de
# ESPERA_MAX_H esperando, corre igual. Menos "grupos": renumera todos los grupos
# durante casi una hora y le cambiaria el grupo en la mano a quien juega; ese
# espera siempre, y salud() avisa si se traba (paso_esperando).
ESPERA_MAX_H = 6
# CADA CUANTO PUEDE REPETIR UN PASO (14/9/2026, revisado el 15/9)
#
# Nacio porque Supabase aviso que la organizacion paso el egress del plan
# gratis (7,87 de 5 GB en diez dias): con gente jugando todo el dia, cada
# respuesta cambia db:identificaciones, y no_es, si_es, referencias,
# sugerencias, agrupamiento e identicas corrian TODAS las horas, releyendo
# photo_people, face_suggestions y photos enteras (hasta 2,6 GB en un dia).
#
# Con el plan Pro (250 GB de egress) el tope sobra, asi que vuelve a 0: lo que
# se contesta a las 10 esta en las sugerencias a las 11. Lo que queda es el
# gzip de build/sb.py, que baja los mismos datos a un tercio. Si el egress
# vuelve a apretar, poner 3 y cada paso corre a lo sumo cada 3 horas.
# Forzar un paso (--paso) o un reintento tras un fallo nunca espera.
CADA_H = 0
REINTENTO_LENTO_H = 6     # despues de 3 fallos seguidos
TOPE_H = 6                # un paso que tarda mas que esto se corta
LOCK_VIEJO_H = 12


def paso(nombre, args, entradas, quieto=True, espera_max_h=ESPERA_MAX_H, cada_h=CADA_H,
         sin_esperar_si=()):
    """espera_max_h=None: espera a que nadie juegue, por mas que tarde.
    cada_h: minimo de horas entre dos corridas buenas (0: sin minimo).
    sin_esperar_si: entradas que, si cambiaron, hacen correr el paso aunque
    alguien este jugando (ver SIN ESPERAR DESPUES DE REAGRUPAR)."""
    return {"nombre": nombre, "args": args, "entradas": entradas, "quieto": quieto,
            "espera_max_h": espera_max_h, "cada_h": cada_h,
            "sin_esperar_si": tuple(sin_esperar_si)}


# SIN ESPERAR DESPUES DE REAGRUPAR (19/9/2026)
#
# Reagrupar renumera todos los grupos. Hasta que corren parecidos y resolver,
# los grupos que el sistema ya habia cerrado quedan sin marca (el panel decia
# "0 cerrados solos" con 8.126), y los pares del juego apuntan a numeros que no
# existen. Esos dos pasos esperaban a que nadie jugara, hasta ESPERA_MAX_H: el
# 19/9 John jugo toda la tarde y quedaron parados. Diego: que corran siempre
# enseguida despues de reagrupar. No renumeran nada, asi que no le cambian el
# grupo en la mano a nadie; solo lo cierran si ya se sabe de quien es.
DESPUES_DE_GRUPOS = ("paso:grupos",)


# Entradas:  db:<clave de pulso_firmas>   npz:<nombre base, sin motor>
#            npz:<base>* (patron)          paso:<nombre> (cuando termino bien)
PASOS = [
    paso("fuentes",        ["fuentes_procesar.py", "--aplicar"], ["db:fuentes"], cada_h=0),
    # Las listas de personas del panel de Fuentes (alumnos, exalumnos, staff):
    # se vuelven a cargar cuando cambia su link u hoja o Diego toca "Leer de
    # nuevo" (sql/fuentes_panel.sql). Antes que las caras: sin la persona en el
    # padron no hay a quien asignarle una cara.
    paso("listas",         ["listas_cargar.py", "--aplicar"], ["db:listas"], cada_h=0),
    paso("grupales",       ["retratos_grupales.py", "--aplicar"], ["db:retratos"]),
    paso("huellas",        ["faces_huellas.py"], ["db:retratos"]),
    # Los albumes de evento que no pasaron enteros por el detector: sin esto
    # sus caras no tienen recuadro, grupo ni "+ ¿Quién es?" (13/9/2026, la
    # cara de Persona E; sql/caras_revisadas.sql). Antes de los que leen
    # _caras_todas, asi la misma vuelta los agrupa y les pone recuadro.
    paso("caras",          ["caras_revisar.py", "--aplicar"], ["db:caras_pendientes"]),
    # FUERA DEL PULSO desde el 12/9/2026: incoherentes y malarchivadas.
    #
    # Las dos juzgan contra las fotos oficiales, y esas fotos estaban
    # contaminadas. Medido con un juez que solo usa "Si" humanos intactos:
    # de los 1.138 "Si" humanos que los scripts dieron vuelta, incoherentes
    # se equivoco 471 veces contra 0 aciertos, y malarchivadas 271 contra 77.
    # Ademas malarchivadas seguia corriendo cada hora: a las 22:20 dio vuelta
    # 5 mas.
    #
    # No vuelven hasta que un script no pueda pisar una respuesta humana
    # (sql/disputas.sql) y su juez deje de ser la foto oficial. Las dos cosas
    # se cumplieron el 13/9/2026: las reemplaza el paso "auditoria", al final.
    paso("no_es",          ["caras_no_es.py", "--aplicar"],
         ["db:rechazos", "npz:_caras_evento"]),
    paso("si_es",          ["caras_si_es.py", "--aplicar"],
         ["db:identificaciones", "db:rechazos", "npz:_caras_evento"]),
    paso("referencias",    ["faces_referencias.py"],
         ["db:identificaciones", "npz:_huellas_retratos"]),
    paso("contarrefs",     ["referencias_contar.py"], ["npz:_referencias"]),
    paso("sugerencias",    ["faces_sugerir.py"],
         ["npz:_referencias*", "npz:_huellas_retratos", "db:rechazos"]),
    paso("agrupamiento",   ["faces_agrupar.py", "--aplicar"],
         ["db:sugerencias", "db:identificaciones", "npz:_huellas_retratos"]),
    paso("identicas",      ["identicas_aprobar.py", "--aplicar"], ["db:sugerencias"]),
    # Renumera todos los grupos: nunca corre con alguien jugando (ESPERA_MAX_H).
    paso("grupos",         ["caras_agrupar_archivo.py", "--aplicar", "--corte", "0.70"],
         ["db:identificaciones", "db:rechazos", "npz:_caras_todas",
          "npz:_caras_evento", "npz:_referencias"], quieto=True, espera_max_h=None),
    # Las caras que no quedaron en ningun grupo, compactas, para que el visor
    # les dibuje recuadro y se puedan nombrar (sql/caras_mapa.sql).
    paso("mapa",           ["caras_mapa.py", "--aplicar"],
         ["paso:grupos", "npz:_caras_todas"], quieto=True),
    # Por album, cuantas caras no tienen nombre: el filtro "+ ¿Quién es?" de la
    # home (sql/filtro_quien_es.sql). Cambia con los grupos, el mapa y las
    # sugerencias.
    paso("quienes",        ["quien_es_refrescar.py"],
         ["paso:mapa", "db:sugerencias", "db:identificaciones"], quieto=True),
    # El juego de Grupos: que grupo sin nombre se parece a uno que ya tiene
    # caras confirmadas. Despues de grupos, porque usa sus numeros; y lee las
    # respuestas, porque un "Si" nuevo convierte un grupo en referencia.
    #
    # ANTES DEL RESOLVER DESDE EL 18/9/2026: el resolver cierra solo los pares
    # con parecido alto y claro (nivel 4, sql/resolver_por_parecido.sql), asi
    # que necesita los pares de ESTE agrupamiento. Con el orden de antes leia
    # los de la vuelta anterior, cuyos numeros de grupo ya no existen.
    paso("parecidos",      ["grupos_parecidos.py", "--aplicar"],
         ["paso:grupos", "db:identificaciones", "db:rechazos"], quieto=True,
         sin_esperar_si=DESPUES_DE_GRUPOS),
    paso("resolver",       ["grupos_resolver.py", "--escribir"],
         ["paso:grupos", "paso:parecidos", "db:identificaciones"], quieto=True,
         sin_esperar_si=DESPUES_DE_GRUPOS),
    paso("gruposconocidos", ["grupos_conocidos.py"],
         ["paso:grupos", "npz:_referencias"], quieto=True),
    # Un carnet es de una persona: cuando figuran dos y la cara decide, se saca
    # la etiqueta de maquina que pierde. Lo que tiene autoridad no se toca.
    paso("carnets",        ["carnets_dos_personas.py", "--aplicar"],
         ["db:identificaciones", "db:rechazos", "npz:_referencias",
          "npz:_huellas_retratos"], quieto=True),
    # VUELVEN las auditorias, en un solo paso y con otro juez (13/9/2026).
    # incoherentes y malarchivadas juzgaban contra las fotos oficiales, que
    # estaban contaminadas (ver arriba). caras_auditoria_humana.py juzga contra
    # los "Si" humanos intactos, se valida contra ellos en cada corrida y no
    # escribe si marcaria mas del 1% (primera corrida: 0,10%). No toca nada que
    # haya puesto una persona.
    paso("auditoria",      ["caras_auditoria_humana.py", "--aplicar"],
         ["db:identificaciones", "db:rechazos", "npz:_caras_evento",
          "npz:_huellas_retratos"], quieto=True),
    # Le busca la cara a quien esta identificado sin recuadro: los que se
    # corrigen desde el visor ("esta cara no es X", sql/no_es_esta_cara.sql)
    # y los que vienen sin saber cual es. Validado el 13/9/2026 contra 5.102
    # recuadros humanos: 100% con una cara, 98,3% con varias.
    paso("recuadros",      ["recuadros_faltantes.py", "--aplicar"],
         ["db:identificaciones", "npz:_caras_todas", "npz:_referencias"], quieto=True),
    # Los pares del juego Fusionar, ordenados por el parecido de las caras de
    # las dos fichas (sql/fusionar_juego.sql).
    paso("fichas",         ["fichas_parecidas.py", "--aplicar"],
         ["db:identificaciones", "npz:_referencias"], quieto=True),
]


def ahora():
    return dt.datetime.now(dt.timezone.utc)


def hora_local():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def di(texto):
    print("%s  %s" % (hora_local(), texto), flush=True)


def fecha(iso):
    if not iso:
        return None
    try:
        return dt.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return None


# --- entradas -----------------------------------------------------------------

def archivos(base):
    """Los .npz de este motor que corresponden a una entrada, con su firma.

    Se arma el nombre a mano y no con faces.npz(): lo que se quiere es el
    archivo EXACTO del motor, y un patron tiene que dejar afuera las copias de
    respaldo como _referencias_arcface.ANTES_DEL_ARREGLO.npz.
    """
    if base.endswith("*"):
        patron = os.path.join(HERE, "%s*_%s.npz" % (base[:-1], MOTOR))
    else:
        patron = os.path.join(HERE, "%s_%s.npz" % (base, MOTOR))
    out = []
    for ruta in sorted(glob.glob(patron)):
        st = os.stat(ruta)
        out.append("%s:%d:%d" % (os.path.basename(ruta), st.st_size, int(st.st_mtime)))
    return out


def firma(p, db, estados):
    f = {}
    for e in p["entradas"]:
        tipo, nombre = e.split(":", 1)
        if tipo == "db":
            f[e] = db.get(nombre)
        elif tipo == "npz":
            f[e] = archivos(nombre)
        elif tipo == "paso":
            f[e] = (estados.get(nombre) or {}).get("ultimo_ok")
    return f


# --- procesos -----------------------------------------------------------------

def procesos_python():
    """Las lineas de comando de los python que estan corriendo."""
    try:
        if os.name == "nt":
            cmd = ["powershell", "-NoProfile", "-Command",
                   "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" "
                   "| ForEach-Object { $_.CommandLine }"]
        else:
            cmd = ["ps", "-eo", "args"]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=90,
                              errors="replace").stdout
    except Exception:
        return ""


def ya_corre(script):
    return any(script in linea and "pulso.py" not in linea
               for linea in procesos_python().splitlines())


# --- estado en la base ----------------------------------------------------------

def leer_estados(cli):
    out = {}
    for r in cli.select("corridas", select="etiqueta,estado,mensaje", tarea="eq.pulso"):
        try:
            m = json.loads(r.get("mensaje") or "{}")
        except ValueError:
            m = {}
        m["estado"] = r.get("estado")
        out[r["etiqueta"]] = m
    return out


def anotar(cli, etiqueta, estado, mensaje, tarea="pulso", total=None, arranco=False):
    fila = {"tarea": tarea, "etiqueta": etiqueta, "estado": estado,
            "mensaje": json.dumps(mensaje, ensure_ascii=False) if isinstance(mensaje, dict) else mensaje,
            "total": total, "hechas": 0, "encontradas": 0, "con_gente": 0,
            "actualizado_at": ahora().isoformat()}
    if arranco:
        fila["arranco_at"] = fila["actualizado_at"]
    try:
        cli.upsert("corridas", [fila], on_conflict="tarea,etiqueta")
    except Exception as e:
        # Informar no puede voltear la vuelta: el trabajo importa mas que el aviso.
        di("  (no se pudo anotar %s/%s en corridas: %s)" % (tarea, etiqueta, str(e)[:120]))


# --- decidir y correr -------------------------------------------------------------

def decidir(p, db, estados, forzar):
    """(correr?, motivo, firma, esperando_desde)

    esperando_desde: desde cuando el paso tiene cambios y no corre porque alguien
    juega, o None si no esta en esa situacion.
    """
    st = estados.get(p["nombre"]) or {}
    f = firma(p, db, estados)
    if forzar:
        return True, "forzado", f, None

    fallos = int(st.get("fallos") or 0)
    if fallos >= 3:
        ultimo = fecha(st.get("ultimo_intento"))
        if ultimo and ahora() - ultimo < dt.timedelta(hours=REINTENTO_LENTO_H):
            return False, "fallo %d veces seguidas; reintenta cada %d h" % (
                fallos, REINTENTO_LENTO_H), f, None

    if not fallos and st.get("firma") == f:
        return False, "sin cambios", f, None

    ultimo_ok = fecha(st.get("ultimo_ok"))
    if not fallos and p.get("cada_h") and ultimo_ok and \
            ahora() - ultimo_ok < dt.timedelta(hours=p["cada_h"]):
        return False, "tiene cambios, pero corrio hace %d min (a lo sumo cada %d h: egress)" % (
            (ahora() - ultimo_ok).total_seconds() // 60, p["cada_h"]), f, None

    tarde = ""
    viejo = st.get("firma") or {}
    urgente = [k for k in p.get("sin_esperar_si", ()) if k in f and viejo.get(k) != f[k]]
    if urgente:
        tarde = " (cambio %s: corre aunque alguien juegue)" % ", ".join(urgente)
    elif p["quieto"]:
        act = fecha(db.get("actividad"))
        if act and ahora() - act < dt.timedelta(minutes=QUIETO_MIN):
            desde = fecha(st.get("esperando_desde")) or ahora()
            espera_h = (ahora() - desde).total_seconds() / 3600
            if p["espera_max_h"] is None or espera_h < p["espera_max_h"]:
                return False, "alguien esta jugando (ultima respuesta hace %d min, espera hace %d h)" % (
                    (ahora() - act).total_seconds() // 60, espera_h), f, desde
            tarde = " (esperaba hace %d h: corre aunque alguien juegue)" % espera_h

    if ya_corre(p["args"][0]):
        return False, "%s ya esta corriendo" % p["args"][0], f, None

    cambio = [k for k in f if (st.get("firma") or {}).get(k) != f[k]]
    motivo = ("reintento tras %d fallo%s" % (fallos, "" if fallos == 1 else "s")) if fallos \
        else ("cambio: " + ", ".join(cambio) if st.get("firma") else "primera vez")
    return True, motivo + tarde, f, None


def final_del_log(ruta, lineas=8):
    try:
        with open(ruta, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(fh.tell() - 8000, 0))
            texto = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    # los avisos de OpenCV llenan el final y no dicen nada
    utiles = [l for l in texto.splitlines() if l.strip() and "WARN:" not in l
              and "not supported by the new graph engine" not in l]
    return "\n".join(utiles[-lineas:])[-600:]


def correr(cli, p, f, motivo, estados):
    nombre = p["nombre"]
    args = list(p["args"])
    # Si el motor cambio desde la ultima vez, las sugerencias que hay son del
    # reconocedor viejo, con otra escala de puntajes (ver la historia de git de
    # nocturno.ps1, retirado el 13/9/2026).
    motor_cambio = False
    if nombre == "sugerencias":
        anterior = open(MARCA_MOTOR).read().strip() if os.path.exists(MARCA_MOTOR) else ""
        if anterior != MOTOR:
            args.append("--rehacer")
            motor_cambio = True

    st = dict(estados.get(nombre) or {})
    st["ultimo_intento"] = ahora().isoformat()
    st["motivo"] = motivo
    st.pop("esperando_desde", None)
    anotar(cli, nombre, "corriendo", st, arranco=True)

    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, nombre + ".log")
    # cuanto bajo el paso de la base: lo escribe sb.py al terminar (egress)
    egress = os.path.join(LOGS, nombre + ".egress")
    if os.path.exists(egress):
        os.remove(egress)
    env = dict(os.environ, FACES_MOTOR=MOTOR, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               SB_EGRESS=egress)
    t0 = time.time()
    with open(log, "wb") as fh:
        fh.write(("%s  %s %s\n" % (hora_local(), nombre, motivo)).encode("utf-8"))
        fh.flush()
        try:
            codigo = subprocess.run(
                [sys.executable, "-u", os.path.join(HERE, args[0])] + args[1:],
                cwd=HERE, stdout=fh, stderr=subprocess.STDOUT, env=env,
                timeout=TOPE_H * 3600).returncode
        except subprocess.TimeoutExpired:
            codigo = "tope de %d h" % TOPE_H
    minutos = int((time.time() - t0) / 60)
    try:
        with open(egress) as fh:
            st["mb"] = round(int(fh.read().strip() or 0) / 1e6, 1)
    except (OSError, ValueError):
        st["mb"] = 0

    if codigo == 0:
        st.update(firma=f, fallos=0, ultimo_ok=ahora().isoformat(), minutos=minutos)
        st.pop("error", None)
        anotar(cli, nombre, "listo", st)
        if motor_cambio:
            with open(MARCA_MOTOR, "w") as fh:
                fh.write(MOTOR)
        di("  %-16s OK en %d min, %.1f MB bajados" % (nombre, minutos, st["mb"]))
    else:
        st.update(fallos=int(st.get("fallos") or 0) + 1, minutos=minutos,
                  error="codigo %s\n%s" % (codigo, final_del_log(log)))
        anotar(cli, nombre, "error", st)
        di("  %-16s FALLO (%s) en %d min, intento %d. Ver _pulso/%s.log"
           % (nombre, codigo, minutos, st["fallos"], nombre))
    estados[nombre] = dict(st, estado="listo" if codigo == 0 else "error")
    return codigo == 0


def salud(cli):
    """Corre salud() y anota cada chequeo. Devuelve los que estan mal."""
    try:
        filas = cli.rpc("salud") or []
    except Exception as e:
        di("  salud(): no se pudo leer: %s" % str(e)[:160])
        return None
    mal = []
    for r in filas:
        grav, n = r["gravedad"], int(r["cantidad"] or 0)
        if r["chequeo"] == "base_mb":
            violada = grav in ("aviso", "error")
        else:
            violada = n > 0 and grav in ("aviso", "error")
        estado = grav if violada else "ok"
        anotar(cli, r["chequeo"], estado, r["que_significa"], tarea="salud", total=n)
        if violada:
            mal.append("%s=%d (%s)" % (r["chequeo"], n, grav))
    return mal


# Lo que la app pide al abrir un juego. En frio la primera llamada llego a 6,8 s
# contra 86 ms la segunda, y el navegador corta a los 3 s: Diego veia "la base
# tardo demasiado" por algo que se arreglaba solo al reintentar. Pedirlas al
# final de cada vuelta deja sus paginas en la cache de la base. Son lecturas y
# no escriben nada; si una falla no importa.
CALENTAR = [
    ("search_events", {"q": ""}),          # la pantalla de inicio
    ("grupos_pendientes", {}),
    ("grupo_para_nombrar", {}),
    ("grupos_cuentas", {}),
    ("grupo_par_siguiente", {}),
    ("grupos_pares_pendientes", {}),
]


def jugando(db):
    """Minutos desde la ultima respuesta humana, si fue hace menos de QUIETO_MIN."""
    act = fecha((db or {}).get("actividad"))
    if act and ahora() - act < dt.timedelta(minutes=QUIETO_MIN):
        return int((ahora() - act).total_seconds() // 60)
    return None


def salud_reciente(cli):
    """Si la ultima salud() anotada tiene menos de SALUD_MAX_H horas."""
    try:
        filas = cli.select("corridas", select="actualizado_at", tarea="eq.salud",
                           order="actualizado_at.desc", limit="1")
    except Exception:
        return False
    ultima = fecha(filas[0]["actualizado_at"]) if filas else None
    return bool(ultima and ahora() - ultima < dt.timedelta(hours=SALUD_MAX_H))


def calentar(cli):
    tiempos = []
    for fn, args in CALENTAR:
        t = time.time()
        try:
            cli.rpc(fn, args)
            tiempos.append("%s %d ms" % (fn, (time.time() - t) * 1000))
        except Exception as e:
            tiempos.append("%s fallo (%s)" % (fn, str(e)[:60]))
    di("  calentar: " + ", ".join(tiempos))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensayo", action="store_true")
    ap.add_argument("--paso", action="append", default=[],
                    help="forzar este paso (se puede repetir)")
    a = ap.parse_args()

    nombres = {p["nombre"] for p in PASOS}
    for n in a.paso:
        if n not in nombres:
            sys.exit("No hay un paso '%s'. Son: %s" % (n, ", ".join(sorted(nombres))))

    if os.path.exists(FRENO) and not a.ensayo:
        di("=== NO CORRE: existe _PULSO_FRENO.txt ===")
        return
    if os.path.exists(LOCK_NOCTURNO) and \
            time.time() - os.path.getmtime(LOCK_NOCTURNO) < 20 * 3600:
        di("=== NO CORRE: el nocturno esta corriendo ===")
        return
    if os.path.exists(LOCK) and not a.ensayo:
        edad_h = (time.time() - os.path.getmtime(LOCK)) / 3600
        if edad_h < LOCK_VIEJO_H:
            di("=== la vuelta anterior sigue corriendo (hace %d min): salgo ===" % (edad_h * 60))
            return
        di("lock de hace %d h: la vuelta anterior murio, lo piso" % edad_h)

    cli = sb.SB()
    if not a.ensayo:
        with open(LOCK, "w") as fh:
            fh.write(str(os.getpid()))
    t0 = time.time()
    corridos, fallados, salteados = [], [], []
    try:
        estados = leer_estados(cli)
        di("=== vuelta%s ===" % (" (ensayo)" if a.ensayo else ""))
        # La señal de vida va al ARRANCAR, no solo al final. salud() corre
        # antes de anotar el resumen de la vuelta, asi que sin esto
        # pulso_parado veia siempre la vuelta anterior, y la primera vuelta
        # despues de un corte se marcaba parada a si misma.
        if not a.ensayo:
            anotar(cli, "_vuelta", "corriendo", {"arranco": ahora().isoformat()},
                   arranco=True)
        for p in PASOS:
            # La foto se saca de nuevo antes de cada paso: el anterior puede
            # haber cambiado justo lo que este lee.
            db = cli.rpc("pulso_firmas") or {}
            ok, motivo, f, desde = decidir(p, db, estados, p["nombre"] in a.paso)
            if not ok:
                salteados.append(p["nombre"])
                di("  %-16s -  %s" % (p["nombre"], motivo))
                st = estados.get(p["nombre"]) or {}
                # Desde cuando espera por alguien jugando: lo lee la vuelta
                # siguiente (ESPERA_MAX_H) y salud() (paso_esperando). Se anota
                # solo la primera vez, y se borra cuando el paso corre.
                if desde and not st.get("esperando_desde") and not a.ensayo:
                    st = dict(st, esperando_desde=desde.isoformat())
                    estados[p["nombre"]] = st
                    anotar(cli, p["nombre"], st.get("estado") or "listo",
                           {k: v for k, v in st.items() if k != "estado"})
                continue
            if a.ensayo:
                di("  %-16s >  correria: %s" % (p["nombre"], motivo))
                continue
            di("  %-16s >  %s" % (p["nombre"], motivo))
            (corridos if correr(cli, p, f, motivo, estados) else fallados).append(p["nombre"])

        mal = None
        if not a.ensayo:
            # salud() tarda 10 a 30 s y calentar() pide las colas de los juegos:
            # con alguien jugando, la cache ya esta caliente y la salud puede
            # esperar, salvo que la ultima sea vieja (pulso_parado y compania
            # tienen que seguir viendose).
            min_juego = jugando(cli.rpc("pulso_firmas") or {})
            if min_juego is not None and salud_reciente(cli):
                di("  salud y calentar -  alguien esta jugando (ultima respuesta hace %d min)"
                   % min_juego)
            else:
                mal = salud(cli)
                calentar(cli)
                # La foto del avance para la curva de largo plazo: caras con
                # nombre por recuadro tarda segundos y no va en vivo
                # (sql/avance_historia.sql). Una fila por dia; la del dia se pisa.
                try:
                    f = cli.rpc("avance_foto") or {}
                    di("  avance: %s de %s caras con nombre, %s por nombrar, %s personas reconocibles"
                       % (f.get("caras_con_nombre"), f.get("caras_mapa"),
                          f.get("caras_por_nombrar"), f.get("personas_reconocibles")))
                except Exception as e:
                    di("  avance_foto(): no se pudo: %s" % str(e)[:160])
    finally:
        if not a.ensayo and os.path.exists(LOCK):
            os.remove(LOCK)

    if a.ensayo:
        di("=== ensayo: no se corrio nada ===")
        return
    resumen = {"corrieron": corridos, "fallaron": fallados,
               "sin_cambios": len(salteados), "minutos": int((time.time() - t0) / 60),
               "salud": mal}
    anotar(cli, "_vuelta", "error" if (fallados or mal) else "listo", resumen)
    di("=== vuelta en %d min: %d corrieron, %d fallaron, %d sin cambios. Salud: %s ==="
       % (resumen["minutos"], len(corridos), len(fallados), len(salteados),
          "sin leer" if mal is None else (", ".join(mal) if mal else "todo bien")))


if __name__ == "__main__":
    main()
