# -*- coding: utf-8 -*-
"""
Procesa la cola de fuentes que se anotan desde la pantalla del padron.

    python fuentes_procesar.py --ensayo          # dice que haria con las pendientes
    python fuentes_procesar.py --aplicar
    python fuentes_procesar.py --aplicar --id 7  # solo esa

QUE PROBLEMA RESUELVE

Las cinco fuentes que entraron hasta ahora -los anuarios de papel, el anuario
en PDF, los Valete del Drive de Marketing, la planilla de fotos de staff, la de
equipo- fueron cinco scripts, y cada una hubo que pedirla. La tabla "fuentes"
es la cola de entrada: Diego o John pegan un link y dicen de que año es, de que
sede y si son alumnos o staff. Esto es lo que la consume.

Sin esto, anotar una fuente deja el link guardado y nada mas. Era un buzon sin
nadie del otro lado.

QUE TIPO SABE LEER

    carpeta        una foto por persona con el nombre en el archivo. Es la
                   forma de los Valete y de los retratos del colegio, y la
                   unica que se puede procesar sola de punta a punta.

    planilla       una hoja de calculo con nombre y link a la foto.
    anuario_pdf    un PDF con las fotos de curso y los nombres al pie.
    anuario_papel  fotos de paginas impresas.

Los tres ultimos SI se pueden anotar: quedan en estado 'a_mano', con el link
guardado y su año y sede, esperando que alguien los procese. No es que falte
escribir el codigo: el anuario de papel necesita que alguien transcriba los
nombres de la pagina a ojo -las 280 paginas que ya entraron se transcribieron
una por una- y la planilla necesita saber en que columna esta cada cosa, que
cambia en cada hoja. Fingir que se procesan solos seria peor que decir que no,
y perder el link seria peor todavia.

COMO DECIDE DE QUIEN ES CADA FOTO

Con el mismo lector de los Valete (valete_cargar.py), que no adivina el orden
del nombre: prueba las permutaciones contra el padron y gana la que existe.
Filtra por la sede de la fuente, asi un "Persona T" de Quilmes no matchea con
el Persona U de North.

A quien no esta en el padron se le da de alta, que es lo que hace que una
fuente nueva sirva. Con un tope: si una sola carpeta fuera a crear mas de 200
personas, algo esta mal -no son nombres, o es la carpeta equivocada- y se para
para que alguien mire.

QUE ESCRIBE

Las huellas van a _referencias_fuente_<id>_<motor>.npz, uno por fuente, que
faces_sugerir.py levanta solo. En la fila de la fuente quedan el estado, cuantas
caras y cuantas personas salieron, y el motivo si no se pudo.

Las fotos NO entran a la biblioteca como fotos: para eso habria que indexarlas,
generar miniaturas y deduplicarlas contra las 490.884 que ya estan, y es otra
tarea. Lo que entra es la cara como referencia, que es lo que sirve para
reconocer a esa persona en las fotos de evento que ya estan cargadas.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
from collections import Counter

import numpy as np

import faces
import sb
from index_drive import Drive, SA_PATH
from retratos_nombres import clave
from valete_cargar import (FOL, a_quien_apunta, imagenes_recursivo, indexar_padron,
                           medir_caras)

HERE = os.path.dirname(os.path.abspath(__file__))

# Un link de Drive viene de muchas formas y todas traen el id en el medio:
#   /drive/folders/<id>            una carpeta
#   /file/d/<id>/view              un archivo
#   /open?id=<id>  /uc?id=<id>     los links viejos
# Tambien se acepta el id pelado, porque alguien lo va a pegar asi.
LINK = re.compile(r"(?:/folders/|/file/d/|/d/|[?&]id=)([A-Za-z0-9_-]{15,})")
ID_SOLO = re.compile(r"^[A-Za-z0-9_-]{15,}$")


def id_de_link(url):
    u = (url or "").strip()
    m = LINK.search(u)
    if m:
        return m.group(1)
    if ID_SOLO.match(u):
        return u
    return None


def salida_de(fuente_id):
    return faces.npz(os.path.join(HERE, "_referencias_fuente_%d.npz" % fuente_id))


def titulo_nombre(t):
    """"malen esteves" -> "Malen Esteves", sin tocar lo que ya esta bien.

    Solo sube la primera letra de cada palabra cuando la palabra viene entera
    en minuscula. Asi no rompe "Persona V", "O'Brien" ni "de la Torre": una
    palabra que ya trae mayusculas adentro la escribio alguien a proposito.
    """
    out = []
    for p in (t or "").split():
        out.append(p if any(c.isupper() for c in p) else p.capitalize())
    return " ".join(out)


def cuenta_servicio():
    """El mail con el que hay que compartir: sale de la credencial, no se copia."""
    try:
        with open(SA_PATH, encoding="utf-8") as fh:
            return json.load(fh)["client_email"]
    except Exception:
        return "la cuenta de servicio del proyecto"


def nombre_de_nota(nota):
    """'Persona W (2004)' -> 'Persona W'. Saca lo que va entre
    parentesis y los numeros sueltos; el resto lo decide a_quien_apunta."""
    t = nota or ""
    while "(" in t and ")" in t[t.index("("):]:
        i = t.index("(")
        t = t[:i] + " " + t[t.index(")", i) + 1:]
    t = " ".join(w for w in t.split() if not w.isdigit())
    return t.strip()


def procesar_carpeta(d, cli, idx, f, escribir, limite=0):
    """Lee una carpeta de Drive con una foto por persona.

    Devuelve (estado, resultado, caras, personas). No levanta: cualquier
    problema vuelve como estado 'error' con el motivo escrito en castellano,
    porque el que lo va a leer es Diego en la pantalla y no un log.
    """
    fid = id_de_link(f["url"])
    if not fid:
        return "error", "El link no parece de Drive: no se le encuentra el id.", None, None

    # PRIMERO QUE ES EL LINK (22/9/2026)
    #
    # Se listaba directo lo que cuelga del id. Si el id no se ve -no esta
    # compartido con la cuenta del proyecto- Drive no da error al listar: da
    # vacio. Y si el link es de UNA foto, tambien da vacio, porque de un archivo
    # no cuelga nada. En los dos casos la fuente decia "la carpeta se abre pero
    # no tiene imagenes", que era falso: la de Persona W (una foto suelta,
    # sin compartir) se quedo asi. Ahora se mira el id antes de listar.
    try:
        meta = d.get("files/" + fid, fields="id,name,mimeType")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            return ("error", "Drive no deja abrir el link (%s)." % str(e)[:120],
                    None, None)
        return ("error",
                "Drive no encuentra el link con la cuenta del sistema: no esta "
                "compartido. Compartilo como lector con %s y volve a anotarlo."
                % cuenta_servicio(), None, None)

    suelta = (meta.get("mimeType") or "").startswith("image/")
    if suelta:
        img = [meta]
    elif meta.get("mimeType") == FOL:
        try:
            img = imagenes_recursivo(d, fid)
        except Exception as e:
            return ("error",
                    "No se puede abrir la carpeta. Compartila con %s como "
                    "lectora. (%s)" % (cuenta_servicio(), str(e)[:120]),
                    None, None)
    else:
        return ("error",
                "El link es un archivo pero no una imagen (%s): aca va una "
                "carpeta o una foto." % meta.get("mimeType"), None, None)

    if not img:
        return ("error",
                "La carpeta se abre pero no tiene imagenes, ni en sus "
                "subcarpetas (se baja hasta tres niveles).", 0, 0)

    campus = f.get("campus")
    coh = f.get("anio")
    fotos = [(x["id"], x["name"], campus, coh) for x in img]
    # Una foto suelta viene con el nombre en la nota ("Persona W (2004)"),
    # que escribio quien la cargo: el que carga es el juez de quien es la foto.
    # Se prueba la nota primero y el nombre del archivo despues, que en una
    # foto suelta suele ser un codigo de camara.
    nota = nombre_de_nota(f.get("nota"))
    if suelta and nota:
        resuelto, sin_padron, motivos = a_quien_apunta(
            idx, [(meta["id"], nota, campus, coh)])
        if not resuelto:
            resuelto, sin_padron, motivos = a_quien_apunta(idx, fotos)
    else:
        resuelto, sin_padron, motivos = a_quien_apunta(idx, fotos)

    # 'descartada' y no 'error' (13/9/2026): no se rompio nada, la fuente se
    # leyo bien y el veredicto es que no sirve. Como error inflaba el aviso
    # fuentes_con_error de salud() con 30 carpetas que no hay que arreglar.
    # Y desde el 19/9/2026 tampoco 'descartada': Diego saco ese estado ("nada
    # anotado en Fuentes se descarta"). Sin nombres en los archivos, la fuente
    # espera a que alguien diga quien esta: eso es 'a_mano'.
    if not resuelto and not sin_padron:
        return ("a_mano",
                "Ninguno de los %d archivos trae un nombre: son codigos de "
                "camara. Esta fuente no sirve por el nombre del archivo."
                % len(img),
                0, 0)

    # BUSCAR FOTOS POR NOMBRE NO DA DE ALTA PERSONAS.
    #
    # Esto tomaba el nombre de archivo de cada imagen y, si no lo encontraba en
    # el padron, creaba la persona. Asi entraron al padron "Nini Supermercado"
    # -un aviso de sponsor de un anuario- y "Proyecto Institucional Festejo
    # Solidario" -el titulo de una pagina-, y aparecieron despues en la lista
    # de alumnos sin foto, que es donde Diego los encontro.
    #
    # La regla la fijo el: "el buscador de fotos con nombre no deberia crear
    # personas. Las personas son las que estan. Las fotos, si tienen nombre y
    # matchean bien se asignan, y si no son fotos sin identificar, no personas
    # nuevas + foto".
    #
    # Es la direccion correcta del flujo. El padron sale de las planillas del
    # colegio, que son una lista cerrada de quienes existen; una carpeta de
    # Drive es un monton de archivos con nombres, y algunos son personas. Dejar
    # que la segunda le agregue filas a la primera invierte quien manda.
    #
    # Lo que no matchea no se pierde: la foto queda sin identificar, que es lo
    # que es, y el numero sale en el resultado de la fuente para que se vea si
    # una carpeta trae muchos nombres que el padron no conoce -que casi siempre
    # significa que es la carpeta equivocada, no que falten personas-.
    # UNA FUENTE MARCADA COMO BASE SI DA DE ALTA.
    #
    # Por defecto no, y esa es la regla de Diego: "el buscador de fotos con
    # nombre no deberia crear personas, las personas son las que estan". Asi
    # entraron "Nini Supermercado" y "Persona X".
    #
    # Pero el mismo dia paso una carpeta con 60 fotos de ex staff, una por
    # persona y con el nombre en el archivo, y dijo "incorporalas al sistema".
    # Eso no es una carpeta encontrada: es una BASE, designada por el que manda
    # sobre el padron, igual que las planillas de alumnos y de staff.
    #
    # La diferencia no esta en el contenido -las dos son archivos con nombres-
    # ni en este codigo, que es el mismo: esta en quien dijo que eso es una
    # lista de personas. Por eso vive en la fila de la fuente, en da_de_alta, y
    # no en una decision del script.
    # fuera del if: el resultado cuenta las altas aunque la fuente no de de
    # alta, y sin esto una fuente comun cortaba al final con UnboundLocalError
    # (22/9/2026, la de Persona W) despues de haber guardado las caras
    nuevos = []
    if f.get("da_de_alta") and sin_padron:
        if not escribir:
            return ("pendiente",
                    "ENSAYO — %d imagenes · %d del padron · %d se darian de alta"
                    % (len(img), len(resuelto), len(sin_padron)), None, None)
        for _, arch, nom, camp, c in sin_padron:
            # El nombre sale de como se llama el archivo, y eso viene como
            # viene: "malen esteves .jpg" entro al padron como "malen esteves",
            # en minuscula, al lado de 7.000 fichas escritas con mayuscula. Se
            # emproliza antes de guardar, no despues.
            nom = titulo_nombre(nom)
            pid = cli.rpc("upsert_yearbook_person", {
                "p_display_name": nom, "p_norm_name": clave(nom),
                "p_house": None, "p_cohort": c, "p_desde": None,
                "p_campus": camp})
            resuelto.append((_, arch, {"id": pid, "display_name": nom}, camp, c))
            nuevos.append(int(pid))
        # upsert_yearbook_person crea a todos como alumnos, que es lo que
        # corresponde a un anuario. La fuente dice de quien es.
        cambios = {}
        if f.get("kind") == "staff":
            cambios["kind"] = "staff"
        if f.get("source_alta"):
            cambios["source"] = f["source_alta"]
        if cambios:
            for pid in nuevos:
                cli.update("people", cambios, id="eq.%d" % pid)
        print("   dadas de alta: %d (source %s, kind %s)"
              % (len(nuevos), f.get("source_alta") or "yearbook",
                 f.get("kind") or "student"))
        sin_padron = []

    if not resuelto:
        return ("error",
                "Ninguno de los %d archivos coincide con alguien del padron. "
                "Los %d nombres que trae no estan en las planillas: "
                "probablemente no sea una carpeta de gente."
                % (len(img), len(sin_padron)),
                0, 0)

    detalle = "%d imagenes · %d del padron · %d sin identificar" % (
        len(img), len(resuelto), len(sin_padron))
    if not escribir:
        return "pendiente", "ENSAYO — " + detalle, None, None

    vecs, personas, anios, err = medir_caras(d, resuelto, limite)
    # Ninguna cara en NINGUNA foto no es una falla: es una carpeta sin gente.
    # El 13/9/2026 las 30 "fuentes con error" eran esto -"isologo - AleoBold",
    # "Logos", "Comprobante de pago", "Graphs", "Dibujos separados"- que
    # drive_faltantes.py encontro porque los archivos tenian nombres. Error
    # queda para lo que si hay que arreglar: bajar la foto, abrir la carpeta.
    # 19/9/2026: queda 'listo' con 0 caras (se leyo entera y no habia gente),
    # no 'descartada', que ya no existe.
    if not vecs and set(err) == {"no se detecto cara"}:
        return ("listo",
                "Se resolvieron %d nombres pero no hay ninguna cara en las "
                "fotos (%d sin cara): no es una carpeta de gente."
                % (len(resuelto), err["no se detecto cara"]),
                0, 0)
    if not vecs:
        return ("error",
                "Se resolvieron %d nombres pero no se pudo sacar ninguna cara "
                "(%s)." % (len(resuelto),
                           ", ".join("%s: %d" % (k, v) for k, v in err.most_common())),
                0, 0)

    np.savez_compressed(salida_de(int(f["id"])),
                        vecs=np.array(vecs, dtype=np.float32),
                        personas=np.array(personas),
                        anios=np.array(anios))

    resultado = "%d caras de %d personas · %d altas nuevas" % (
        len(vecs), len(set(personas)), len(nuevos))
    if err:
        resultado += " · sin cara: %d" % sum(err.values())
    return "listo", resultado, len(vecs), len(set(personas))


# Los tipos que no se leen solos. Quedan en estado 'a_mano', que NO es un
# error: la fuente esta bien anotada, el link guardado, y espera que alguien la
# procese. La primera version las marcaba 'error' y Diego lo vio enseguida:
# parecia que se habia roto algo y que no quedaba nada por hacer.
#
# Tampoco vuelven a la cola de pendientes, porque el procesador las tomaria
# todas las noches para decir lo mismo.
FALTA = {
    "planilla":
        "Anotada, esperando que se procese a mano: hay que decirle en que "
        "columna esta el nombre y en cual el link a la foto, y eso cambia en "
        "cada hoja.",
    "anuario_pdf":
        "Anotada, esperando que se procese a mano: el PDF hay que recortarlo "
        "por pagina y atar cada foto de curso con los nombres del pie.",
    "anuario_papel":
        "Anotada, esperando que se procese a mano: los nombres de una pagina "
        "impresa hay que transcribirlos a ojo, como las 280 paginas que ya "
        "entraron.",
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--ensayo", action="store_true")
    ap.add_argument("--id", type=int, default=0, help="procesar solo esa fuente")
    ap.add_argument("--limite", type=int, default=0,
                    help="cuantas caras medir por fuente, para probar")
    a = ap.parse_args()
    print("Motor: %s" % faces.MOTOR)

    cli = sb.SB()
    filtros = {"select": "id,url,tipo,anio,campus,kind,nota,estado,da_de_alta,source_alta"}
    if a.id:
        filtros["id"] = "eq.%d" % a.id
    else:
        filtros["estado"] = "eq.pendiente"
    cola = sorted(cli.select("fuentes", **filtros), key=lambda f: f["id"])

    if not cola:
        print("No hay fuentes pendientes.")
        return
    print("Fuentes a procesar: %d" % len(cola))
    for f in cola:
        print("   #%-3s %-14s %-8s %-8s %s"
              % (f["id"], f["tipo"], f.get("anio") or "-",
                 f.get("campus") or "-", (f["url"] or "")[:52]))
    print()

    d = Drive(SA_PATH)
    idx = indexar_padron(cli)
    print("Padron indexado: %d personas\n" % idx["n"])

    for f in cola:
        print("--- #%s  %s ---" % (f["id"], f["tipo"]))
        if a.aplicar:
            # se marca antes de empezar: bajar y medir una carpeta tarda, y dos
            # corridas simultaneas la procesarian dos veces
            cli.update("fuentes", {"estado": "procesando"}, id="eq.%s" % f["id"])

        if f["tipo"] == "carpeta":
            estado, resultado, caras, personas = procesar_carpeta(
                d, cli, idx, f, a.aplicar, a.limite)
        else:
            estado, resultado, caras, personas = (
                "a_mano", FALTA[f["tipo"]], None, None)

        print("   %-9s %s" % (estado, resultado))
        if not a.aplicar:
            continue
        campos = {"estado": estado, "resultado": resultado,
                  "procesado_at": "now()"}
        if caras is not None:
            campos["caras"] = caras
        if personas is not None:
            campos["personas"] = personas
        cli.update("fuentes", campos, id="eq.%s" % f["id"])

    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar.")
        return
    print("\nCorrer despues:  faces_sugerir.py  y  faces_aprobar.py --aplicar")
    print("(las caras nuevas no sugieren nada hasta que corra faces_sugerir)")


if __name__ == "__main__":
    main()
