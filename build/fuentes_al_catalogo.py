# -*- coding: utf-8 -*-
"""Que las carpetas anotadas en Fuentes entren al catalogo, no solo al padron.

    python fuentes_al_catalogo.py            # clasifica y no escribe
    python fuentes_al_catalogo.py --aplicar

Paso "catalogo" del pulso desde el 21/10/2026, despues de "fuentes". Hace tres
cosas, las tres sin pisar nada:

    1. anota en fuentes_fotos las carpetas de Fuentes que no estan en el
       catalogo (lo de abajo);
    2. le pone año a las raices de fuentes carpeta que no lo tienen, con la
       misma regla que sync_drive.anio_de_raiz (ver ahi el porque);
    3. a las fuentes cuyas fotos van al reconocimiento sin dar caras de
       referencia (fuentes_procesar.AL_RECONOCIMIENTO) les escribe en que parte
       del circuito estan: fuera del catalogo, sin año, esperando el detector o
       pasadas por el detector.

EL AGUJERO

Hay dos tablas que se llaman parecido y no se hablan:

    fuentes_fotos   los drives y carpetas que sync_drive.py recorre. Lo que
                    esta aca entra al catalogo: folders y photos.
    fuentes         lo que Diego anota desde el Padron. Solo lo lee
                    fuentes_procesar.py, y solo para sacar NOMBRES.

Asi que anotar una fuente nunca metio una foto en la biblioteca. Medido: de las
191 fuentes anotadas, 190 no estaban en el catalogo.

Diego, 16/9/2026: "todo lo cargado en 'Fuentes: fotos de la biblioteca y listas
de personas' se tiene que procesar e indexar y pasar al reconocimiento siempre.
Si vas a NO hacer algo, que sentido tiene que yo te defina esas fuentes?".

Y lo vio de la forma mas concreta: cargo una foto suya de un alumno, se guardo
como cara de referencia, y "la foto que cargue en el padron no quedo en el
album cuando lo buscas a el, y deberia".

POR QUE ESTO Y NO UN INDEXADOR NUEVO

La maquinaria existe: sync_drive.py ya recorre una carpeta comun de My Drive
-es lo que hace con la de Quilmes, raiz='carpeta'- y escribe folders y photos.
Faltaba anotar estas carpetas ahi. Las fuentes viven en My Drive, no en la
unidad compartida -driveId viene nulo en todas- y por eso el indexador, que
recorre dos raices compartidas, nunca las alcanzo.

LA SEDE Y EL MARKETING SALEN DEL CAMINO DE DRIVE, NO DE LA NOTA

fuentes_fotos.sede es obligatorio y 115 de las 173 no tienen sede anotada.
Ponerles North por defecto marcaria como North carpetas de Quilmes y ensuciaria
todos los filtros.

El camino de la carpeta en Drive lo dice mejor que cualquier otra cosa:

    Roberts < Prep Houses < Houses < This is Us < Quilmes < Contenido
    LOCKWOOD-... < 2021-Houses-Secondary North-COMPLETO- < 3.SECONDARY North
    Exchanges 2018 < Georgian 2018 < Georgians < MKT < Marketing & Comms

El tercero ademas muestra por que la nota no alcanza para el filtro de
marketing: el nombre de la carpeta no dice nada y el camino dice MKT.

Lo que no se puede decidir no se decide: las que quedan sin sede se listan con
su link para que las mire una persona, en vez de entrar con una sede inventada.
Pasa con diez: cinco no tienen madre en Drive -estan compartidas sueltas- y las
otras cinco cuelgan de arboles que no nombran sede (Georgians, One School).
"""
import argparse
import re
import sys

import sb
from fuentes_procesar import AL_RECONOCIMIENTO
from index_drive import Drive, SA_PATH
from sync_drive import HONDO_ANIO, anio_del_camino

# MARKETING ES DUENIO DEL DRIVE DE FOTOS, ASI QUE NO SE MIRA EL CAMINO.
#
# La primera version buscaba "MKT" y "marketing" en el camino entero y
# descartaba casi todo el archivo:
#
#   #12  Fotos Individuales Year 6 < Primary < ... < MKT < Marketing & Comms
#   #13  Valete 2020 < Secondary < ... < MKT < Marketing & Comms
#   #26  Graduates < Prep School < ... < MKT < Marketing & Comms
#
# Todo el arbol "Georgians" cuelga del drive de Marketing & Communications, que
# es el area que saca las fotos. Colgar de ahi no dice nada sobre el contenido.
#
# Lo que sirve es el nombre de la carpeta y el de su madre inmediata: una
# carpeta de logos se llama "Logos", y una de flyers cuelga de "02_Flyer_...".
MARKETING = re.compile(
    r"logo|isologo|flyer|banner|stories|instagram|linkedin|posteo|comprobante"
    r"|pantalla|placa|rrss|newsletter|tarjeta|firma de mail|redes sociales"
    r"|imagen institucional|piezas", re.I)
# "Norte" tambien: siete carpetas quedaban sin sede por estar escritas en
# castellano en el camino de Drive.
QUILMES = re.compile(r"\bquilmes\b", re.I)
NORTH = re.compile(r"\bnorth\b|\bnorte\b", re.I)
HONDO = 8


def id_de(url):
    m = (re.search(r"/folders/([A-Za-z0-9_-]{20,})", url or "")
         or re.search(r"/d/([A-Za-z0-9_-]{20,})", url or ""))
    return m.group(1) if m else None


def camino(d, fid, hondo=HONDO):
    """Los nombres desde la carpeta hasta la raiz, de adentro hacia afuera."""
    out, cur = [], fid
    for _ in range(hondo):
        try:
            m = d.get("files/" + cur, fields="name,parents")
        except Exception:
            break
        out.append(m.get("name") or "")
        ps = m.get("parents") or []
        if not ps:
            break
        cur = ps[0]
    return out


def anios_de_raices(cli, d, fuentes):
    """Las raices de fuentes carpeta sin año que lo pueden tener: [(id, año, de_donde, nombre)].

    Solo las que no tienen año: completar, no pisar. Y solo raices (sin madre en
    la base): una hija lo hereda de su madre por el trigger."""
    raices = {r["id"] for r in cli.select("fuentes_fotos", select="id,raiz")
              if r.get("raiz") == "carpeta"}
    sin_anio = [r for r in cli.select("folders", select="id,name,parent_id,year",
                                      year="is.null", parent_id="is.null")
                if r["id"] in raices]
    del_panel = {}
    for f in fuentes:
        fid = id_de(f["url"])
        if fid and f.get("anio"):
            del_panel.setdefault(fid, set()).add(int(f["anio"]))
    out, sin = [], []
    for r in sorted(sin_anio, key=lambda r: r.get("name") or ""):
        anios = del_panel.get(r["id"]) or set()
        if len(anios) == 1:
            out.append((r["id"], anios.pop(), "Fuentes", r.get("name") or ""))
            continue
        y = anio_del_camino(camino(d, r["id"], HONDO_ANIO)[1:])
        if y:
            out.append((r["id"], y, "camino de Drive", r.get("name") or ""))
        else:
            sin.append(r)
    return out, sin


CIRCUITO = " En el circuito: "


def circuito(cli, fuentes, motivo_fuera):
    """[(fuente, resultado nuevo)] de las fuentes cuyas fotos van al
    reconocimiento, con en que parte del circuito estan hoy. Se mide, no se
    supone: el catalogo, el año y carpetas_caras_revisadas."""
    van = [f for f in fuentes if AL_RECONOCIMIENTO in (f.get("resultado") or "")]
    if not van:
        return []
    ids = sorted({id_de(f["url"]) for f in van if id_de(f["url"])})
    fol, rev = {}, {}
    for i in range(0, len(ids), 100):
        lote = "in.(%s)" % ",".join(ids[i:i + 100])
        for r in cli.select("folders", select="id,year,is_mugshot,photo_count", id=lote):
            fol[r["id"]] = r
        for r in cli.select("carpetas_caras_revisadas", select="folder_id,procesadas",
                            folder_id=lote):
            rev[r["folder_id"]] = r
    ff = {r["id"]: r for r in cli.select("fuentes_fotos", select="id,sincroniza")}
    out = []
    for f in van:
        fid = id_de(f["url"])
        c = fol.get(fid)
        if c is None:
            if fid in ff and ff[fid].get("sincroniza"):
                txt = "anotada para el catalogo; entra con el proximo sync de Drive."
            else:
                txt = "fuera del catalogo (%s)." % (motivo_fuera.get(fid)
                                                   or "no esta en fuentes_fotos")
        elif c.get("is_mugshot"):
            txt = "%d fotos en el catalogo, como carnets: las lee el paso de retratos." % (
                c.get("photo_count") or 0)
        elif c.get("year") is None:
            txt = ("%d fotos en el catalogo, sin año: el detector todavia no pasa por "
                   "carpetas sin año." % (c.get("photo_count") or 0))
        elif fid in rev:
            txt = "%d fotos en el catalogo (%d), %d pasadas por el detector." % (
                c.get("photo_count") or 0, c["year"], rev[fid].get("procesadas") or 0)
        else:
            txt = "%d fotos en el catalogo (%d), esperando el detector (paso caras)." % (
                c.get("photo_count") or 0, c["year"])
        base = (f.get("resultado") or "").split(CIRCUITO)[0]
        nuevo = base + CIRCUITO + txt
        if nuevo != f.get("resultado"):
            out.append((f, nuevo))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    # LA SEDE NO SE INVENTA. Diego, 16/9/2026, cuando le puse North a las diez
    # que no se podian deducir: "no no no quien te dijo eso? No tomes
    # definiciones que nadie te dio, quien te dijo que north es el campus
    # principal?". Tenia razon: que North sea "el campus principal" lo supuse
    # yo, no salio de ningun lado.
    #
    # Las carpetas compartidas sueltas ademas no tienen madre -Drive no
    # devuelve parents-, asi que no hay root del que leer la sede, y 'Ambas' lo
    # rechaza el check de la columna. No hay de donde sacarla: la dice alguien.
    #
    # Por eso esto no es un default. Es para cuando una persona ya miro la
    # lista de abajo -que sale con el link de cada carpeta- y dice "estas son
    # de tal lado". Sin --sede, las que no se pueden deducir quedan afuera.
    ap.add_argument("--sede", default=None,
                    help="solo si una persona ya decidio la sede de las que "
                         "quedan sin deducir; sin esto se listan y no entran")
    a = ap.parse_args()

    cli = sb.SB()
    d = Drive(SA_PATH)
    fuentes = cli.select("fuentes", select="id,url,nota,campus,anio,tipo,resultado")
    ya_cat = {r["id"] for r in cli.select("folders", select="id")}
    ya_sync = {r["id"] for r in cli.select("fuentes_fotos", select="id")}

    candidatas, vistos = [], set()
    for f in fuentes:
        fid = id_de(f["url"])
        if not fid or fid in ya_cat or fid in ya_sync or fid in vistos:
            continue
        vistos.add(fid)
        candidatas.append((fid, f))
    print("fuentes sin catalogar: %d   (mirando el camino de cada una…)"
          % len(candidatas), flush=True)

    entran, marketing, sin_sede = [], [], []
    for i, (fid, f) in enumerate(candidatas):
        ruta = camino(d, fid)
        texto = " < ".join(ruta) + " " + (f.get("nota") or "")
        # la carpeta, su madre y su abuela: ver el comentario de MARKETING.
        # Tres y no dos porque "Blanco < Jpg < Logos" y "Rojo < Png < Logos"
        # son variantes de un logo y el nombre propio no lo dice.
        cerca = " < ".join(ruta[:3]) + " " + (f.get("nota") or "")
        if MARKETING.search(cerca):
            marketing.append((fid, f, ruta))
            continue
        sede = (f.get("campus") or "").strip()
        if not sede:
            if QUILMES.search(texto):
                sede = "Quilmes"
            elif NORTH.search(texto):
                sede = "North"
        if not sede and a.sede:
            sede = a.sede
        if not sede:
            sin_sede.append((fid, f, ruta))
            continue
        entran.append((fid, f, sede, ruta))
        if (i + 1) % 40 == 0:
            print("   %d/%d…" % (i + 1, len(candidatas)), flush=True)

    print("\n  ENTRAN:                %d" % len(entran))
    print("  afuera por marketing:  %d" % len(marketing))
    print("  sin sede, sin decidir: %d" % len(sin_sede))
    from collections import Counter
    print("  sedes:", Counter(s for _, _, s, _ in entran).most_common())

    print("\n--- afuera por marketing ---")
    for fid, f, ruta in marketing[:20]:
        print("   #%-4s %s" % (f["id"], " < ".join(ruta)[:78]))

    # con el link, porque lo que hace falta es que alguien las abra y mire
    print("\n--- sin sede: las mira una persona ---")
    for fid, f, ruta in sin_sede:
        print("   #%-4s %s" % (f["id"], (ruta[0] if ruta else "")[:70]))
        print("         %s" % f["url"])
        print("         %s" % (" < ".join(ruta)[:78] or "(sin madre en Drive)"))

    # 2. el año de las raices
    anios, sin_anio = anios_de_raices(cli, d, fuentes)
    print("\n--- raices de fuentes carpeta sin año: %d con año para completar, "
          "%d sin año que deducir ---" % (len(anios), len(sin_anio)))
    print("   de Fuentes: %d   del camino de Drive: %d"
          % (sum(1 for x in anios if x[2] == "Fuentes"),
             sum(1 for x in anios if x[2] != "Fuentes")))
    for fid, y, de, nombre in anios:
        print("   %d  %-16s %s" % (y, de, nombre[:60]))
    # la lista de los que no cruzaron: sin año siguen fuera del detector
    for r in sin_anio:
        print("   sin año: %s  https://drive.google.com/drive/folders/%s"
              % ((r.get("name") or "")[:50], r["id"]))

    # 3. en que parte del circuito esta cada fuente que va al reconocimiento
    motivo = {}
    for fid, _f, _r in marketing:
        motivo[fid] = ("pieza de diseño: el filtro de marketing de "
                       "fuentes_al_catalogo.py la deja afuera")
    for fid, _f, _r in sin_sede:
        motivo[fid] = "sin sede: la tiene que decir una persona"
    for r in cli.select("fuentes_fotos", select="id,sincroniza,nota"):
        if not r.get("sincroniza"):
            motivo.setdefault(r["id"], "apagada en fuentes_fotos: %s" % (r.get("nota") or "")[:80])
    estados = circuito(cli, fuentes, motivo)
    print("\n--- fuentes que van al reconocimiento: %d resultados cambian ---" % len(estados))
    for f, nuevo in estados:
        print("   #%-4s %s" % (f["id"], nuevo.split(CIRCUITO)[-1][:100]))

    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar.")
        return

    for fid, y, de, nombre in anios:
        # year=is.null: si alguien se lo puso entre la lectura y aca, no se pisa
        cli.update("folders", {"year": y}, id="eq.%s" % fid, year="is.null")
    print("\naño puesto a %d raices (las hijas sin año lo heredan por el trigger)" % len(anios))
    for f, nuevo in estados:
        cli.update("fuentes", {"resultado": nuevo}, id="eq.%s" % f["id"])
    print("resultado actualizado en %d fuentes" % len(estados))

    filas = []
    for fid, f, sede, ruta in entran:
        nombre = (ruta[0] if ruta else "") or ("fuente %s" % f["id"])
        filas.append({
            "id": fid,
            "nombre": nombre[:80],
            "proveedor": "drive",
            "raiz": "carpeta",
            "sede": sede,
            "sincroniza": True,
            "nota": "Anotada en Fuentes (#%s). Entra al catalogo, no solo al "
                    "padron. Camino: %s" % (f["id"], " < ".join(ruta)[:160]),
        })
    for i in range(0, len(filas), 100):
        cli.upsert("fuentes_fotos", filas[i:i + 100], on_conflict="id")
    print("\nregistradas %d carpetas en fuentes_fotos" % len(filas))
    print("El proximo sync_drive.py las recorre y sus fotos entran al catalogo.")


if __name__ == "__main__":
    main()
