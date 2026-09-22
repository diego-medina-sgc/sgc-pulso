# -*- coding: utf-8 -*-
"""
Carga el padron de alumnos desde las planillas del colegio.

    python padron_cargar.py --ensayo      # dice que haria
    python padron_cargar.py

DE DONDE SALE

Las listas del panel de Fuentes, en modo lectura -no se toca la fuente-. Desde
el 14/9/2026 el link y la hoja salen de padron_fuentes (sql/fuentes_panel.sql,
build/listas.py): cambiar el link en el panel cambia lo que se lee aca. Los
exalumnos pasaron al Alumni Information Form (hoja "Form responses 1").
Antes eran dos planillas fijas en el codigo:

  ALUMNOS ACTUALES  hoja IMP_students, exportada de iSAMS. Trae "Year Code"
                    como QP4C: la primera letra es la sede (Q o N), el resto
                    el nivel y la division. 855 de Quilmes y 560 de North.

  EXALUMNOS         hoja Alumni. Trae campus, casa y los años de cursada.
                    1.538 de Quilmes y 692 de North.

                    La cursada viene en DOS pares de columnas, uno por sede:
                    "NS desde/hasta" para North y "QS desde/hasta" para
                    Quilmes. La primera version leia solo el par de North, que
                    esta lleno en 673 filas de 2.247, y daba por sentado que
                    el resto no tenia el dato. El par de Quilmes esta lleno en
                    1.529: eran 1.053 exalumnos sin ningun año, 980 de ellos
                    de Quilmes, que no entraban en ninguna camada y que
                    faces_eventos.py no podia poner en ninguna foto.

                    Quien estuvo en las dos sedes tiene los dos pares. Se usa
                    la union, que es la cursada de verdad.

                    Sin año no entraban en ninguna camada. Para el
                    reconocimiento de eventos el efecto es el contrario del
                    que parece: faces_eventos.py trata la ventana vacia como
                    "pudo estar en cualquier año", asi que esas personas eran
                    elegibles siempre. Ponerles la cursada no las habilita,
                    las acota. De las 1.783 que ahora tienen ventana, 503
                    tienen cara de referencia y son las unicas donde eso
                    cambia algo; a las otras 1.280 les sirve para la camada
                    del padron y para proyectarles el nivel en la lista de
                    candidatos del juego.

POR QUE IMPORTA

El padron tenia 216 personas de Quilmes para 3.015 retratos, y por eso el 97%
de esa sede estaba sin identificar: no habia a quien asignarle las caras, ni
nombres que ofrecerle a quien juega. Con esto, la lista de candidatos del
juego pasa de "todos los de 2017" a los veinticinco de esa division.

Los años de cursada, ademas, son los que faces_eventos.py usa para decidir
quien pudo estar en una foto: hasta ahora salian de las fotos que la persona
ya tenia, con una ventana mediana de un año.

QUE NO HACE

No crea a nadie que ya este. El nombre se busca normalizado y ademas se
prueba con un apellido de mas o de menos, que es como varia entre fuentes.
Ante dos candidatos posibles no elige: lo deja para mirar.
"""
import argparse
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import sb
from listas import Listas, indice, texto

HERE = os.path.dirname(os.path.abspath(__file__))

# El anio lectivo de los que figuran como alumnos actuales.
ANIO_ACTUAL = 2026

# La escalera del colegio, de sala a ultimo anio: cinco de kinder, seis de
# primaria y seis de secundaria. Sirve para saber desde cuando pudo estar en el
# colegio alguien que hoy cursa: si esta en P4 en 2026, esta en el noveno
# escalon, y en 2018 recien podia estar entrando a K1. No dice cuando entro
# -pocos empiezan en K1- sino que antes de ese anio no estuvo, que es el limite
# que hace falta para no buscarlo en fotos de cuando no habia nacido.
ESCALERA = (["K%d" % n for n in range(1, 6)]
            + ["P%d" % n for n in range(1, 7)]
            + ["C%d" % n for n in range(1, 7)])


def cursada_actual(nivel):
    """(primer anio posible, ANIO_ACTUAL) para un alumno que hoy cursa."""
    if not nivel or nivel not in ESCALERA:
        return None, ANIO_ACTUAL
    return ANIO_ACTUAL - ESCALERA.index(nivel), ANIO_ACTUAL


def dato(fila, idx, col):
    """La celda de esa columna como texto, o '' si la fila es mas corta."""
    k = idx.get(col)
    return texto(fila[k]) if k is not None and k < len(fila) else ""


def clave(s):
    t = "".join(c for c in unicodedata.normalize("NFD", s or "")
                if unicodedata.category(c) != "Mn").lower()
    return " ".join("".join(c if c.isalnum() or c == " " else " " for c in t).split())


def titulo(s):
    """'Persona Y' -> 'Persona Y'. La planilla trae el apellido en mayusculas."""
    out = []
    for p in (s or "").split():
        out.append(p if (len(p) > 1 and p[1:].islower()) else p.capitalize())
    return " ".join(out)


def leer_actuales(L, lecturas):
    out = []
    for lista in L.de_tipo("alumnos"):
        filas = L.filas(lista)
        # la hoja de iSAMS trae una fila de titulo antes del encabezado
        fila_enc = next((k for k, f in enumerate(filas[:6])
                         if {"year code", "forename", "surname"} <= set(indice(f))), None)
        if fila_enc is None:
            raise RuntimeError("%s: no encuentro el encabezado (Year Code, Forename, Surname)"
                               % lista["titulo"])
        n = len(out)
        out += leer_actuales_filas(filas[fila_enc + 1:], indice(filas[fila_enc]), lista["clave"])
        lecturas.append((lista, len(out) - n))
    return out


def leer_actuales_filas(filas, i, origen):
    out = []
    for f in filas:
        if not dato(f, i, "forename") or not dato(f, i, "surname"):
            continue
        yc = dato(f, i, "year code")
        campus = {"Q": "Quilmes", "N": "North"}.get(yc[:1])
        nombre = titulo("%s %s" % (dato(f, i, "forename"), dato(f, i, "surname")))
        # QP4C -> nivel P4, division C; QC1 -> nivel C1, sin division
        curso = yc[1:] if len(yc) > 1 else ""
        nivel = curso[:2] if curso[:2] in ESCALERA else None
        desde, hasta = cursada_actual(nivel)
        out.append({
            "nombre": nombre,
            "campus": campus,
            "nivel": nivel,
            "division": curso[2:] or None,
            # El codigo entero -QP4C- se guarda tal cual. El nivel de cualquier
            # año anterior sale de una resta contra la escalera, sin tabla
            # intermedia: quien esta en P4 en 2026 estaba en P3 en 2025.
            "year_code": yc or None,
            "genero": dato(f, i, "gender")[:1].upper() or None,
            "desde": desde, "hasta": hasta, "house": None,
            "origen": origen,
        })
    return out


def camada_por_nacimiento(fecha):
    """La camada de un exalumno, a partir de su fecha de nacimiento.

    POR QUE NO SE USA LA COLUMNA DE LA PLANILLA

    Diego miro la camada 2004 en la pantalla del padron y dijo que estaba mal.
    Tenia razon y era peor de lo que parecia: de 2.228 filas de la hoja Alumni,
    solo 593 tienen una edad coherente en "NS/QS hasta". Hay gente nacida en
    2006 que figura dejando el colegio en 2003, y una chica nacida en 2004 con
    "hasta 2004".

    Medido contra las 564 personas cuya camada sale de un anuario -que si es un
    dato duro, porque el anuario de un año fotografia a los que se egresan ese
    año-:

        la columna "hasta" de la planilla     21% dentro de +-1 año
        la fecha de nacimiento                98% dentro de +-1 año

    EL CORTE ES EL 30 DE JUNIO

    Los nacidos de enero a junio se egresan a los 17 y los de julio a diciembre
    a los 18. No es una suposicion: sale de mirar esos 564 casos mes por mes, y
    la separacion es limpia.

    Da la camada exacta en el 74% y dentro de un año en el 98%. El resto son
    repitentes y adelantados, que ninguna formula puede adivinar.
    """
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", (fecha or "").strip())
    if not m:
        return None
    anio, mes = int(m.group(1)), int(m.group(2))
    if not (1900 < anio < ANIO_ACTUAL):
        return None
    return anio + (17 if mes <= 6 else 18)


def leer_exalumnos(L, lecturas):
    """El Alumni Information Form (hoja "Form responses 1"), desde el 14/9/2026.

    El formulario pregunta la cursada de tres maneras segun lo que conteste la
    persona -una sede, la sede principal, primera y segunda sede-, asi que los
    años salen de TODAS las columnas "Started/Left ... (year)" y se usa la
    union, igual que antes con los pares NS y QS.
    """
    out = []
    for lista in L.de_tipo("exalumnos"):
        filas = L.filas(lista)
        i = indice(filas[0])
        col_anios = [k for c, k in i.items() if "year" in c and ("started" in c or "left" in c)]
        col_casas = [k for c, k in i.items() if c.startswith("house")]
        n = 0
        for f in filas[1:]:
            nom, ape = dato(f, i, "first name (s)"), dato(f, i, "surname")
            if not nom or not ape:
                continue
            sede = (dato(f, i, "which site did you primarily attend?") or dato(f, i, "site")).lower()
            campus = "Quilmes" if "quilmes" in sede else ("North" if "north" in sede else None)
            # El tope es el año en curso porque estos años son de cursada: nadie
            # cursó en 2093 (paso con la planilla anterior).
            anios = [int(texto(f[k])) for k in col_anios
                     if k < len(f) and texto(f[k]).isdigit() and 1900 <= int(texto(f[k])) <= ANIO_ACTUAL]
            casas = [texto(f[k]) for k in col_casas if k < len(f) and texto(f[k])]
            cam = dato(f, i, "camada")
            out.append({
                "nombre": titulo("%s %s" % (nom, ape)),
                "campus": campus,
                "nivel": None,
                "division": None,
                "year_code": None,
                "genero": {"Male": "M", "Female": "F"}.get(dato(f, i, "gender")),
                "desde": min(anios) if anios else None,
                "hasta": max(anios) if anios else None,
                "house": casas[0] if casas else None,
                "cohort": (int(cam) if cam.isdigit() and 1900 < int(cam) <= ANIO_ACTUAL + 1
                           else camada_por_nacimiento(dato(f, i, "date of birth"))),
                "origen": lista["clave"],
            })
            n += 1

        # TOTAL ARCHIVE (14/9/2026). El formulario solo tiene a quienes lo
        # respondieron (2.760). La hoja Total Archive del mismo archivo es el
        # archivo completo -unos 4.800 exalumnos con camada y sede- y es la que
        # le da nombre a caras de exalumnos que nunca llenaron el formulario.
        # Diego lo encontro etiquetando: "se quienes son pero no me deja
        # agregarlos porque no estan en las listas".
        # Solo aporta a quien no esta en el formulario; la busqueda contra el
        # padron (main) decide despues si ya existe. No entran las camadas del
        # año en curso en adelante: son alumnos que todavia cursan y los trae la
        # lista de iSAMS. Y un mismo nombre con dos camadas distintas no entra:
        # no hay forma de saber cual es cual.
        try:
            archivo = L.filas(lista, "Total Archive")
        except RuntimeError:
            archivo = []
        ya = {clave(p["nombre"]) for p in out}
        por_nombre = defaultdict(list)
        for f in archivo[1:]:
            ape, nom, cam, sede = (texto(x) for x in (list(f) + [None] * 4)[:4])
            if not ape or not nom or not cam.isdigit():
                continue
            nombre = titulo("%s %s" % (nom, ape))
            por_nombre[clave(nombre)].append((nombre, int(cam), sede))
        m = 0
        for k, filas_k in por_nombre.items():
            if not k or k in ya or len({c for _, c, _ in filas_k}) > 1:
                continue
            nombre, anio, sede = filas_k[0]
            if not (1900 < anio < ANIO_ACTUAL):
                continue
            out.append({
                "nombre": nombre,
                "campus": sede if sede in ("North", "Quilmes") else None,
                "nivel": None, "division": None, "year_code": None, "genero": None,
                # la ventana la rehace alumni_camadas.py con la escalera y las fotos
                "desde": anio - 16, "hasta": anio,
                "house": None, "cohort": anio,
                "origen": lista["clave"],
            })
            m += 1
        print("  %s: %d del formulario y %d mas de Total Archive" % (lista["titulo"], n, m))
        lecturas.append((lista, n + m))
    return out


def ensanchar(cli, existentes):
    """Completa la cursada y la sede de las que ya estaban en el padron.

    Sus años salian de las fotos que la persona ya tenia identificadas, que es
    una ventana de un año para quien estuvo catorce. La planilla dice la
    cursada de verdad, asi que se usa la union de las dos: nunca se recorta el
    rango que ya habia, solo se ensancha. Recortarlo dejaria a alguien afuera
    de una foto en la que ya sabemos que esta.

    Se actualiza por id y no por nombre: el nombre pudo haber coincidido con un
    apellido de mas o de menos, y un upsert por norm_name crearia una ficha
    repetida en vez de completar la que ya existe.

    Y se acumula por persona antes de escribir. Hay gente que matchea con dos
    filas de las planillas -figura como alumno actual y tambien en Alumni, o
    tiene una fila por cada sede- y mandar el mismo id dos veces en el mismo
    lote hace fallar todo el lote:

        ON CONFLICT DO UPDATE command cannot affect row a second time

    Acumular es ademas lo que corresponde: si viene en dos filas, la cursada
    es la union de las dos, no la de la ultima que se leyo.
    """
    junto = {}
    for viejo, nuevo in existentes:
        pid = viejo["id"]
        v = junto.get(pid)
        if v is None:
            v = junto[pid] = {
                "viejo": viejo,
                "desde": viejo.get("first_seen"), "hasta": viejo.get("last_seen"),
                "campus": viejo.get("campus"), "yc": viejo.get("year_code")}
        # 'desde' de una fila tambien puede ensanchar el 'hasta': una fila con
        # un solo año puesto dice que en ese año la persona estaba
        v["desde"] = min([x for x in (v["desde"], nuevo["desde"]) if x], default=None)
        v["hasta"] = max([x for x in (v["hasta"], nuevo["desde"], nuevo["hasta"])
                          if x], default=None)
        v["campus"] = v["campus"] or nuevo["campus"]
        v["yc"] = v["yc"] or nuevo["year_code"]

    filas = []
    for pid, v in junto.items():
        viejo = v["viejo"]
        if (v["desde"], v["hasta"], v["campus"], v["yc"]) == (
                viejo.get("first_seen"), viejo.get("last_seen"),
                viejo.get("campus"), viejo.get("year_code")):
            continue
        filas.append({"id": pid,
                      "display_name": viejo["display_name"],
                      "norm_name": viejo.get("norm_name") or clave(viejo["display_name"]),
                      "kind": viejo.get("kind") or "student",
                      "first_seen": v["desde"], "last_seen": v["hasta"],
                      "campus": v["campus"], "year_code": v["yc"]})
    for j in range(0, len(filas), 200):
        cli.upsert("people", filas[j:j + 200], on_conflict="id")
    return len(filas)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensayo", action="store_true")
    a = ap.parse_args()

    cli = sb.SB()
    L = Listas(cli)
    lecturas = []
    gente = leer_actuales(L, lecturas) + leer_exalumnos(L, lecturas)
    print("Planillas: %d personas  (%s)"
          % (len(gente), dict(Counter(p["origen"] for p in gente))))
    print("  por campus:", dict(Counter(str(p["campus"]) for p in gente)))

    padron = [p for p in cli.select(
        "people", select="id,display_name,norm_name,kind,campus,first_seen,last_seen,year_code,aliases")
        if p.get("kind") != "noise"]
    por_clave = defaultdict(list)
    por_pal = defaultdict(list)
    for p in padron:
        # el nombre de hoy y los anteriores (rename_person los guarda como
        # alias): una lista que sigue escribiendo "Perruzzini" tiene que
        # encontrar a la ficha que Diego corrigio a "Pieruzzini"
        for k in {clave(p.get("display_name"))} | {clave(a) for a in (p.get("aliases") or [])}:
            if k:
                por_clave[k].append(p)
                por_pal[frozenset(k.split())].append(p)
    print("Padron actual: %d personas\n" % len(padron))

    nuevas, existentes, ambiguas = [], [], 0
    vistas = set()
    for p in gente:
        k = clave(p["nombre"])
        if not k or k in vistas:
            continue
        cand = por_clave.get(k, [])
        if not cand:
            pal = set(k.split())
            cand = [x for kk, xs in por_pal.items()
                    if pal and (pal <= set(kk) or set(kk) <= pal) for x in xs]
            cand = [x for x in cand
                    if clave(x.get("display_name")).split()[:1] == k.split()[:1]]
        if len(cand) == 1:
            existentes.append((cand[0], p))
        elif len(cand) > 1:
            ambiguas += 1
        else:
            vistas.add(k)
            nuevas.append(p)

    print("Ya estan en el padron:  %d" % len(existentes))
    print("Ambiguas, no se tocan:  %d" % ambiguas)
    print("Para dar de alta:       %d" % len(nuevas))
    print("  por campus:", dict(Counter(str(p["campus"]) for p in nuevas)))
    print("  con años de cursada:", sum(1 for p in nuevas if p["desde"] or p["hasta"]))
    print("\nEjemplos:")
    for p in nuevas[:8]:
        print("   %-34s %-8s nivel=%-5s %s-%s"
              % (p["nombre"][:34], p["campus"], p["nivel"], p["desde"], p["hasta"]))

    if a.ensayo:
        print("\nEnsayo: no se escribio nada.")
        return

    # Todas las filas con las mismas claves, aunque el valor sea None:
    # PostgREST rechaza un lote cuyos objetos no coinciden ("All object keys
    # must match") y corta a la mitad.
    filas = [{"display_name": p["nombre"],
              "norm_name": clave(p["nombre"]),
              "kind": "student",
              "source": p["origen"],
              "campus": p["campus"] or None,
              "gender_hint": (p["genero"] or "").lower() or None,
              "house": p["house"] or None,
              "first_seen": p["desde"],
              "last_seen": p["hasta"],
              "year_code": p["year_code"]}
             for p in nuevas]
    for j in range(0, len(filas), 200):
        cli.upsert("people", filas[j:j + 200], on_conflict="norm_name,kind")
    print("\nDadas de alta %d personas." % len(filas))

    actualizadas = ensanchar(cli, existentes)
    print("Actualizadas %d de las que ya estaban." % actualizadas)
    altas = Counter(p["origen"] for p in nuevas)
    for lista, n in lecturas:
        L.anotar(lista, n, "%d altas nuevas; %d fichas del padron completadas en total"
                 % (altas.get(lista["clave"], 0), actualizadas))
    print("Conviene correr despues:  faces_sugerir.py --rehacer")


if __name__ == "__main__":
    main()
