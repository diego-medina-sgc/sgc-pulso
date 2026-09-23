# -*- coding: utf-8 -*-
"""Marcar que grupos son de alguien que el padron ya sabe reconocer.

POR QUE CAMBIA EL ORDEN DEL JUEGO

El juego de grupos elige el grupo mas grande que quede sin nombre. Es un buen
criterio -nombrar 300 caras de una vale mas que nombrar 3- pero no dice nada
sobre lo que se GANA, y no es lo mismo.

Si las caras del grupo ya se parecen a alguien del padron, ponerle el nombre
agrega la referencia numero 21 a quien tiene 20. Medido dejandole k referencias
a las 610 personas que tienen 8 o mas: con 6 el reconocimiento acierta 91,10% y
con 8, 90,95%. De 6 para arriba no se gana nada.

Si no se parecen a nadie, ese grupo es la unica forma de que esa persona exista
para el sistema. Hay 1.409 personas en el padron con CERO referencias, y en
confirmar no pueden aparecer nunca: para que haya pregunta hace falta una
sugerencia, y para la sugerencia hace falta una referencia. Este juego es el
unico lugar donde asoman, porque agrupa por parecido sin necesitar saber quien
es nadie.

LO QUE NO HAY QUE HACER CON ESTO

Filtrar. La primera idea fue separar los grupos que son personas de los que son
caras borrosas o de gente ajena, y medirlo fallo dos veces:

  coherencia interna    1,00 en los 9.777 grupos, conocidos y desconocidos por
                        igual. El agrupamiento une a 0,70, asi que por
                        construccion todos salen compactos.

  tamanio de la cara    mediana 0,0112 en los conocidos y 0,0106 en los
  albumes, anios        desconocidos; albumes 1/1/2 en los dos; anios 1/1/1 en
  caras por grupo       los dos.

O sea que los desconocidos son indistinguibles de los conocidos en todo lo que
se pudo medir, y los conocidos son personas de verdad por definicion. No hay
evidencia de que el monton de desconocidos sea basura, asi que no se descarta
nada: se ordena.

COMO

Un grupo es "conocido" si alguna de sus caras llega a 0,50 -el umbral con el
que este proyecto aprueba una cara- contra alguna referencia del padron. Se
escribe en face_group_meta.conocido y grupo_para_nombrar ordena por ahi antes
que por tamanio.

Corre despues del agrupamiento, que es quien arma face_group_meta.
"""
import os
import sys
from collections import defaultdict

import numpy as np

import faces
import sb
from faces_sugerir import todas_las_referencias

CERCA = 0.50
BLOQUE = 4096
LOTE = 500
TODAS = "_caras_todas_arcface.npz"


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    # El mismo freno que referencias_contar.py, y por lo mismo: los .npz llevan
    # el sufijo del motor, asi que sin la variable se leen las huellas de otro
    # motor y sale un resultado sin sentido en vez de un error.
    if os.environ.get("FACES_MOTOR", "").lower() != faces.MOTOR or faces.MOTOR == "sface":
        sys.exit("Falta FACES_MOTOR. En PowerShell: $env:FACES_MOTOR='arcface'")

    cli = sb.SB()

    print("Leyendo referencias…", flush=True)
    vecs, personas, _ = todas_las_referencias(cli)
    R = vecs.astype(np.float32)
    R = R / np.clip(np.linalg.norm(R, axis=1, keepdims=True), 1e-9, None)
    print("  %d caras de referencia" % len(R))

    if not os.path.exists(TODAS):
        sys.exit("Falta %s: lo escribe faces_eventos.py" % TODAS)
    z = np.load(TODAS, allow_pickle=True)
    pid = [str(x) for x in z["photo_ids"]]
    caja = z["cajas"]
    H = z["vecs"].astype(np.float32)
    H = H / np.clip(np.linalg.norm(H, axis=1, keepdims=True), 1e-9, None)
    print("  %d caras en el archivo" % len(H))

    donde = {}
    for i, p in enumerate(pid):
        b = caja[i]
        donde[(p, round(float(b[0]), 2), round(float(b[1]), 2))] = i

    print("Leyendo grupos…", flush=True)
    grupos = defaultdict(list)
    for r in cli.select("face_groups", select="photo_id,bx,by,grupo"):
        i = donde.get((r["photo_id"], round(float(r["bx"]), 2),
                       round(float(r["by"]), 2)))
        if i is not None:
            grupos[r["grupo"]].append(i)
    print("  %d grupos con caras ubicadas" % len(grupos))

    # UNA SOLA MULTIPLICACION GRANDE Y NO UNA POR GRUPO.
    #
    # Preguntarle a cada grupo por separado son 24.000 productos chicos contra
    # una matriz de 19.727x512, y tarda un cuarto de hora. Se juntan todas las
    # caras de todos los grupos, se hace el producto en bloques y despues se
    # reparte por grupo: es el mismo calculo en menos de un minuto.
    orden = sorted(grupos)
    idx, corte = [], []
    for g in orden:
        corte.append(len(idx))
        idx.extend(grupos[g])
    corte.append(len(idx))

    print("Comparando %d caras contra el padron…" % len(idx), flush=True)
    mejor = np.empty(len(idx), dtype=np.float32)
    V = H[idx]
    for a in range(0, len(V), BLOQUE):
        mejor[a:a + BLOQUE] = (V[a:a + BLOQUE] @ R.T).max(axis=1)

    # DE QUIEN SE PARECE, Y NO SOLO SI SE PARECE (23/9/2026)
    #
    # El maximo por cara decia "este grupo se parece a alguien del padron", que
    # es lo que ordena la cola del juego. Falta lo otro: DE QUIEN. Diego lo
    # pidio al ver que Persona Y y Persona Z tenian
    # 4 y 2 fotos mientras sus companeros tenian 30 o 200: sus caras estaban en
    # un grupo que nadie habia nombrado, y no habia forma de ir de la persona a
    # su grupo sin esperar a que la cola lo ofreciera.
    #
    # Se guarda el mejor puntaje de cada (persona, grupo) por encima del corte:
    # con eso el padron puede llevar de una ficha a sus grupos.
    quien = np.empty(len(idx), dtype=np.int64)
    for a in range(0, len(V), BLOQUE):
        s = V[a:a + BLOQUE] @ R.T
        mejor[a:a + BLOQUE] = s.max(axis=1)
        quien[a:a + BLOQUE] = np.asarray(personas)[s.argmax(axis=1)]

    reparto = {0: [], 1: []}
    filas_pg = []
    for k, g in enumerate(orden):
        trozo = mejor[corte[k]:corte[k + 1]]
        reparto[int(len(trozo) and float(trozo.max()) >= CERCA)].append(g)
        # una fila por persona que aparece en el grupo, con su mejor cara
        por_persona = {}
        for pos in range(corte[k], corte[k + 1]):
            if mejor[pos] < CERCA:
                continue
            pid = int(quien[pos])
            if float(mejor[pos]) > por_persona.get(pid, 0):
                por_persona[pid] = float(mejor[pos])
        for pid, sc in por_persona.items():
            filas_pg.append({"person_id": pid, "grupo": int(g),
                             "puntaje": round(sc, 4),
                             "caras": int(corte[k + 1] - corte[k])})

    print("\n%d conocidos, %d que no se parecen a nadie"
          % (len(reparto[1]), len(reparto[0])))

    # VA POR PATCH Y NO POR UPSERT, Y NO ES UN DETALLE.
    #
    # PostgREST traduce el upsert a INSERT ... ON CONFLICT, y Postgres valida
    # los NOT NULL al armar la tupla del INSERT, antes de enterarse de que
    # habia conflicto. Mandar {"grupo": 267, "conocido": 1} sobre una fila que
    # existe falla igual, pidiendo las columnas que no vienen en el payload:
    #
    #   23502  null value in column "caras" violates not-null constraint
    #          Failing row contains (267, null, null, null, null, null, 1)
    #
    # Esta explicado en el docstring de sb.update y lo pise igual.
    #
    # Y el PATCH ademas hace lo correcto con los grupos que NO estan en
    # face_group_meta -el 267 es uno-: no los inventa, no los toca. Esa tabla
    # la arma el agrupamiento y solo guarda los grupos de mas de una cara; los
    # de una sola no tienen fila y tampoco tienen por que tenerla.
    #
    # Son dos valores nada mas, asi que se agrupa por valor en vez de escribir
    # fila por fila: 25.000 llamadas se vuelven unas cincuenta.
    tocadas = 0
    for valor, ids in reparto.items():
        for i in range(0, len(ids), LOTE):
            trozo = ids[i:i + LOTE]
            cli.update("face_group_meta", {"conocido": valor},
                       grupo="in.(%s)" % ",".join(str(x) for x in trozo))
            tocadas += len(trozo)
    print("marcados %d grupos en face_group_meta" % tocadas)

    # persona -> grupos. Se reemplaza entera: los numeros de grupo cambian en
    # cada reagrupamiento, asi que una fila vieja apunta a un grupo que ya no
    # es el mismo (es el mismo motivo por el que face_group_names se vacia).
    cli.rpc("persona_grupos_vaciar")
    print("Guardando %d filas de persona -> grupo…" % len(filas_pg), flush=True)
    for i in range(0, len(filas_pg), 500):
        cli.upsert("persona_grupos", filas_pg[i:i + 500], on_conflict="person_id,grupo")
    print("persona_grupos: %d filas" % len(filas_pg))

    # PRIMERO LOS QUE LE FALTAN A QUIEN CASI NO TIENE FOTOS (23/9/2026)
    #
    # La cola ordenaba por tamaño: los grupos grandes primero, porque un "si"
    # ahi etiqueta mas caras. Pero el que sufre no es el archivo, es la persona
    # que tiene dos fotos mientras sus compañeros tienen doscientas -John
    # encontro dos asi en K5 North-. Con persona_grupos ya se sabe de quien es
    # cada grupo; aca se guarda cuantas fotos tiene el peor cubierto de ellos y
    # grupo_para_nombrar ordena por ese numero.
    #
    # Se guarda en face_group_meta y no se calcula en la consulta: la cola se
    # pide en cada vuelta del juego y contar fotos de 23.000 grupos ahi seria
    # pagarlo en cada pregunta.
    fotos = {int(r["person_id"]): int(r["n"]) for r in cli.rpc("fotos_por_persona") or []}
    peor = {}
    for f in filas_pg:
        n_f = fotos.get(f["person_id"], 0)
        g = f["grupo"]
        if g not in peor or n_f < peor[g]:
            peor[g] = n_f
    print("Guardando el peor cubierto de %d grupos…" % len(peor), flush=True)
    # agrupados por valor, como el conocido de arriba: son pocos valores
    # distintos y asi son decenas de llamadas en vez de miles
    por_valor = defaultdict(list)
    for g, n_f in peor.items():
        por_valor[n_f].append(g)
    for valor, ids in por_valor.items():
        for i in range(0, len(ids), LOTE):
            trozo = ids[i:i + LOTE]
            cli.update("face_group_meta", {"fotos_min": valor},
                       grupo="in.(%s)" % ",".join(str(x) for x in trozo))


if __name__ == "__main__":
    main()
