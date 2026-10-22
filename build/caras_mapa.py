# -*- coding: utf-8 -*-
"""Los recuadros de las caras que no estan en ningun grupo, para el visor.

    python caras_mapa.py              # mide cuanto ocuparia y no escribe
    python caras_mapa.py --aplicar

POR QUE

El visor deja nombrar cualquier cara detectada ("+ ¿Quién es?"), pero leia los
recuadros de face_groups, que solo guarda las caras que cayeron en un grupo de
2 o mas. Las demas -una cara que aparece una sola vez en el archivo- no tenian
recuadro. Estan en _caras_todas.npz; aca se suben compactas a caras_mapa, una
fila por foto (sql/caras_mapa.sql).

Corre en el pulso despues de "grupos": lo que no quedo en un grupo es lo que
va aca. Reescribe la tabla entera (TRUNCATE y carga), asi no se infla.

Desde el 21/10/2026 sube tambien los recuadros chicos (40-59 px, sin huella)
de _caras_chicas.npz a caras_mapa.chicas: se ven y se pueden nombrar, pero no
son referencia (faces.LADO_RECUADRO).
"""
import argparse
import os
import sys
import time
from collections import defaultdict

if os.environ.get("FACES_MOTOR", "").lower() != "arcface":
    sys.exit("Falta FACES_MOTOR. En PowerShell: $env:FACES_MOTOR='arcface'")

import numpy as np  # noqa: E402

import faces  # noqa: E402
import sb  # noqa: E402

# Freno contra un calculo desbocado, no control de disco: ese vive en salud()
# (renglon base_mb). 300 MB es el doble del peor caso medido el 15/10/2026
# con el piso mas bajo, sobre 8 GB de Supabase Pro.
TOPE_MB = 300


def columna_chicas(cli):
    """Si caras_mapa ya tiene la columna chicas. Avisa una vez si no."""
    try:
        cli.select("caras_mapa", select="photo_id,chicas", limit="1")
        return True
    except Exception as e:
        print("  (caras_mapa no tiene la columna chicas todavia: %s; los recuadros "
              "chicos no se suben)" % str(e)[:120], flush=True)
        return False


def leer_chicas():
    """{photo_id: [bx, by, bw, bh, ...]} de faces.CACHE_CHICAS. Vacio si no esta."""
    ruta = faces.npz(faces.CACHE_CHICAS)
    out = defaultdict(list)
    if not os.path.exists(ruta):
        print("  (no hay %s todavia: el detector no corrio con el piso del "
              "recuadro)" % os.path.basename(ruta), flush=True)
        return out
    with np.load(ruta, allow_pickle=True) as z:
        F = z["photo_ids"].astype(str)
        C = z["cajas"].astype(np.float32)
    for i in range(len(F)):
        bx, by, bw, bh = (round(float(x), 4) for x in C[i])
        if bw > 0 and bh > 0:
            out[F[i]].extend([bx, by, bw, bh])
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    cli = sb.SB()

    print("Leyendo las caras agrupadas…", flush=True)
    agrupadas = {(r["photo_id"], round(float(r["bx"]), 4), round(float(r["by"]), 4))
                 for r in cli.select("face_groups", select="photo_id,bx,by")}
    print("  en grupos: %d   (%.0f s)" % (len(agrupadas), time.time() - t0), flush=True)

    with np.load(faces.npz("_caras_todas.npz"), allow_pickle=True) as z:
        F = z["photo_ids"].astype(str)
        C = z["cajas"].astype(np.float32)
    por_foto = defaultdict(list)
    sueltas = 0
    for i in range(len(F)):
        bx, by, bw, bh = (round(float(x), 4) for x in C[i])
        if bw <= 0 or bh <= 0:
            continue
        if (F[i], bx, by) in agrupadas:
            continue
        por_foto[F[i]].extend([bx, by, bw, bh])
        sueltas += 1

    # Los recuadros chicos (faces.LADO_RECUADRO a LADO_EVENTOS-1 px, sin
    # huella; 21/10/2026) van a la columna chicas, con el mismo formato. Una
    # foto que solo tiene chicas lleva cajas vacio. Si la columna no existe
    # todavia (pedido a base), se avisa y se escribe como antes.
    con_chicas = columna_chicas(cli)
    chicas_de = leer_chicas() if con_chicas else {}
    claves = set(por_foto) | set(chicas_de)
    if con_chicas:
        filas = [{"photo_id": f, "cajas": por_foto.get(f, []), "chicas": chicas_de.get(f, [])}
                 for f in sorted(claves)]
    else:
        filas = [{"photo_id": f, "cajas": c} for f, c in sorted(por_foto.items())]
    # caras_mapa no tiene clave foranea a photos, asi que nada frena las caras
    # de fotos borradas en Drive: el 13/10/2026 habia 2.307 filas sin foto. Se
    # dejan afuera aca, tambien en el ensayo para contarlas.
    filas = cli.solo_fotos_vivas(filas, "caras_mapa")
    sueltas = sum(len(f["cajas"]) // 4 for f in filas)
    n_chicas = sum(len(f.get("chicas") or []) // 4 for f in filas)
    solo_chicas = sum(1 for f in filas if not f["cajas"])
    # fila: ~24 de cabecera + id de 33 + arreglo (24 + 4 por numero); indice ~45 por fila
    bytes_tabla = sum(24 + 34 + 24 + 4 * len(f["cajas"])
                      + (24 + 4 * len(f["chicas"]) if "chicas" in f else 0) for f in filas)
    mb = (bytes_tabla + 45 * len(filas)) / 1048576
    print("  caras detectadas: %d   sin grupo: %d   en %d fotos" % (len(F), sueltas, len(filas)))
    if con_chicas:
        print("  recuadros chicos (%d-%d px, sin huella): %d   fotos que solo tienen chicos: %d"
              % (faces.LADO_RECUADRO, faces.LADO_EVENTOS - 1, n_chicas, solo_chicas))
    print("  ocuparia ~%.1f MB (tope %d MB)" % (mb, TOPE_MB))

    if mb > TOPE_MB:
        # Sale con error: un return dejaba el paso como "listo" en el pulso, con
        # firma nueva, y el mapa quedaba viejo sin que nadie se enterara.
        sys.exit("NO SE ESCRIBE: ocuparia ~%.1f MB, mas del tope de %d MB." % (mb, TOPE_MB))
    if not a.aplicar:
        print("\nEnsayo: no se escribio nada. Agregar --aplicar. (%.0f s)" % (time.time() - t0))
        return
    cli.rpc("vaciar_caras_mapa")
    for i in range(0, len(filas), 2000):
        cli.upsert("caras_mapa", filas[i:i + 2000], on_conflict="photo_id")
        if (i // 2000) % 10 == 0:
            print("  %d/%d" % (min(i + 2000, len(filas)), len(filas)), flush=True)
    print("\nescritas %d fotos con %d caras (%.0f s)" % (len(filas), sueltas, time.time() - t0))


if __name__ == "__main__":
    main()
