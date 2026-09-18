# -*- coding: utf-8 -*-
"""Recalcula, por album, cuantas caras detectadas no tienen nombre.

    python quien_es_refrescar.py

Es lo que usa el filtro "+ ¿Quién es?" de la home para elegir albumes
(sql/filtro_quien_es.sql). Contarlo en vivo sobre todo el archivo tarda ~11 s y
el navegador corta a los 3, asi que lo deja hecho el pulso despues de agrupar,
del mapa de caras sueltas y de las sugerencias. nombrar_cara rehace al instante
el album que toca.
"""
import sys
import time

import sb


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    n = sb.SB().rpc("refrescar_quien_es")
    print("albumes con caras sin nombre: %s   (%.0f s)" % (n, time.time() - t0))


if __name__ == "__main__":
    main()
