# -*- coding: utf-8 -*-
"""Cierra solo los grupos cuya respuesta ya esta en la base.

Corre despues de agrupar, todas las noches. Lo que hace esta explicado con
detalle en sql/grupos_resolver.sql; el resumen es que un grupo cuyas caras ya
estan confirmadas de una persona no se pregunta, se cierra.

    python build/grupos_resolver.py            # ensayo, no escribe nada
    python build/grupos_resolver.py --escribir
    python build/grupos_resolver.py --escribir --cohesion 0.80

El corte de cohesion por defecto es 0.75. Medido sobre los grupos que tienen
tres o mas caras univocas: sin corte, 47 de 1.627 tienen dos personas adentro
(2,9%); con 0.75, 9 de 594 (1,5%); con 0.80, 4 de 421 (0,95%) pero se resuelven
la mitad de los grupos. 0.75 es el punto donde el error queda en lo que Diego
acepta para no tener que revisar a mano.
"""
import argparse

import sb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escribir", action="store_true")
    ap.add_argument("--cohesion", type=float, default=0.75)
    ap.add_argument("--limite", type=int, default=20000)
    # nivel 3 (sql/grupos_resolver.sql): validado 99,8% con 0,95 y 3 caras
    ap.add_argument("--coincidencia", type=float, default=0.95)
    ap.add_argument("--min-caras", dest="min_caras", type=int, default=3)
    # sin cobertura, 3 caras nombradas le ponian nombre a grupos de 150
    ap.add_argument("--cobertura", type=float, default=0.25)
    # o 10% con al menos 10 caras nombradas: 0 contradicciones en 593 respuestas
    ap.add_argument("--cobertura-baja", dest="cobertura_baja", type=float, default=0.10)
    ap.add_argument("--min-nombradas-baja", dest="min_nombradas_baja", type=int, default=10)
    # nivel 4 (sql/resolver_por_parecido.sql, 18/9/2026): el grupo se parece a
    # las caras confirmadas de una persona. De 0,65 para arriba las personas
    # contestaron 338 "si" y ningun "no"; 0,70 deja un escalon de margen.
    # 0,65 desde el 19/9/2026: John contesto 36 grupos mas, todos "si", seis
    # de ellos entre 0,60 y 0,65. Seguian en la cola 1.665 grupos entre 0,65 y
    # 0,70 que casi seguro eran "si".
    ap.add_argument("--parecido", type=float, default=0.65)
    ap.add_argument("--margen", type=float, default=0.10)
    # y solo con referencia solida: 10 fotos o mas, y no mas caras que esas
    ap.add_argument("--min-fotos", dest="min_fotos", type=int, default=10)
    a = ap.parse_args()

    cli = sb.SB()

    # Hasta que no quede nada, y no una sola vez.
    #
    # Cada vuelta escribe photo_people, y eso tapa fotos que antes no estaban
    # tapadas: grupos que en la primera pasada no llegaban a "todas sus caras
    # confirmadas" pasan a llegar en la segunda. La primera corrida cerro 1.314
    # grupos y la segunda pasada encontro 216 mas sin haber hecho nada nuevo.
    # Se cae solo despues de unas pocas vueltas.
    g1 = g2 = g3 = g4 = c1 = c2 = c3 = c4 = nuevas3 = nuevas4 = fotos = 0
    for vuelta in range(1, 11):
        r = cli.rpc("grupos_resolver", {
            "p_ensayo": not a.escribir,
            "p_cohesion": a.cohesion,
            "p_limite": a.limite,
            "p_coincidencia": a.coincidencia,
            "p_min_caras": a.min_caras,
            "p_cobertura": a.cobertura,
            "p_cobertura_baja": a.cobertura_baja,
            "p_min_nombradas_baja": a.min_nombradas_baja,
            "p_parecido": a.parecido,
            "p_margen": a.margen,
            "p_min_fotos": a.min_fotos,
        }) or {}
        n1, n2, n3 = r.get("ya_contestado", 0), r.get("por_mayoria", 0), r.get("por_caras", 0)
        n4 = r.get("por_parecido", 0)
        print("vuelta %d: ya contestados %d (%s caras), por mayoria %d (%s caras), "
              "por caras nombradas %d (%s caras, %s sin nombre)"
              % (vuelta, n1, r.get("caras_ya_contestado", 0),
                 n2, r.get("caras_por_mayoria", 0),
                 n3, r.get("caras_por_caras", 0), r.get("caras_nuevas_por_caras", 0)))
        print("         por parecido con una persona %d (%s caras, %s sin nombre)"
              % (n4, r.get("caras_por_parecido", 0), r.get("caras_nuevas_por_parecido", 0)))
        g1 += n1; g2 += n2; g3 += n3; g4 += n4
        c4 += int(r.get("caras_por_parecido", 0) or 0)
        nuevas4 += int(r.get("caras_nuevas_por_parecido", 0) or 0)
        c1 += int(r.get("caras_ya_contestado", 0) or 0)
        c2 += int(r.get("caras_por_mayoria", 0) or 0)
        c3 += int(r.get("caras_por_caras", 0) or 0)
        nuevas3 += int(r.get("caras_nuevas_por_caras", 0) or 0)
        fotos += int(r.get("fotos_etiquetadas", 0) or 0)
        if not a.escribir or (n1 == 0 and n2 == 0 and n3 == 0 and n4 == 0):
            break

    print("")
    print("grupos que ya estaban contestados : %5d  (%d caras)" % (g1, c1))
    print("grupos que decide el propio grupo : %5d  (%d caras)" % (g2, c2))
    print("grupos por sus caras ya nombradas : %5d  (%d caras, %d nuevas)" % (g3, c3, nuevas3))
    print("grupos por parecido con alguien   : %5d  (%d caras, %d nuevas)" % (g4, c4, nuevas4))
    if a.escribir:
        print("fotos etiquetadas                 : %5d" % fotos)
    else:
        print("ENSAYO: no se escribio nada. Agregale --escribir.")


if __name__ == "__main__":
    main()
