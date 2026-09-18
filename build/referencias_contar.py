# -*- coding: utf-8 -*-
"""Cuantas caras de referencia tiene cada persona, escrito en la base.

POR QUE HACE FALTA

La cola de confirmar necesita saber a quien ya reconoce bien para dejar de
preguntarle. Ese numero no esta en la base: las referencias viven en los .npz
del archivo, que es donde el reconocimiento las busca.

Y no se puede reemplazar contando photo_people. Ahi hay 10.032 etiquetas
puestas desde los grupos y 2.874 desde el clustering, y una etiqueta no es una
referencia: faces_referencias.py solo archiva la cara cuando puede senialar
CUAL de las caras de la foto es -una sola cara en la foto, o un recuadro
guardado-. Una foto de acto con treinta personas etiquetada entera no aporta
ninguna referencia, y contarla como si aportara silenciaria preguntas que si
habia que hacer.

POR QUE 6

Medido sobre las 610 personas con 8 o mas referencias, dejandoles k y
escondiendoles el resto:

    1 referencia   82,72%      4 referencias   90,26%
    2 referencias  88,30%      6 referencias   91,10%
    3 referencias  89,62%      8 referencias   90,95%

De 1 a 3 se ganan 7 puntos; de 6 a 8, nada. La septima referencia de una
persona no ayuda a encontrarla en ningun lado.

Se corre despues de faces_referencias.py, que es quien arma el archivo del que
esto lee. Si corre antes, cuenta lo de ayer, y lo de ayer solo puede ser menos:
el error es preguntar de mas, que es el que ya teniamos.
"""
import os
import sys
from collections import Counter

import faces
import sb
from faces_sugerir import todas_las_referencias

TOPE = 6      # el mismo que el de sql/referencias_cola.sql
LOTE = 300


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    # NO ARRANCA SIN QUE LE DIGAN EL MOTOR, Y NO ES BUROCRACIA.
    #
    # faces.MOTOR sale de FACES_MOTOR y su default es "sface". Los nombres de
    # archivo llevan el sufijo del motor -_referencias.npz para sface,
    # _referencias_arcface.npz para arcface- asi que olvidarse la variable no
    # da un error: lee OTRAS referencias, de otro motor, y sigue como si nada.
    #
    # Paso el 12/9/2026 y salio caro. Diego corrio faces_referencias.py y este
    # script con un comando que yo le di sin la variable. El primero se paso
    # horas reconstruyendo el archivo de sface, que no usa nadie. El segundo
    # conto esas referencias y reescribio la tabla entera: de 4.763 personas
    # quedaron 3.148, borro 1.615 filas, y el tope de confirmar se solto para
    # toda esa gente sin que nada avisara. Lo unico que se vio fue que dos
    # numeros de la pantalla se movieron para el lado contrario.
    #
    # Este script BORRA lo que no cuenta, asi que es el que mas dano hace con
    # el motor equivocado. Por eso el freno va aca y es duro: si la variable no
    # esta puesta a proposito, no corre.
    if os.environ.get("FACES_MOTOR", "").lower() != faces.MOTOR or faces.MOTOR == "sface":
        sys.exit(
            "Falta FACES_MOTOR. Este script reescribe persona_referencias y "
            "borra lo que no cuenta, asi que leer las referencias del motor "
            "equivocado deja la tabla mal sin avisar.\n"
            "  PowerShell:  $env:FACES_MOTOR='arcface'; "
            ".venv-faces\\Scripts\\python.exe referencias_contar.py\n"
            "  cmd:         set FACES_MOTOR=arcface")

    cli = sb.SB()

    print("Leyendo el juego de referencias…")
    _, personas, _ = todas_las_referencias(cli)
    cuenta = Counter(int(p) for p in personas)
    print("\n%d referencias de %d personas" % (sum(cuenta.values()), len(cuenta)))

    tramos = Counter()
    for n in cuenta.values():
        tramos["1-2" if n < 3 else "3-5" if n < TOPE else "%d+" % TOPE] += 1
    print("  %s" % "   ".join("%s: %d" % (k, tramos[k])
                              for k in ("1-2", "3-5", "%d+" % TOPE)))

    filas = [{"person_id": p, "n": n} for p, n in sorted(cuenta.items())]
    cli.upsert("persona_referencias", filas, on_conflict="person_id", lote=LOTE)
    print("escritas %d filas" % len(filas))

    # LO QUE YA NO ESTA TAMBIEN HAY QUE ESCRIBIRLO.
    #
    # El upsert solo toca lo que le mando. Una persona que perdio sus
    # referencias -se le borro una cara mal archivada, se fusiono con otra
    # ficha, quedo marcada como ruido- se quedaria con el numero viejo, y el
    # numero viejo es el unico error que importa aca: un 7 desactualizado la
    # saca de la cola para siempre sin que nadie se entere.
    #
    # Es la misma poda que hace faces_referencias.py con el .npz, y por el
    # mismo motivo: de una referencia se sale.
    hay = {int(r["person_id"])
           for r in cli.select("persona_referencias", select="person_id")}
    sobran = sorted(hay - set(cuenta))
    for i in range(0, len(sobran), LOTE):
        trozo = sobran[i:i + LOTE]
        cli.delete("persona_referencias",
                   person_id="in.(%s)" % ",".join(str(x) for x in trozo))
    if sobran:
        print("borradas %d que ya no tienen ninguna referencia" % len(sobran))

    silenciadas = sum(1 for n in cuenta.values() if n >= TOPE)
    print("\n%d personas quedan fuera de la cola de confirmar por tener %d o mas"
          % (silenciadas, TOPE))


if __name__ == "__main__":
    main()
