# -*- coding: utf-8 -*-
"""
Fase 1 — Convierte los chunks NDJSON del indexador en filas de `photos`.

Los chunks vienen envueltos en el JSON del conector de Drive
({content: base64, ...}), así que este script acepta tanto ese formato como
NDJSON crudo.

La fecha de cada foto se hereda de la carpeta, NUNCA de modifiedTime: Drive
guarda ahí la fecha de subida, y hay material de 2014 subido en 2019.

Uso:
    python parse_photos.py <dir_con_chunks> --folders ../data/folders.json --out ../data
"""
import argparse
import base64
import glob
import json
import os
import re
import sys
from collections import Counter
from datetime import date, datetime


EXIF_RE = re.compile(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}:\d{2}:\d{2})")


def exif_iso(value):
    """'2012:08:21 10:10:57' -> '2012-08-21 10:10:57'.

    Descarta relojes de camara sin poner en hora, que dan fechas imposibles.

    Y tambien HORAS imposibles. Esto miraba solo el dia y dejaba pasar la hora
    tal cual, asi que "2020:11:10 12:19:72" salia entero y lo rechazaba
    Postgres al escribir:

        22008  date/time field value out of range: "2020-11-10 12:19:72"

    Setenta y dos segundos. Son camaras que escriben mal el EXIF, y hay pocas,
    pero el 16/9/2026 seis fuentes enteras se cayeron sin guardar nada -unas
    300 fotos- porque el lote va en un solo POST y una fila rota lo tira todo.

    No se arregla el segundo poniendole 59: eso seria inventar una hora. La
    foto entra igual, sin fecha de EXIF, como cualquier otra que no la trae.
    """
    m = EXIF_RE.match(str(value or ""))
    if not m:
        return None
    y = int(m.group(1))
    if y < 1990 or y > 2100:
        return None
    try:
        h, mi, sec = (int(x) for x in m.group(4).split(":"))
        datetime(y, int(m.group(2)), int(m.group(3)), h, mi, sec)
    except ValueError:
        return None
    return "%s-%s-%s %s" % (m.group(1), m.group(2), m.group(3), m.group(4))


def read_chunk(path):
    """Devuelve la lista de registros de un chunk, venga envuelto o crudo."""
    with open(path, "rb") as fh:
        head = fh.read(1)
        fh.seek(0)
        raw = fh.read()

    text = None
    if head == b"{":
        try:
            obj = json.loads(raw.decode("utf-8"))
            if isinstance(obj, dict) and "content" in obj:
                try:
                    text = base64.b64decode(obj["content"]).decode("utf-8")
                except Exception:
                    text = obj["content"]
        except json.JSONDecodeError:
            pass

    if text is None:
        text = raw.decode("utf-8", "replace")

    out = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks_dir")
    ap.add_argument("--folders", required=True)
    ap.add_argument("--out", default="../data")
    ap.add_argument("--pattern", default="*.txt")
    a = ap.parse_args()

    with open(a.folders, encoding="utf-8") as fh:
        folders = json.load(fh)

    # fecha y ruido por carpeta, para heredarlos a cada foto
    meta = {}
    for f in folders:
        y = f.get("year")
        meta[f["folder_id"]] = {
            "year": int(y) if y not in ("", None) else None,
            "date": (f["event_date"] if f.get("event_date") else None),
            "noise": f.get("noise") or None,
        }

    # event_date no viene en folders.json del parser: se recompone acá
    for f in folders:
        m = meta[f["folder_id"]]
        if f.get("date_precision") == "day" and f.get("year") and f.get("month") and f.get("day"):
            try:
                d = date(int(f["year"]), int(f["month"]), int(f["day"]))
                m["date"] = d.isoformat()
            except ValueError:
                m["date"] = None
        elif f.get("date_precision") == "month" and f.get("year") and f.get("month"):
            m["date"] = "%04d-%02d-01" % (int(f["year"]), int(f["month"]))

    files = sorted(glob.glob(os.path.join(a.chunks_dir, a.pattern)))
    print("Chunks encontrados: %d" % len(files))

    rows, stats = [], Counter()
    seen = set()
    for path in files:
        for r in read_chunk(path):
            fid = r.get("parent")
            if not fid:
                stats["sin_parent"] += 1
                continue
            if r["id"] in seen:
                stats["duplicados"] += 1
                continue
            seen.add(r["id"])

            m = meta.get(fid)
            if m is None:
                stats["carpeta_desconocida"] += 1
                m = {"year": None, "date": None, "noise": None}
            if m["noise"]:
                stats["en_carpeta_basura"] += 1
                continue

            rows.append({
                "id": r["id"],
                "folder_id": fid,
                "name": r.get("name") or "",
                "mime_type": r.get("mimeType"),
                "web_view_link": r.get("webViewLink"),
                "photo_date": m["date"],
                "year": m["year"],
                "drive_modified_time": r.get("modifiedTime"),
                # EXIF: la fecha real de la toma. Viene como "2012:08:21
                # 10:10:57" y hay que pasarla a ISO. Es mejor fuente que el
                # nombre de la carpeta, que a su vez es mejor que
                # modifiedTime, que es la fecha de subida a Drive.
                "exif_time": exif_iso(r.get("exifTime")),
                "width": r.get("width"),
                "height": r.get("height"),
                "size_bytes": int(r["size"]) if str(r.get("size") or "").isdigit() else None,
            })
            stats[r.get("mimeType") or "?"] += 1

    os.makedirs(a.out, exist_ok=True)
    dest = os.path.join(a.out, "photos.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, separators=(",", ":"))

    print("\nFotos listas: %d" % len(rows))
    for k, v in stats.most_common():
        print("   %-24s %7d" % (k, v))
    con_exif = sum(1 for r in rows if r.get("exif_time"))
    print("\n   con EXIF (fecha real de toma): %7d  (%.1f%%)"
          % (con_exif, 100 * con_exif / max(len(rows), 1)))
    con_fecha = sum(1 for r in rows if r["photo_date"])
    print("\n   con fecha heredada:      %7d  (%.1f%%)"
          % (con_fecha, 100 * con_fecha / max(len(rows), 1)))
    print("\nEscrito: %s" % dest)


if __name__ == "__main__":
    main()
