# -*- coding: utf-8 -*-
"""
Saneamiento Integral de Etiquetas Faciales Obsoletas en apple_db.json.
Elimina cualquier residuo de 'Persona Nueva' en caras de fotos que ya pertenecen
a personas conocidas (Conocidos, Familia, YO) o a Desconocidos.
"""

import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
APPLE_DB_PATH = ROOT_DIR / "apple_db.json"

def sanitize_apple_db_faces():
    with open(APPLE_DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    cleaned_faces_count = 0
    cleaned_entries_count = 0

    for path, info in db.items():
        if not isinstance(info, dict):
            continue
        main_id = str(info.get("identity", "")).strip()
        cat = str(info.get("category", "")).strip()
        
        is_known = (
            cat in ("Conocidos", "Familia", "YO", "Profesores", "Mascotas")
            or main_id.startswith(("C.", "C ", "F.", "F ", "YO", "P.", "M."))
        )
        is_unknown = (
            "desconocid" in main_id.lower()
            or "desconocid" in cat.lower()
            or main_id in ("❓ Desconocidos (Por Identificar)", "👤 Desconocidos / Caras Aisladas", "📷 Fotos Sin Rostro")
        )

        faces = info.get("faces", [])
        modified = False

        for f in faces:
            fid = str(f.get("identity", ""))
            if "Persona Nueva" in fid or "persona nueva" in fid.lower() or "persona #" in fid.lower():
                if is_known:
                    # Si la foto es de una persona conocida, la cara principal es esa persona o Desconocido
                    f["identity"] = main_id
                    modified = True
                    cleaned_faces_count += 1
                elif is_unknown:
                    # Si la foto es Desconocido, la cara no debe crear un álbum de Persona Nueva
                    f["identity"] = "Desconocido"
                    modified = True
                    cleaned_faces_count += 1
                elif "(B)" in fid or "(C)" in fid or "(D)" in fid or "(E)" in fid or "(F)" in fid or "(G)" in fid or "(H)" in fid or "(I)" in fid or "(J)" in fid:
                    # Letras sintéticas huérfanas
                    f["identity"] = "Desconocido"
                    modified = True
                    cleaned_faces_count += 1

        if modified:
            cleaned_entries_count += 1

    with open(APPLE_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"✓ Saneamiento completado: {cleaned_faces_count} caras limpiadas en {cleaned_entries_count} entradas de apple_db.json.")

if __name__ == "__main__":
    sanitize_apple_db_faces()
