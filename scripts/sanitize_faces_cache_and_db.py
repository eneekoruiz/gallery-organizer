"""
Sanitize faces_cache.pkl and apple_db.json:
1. Purge all 24,000+ phantom face entries with missing/zero coordinates.
2. Remove rogue 'YO' tags on F. Uxue and other single-subject family member photos.
3. Validate and re-save cleaned data with full backups.
"""

import pickle
import json
import os
import shutil
from datetime import datetime

BACKUP_DIR = os.path.join('backups', 'pre_phantom_purge')
os.makedirs(BACKUP_DIR, exist_ok=True)

def is_valid_face(f):
    if not isinstance(f, dict):
        return False
    x = f.get('x')
    y = f.get('y')
    w = f.get('width')
    h = f.get('height')
    if w is None and f.get('box') and len(f.get('box')) >= 4:
        w = f['box'][2]
        h = f['box'][3]
    try:
        w_val = float(w) if w is not None else 0
        h_val = float(h) if h is not None else 0
        return x is not None and y is not None and w_val > 0 and h_val > 0
    except (ValueError, TypeError):
        return False

def sanitize_faces_cache():
    cache_path = 'faces_cache.pkl'
    if not os.path.exists(cache_path):
        print(f"File {cache_path} not found.")
        return

    # Backup
    backup_file = os.path.join(BACKUP_DIR, f'faces_cache_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
    shutil.copy2(cache_path, backup_file)
    print(f"Backed up {cache_path} to {backup_file}")

    with open(cache_path, 'rb') as f:
        fc = pickle.load(f)

    cleaned_fc = {}
    purged_phantoms = 0
    purged_yo_in_family = 0
    retained_valid_faces = 0

    for path, faces in fc.items():
        if not isinstance(faces, list):
            continue

        norm_p = path.replace('/', '\\')
        is_family_child_album = any(fld in norm_p for fld in ['\\Familia\\F. Uxue', '\\Familia\\F. Zuriñe', '\\Familia\\F. Naroa', '\\Familia sin mí'])

        valid_faces = []
        for f in faces:
            if not is_valid_face(f):
                purged_phantoms += 1
                continue

            # Check if this face has 'YO' inside a sister / family without me folder
            if f.get('identity') == 'YO' and is_family_child_album:
                # If there is only 1 face in this photo, it's definitively the family member, not YO
                # Change identity to folder identity or Desconocido
                if 'F. Uxue' in norm_p:
                    f['identity'] = 'F. Uxue'
                elif 'F. Zuriñe' in norm_p:
                    f['identity'] = 'F. Zuriñe'
                elif 'F. Naroa' in norm_p:
                    f['identity'] = 'F. Naroa'
                else:
                    f['identity'] = 'Desconocido'
                purged_yo_in_family += 1

            valid_faces.append(f)
            retained_valid_faces += 1

        if valid_faces:
            cleaned_fc[path] = valid_faces

    # Atomic save
    temp_path = cache_path + '.tmp'
    with open(temp_path, 'wb') as f:
        pickle.dump(cleaned_fc, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp_path, cache_path)

    print(f"Cleaned {cache_path}:")
    print(f"  Purged phantom faces: {purged_phantoms}")
    print(f"  Purged rogue YO tags in family: {purged_yo_in_family}")
    print(f"  Retained valid faces: {retained_valid_faces}")
    print(f"  Total files in cleaned cache: {len(cleaned_fc)}")

def sanitize_apple_db():
    db_path = 'apple_db.json'
    if not os.path.exists(db_path):
        print(f"File {db_path} not found.")
        return

    backup_file = os.path.join(BACKUP_DIR, f'apple_db_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    shutil.copy2(db_path, backup_file)
    print(f"Backed up {db_path} to {backup_file}")

    with open(db_path, 'r', encoding='utf-8') as f:
        adb = json.load(f)

    purged_phantoms = 0
    purged_yo_in_family = 0

    for path, info in adb.items():
        if not isinstance(info, dict):
            continue
        norm_p = path.replace('/', '\\')
        is_family_child_album = any(fld in norm_p for fld in ['\\Familia\\F. Uxue', '\\Familia\\F. Zuriñe', '\\Familia\\F. Naroa', '\\Familia sin mí'])

        faces = info.get('faces', [])
        if faces:
            valid_faces = []
            for f in faces:
                if not is_valid_face(f):
                    purged_phantoms += 1
                    continue
                if f.get('identity') == 'YO' and is_family_child_album:
                    if 'F. Uxue' in norm_p:
                        f['identity'] = 'F. Uxue'
                    else:
                        f['identity'] = 'Desconocido'
                    purged_yo_in_family += 1
                valid_faces.append(f)
            info['faces'] = valid_faces

    temp_path = db_path + '.tmp'
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(adb, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, db_path)

    print(f"Cleaned {db_path}:")
    print(f"  Purged phantom faces in apple_db: {purged_phantoms}")
    print(f"  Purged rogue YO tags in apple_db: {purged_yo_in_family}")

if __name__ == '__main__':
    sanitize_faces_cache()
    sanitize_apple_db()
