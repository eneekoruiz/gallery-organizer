import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "config.json"

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    cfg = json.load(f)

local_path = Path(cfg["local_path"])
resultados_dir = local_path / "Resultados"
espaldas_dir = resultados_dir / "Revision_Interactiva" / "Personas_De_Espaldas"
espaldas_dir.mkdir(parents=True, exist_ok=True)

apple_db_path = ROOT / "apple_db.json"
with open(apple_db_path, "r", encoding="utf-8") as f:
    apple_db = json.load(f)

moved = 0
new_apple_db = {}
for p_str, info in apple_db.items():
    ident = str(info.get("identity", ""))
    if "espalda" in ident.lower():
        src_path = Path(p_str)
        if src_path.exists():
            dst_path = espaldas_dir / src_path.name
            if src_path.resolve() != dst_path.resolve():
                shutil.move(str(src_path), str(dst_path))
                moved += 1
                print(f"Moved {src_path.name} -> {dst_path}")
            new_apple_db[str(dst_path)] = info
            info["path"] = str(dst_path)
            info["category"] = "📋 Revisión / Por Organizar"
            info["identity"] = "👤 Personas de Espaldas"
            continue
    new_apple_db[p_str] = info

with open(apple_db_path, "w", encoding="utf-8") as f:
    json.dump(new_apple_db, f, indent=2, ensure_ascii=False)

print(f"Total moved to Personas_De_Espaldas: {moved}")
