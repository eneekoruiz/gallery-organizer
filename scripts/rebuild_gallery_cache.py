#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/rebuild_gallery_cache.py
Rebuilds gallery_cache.json cleanly with get_gallery(force_reload=True).
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import create_app
from app.services.gallery_service import get_gallery
import json
import time

app = create_app()
with app.app_context():
    print("Rebuilding gallery cache cleanly...")
    t0 = time.time()
    gallery = get_gallery(force_reload=True)
    t1 = time.time()
    print(f"Gallery built in {t1 - t0:.2f} seconds!")
    
    with open(BASE_DIR / 'gallery_cache.json', 'r', encoding='utf-8') as f:
        gal_data = json.load(f)

    # Print key statistics
    por_org = gal_data.get("📋 Revisión / Por Organizar", {})
    print("\n--- 📋 Revisión / Por Organizar ---")
    for k, v in list(por_org.items())[:10]:
        print(f"  {k}: {len(v)} elements")
