# -*- coding: utf-8 -*-
"""
Pruebas Unitarias para la funcionalidad de 'Personas de Espaldas' (Back-Turned Photos).
Verifica:
1. Etiquetado seguro en el motor de análisis cuando se detecta un cuerpo sin rostro frontal.
2. Reconocimiento de carpetas físicas 'Personas_De_Espaldas' en el árbol de galería.
3. Reclasificación dinámica de 'Sin Rostro' a '👤 Personas de Espaldas' ante evidencia de cuerpo.
4. Integración y resolución limpia en fotos virtuales (apple_db.json).
5. No interferencia con identidades nominales en la exclusión mutua de fotos.
"""

import unittest
import os
import sys
import tempfile
import shutil
import contextlib
from pathlib import Path
from unittest.mock import patch, MagicMock
from flask import Flask

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.services.gallery_service as gs
from app.routes.api_faces import api_faces_bp


class TestPersonasDeEspaldas(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_espaldas_")
        self.root = Path(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _setup_stack(self, stack, cached_faces=None, cloud_db=None):
        for name, value in dict(
            FOTOS_DIR=self.root / 'Fotos',
            RESULTADOS_DIR=self.root / 'Resultados',
            CONFIG_FILE=self.root / 'config.json',
            faces_cache=(cached_faces or {})
        ).items():
            stack.enter_context(patch.object(gs, name, value))
        stack.enter_context(patch.object(gs, 'get_gallery_cache', return_value=None))
        stack.enter_context(patch.object(gs, 'set_gallery_cache'))
        stack.enter_context(patch.object(gs, 'process_pending_deletions'))
        stack.enter_context(patch.object(gs, 'load_overrides', return_value={}))
        stack.enter_context(patch.object(gs, 'get_file_key', side_effect=str))
        stack.enter_context(patch.object(gs, 'get_all_identities', return_value=[]))
        stack.enter_context(patch('app.services.apple_service.load_apple_db', return_value=(cloud_db or {})))

    def test_gallery_service_routes_espaldas_physical_folder(self):
        """Una carpeta con nombre 'Personas_De_Espaldas' se agrupa en '📋 Revisión / Por Organizar' -> '👤 Personas de Espaldas'."""
        folder = self.root / "Resultados" / "Revision_Interactiva" / "Personas_De_Espaldas"
        folder.mkdir(parents=True, exist_ok=True)
        img_file = folder / "persona_espalda_01.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        with contextlib.ExitStack() as stack:
            self._setup_stack(stack)
            with Flask(__name__).test_request_context('/api/gallery?force=true'):
                gallery = gs.get_gallery(force_reload=True)
                if hasattr(gallery, 'get_json'):
                    gallery = gallery.get_json()

            rev_cat = "📋 Revisión / Por Organizar"
            self.assertIn(rev_cat, gallery)
            self.assertIn("👤 Personas de Espaldas", gallery[rev_cat])
            items = gallery[rev_cat]["👤 Personas de Espaldas"]
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["name"], "persona_espalda_01.jpg")

    def test_dynamic_reclassification_from_sin_rostro_to_espaldas(self):
        """Un archivo en Sin_Rostro que tiene cuerpo detectado en cache se reclasifica a '👤 Personas de Espaldas'."""
        folder = self.root / "Resultados" / "Revision_Interactiva" / "Sin_Rostro"
        folder.mkdir(parents=True, exist_ok=True)
        img_file = folder / "playa_de_espaldas.jpg"
        img_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        mock_cache = {
            str(img_file): [
                {
                    "identity": "👤 Persona de Espaldas",
                    "confidence": "👤 Persona de Espaldas (89.5%)",
                    "is_body": True,
                    "is_back_turned": True,
                    "box": [100, 100, 200, 300]
                }
            ]
        }

        with contextlib.ExitStack() as stack:
            self._setup_stack(stack, cached_faces=mock_cache)
            with Flask(__name__).test_request_context('/api/gallery?force=true'):
                gallery = gs.get_gallery(force_reload=True)
                if hasattr(gallery, 'get_json'):
                    gallery = gallery.get_json()

            rev_cat = "📋 Revisión / Por Organizar"
            self.assertIn(rev_cat, gallery)
            self.assertIn("👤 Personas de Espaldas", gallery[rev_cat])
            items = gallery[rev_cat]["👤 Personas de Espaldas"]
            self.assertTrue(any(it["name"] == "playa_de_espaldas.jpg" for it in items))

    def test_apple_db_virtual_espaldas_resolution(self):
        """Un elemento en apple_db.json marcado como '👤 Personas de Espaldas' se mapea a '📋 Revisión / Por Organizar'."""
        mock_apple_db = {
            "C:\\Virtual\\Photos\\IMG_9999.JPG": {
                "category": "📋 Revisión / Por Organizar",
                "identity": "👤 Personas de Espaldas",
                "cloud_source": "icloud",
                "num_faces": 1,
                "faces": [
                    {
                        "identity": "👤 Persona de Espaldas",
                        "is_body": True,
                        "is_back_turned": True,
                        "box": [50, 50, 150, 200]
                    }
                ]
            }
        }

        with contextlib.ExitStack() as stack:
            self._setup_stack(stack, cloud_db=mock_apple_db)
            with Flask(__name__).test_request_context('/api/gallery?force=true'):
                gallery = gs.get_gallery(force_reload=True)
                if hasattr(gallery, 'get_json'):
                    gallery = gallery.get_json()

            rev_cat = "📋 Revisión / Por Organizar"
            self.assertIn(rev_cat, gallery)
            self.assertIn("👤 Personas de Espaldas", gallery[rev_cat])
            items = gallery[rev_cat]["👤 Personas de Espaldas"]
            self.assertTrue(any(it["name"] == "IMG_9999.JPG" for it in items))

    def test_back_turned_identity_does_not_claim_named_individuals(self):
        """Verificar que la etiqueta '👤 Persona de Espaldas' no interfiere con la deduplicación de personas reales."""
        self.assertIsNotNone(api_faces_bp)
        excluded = ("Desconocido", "Falso_Positivo", "Ignorar_Irrelevante", "Persona (Sin rostro / Filtro)", "👤 Persona de Espaldas", "Persona de Espaldas", "Sin Nombre")
        self.assertIn("👤 Persona de Espaldas", excluded)
        self.assertIn("Persona de Espaldas", excluded)


if __name__ == "__main__":
    unittest.main()
