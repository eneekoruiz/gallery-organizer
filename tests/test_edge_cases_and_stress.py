# -*- coding: utf-8 -*-
"""
test_edge_cases_and_stress.py - Suite de pruebas de casos borde y estrés para el motor facial.
Verifica:
1. Fotos sin caras (0 rostros) retornan estructura limpia y no rompen el pipeline.
2. Multitud con micro-caras diminutas en el fondo (< 20px) son descartadas por FIQA.
3. Formatos especiales (WebP, PNG transparente RGBA, escala de grises) decodifican a 3 canales BGR.
4. Archivos corruptos, truncados o vacíos (0 bytes) no cuelgan workers ni lanzan 500 no capturados.
5. Cola de tareas (task_service) nunca se queda colgada en 'running' infinito ante excepciones.
"""
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
from io import BytesIO
from PIL import Image

from app.utils.images import decode_image
from app.services.biometric_engine import estimate_face_quality_and_pose
from app.services.task_service import submit_task, get_task
import time


class TestEdgeCasesAndStress(unittest.TestCase):

    def test_corrupt_and_empty_images(self):
        """Archivos vacíos, corruptos o nulos retornan None de forma segura sin excepciones."""
        self.assertIsNone(decode_image(None))
        self.assertIsNone(decode_image(b""))
        self.assertIsNone(decode_image(b"\x00" * 128))
        self.assertIsNone(decode_image(b"CORRUPTED_NON_IMAGE_HEADER_TEST_BYTES"))

    def test_special_formats_decoding(self):
        """PNG transparente RGBA, WebP y escala de grises decodifican a BGR 3 canales."""
        # 1. RGBA PNG
        rgba = Image.new("RGBA", (80, 80), color=(10, 132, 255, 128))
        buf_rgba = BytesIO()
        rgba.save(buf_rgba, format="PNG")
        decoded_rgba = decode_image(buf_rgba.getvalue())
        self.assertIsNotNone(decoded_rgba)
        self.assertEqual(decoded_rgba.shape, (80, 80, 3))

        # 2. WebP
        buf_webp = BytesIO()
        rgba.convert("RGB").save(buf_webp, format="WEBP")
        decoded_webp = decode_image(buf_webp.getvalue())
        self.assertIsNotNone(decoded_webp)
        self.assertEqual(decoded_webp.shape, (80, 80, 3))

        # 3. Grayscale (1-channel)
        gray = Image.new("L", (60, 60), color=200)
        buf_gray = BytesIO()
        gray.save(buf_gray, format="PNG")
        decoded_gray = decode_image(buf_gray.getvalue())
        self.assertIsNotNone(decoded_gray)
        self.assertEqual(decoded_gray.shape, (60, 60, 3))

    def test_micro_faces_fiqa_rejection(self):
        """20 caras diminutas al fondo (< 20px IOD o < 16px) son descartadas por FIQA para clustering."""
        img = np.zeros((800, 1000, 3), dtype=np.uint8)
        # Simular 20 micro-rostros lejanos
        for i in range(20):
            x = 50 + (i * 45)
            y = 100
            w, h = 14, 15  # Caras minúsculas en el fondo
            face_vec = np.array([
                x, y, w, h,
                x + 3, y + 4,   # ojo izq
                x + 10, y + 4,  # ojo der
                x + 7, y + 8,   # nariz
                x + 4, y + 12,  # boca izq
                x + 9, y + 12,  # boca der
                0.7             # score
            ], dtype=np.float32)
            quality = estimate_face_quality_and_pose(img, face_vec)
            self.assertFalse(quality["is_usable_for_clustering"], f"Micro-cara #{i} debió ser rechazada por FIQA")
            self.assertEqual(quality["quality_score"], 0.0)

    def test_zero_faces_blank_scene(self):
        """Una escena sin caras (ej. paisaje blanco o negro) no genera excepciones."""
        blank_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
        dummy_face = np.zeros(15, dtype=np.float32)
        quality = estimate_face_quality_and_pose(blank_img, dummy_face)
        self.assertFalse(quality["is_usable_for_clustering"])
        self.assertEqual(quality["quality_score"], 0.0)

    def test_task_queue_never_hangs_on_error(self):
        """Tareas en background con fallos o excepciones transicionan limpiamente a 'error' y no quedan colgadas."""
        def failing_task(task_inst):
            time.sleep(0.05)
            raise RuntimeError("Simulated crash in background worker")

        task_id = submit_task("failing_worker_test", failing_task, retry=False)
        self.assertIsNotNone(task_id)

        # Esperar a que el worker finalice
        max_wait = 3.0
        start = time.time()
        final_status = None
        while time.time() - start < max_wait:
            t = get_task(task_id)
            if t and t.status in ("error", "done"):
                final_status = t.status
                break
            time.sleep(0.05)

        self.assertEqual(final_status, "error")
        task_obj = get_task(task_id)
        self.assertIn("Simulated crash", task_obj.error)
        self.assertIsNotNone(task_obj.finished_at)


if __name__ == "__main__":
    unittest.main()
