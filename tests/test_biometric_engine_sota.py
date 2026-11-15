r"""Tests unitarios e integración para el Motor Biométrico Forense SOTA y Tabula Rasa."""

import os
import sys
import unittest
import numpy as np
import cv2
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.biometric_engine import (
    estimate_face_quality_and_pose,
    extract_features_with_tta,
    extract_feature_from_aligned_tta
)
from app.utils.clustering import (
    cluster_faces_upcac,
    cluster_faces_consensus_upcac,
    audit_and_split_contaminated_clusters
)


class BiometricEngineSOTATests(unittest.TestCase):
    def setUp(self):
        # Generar imagen sintética de prueba con rostro de alta nitidez
        self.img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(self.img, (200, 200), 60, (220, 220, 220), -1)
        # Ojos
        cv2.circle(self.img, (180, 185), 8, (40, 40, 40), -1)
        cv2.circle(self.img, (220, 185), 8, (40, 40, 40), -1)
        # Nariz
        cv2.circle(self.img, (200, 205), 4, (60, 60, 60), -1)
        # Boca
        cv2.line(self.img, (185, 230), (215, 230), (30, 30, 30), 3)

        # Landmarks sintéticos coherentes
        self.frontal_face = np.array([
            140, 140, 120, 120,   # x, y, w, h
            180, 185,             # Right eye
            220, 185,             # Left eye
            200, 205,             # Nose tip
            185, 230,             # Right mouth
            215, 230,             # Left mouth
            0.98                  # Confidence score
        ], dtype=np.float32)

    def test_fiqa_quality_assessment_frontal_face(self):
        res = estimate_face_quality_and_pose(self.img, self.frontal_face)
        self.assertIn("quality_score", res)
        self.assertIn("sharpness", res)
        self.assertIn("yaw", res)
        self.assertIn("pitch", res)
        self.assertIn("roll", res)
        self.assertIn("iod", res)
        self.assertGreaterEqual(res["quality_score"], 0.0)
        self.assertLessEqual(res["quality_score"], 1.0)
        self.assertEqual(res["iod"], 40.0)

    def test_fiqa_rejects_micro_faces(self):
        micro_face = np.array([
            10, 10, 8, 8,
            12, 12, 16, 12, 14, 14, 12, 16, 16, 16, 0.90
        ], dtype=np.float32)
        res = estimate_face_quality_and_pose(self.img, micro_face)
        self.assertFalse(res["is_usable_for_clustering"])
        self.assertIn("minúscula", res["reason"].lower())

    def test_extract_feature_from_aligned_tta_unit_norm(self):
        rec_path = ROOT_DIR / "models" / "face_recognition_sface.onnx"
        if not rec_path.exists():
            self.skipTest("Modelo SFace ONNX no presente para test de inferencia")

        rec = cv2.FaceRecognizerSF.create(str(rec_path), "")
        aligned_crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        feat = extract_feature_from_aligned_tta(aligned_crop, rec)
        self.assertEqual(feat.shape, (128,))
        norm = np.linalg.norm(feat)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_audit_and_split_bimodal_cluster(self):
        # Dos identidades sintéticas con 5 puntos cada una
        # Núcleo A centrado en eje X, Núcleo B centrado en eje Y
        np.random.seed(42)
        cluster_a = np.tile([1.0, 0.0, 0.0], (5, 1)) + np.random.normal(0, 0.04, (5, 3))
        cluster_b = np.tile([0.0, 1.0, 0.0], (5, 1)) + np.random.normal(0, 0.04, (5, 3))
        embs = np.vstack([cluster_a, cluster_b])
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

        # Inicialmente asignados erróneamente al mismo clúster (label 0)
        labels = np.zeros(10, dtype=int)

        # Auditoría espectral debe detectar la bimodalidad y escindirlo
        audited = audit_and_split_contaminated_clusters(embs, labels, min_cluster_size=6, fiedler_threshold=0.15)
        self.assertEqual(len(np.unique(audited)), 2, "El clúster bimodal debió ser dividido en 2 grupos puros")
        # Verificar que la partición es coherente: primeros 5 en un grupo, últimos 5 en otro
        self.assertEqual(len(np.unique(audited[:5])), 1)
        self.assertEqual(len(np.unique(audited[5:])), 1)
        self.assertNotEqual(audited[0], audited[5])

    def test_cannot_link_prevents_same_photo_clustering(self):
        # Dos caras idénticas que provienen de la misma foto
        embs = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], dtype=np.float32)
        photos = ["same_photo.jpg", "same_photo.jpg"]
        labels = cluster_faces_consensus_upcac(embs, photo_ids=photos)
        self.assertNotEqual(labels[0], labels[1], "Caras de la misma foto jamás pueden fusionarse")


if __name__ == "__main__":
    unittest.main(verbosity=2)
