r"""Motor Biométrico Forense SOTA para Detección, FIQA, TTA y Extracción Facial.
Soporta arquitecturas de alta precisión, evaluación de calidad y normalización esférica.
"""

import math
import cv2
import numpy as np
from pathlib import Path

# Modelo 3D canónico simplificado (puntos antropométricos faciales en mm en coordenadas de cámara OpenCV: X der, Y abajo, Z adelante)
# Ojo Der, Ojo Izq, Punta Nariz, Comisura Boca Der, Comisura Boca Izq
CANONICAL_FACE_3D = np.array([
    [-34.0, -32.0, -18.0],  # Right eye (subject's right, x < 0, y above nose)
    [ 34.0, -32.0, -18.0],  # Left eye (subject's left, x > 0, y above nose)
    [  0.0,   0.0,   0.0],  # Nose tip
    [-24.0,  38.0, -14.0],  # Right mouth corner (y below nose)
    [ 24.0,  38.0, -14.0]   # Left mouth corner (y below nose)
], dtype=np.float64)


def estimate_face_quality_and_pose(image: np.ndarray, face_data: np.ndarray) -> dict:
    r"""Evalúa la calidad geométrica, nitidez (FIQA heurístico) y ángulos de pose 3D (Yaw/Pitch/Roll).

    Args:
        image: Imagen BGR completa.
        face_data: Vector de 14 o 15 elementos de YuNet/SCRFD:
                   [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rm, y_rm, x_lm, y_lm, (score)]

    Returns:
        dict con metricas:
            - quality_score: float [0.0, 1.0]
            - sharpness: float
            - is_usable_for_clustering: bool
            - yaw, pitch, roll: float en grados
            - iod: float (interocular distance en px)
            - reason: str (si se rechaza)
    """
    x, y, w, h = [int(v) for v in face_data[:4]]
    img_h, img_w = image.shape[:2]

    # Validar coordenadas de recorte
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w), min(img_h, y + h)

    if (x2 - x1) < 16 or (y2 - y1) < 16:
        return {
            "quality_score": 0.0,
            "sharpness": 0.0,
            "is_usable_for_clustering": False,
            "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
            "iod": 0.0,
            "reason": "Resolución minúscula (< 16px)"
        }

    face_crop = image[y1:y2, x1:x2]

    # 1. Nitidez (Varianza del Laplaciano en canal de luminancia)
    gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    sharpness_norm = min(1.0, laplacian_var / 350.0)

    # 2. Distancia interocular (IOD)
    x_re, y_re = float(face_data[4]), float(face_data[5])
    x_le, y_le = float(face_data[6]), float(face_data[7])
    iod = float(math.sqrt((x_re - x_le)**2 + (y_re - y_le)**2))

    # 3. Estimación de Pose 3D mediante solvePnP con modelo antropométrico canónico
    landmarks_2d = np.array([
        [x_re, y_re],
        [x_le, y_le],
        [float(face_data[8]), float(face_data[9])],   # Nose
        [float(face_data[10]), float(face_data[11])], # Right mouth
        [float(face_data[12]), float(face_data[13])]  # Left mouth
    ], dtype=np.float64)

    # Matriz intrínseca sintética de cámara centrada
    focal_length = img_w
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    yaw, pitch, roll = 0.0, 0.0, 0.0
    try:
        success, rvec, tvec = cv2.solvePnP(
            CANONICAL_FACE_3D,
            landmarks_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )
        if success:
            rmat, _ = cv2.Rodrigues(rvec)
            # Descomposición de Euler Tait-Bryan (Z-Y-X)
            sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
            if sy > 1e-6:
                pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
                yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
                roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
            else:
                pitch = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
                yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
                roll = 0.0
    except Exception:
        pass

    # 4. Evaluación de Calidad Combinada (FIQA)
    # Penalizaciones por pose extrema
    pose_penalty = 1.0
    if abs(yaw) > 35.0:
        pose_penalty *= max(0.0, 1.0 - (abs(yaw) - 35.0) / 25.0)
    if abs(pitch) > 25.0:
        pose_penalty *= max(0.0, 1.0 - (abs(pitch) - 25.0) / 25.0)

    # Penalización por tamaño de rostro
    size_factor = min(1.0, max(0.0, (iod - 18.0) / 30.0))

    # Detector confidence score (si está presente como elemento 14)
    det_score = float(face_data[14]) if len(face_data) > 14 else 0.90

    # Score final en [0.0, 1.0]
    quality_score = float(np.clip(
        (0.35 * sharpness_norm + 0.35 * size_factor + 0.30 * det_score) * pose_penalty,
        0.0, 1.0
    ))

    # Criterio estricto para clustering de alta precisión
    is_usable = (
        quality_score >= 0.45 and
        iod >= 20.0 and
        abs(yaw) <= 45.0 and
        abs(pitch) <= 35.0 and
        laplacian_var >= 20.0
    )

    reason = "Aceptable"
    if not is_usable:
        reasons = []
        if quality_score < 0.45: reasons.append(f"FIQA insuficiente ({quality_score:.2f})")
        if iod < 20.0: reasons.append(f"IOD bajo ({iod:.1f}px)")
        if abs(yaw) > 45.0: reasons.append(f"Yaw extremo ({yaw:.1f}°)")
        if abs(pitch) > 35.0: reasons.append(f"Pitch extremo ({pitch:.1f}°)")
        if laplacian_var < 20.0: reasons.append(f"Desenfoque severo (var={laplacian_var:.1f})")
        reason = " | ".join(reasons)

    return {
        "quality_score": quality_score,
        "sharpness": laplacian_var,
        "is_usable_for_clustering": is_usable,
        "yaw": round(yaw, 2),
        "pitch": round(pitch, 2),
        "roll": round(roll, 2),
        "iod": round(iod, 2),
        "reason": reason
    }


def extract_feature_from_aligned_tta(aligned: np.ndarray, recognizer) -> np.ndarray:
    r"""Extrae embedding con TTA (Test-Time Augmentation) sobre un recorte facial ya alineado.
    Promedia el vector de la imagen canónica con el de su reflexión especular horizontal."""
    feat_arr = recognizer.feature(aligned)
    if feat_arr is None or len(feat_arr) == 0:
        return np.zeros(128, dtype=np.float32)
    feat_orig = np.asarray(feat_arr[0], dtype=np.float32).flatten()
    norm_orig = np.linalg.norm(feat_orig)
    if norm_orig > 1e-7:
        feat_orig = feat_orig / norm_orig

    aligned_flip = cv2.flip(aligned, 1)
    feat_flip_arr = recognizer.feature(aligned_flip)
    if feat_flip_arr is not None and len(feat_flip_arr) > 0:
        feat_flip = np.asarray(feat_flip_arr[0], dtype=np.float32).flatten()
        norm_flip = np.linalg.norm(feat_flip)
        if norm_flip > 1e-7:
            feat_flip = feat_flip / norm_flip
        fused = feat_orig + feat_flip
        norm_fused = np.linalg.norm(fused)
        if norm_fused > 1e-7:
            return (fused / norm_fused).astype(np.float32)
    return feat_orig.astype(np.float32)


def extract_features_with_tta(image: np.ndarray, face_data: np.ndarray, recognizer) -> np.ndarray:
    r"""Extrae embeddings utilizando Test-Time Augmentation (TTA) sobre la esfera unidad.
    Alinea la cara original y su reflejo horizontal especular, promediando y normalizando L2.

    Args:
        image: Imagen BGR.
        face_data: Vector de caja y landmarks de la cara detectada.
        recognizer: Instancia cv2.FaceRecognizerSF o extractor compatible.

    Returns:
        np.ndarray vector normalizado unitario L2 en S^{D-1}.
    """
    # 1. Extracción cara canónica original
    aligned_orig = recognizer.alignCrop(image, face_data)
    feat_orig = recognizer.feature(aligned_orig)[0]
    feat_orig = np.asarray(feat_orig, dtype=np.float32).flatten()
    norm_orig = np.linalg.norm(feat_orig)
    if norm_orig > 1e-7:
        feat_orig = feat_orig / norm_orig

    # 2. Test-Time Augmentation (Reflexión Horizontal Especular)
    aligned_flip = cv2.flip(aligned_orig, 1)
    # Re-alinear o extraer sobre el recorte volteado
    # Para SFace, pasar un recorte ya alineado a .feature funciona directamente
    feat_flip = recognizer.feature(aligned_flip)[0]
    feat_flip = np.asarray(feat_flip, dtype=np.float32).flatten()
    norm_flip = np.linalg.norm(feat_flip)
    if norm_flip > 1e-7:
        feat_flip = feat_flip / norm_flip

    # 3. Fusión esférica simétrica
    fused = feat_orig + feat_flip
    norm_fused = np.linalg.norm(fused)
    if norm_fused > 1e-7:
        return (fused / norm_fused).astype(np.float32)
    return feat_orig.astype(np.float32)
