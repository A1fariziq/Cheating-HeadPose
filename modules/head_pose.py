import cv2
import numpy as np


# ==================================================
# 1. LANDMARK WAJAH YANG DIGUNAKAN
# ==================================================
# Hanya titik yang relatif kaku (dahi, alis, mata, hidung, pipi).
# Titik mulut dan dagu sengaja TIDAK dipakai karena ikut bergerak
# saat rahang/ekspresi berubah dan merusak estimasi rotasi.

RIGID_LANDMARK_IDS = [
    10, 109, 338, 67, 297,          # dahi
    9, 8, 168, 6, 197, 195, 5, 4, 1,  # garis tengah hidung (1 = ujung hidung)
    98, 327, 129, 358,              # sisi hidung
    33, 133, 263, 362,              # sudut mata
    70, 105, 107, 336, 334, 300,    # alis
    234, 454, 127, 356, 162, 389,   # pipi / pelipis
    21, 251                         # pelipis atas
]

NOSE_TIP_INDEX = RIGID_LANDMARK_IDS.index(1)


# ==================================================
# 2. MATRIKS KAMERA
# ==================================================

def get_camera_matrix(frame_width, frame_height):

    focal_length = frame_width

    return np.array([
        [focal_length, 0, frame_width / 2],
        [0, focal_length, frame_height / 2],
        [0, 0, 1]
    ], dtype=np.float64)


# ==================================================
# 3. KONVERSI MATRIKS ROTASI KE EULER
# ==================================================
# Sumbu mengikuti kamera OpenCV pada posisi netral:
#   X = kanan layar, Y = bawah, Z = menjauhi kamera
#
# Konvensi: R = Rz(roll) Ry(yaw) Rx(pitch)
#
# Tanda sudut (pada tampilan yang sudah di-mirror):
#   pitch positif = kepala menunduk
#   yaw   positif = kepala menoleh ke kanan layar
#   roll  positif = kepala miring ke kanan layar

def rotation_matrix_to_euler(rotation_matrix):

    sy = np.sqrt(
        rotation_matrix[0, 0] ** 2 +
        rotation_matrix[1, 0] ** 2
    )

    singular = sy < 1e-6

    if not singular:

        pitch = np.arctan2(
            rotation_matrix[2, 1],
            rotation_matrix[2, 2]
        )

        yaw = np.arctan2(
            -rotation_matrix[2, 0],
            sy
        )

        roll = np.arctan2(
            rotation_matrix[1, 0],
            rotation_matrix[0, 0]
        )

    else:

        pitch = np.arctan2(
            -rotation_matrix[1, 2],
            rotation_matrix[1, 1]
        )

        yaw = np.arctan2(
            -rotation_matrix[2, 0],
            sy
        )

        roll = 0.0

    return {
        "pitch": float(np.degrees(pitch)),
        # Tanda yaw dibalik agar positif = menoleh ke kanan layar
        "yaw": float(-np.degrees(yaw)),
        "roll": float(np.degrees(roll))
    }


# ==================================================
# 4. AMBIL TITIK LANDMARK (2D DAN 3D)
# ==================================================
# 2D : koordinat piksel (x, y)
# 3D : (x, y, z) dalam skala piksel. z MediaPipe berskala sama dengan x,
#      sehingga dikali lebar frame. Sumbunya sudah sejajar dengan
#      sumbu kamera OpenCV (Y ke bawah, Z menjauhi kamera).

def extract_landmark_points(landmarks, frame_width, frame_height):

    count = len(RIGID_LANDMARK_IDS)

    points_2d = np.empty((count, 2), dtype=np.float64)
    points_3d = np.empty((count, 3), dtype=np.float64)

    for i, index in enumerate(RIGID_LANDMARK_IDS):

        landmark = landmarks[index]

        x = landmark.x * frame_width
        y = landmark.y * frame_height
        z = landmark.z * frame_width

        points_2d[i] = (x, y)
        points_3d[i] = (x, y, z)

    return points_2d, points_3d


# ==================================================
# 5. SOLVE PNP: EPNP (AWAL) + ITERATIVE (PENYEMPURNAAN)
# ==================================================

def solve_pose(model_points, image_points, camera_matrix, distortion):

    model_points = np.ascontiguousarray(model_points, dtype=np.float64)
    image_points = np.ascontiguousarray(image_points, dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        return None

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion,
        rotation_vector,
        translation_vector,
        True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    return rotation_vector, translation_vector, rotation_matrix


# ==================================================
# 6. RATA-RATA MATRIKS ROTASI UNTUK KALIBRASI
# ==================================================

def average_rotation_matrices(rotation_matrices):

    average_matrix = np.mean(
        rotation_matrices,
        axis=0
    )

    u, _, vt = np.linalg.svd(average_matrix)

    rotation_reference = u @ vt

    if np.linalg.det(rotation_reference) < 0:

        u[:, -1] *= -1

        rotation_reference = u @ vt

    return rotation_reference


# ==================================================
# 7. KALIBRASI: MEMBANGUN MODEL WAJAH PERSONAL
# ==================================================
# Model 3D dibuat dari rata-rata landmark selama kalibrasi (wajah netral),
# bukan dari 6 titik generik. Dengan begitu proporsi kedalaman model
# cocok dengan wajah dan sistem koordinat MediaPipe, sehingga pitch,
# yaw, dan roll tidak saling bocor.

def build_calibration(calibration_samples, frame_width, frame_height):

    if not calibration_samples:
        return None, None

    mean_points_3d = np.mean(
        [sample[1] for sample in calibration_samples],
        axis=0
    )

    # Pusatkan model ke titik tengahnya
    centered_points = mean_points_3d - mean_points_3d.mean(axis=0)

    face_model = {
        "points": centered_points,
        "nose_tip": centered_points[NOSE_TIP_INDEX].copy()
    }

    camera_matrix = get_camera_matrix(frame_width, frame_height)
    distortion = np.zeros((4, 1), dtype=np.float64)

    rotations = []

    for points_2d, _ in calibration_samples:

        result = solve_pose(
            centered_points,
            points_2d,
            camera_matrix,
            distortion
        )

        if result is not None:
            rotations.append(result[2])

    if not rotations:
        return None, None

    reference_rotation = average_rotation_matrices(rotations)

    return face_model, reference_rotation


# ==================================================
# 8. ESTIMASI ORIENTASI KEPALA
# ==================================================

def estimate_head_pose(landmarks, frame_width, frame_height, face_model):

    if face_model is None:
        return None

    points_2d, _ = extract_landmark_points(
        landmarks,
        frame_width,
        frame_height
    )

    camera_matrix = get_camera_matrix(frame_width, frame_height)

    distortion_coefficients = np.zeros(
        (4, 1),
        dtype=np.float64
    )

    result = solve_pose(
        face_model["points"],
        points_2d,
        camera_matrix,
        distortion_coefficients
    )

    if result is None:
        return None

    rotation_vector, translation_vector, rotation_matrix = result

    angles = rotation_matrix_to_euler(rotation_matrix)

    return {
        "yaw": angles["yaw"],
        "pitch": angles["pitch"],
        "roll": angles["roll"],
        "rotation_matrix": rotation_matrix,
        "rotation_vector": rotation_vector,
        "translation_vector": translation_vector,
        "camera_matrix": camera_matrix,
        "distortion_coefficients": distortion_coefficients,
        "origin": face_model["nose_tip"]
    }


# ==================================================
# 9. MENGHITUNG ORIENTASI RELATIF
# ==================================================

def calculate_relative_pose(current_rotation, reference_rotation):

    # Membandingkan orientasi saat ini dengan orientasi netral

    relative_rotation = (
        reference_rotation.T @ current_rotation
    )

    angles = rotation_matrix_to_euler(relative_rotation)

    return {
        "yaw": angles["yaw"],
        "pitch": angles["pitch"],
        "roll": angles["roll"],
        "rotation_matrix": relative_rotation
    }


# ==================================================
# 10. VISUALISASI SUMBU ORIENTASI KEPALA
# ==================================================

def draw_head_pose_axis(frame, pose, axis_length=100):

    if pose is None:
        return frame

    origin = pose["origin"]

    # Sumbu Z digambar ke arah kamera (keluar dari wajah) = -Z model
    axis_points = np.array([
        origin,
        origin + (axis_length, 0, 0),
        origin + (0, axis_length, 0),
        origin + (0, 0, -axis_length)
    ], dtype=np.float64)

    projected_points, _ = cv2.projectPoints(
        axis_points,
        pose["rotation_vector"],
        pose["translation_vector"],
        pose["camera_matrix"],
        pose["distortion_coefficients"]
    )

    points_2d = projected_points.reshape(-1, 2)

    nose, x_end, y_end, z_end = [
        (int(p[0]), int(p[1])) for p in points_2d
    ]

    # Sumbu X - Merah
    cv2.line(frame, nose, x_end, (0, 0, 255), 3)

    # Sumbu Y - Hijau
    cv2.line(frame, nose, y_end, (0, 255, 0), 3)

    # Sumbu Z - Biru
    cv2.line(frame, nose, z_end, (255, 0, 0), 3)

    return frame
