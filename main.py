import cv2
import mediapipe as mp

from modules.head_pose import (
    extract_landmark_points,
    build_calibration,
    estimate_head_pose,
    draw_head_pose_axis,
    calculate_relative_pose
)


def main():

    # ==================================================
    # 1. INISIALISASI MEDIAPIPE
    # ==================================================

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ==================================================
    # 2. INISIALISASI WEBCAM
    # ==================================================

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Webcam tidak dapat diakses!")
        face_mesh.close()
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam berhasil diaktifkan!")
    print("Tekan C untuk kalibrasi.")
    print("Tekan R untuk mengulang kalibrasi.")
    print("Tekan Q untuk keluar.")

    # ==================================================
    # 3. VARIABEL KALIBRASI
    # ==================================================

    face_model = None
    reference_rotation = None

    calibration_samples = []

    calibration_frames = 30

    calibrating = False

    # ==================================================
    # 4. PROSES UTAMA
    # ==================================================

    while True:

        success, frame = camera.read()

        if not success:
            print("Gagal membaca frame webcam.")
            break

        frame = cv2.flip(frame, 1)

        frame_height, frame_width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                landmarks = face_landmarks.landmark

                # Menggambar Face Mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=1,
                        circle_radius=1
                    )
                )

                # ======================================
                # PROSES KALIBRASI
                # ======================================

                if calibrating:

                    calibration_samples.append(
                        extract_landmark_points(
                            landmarks,
                            frame_width,
                            frame_height
                        )
                    )

                    sample_count = len(calibration_samples)

                    cv2.putText(
                        frame,
                        f"CALIBRATING: {sample_count}/{calibration_frames}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                    if sample_count >= calibration_frames:

                        face_model, reference_rotation = build_calibration(
                            calibration_samples,
                            frame_width,
                            frame_height
                        )

                        calibrating = False
                        calibration_samples.clear()

                        if face_model is None:
                            print("KALIBRASI GAGAL. Tekan C untuk mengulang.")
                        else:
                            print("KALIBRASI BERHASIL!")

                # ======================================
                # MENAMPILKAN HASIL SETELAH KALIBRASI
                # ======================================

                elif face_model is not None:

                    pose = estimate_head_pose(
                        landmarks,
                        frame_width,
                        frame_height,
                        face_model
                    )

                    if pose is not None:

                        # Menggambar sumbu orientasi kepala
                        frame = draw_head_pose_axis(
                            frame,
                            pose
                        )

                        relative_pose = calculate_relative_pose(
                            pose["rotation_matrix"],
                            reference_rotation
                        )

                        cv2.putText(
                            frame,
                            f"Yaw: {relative_pose['yaw']:.2f}",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Pitch: {relative_pose['pitch']:.2f}",
                            (20, 115),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Roll: {relative_pose['roll']:.2f}",
                            (20, 150),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 255),
                            2
                        )

                # ======================================
                # STATUS SEBELUM KALIBRASI
                # ======================================

                else:

                    cv2.putText(
                        frame,
                        "PRESS C TO CALIBRATE",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                cv2.putText(
                    frame,
                    "FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        else:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ==================================================
        # 5. STATUS KALIBRASI
        # ==================================================

        if face_model is not None and not calibrating:

            cv2.putText(
                frame,
                "CALIBRATED",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # ==================================================
        # 6. MENAMPILKAN WEBCAM
        # ==================================================

        cv2.imshow(
            "Skripsi - Head Pose Estimation",
            frame
        )

        # ==================================================
        # 7. INPUT KEYBOARD
        # ==================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):

            calibration_samples.clear()
            face_model = None
            reference_rotation = None
            calibrating = True

            print("Memulai kalibrasi...")
            print("Pertahankan wajah menghadap lurus ke kamera.")

        elif key == ord("r"):

            calibration_samples.clear()
            face_model = None
            reference_rotation = None
            calibrating = True

            print("Kalibrasi diulang.")
            print("Pertahankan wajah menghadap lurus ke kamera.")

        elif key == ord("q"):
            break

    # ==================================================
    # 8. MEMBERSIHKAN RESOURCE
    # ==================================================

    camera.release()
    face_mesh.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
