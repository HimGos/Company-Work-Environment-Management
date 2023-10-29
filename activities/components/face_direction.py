import time
import cv2 as cv
import numpy as np
import mediapipe as mp

mp_face_dir = mp.solutions.face_mesh

# Define a class for detecting face direction
class FaceDirectionDetector:
    def __init__(self)-> None:
        self.face_direction = 'Looking Away'      # Default direction
        self.face_mesh = mp_face_dir.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.timer_running = False
        self.timer_start_time = None
        self.timer_paused_time = 0

    def detect_face_direction(self, frame):
        image = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks  in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    # We gonna work with these 6 points only
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get 2D Coordinates
                        face_2d.append([x, y])

                        # Get 3D Coordinates
                        face_3d.append([x, y, lm.z])

                # Convert it to the Numpy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to Numpy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The Distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # See where the user's head tilting
                if y < -12:
                    self.face_direction = 'Looking Away'
                elif y > 10:
                    self.face_direction = 'Looking Away'
                elif x < -12:
                    self.face_direction = 'Looking Away'
                elif x > 10:
                    self.face_direction = 'Looking Away'
                else:
                    self.face_direction = 'Forward'

                # Display the nose direction
                nose_3d_projection, jacobian = cv.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                 dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv.line(image, p1, p2, (255, 0, 0), 3)

                self.time_update()

    def get_face_direction(self):
        return self.face_direction

    def time_update(self):
        if self.face_direction == 'Forward':
            if self.timer_running:
                self.timer_running = False
                self.timer_paused_time += time.time() - self.timer_start_time
                self.timer_start_time = None
        else:
            if not self.timer_running:
                self.timer_running = True
                if self.timer_start_time is None:
                    self.timer_start_time = time.time()

    def get_elapsed_time(self):
        if self.timer_running:
            return time.time() - self.timer_start_time + self.timer_paused_time
        else:
            return self.timer_paused_time
