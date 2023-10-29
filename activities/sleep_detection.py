# Import necessary libraries
import cv2 as cv
import time
import dlib
import imutils
import utils  # Custom utility functions
import mediapipe as mp
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

# Initialize MediaPipe solutions for face detection and pose estimation
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_face_dir = mp.solutions.face_mesh


# Define a class for tracking time spent with the phone near the ear
class FaceTimeSpend:
    def __init__(self, start_time=time.time()) -> None:
        self.start_time = start_time  # Start time for tracking the session
        self.session_id = 0  # Session ID for tracking multiple sessions
        self.session_time_list = []  # List to store session times
        self.current_session_time = 0  # Current session time

    def calculate_session_time(self):
        """Calculate the current session time"""
        self.current_session_time = time.time() - self.start_time
        return self.current_session_time

    def update_time(self):
        """Update the session time, increment session ID, and add the current session time to the list"""
        self.start_time = time.time()
        if self.current_session_time >= 2.5:
            self.session_id += 1
            self.session_time_list.append(self.current_session_time)
            self.current_session_time = 0

    def format_time(self, seconds):
        """Format seconds into a time string (HH:MM:SS)"""
        return time.strftime("%H:%M:%S", time.gmtime(seconds))

    def get_time(self):
        """Get total time, current session time, and session ID"""
        total_seconds = self.current_session_time + sum(self.session_time_list)
        total_time_formatted = self.format_time(total_seconds)
        session_time_formatted = self.format_time(self.current_session_time)
        return total_time_formatted, session_time_formatted, self.session_id


# Define a class for monitoring phone proximity to the ear
class PhoneEarProximity:
    def __init__(self) -> None:
        self.phone_near_ear = False  # Flag for phone proximity to the ear
        self.timer_running = False  # Flag to track if the timer is running
        self.timer_start_time = None  # Start time for the timer
        self.timer_paused_time = 0  # Accumulated paused time for the timer

    def update(self, left_distance, right_distance):
        if left_distance < 0.2 or right_distance < 0.2:
            # Check if the phone is near the ear
            if not self.timer_running:
                self.timer_running = True
                if self.timer_start_time is None:
                    self.timer_start_time = time.time()  # Start the timer if it was not running
            self.phone_near_ear = True
        else:
            if self.timer_running:
                self.timer_running = False
                self.timer_paused_time += time.time() - self.timer_start_time
                self.timer_start_time = None
            self.phone_near_ear = False

    def get_elapsed_time(self):
        if self.timer_running:
            return time.time() - self.timer_start_time + self.timer_paused_time
        else:
            return self.timer_paused_time


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


class SleepDetection:
    def __init__(self):
        # Global Configuration Variables
        self.FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"  # path to dlib's pre-trained facial landmark predictor
        self.MINIMUM_EAR = 0.2  # Minimum EAR for both the eyes to mark the eyes as open
        self.MAXIMUM_FRAME_COUNT = 50  # Maximum number of consecutive frames in which EAR can remain less than MINIMUM_EAR, otherwise alert drowsiness
        self.activeness_status = 'Drowsy'

        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_finder = dlib.shape_predictor(self.FACIAL_LANDMARK_PREDICTOR)
        self.leftEyeStart, self.leftEyeEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.rightEyeStart, self.rightEyeEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.EYE_CLOSED_COUNTER = 0

        # Timer config
        self.timer_running = False
        self.timer_start_time = None
        self.timer_paused_time = 0

    @staticmethod
    def eye_aspect_ratio(eye):
        p2_minus_p6 = dist.euclidean(eye[1], eye[5])
        p3_minus_p5 = dist.euclidean(eye[2], eye[4])
        p1_minus_p4 = dist.euclidean(eye[0], eye[3])
        ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
        return ear

    def detect_sleep(self, frame):
        gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.face_detector(gray_image, 0)

        for face in faces:
            faceLandmarks = self.landmark_finder(gray_image, face)
            faceLandmarks = face_utils.shape_to_np(faceLandmarks)

            leftEye = faceLandmarks[self.leftEyeStart:self.leftEyeEnd]
            rightEye = faceLandmarks[self.rightEyeStart:self.rightEyeEnd]

            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv.convexHull(leftEye)
            rightEyeHull = cv.convexHull(rightEye)

            cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < self.MINIMUM_EAR:
                self.EYE_CLOSED_COUNTER += 1
            else:
                self.EYE_CLOSED_COUNTER = 0

            # cv.putText(frame, "EAR: {}".format(round(ear, 1)), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if self.EYE_CLOSED_COUNTER >= self.MAXIMUM_FRAME_COUNT:
                # cv.putText(frame, "Drowsiness", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.activeness_status = 'Drowsy'
            else:
                self.activeness_status = 'Awake'

            self.sleep_time()

    def activity_status(self):
        return self.activeness_status

    def sleep_time(self):
        if self.activeness_status == 'Awake':
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
# -----------xxxxxxx-----------------


# Initialize the camera capture using OpenCV
cap = cv.VideoCapture(0, cv.CAP_DSHOW)

# Initialize MediaPipe solutions for face detection and pose estimation with specified confidence thresholds
with mp_face_detection.FaceDetection(
        min_detection_confidence=0.5, model_selection=1
) as face_detector, mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose_detector:
    frame_counter = 0  # Counter for frames
    EYE_CLOSED_COUNTER = 0   # Counter for sleep detection
    start_time = time.time()  # Start time of the program
    face_timer = FaceTimeSpend()  # Initialize the face time tracker
    phone_proximity = PhoneEarProximity()  # Initialize phone proximity tracker
    face_direction_detector = FaceDirectionDetector()   # Initialize Face Direction Tracker
    sleep_detector = SleepDetection()   # Initialize Sleep Detection Tracker

    while True:
        frame_counter += 1  # Increment frame counter
        ret, frame = cap.read()  # Read a frame from the camera
        if ret is False:
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert frame to RGB
        frame_height, frame_width, c = frame.shape

        # Enhancing Performance
        rgb_frame.flags.writeable = False

        # HERE WE WORK ON GETTING FACE LOCATION
        results = face_detector.process(rgb_frame)  # Use face detection model to detect faces

        # Enhancing Performance
        rgb_frame.flags.writeable = True

        if results.detections:
            session_time = face_timer.calculate_session_time()  # Calculate session time
            print(f'session_time: {session_time:.2f}', end='\r')
            for face in results.detections:
                # Extract face bounding box coordinates and display face detection confidence
                # Also, draw bounding boxes around detected faces
                face_react = np.multiply(
                    [
                        face.location_data.relative_bounding_box.xmin,
                        face.location_data.relative_bounding_box.ymin,
                        face.location_data.relative_bounding_box.width,
                        face.location_data.relative_bounding_box.height
                    ],
                    [frame_width, frame_height, frame_width, frame_height]
                ).astype(int)
                utils.rect_corners(image=frame, rect_points=face_react, color=utils.INDIGO, th=2)
                utils.text_with_background(image=frame, text=f"Conf: {(face.score[0] * 100):.2f}",
                                           position=face_react[:2], fonts=cv.FONT_HERSHEY_PLAIN,
                                           color=utils.MAGENTA)
        else:
            face_timer.update_time()  # Update session time when no faces are detected

        # Enhancing Performance
        rgb_frame.flags.writeable = False

        # HERE WE WORK ON GETTING HAND LANDMARKS AND FURTHER GET PHONE PROXIMITY
        pose_results = pose_detector.process(rgb_frame)  # Use pose estimation to track hand positions

        # Enhancing Performance
        rgb_frame.flags.writeable = True

        landmarks = pose_results.pose_landmarks

        if landmarks:
            # Calculate distances between index fingers and ears
            # Update phone proximity information
            left_index_finger = landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
            right_index_finger = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
            left_ear = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR]

            left_distance = np.linalg.norm(
                np.array([left_index_finger.x, left_index_finger.y]) - np.array([left_ear.x, left_ear.y]))
            right_distance = np.linalg.norm(
                np.array([right_index_finger.x, right_index_finger.y]) - np.array([right_ear.x, right_ear.y]))

            phone_proximity.update(left_distance, right_distance)

        # Get session time information
        total_face_time_formatted, session_time_formatted, session_id = face_timer.get_time()
        # Get phone status
        phone_status = "Phone Near Ear" if phone_proximity.phone_near_ear else "Phone Not Near Ear"
        phone_elapsed_time = face_timer.format_time(phone_proximity.get_elapsed_time())

        # HERE WE WORK ON GETTING FACE DIRECTION
        face_direction_detector.detect_face_direction(frame)
        face_direction = face_direction_detector.get_face_direction()
        face_away_time = face_timer.format_time(face_direction_detector.get_elapsed_time())

        # HERE WE WORK ON GETTING SLEEP DETECTION
        sleep_detector.detect_sleep(frame)
        sleep_status = sleep_detector.activity_status()
        sleep_time = face_timer.format_time(sleep_detector.get_elapsed_time())

        # Display relevant information on the frame
        text_to_display = f"Time: {total_face_time_formatted} S-Time: {session_time_formatted} S_ID: {session_id}"
        phone_text_to_display = f"Phone Status: {phone_status} Elapsed Time: {phone_elapsed_time} s"
        face_direction_display = f'Face: {face_direction} | Away Time: {face_away_time}'
        sleep_text_to_display = f'Activeness Status: {sleep_status} | Drowsy: {sleep_time}'

        utils.text_with_background(image=frame, text=text_to_display, position=(20, 60),
                                   fonts=cv.FONT_HERSHEY_PLAIN, color=utils.YELLOW)
        utils.text_with_background(image=frame, text=phone_text_to_display, position=(20, 90),
                                   fonts=cv.FONT_HERSHEY_PLAIN, color=utils.YELLOW)
        utils.text_with_background(image=frame, text=face_direction_display, position=(20, 120),
                                   fonts=cv.FONT_HERSHEY_PLAIN, color=utils.YELLOW)
        utils.text_with_background(image=frame, text=sleep_text_to_display, position=(20, 150),
                                   fonts=cv.FONT_HERSHEY_PLAIN, color=utils.YELLOW)

        fps = frame_counter / (time.time() - start_time)  # Calculate frames per second
        utils.text_with_background(image=frame, text=f"FPS: {fps:.2f}", position=(20, 30), fonts=cv.FONT_HERSHEY_PLAIN)

        # Display the frame with the information
        cv.imshow('frame', frame)

        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cap.release()  # Release the camera
    cv.destroyAllWindows()  # Close OpenCV windows
