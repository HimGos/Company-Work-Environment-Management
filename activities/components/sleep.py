import time
import cv2 as cv
import dlib
import sys
from imutils import face_utils
from scipy.spatial import distance as dist
from activities.exception import CustomException

class SleepDetection:
    def __init__(self):
        try:
            # Global Configuration Variables
            self.FACIAL_LANDMARK_PREDICTOR = "shape_predictor_68_face_landmarks.dat"  # path to dlib's pre-trained facial landmark predictor
            self.MINIMUM_EAR = 0.23  # Minimum EAR for both the eyes to mark the eyes as open
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
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def eye_aspect_ratio(eye):
        try:
            p2_minus_p6 = dist.euclidean(eye[1], eye[5])
            p3_minus_p5 = dist.euclidean(eye[2], eye[4])
            p1_minus_p4 = dist.euclidean(eye[0], eye[3])
            ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
            return ear
        except Exception as e:
            raise CustomException(e, sys)

    def detect_sleep(self, frame):
        try:
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
        except Exception as e:
            raise CustomException(e, sys)

    def activity_status(self):
        try:
            return self.activeness_status
        except Exception as e:
            raise CustomException(e, sys)

    def sleep_time(self):
        try:
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
        except Exception as e:
            raise CustomException(e, sys)

    def get_elapsed_time(self):
        try:
            if self.timer_running:
                return time.time() - self.timer_start_time + self.timer_paused_time
            else:
                return self.timer_paused_time
        except Exception as e:
            raise CustomException(e, sys)
