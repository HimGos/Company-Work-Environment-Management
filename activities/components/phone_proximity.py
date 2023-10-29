import time
import sys
import cv2 as cv
import numpy as np
from activities import utils
import mediapipe as mp
from activities.exception import CustomException

mp_pose = mp.solutions.pose


# Define a class for monitoring phone proximity to the ear
class PhoneEarProximity:
    def __init__(self) -> None:
        try:
            self.pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            self.phone_near_ear = False  # Flag for phone proximity to the ear
            self.timer_running = False  # Flag to track if the timer is running
            self.timer_start_time = None  # Start time for the timer
            self.timer_paused_time = 0  # Accumulated paused time for the timer
        except Exception as e:
            raise CustomException(e, sys)

    def phone(self, frame):
        try:
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert frame to RGB

            # Enhancing Performance
            rgb_frame.flags.writeable = False

            pose_results = self.pose_detector.process(rgb_frame)  # Use pose estimation to track hand positions

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

                self.update(left_distance, right_distance)
        except Exception as e:
            raise CustomException(e, sys)

    def update(self, left_distance, right_distance):
        try:
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
