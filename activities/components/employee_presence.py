import time
import mediapipe as mp
import cv2 as cv
import numpy as np
from activities import utils


mp_face_detection = mp.solutions.face_detection


class FaceTimeSpend:
    def __init__(self, start_time=time.time()) -> None:
        self.face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
        self.start_time = start_time  # Start time for tracking the session
        self.session_id = 0  # Session ID for tracking multiple sessions
        self.session_time_list = []  # List to store session times
        self.current_session_time = 0  # Current session time

    def detect_face_time(self, frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # Convert frame to RGB
        frame_height, frame_width, c = frame.shape
        # Enhancing Performance
        rgb_frame.flags.writeable = False

        # HERE WE WORK ON GETTING FACE LOCATION
        results = self.face_detector.process(rgb_frame)  # Use face detection model to detect faces

        # Enhancing Performance
        rgb_frame.flags.writeable = True

        if results.detections:
            session_time = self.calculate_session_time()  # Calculate session time
            for face in results.detections:
                # Extract face bounding box coordinates and display face detection confidence
                # Also, draw bounding boxes around detected faces
                face_box = np.multiply(
                    [
                        face.location_data.relative_bounding_box.xmin,
                        face.location_data.relative_bounding_box.ymin,
                        face.location_data.relative_bounding_box.width,
                        face.location_data.relative_bounding_box.height
                    ],
                    [frame_width, frame_height, frame_width, frame_height]
                ).astype(int)
                utils.rect_corners(image=frame, rect_points=face_box, color=utils.INDIGO, th=2)
                utils.text_with_background(image=frame, text=f"Conf: {(face.score[0] * 100):.2f}",
                                           position=face_box[:2], fonts=cv.FONT_HERSHEY_PLAIN,
                                           color=utils.MAGENTA)
        else:
            self.update_time()  # Update session time when no faces are detected

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
