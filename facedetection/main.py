import cv2 as cv
import time
import utils
import mediapipe as mp
import numpy as np

mp_face_detection = mp.solutions.face_detection


class FaceTimeSpend:
    def __init__(self, start_time=time.time()) -> None:
        self.start_time = start_time
        self.session_id = 0
        self.session_time_list = []

    def calculate_session_time(self):
        self.current_session_time = time.time() - self.start_time
        return self.current_session_time

    def update_time(self):
        """
        This function helps in restarting the session time and updates session id once a person sits
        for more than 2.5 seconds. It also appends the previous time in list to keep record.
        :return:
        """
        self.start_time = time.time()
        if self.current_session_time >= 2.5:
            self.session_id += 1
            self.session_time_list.append(self.current_session_time)
            self.current_session_time = 0

    def format_time(self, seconds):
        """This function just change the format of time"""
        return time.strftime("%H:%M:%S", time.gmtime(seconds))

    def get_time(self):
        """
        We calculate total time, session time & get session id too
        :return:
        """
        total_seconds = self.current_session_time + sum(self.session_time_list)
        total_time_formatted = self.format_time(total_seconds)
        session_time_formatted = self.format_time(self.current_session_time)
        return total_time_formatted, session_time_formatted, self.session_id


cap = cv.VideoCapture(0)
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5, model_selection=1
) as face_detector:
    # Calculating fps
    frame_counter = 0
    start_time = time.time()
    face_timer = FaceTimeSpend()
    while True:
        frame_counter += 1
        ret, frame = cap.read()
        if ret is False:
            break
        # Since mediapipe require RGB, so we convert cv2 BGR to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_height, frame_width, c = frame.shape
        # Passing this converted frame to mediapipe to detect the face
        results = face_detector.process(rgb_frame)
        # This will give us coordinates of faces
        if results.detections:
            session_time = face_timer.calculate_session_time()
            print(f'session_time: {session_time:.2f}', end='\r')
            # print(results.detections) # Printing coordinates
            for face in results.detections:
                face_react = np.multiply(
                    [
                        face.location_data.relative_bounding_box.xmin,
                        face.location_data.relative_bounding_box.ymin,
                        face.location_data.relative_bounding_box.width,
                        face.location_data.relative_bounding_box.height
                    ],
                    [frame_width, frame_height, frame_width, frame_height]
                ).astype(int)
                # print(face.location_data.relative_bounding_box)
                # Drawing rectangle around the face
                utils.rect_corners(image=frame, rect_points=face_react, color=utils.INDIGO, th=3)
                # Confidence score
                utils.text_with_background(image=frame, text=f"Conf: {(face.score[0]*100):.2f}",
                                           position=face_react[:2], fonts=cv.FONT_HERSHEY_PLAIN,
                                           color=utils.MAGENTA)
        else:
            # If no face then we reset time which impacts session time
            face_timer.update_time()
        # Getting total time and session time
        total_face_time_formatted, session_tim_formatted, session_id = face_timer.get_time()
        # Top left rectangle to show stats
        utils.text_with_background(image=frame,
                                   text=f"Time: {total_face_time_formatted} S-Time: {session_tim_formatted} S_ID: {session_id}",
                                   position=(20, 60), fonts=cv.FONT_HERSHEY_PLAIN, color=utils.YELLOW)
        # Showing fps on another rectangle above the stats rectangle
        fps = frame_counter/(time.time() - start_time)
        utils.text_with_background(image=frame, text=f"FPS: {fps:.2f}", position=(20,30), fonts=cv.FONT_HERSHEY_PLAIN)
        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

