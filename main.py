import cv2 as cv
import time
import mediapipe as mp
from activities import utils
from activities.components.face_direction import FaceDirectionDetector
from activities.components.phone_proximity import PhoneEarProximity
from activities.components.sleep import SleepDetection
from activities.components.employee_presence import FaceTimeSpend

# Initialize MediaPipe solutions for face detection and pose estimation
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_face_dir = mp.solutions.face_mesh


# Initialize the camera capture using OpenCV
cap = cv.VideoCapture(0)

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

    # HERE WE WORK ON GETTING FACE TIME
    face_timer.detect_face_time(frame=frame)
    total_face_time_formatted, session_time_formatted, session_id = face_timer.get_time()

    # HERE WE WORK ON GETTING PHONE PROXIMITY
    phone_proximity.phone(frame=frame)
    phone_status = "Phone Near Ear" if phone_proximity.phone_near_ear else "Phone Not Near Ear"
    phone_elapsed_time = face_timer.format_time(phone_proximity.get_elapsed_time())

    # HERE WE WORK ON GETTING FACE DIRECTION
    face_direction_detector.detect_face_direction(frame=frame)
    face_direction = face_direction_detector.get_face_direction()
    face_away_time = face_timer.format_time(face_direction_detector.get_elapsed_time())

    # HERE WE WORK ON GETTING SLEEP DETECTION
    sleep_detector.detect_sleep(frame=frame)
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
