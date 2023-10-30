import cv2 as cv
import os
import time
import numpy as np
import mediapipe as mp
from activities import utils
from activities.components.face_direction import FaceDirectionDetector
from activities.components.phone_proximity import PhoneEarProximity
from activities.components.sleep import SleepDetection
from activities.components.employee_presence import FaceTimeSpend
from flask import Flask, render_template, Response, request, redirect, url_for


app = Flask(__name__)

camera = cv.VideoCapture(0)

# Initialize MediaPipe solutions for face detection and pose estimation
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_face_dir = mp.solutions.face_mesh

frame_counter = 0  # Counter for frames
EYE_CLOSED_COUNTER = 0   # Counter for sleep detection
start_time = time.time()  # Start time of the program
face_timer = FaceTimeSpend()  # Initialize the face time tracker
phone_proximity = PhoneEarProximity()  # Initialize phone proximity tracker
face_direction_detector = FaceDirectionDetector()   # Initialize Face Direction Tracker
sleep_detector = SleepDetection()   # Initialize Sleep Detection Tracker


@app.route('/')
def index():
    return render_template('imagesave.html')


@app.route('/video')
def video():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/workvideo')
def workvideo():
    return Response(gen_work_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['POST'])
def capture():
    # Capture an image from the webcam
    ret, frame = camera.read()

    # Save the captured image to the "shots" directory
    if not os.path.exists('./shots'):
        os.makedirs('./shots')
    image_path = './shots/captured_image.jpg'
    cv.imwrite(image_path, frame)

    return redirect(url_for('result'))


@app.route('/result')
def result():
    # Load the captured image from the ./shots directory
    image_path = './shots/captured_image.jpg'
    captured_image = cv.imread(image_path)

    # Perform face recognition on the captured image
    result_image, recognized_name = detect_faces(captured_image)

    # Save the result image to a temporary file (e.g., "result.jpg")
    result_image_path = 'static/result.jpg'
    cv.imwrite(result_image_path, result_image)

    return render_template('result.html', name=recognized_name, image=result_image_path)


@app.route('/work.html')
def work():
    return render_template('work.html')


@app.route('/stats')
def stats():
    phone_usage_time = face_timer.format_time(phone_proximity.get_elapsed_time())
    face_away_time = face_timer.format_time(face_direction_detector.get_elapsed_time())
    sleep_time = face_timer.format_time(sleep_detector.get_elapsed_time())

    # Calculate total work duration using start_time and end_time
    if start_time:
        end_time = time.time()  # Get the current time as the end time
        work_duration = end_time - start_time
    else:
        work_duration = 0  # Default to 0 if start_time is not set

    work_duration = face_timer.format_time(work_duration)

    return render_template('stats.html', phone_usage_time=phone_usage_time, face_away_time=face_away_time,
                           sleep_time=sleep_time, work_duration=work_duration)


def generate_frame():
    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen_work_frame():
    global frame_counter

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_counter += 1  # Increment frame counter

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
            utils.text_with_background(image=frame, text=f"FPS: {fps:.2f}", position=(20, 30),
                                       fonts=cv.FONT_HERSHEY_PLAIN)

            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def detect_faces(our_image):
    # Get the full path to the current directory
    current_directory = os.path.abspath(os.path.dirname(__file__))

    # Load the trained recognizer model
    # recognizer = cv2.face.FisherFaceRecognizer_create()
    recognizer = cv.face.LBPHFaceRecognizer_create()

    recognizer_file_path = os.path.join(current_directory, 'recognition_dataset/trainingData.yml')

    if os.path.exists(recognizer_file_path):
        recognizer.read(recognizer_file_path)

    # Load the cascade classifier for face detection
    face_cascade_file = 'recognition_dataset/haarcascade_frontalface_default.xml'
    face_cascade_path = os.path.join(current_directory, face_cascade_file)

    if os.path.exists(face_cascade_path):
        detector = cv.CascadeClassifier(face_cascade_path)
        # print(f'Detector Path: {face_cascade_path} | Detector: {detector}')

    gray = cv.cvtColor(our_image, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    name = 'Unknown'
    for (x, y, w, h) in faces:
        # Drawing rectangle
        cv.rectangle(our_image, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Recognize the face
        face_roi = gray[y:y + h, x:x + w]

        # Resizing
        face_roi = cv.resize(face_roi, (170, 170))
        Id, uncertainity = recognizer.predict(face_roi)
        print(f'ID is: {Id} | Uncertainity is {uncertainity}')

        # Determine if the face belongs to a known person or is unknown
        if uncertainity < 70:  # You may need to adjust this threshold
            if (Id == 1 or Id == 2):
                name = 'Himanshu'
                cv.putText(our_image, name, (x, y - 10), cv.FONT_HERSHEY_DUPLEX, 0.9, utils.GREEN, 2)

        else:
            cv.putText(our_image, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, utils.RED, 2)
        print(f"Name is: {name}")

    return our_image, name


if __name__ == "__main__":
    app.run(debug=False, extra_files=['./shots/captured_image.jpg', 'static/result.jpg'])

