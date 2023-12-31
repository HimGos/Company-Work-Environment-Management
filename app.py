import cv2 as cv
import sys
import os
import time
import json
import mediapipe as mp
from activities import utils
from activities.exception import CustomException
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
start_time = time.time()  # Start time of the program
EMP_NAME = ''
work_duration = 0

# Defining class objects
face_timer = FaceTimeSpend()  # Initialize the face time tracker
phone_proximity = PhoneEarProximity()  # Initialize phone proximity tracker
face_direction_detector = FaceDirectionDetector()   # Initialize Face Direction Tracker
sleep_detector = SleepDetection()   # Initialize Sleep Detection Tracker


@app.route('/')
def index():
    try:
        return render_template('imagesave.html')
    except Exception as e:
        raise CustomException(e, sys)


@app.route('/video')
def video():
    try:
        return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        raise CustomException(e, sys)


@app.route('/workvideo')
def workvideo():
    try:
        return Response(gen_work_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        raise CustomException(e, sys)


@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Capture an image from the webcam
        ret, frame = camera.read()

        # Save the captured image to the "shots" directory
        if not os.path.exists('./shots'):
            os.makedirs('./shots')
        image_path = './shots/captured_image.jpg'
        cv.imwrite(image_path, frame)

        return redirect(url_for('result'))
    except Exception as e:
        raise CustomException(e, sys)


@app.route('/result')
def result():
    try:
        # Load the captured image from the ./shots directory
        image_path = './shots/captured_image.jpg'
        captured_image = cv.imread(image_path)

        # Perform face recognition on the captured image
        result_image, recognized_name = detect_faces(captured_image)

        # Save the result image to a temporary file (e.g., "result.jpg")
        result_image_path = 'static/result.jpg'
        cv.imwrite(result_image_path, result_image)

        return render_template('result.html', name=recognized_name, image=result_image_path)
    except Exception as e:
        raise CustomException(e, sys)


@app.route('/work.html')
def work():
    try:
        return render_template('work.html')
    except Exception as e:
        raise CustomException(e, sys)


@app.route('/stats')
def stats():
    try:
        global work_duration

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

        present_on_screen_time, session_time_formatted, times_away = face_timer.get_time()

        return render_template('stats.html', phone_usage_time=phone_usage_time, face_away_time=face_away_time,
                               sleep_time=sleep_time, work_duration=work_duration, present_on_screen_time=present_on_screen_time,
                               times_away=times_away, emp_name=EMP_NAME)
    except Exception as e:
        raise CustomException(e, sys)


@app.route('/record_employee_stats', methods=['POST'])
def record_employee_stats():
    """This function record employee stats and saves in JSON file."""
    try:
        global work_duration
        present_on_screen_time, session_time_formatted, times_away = face_timer.get_time()
        # Collect the stats data in a JSON file
        stats_data = {
            'Employee Name': EMP_NAME,
            'Work Duration': work_duration,
            'Times Away From Screen': times_away,
            'Present on Screen Duration': present_on_screen_time,
            'Phone Usage': face_timer.format_time(phone_proximity.get_elapsed_time()),
            'Looking Away': face_timer.format_time(face_direction_detector.get_elapsed_time()),
            'Sleepy / Drowsy': face_timer.format_time(sleep_detector.get_elapsed_time())
        }

        # Get the absolute path to your project directory
        project_dir = os.path.abspath(os.path.dirname(__file__))

        # Create a folder 'employee stats' if it doesn't exist
        stats_folder = os.path.join(project_dir, 'employee_stats')
        if not os.path.exists(stats_folder):
            os.makedirs(stats_folder)

        # Define the absolute path to the stats file
        stats_file = os.path.join(stats_folder, f"{EMP_NAME}_stats.json")

        # Save the stats data to the JSON file
        with open(stats_file, 'w') as json_file:
            json.dump(stats_data, json_file)

        return redirect(url_for('stats'))
    except Exception as e:
        raise CustomException(e, sys)


def generate_frame():
    """This function is used on first page to help access webcam and capture employee pic."""
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        raise CustomException(e, sys)


def gen_work_frame():
    """This function is used in work.html page. Helps in getting employee activity data"""
    try:
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
    except Exception as e:
        raise CustomException(e, sys)


def detect_faces(our_image):
    """This function is used in result.html page to detect the employee and return name & image"""
    try:
        global EMP_NAME
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
                    EMP_NAME = name
                    cv.putText(our_image, name, (x, y - 10), cv.FONT_HERSHEY_DUPLEX, 0.9, utils.GREEN, 2)

            else:
                cv.putText(our_image, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, utils.RED, 2)
            print(f"Name is: {name}")

        return our_image, name
    except Exception as e:
        raise CustomException(e, sys)


def cleanup():
    # Release the video capture
    camera.release()

    # Close any OpenCV windows
    cv.destroyAllWindows()


if __name__ == "__main__":
    try:
        app.run(debug=False, extra_files=['./shots/captured_image.jpg', 'static/result.jpg'])
    finally:
        cleanup()   # Ensure cleanup is always called on exit
