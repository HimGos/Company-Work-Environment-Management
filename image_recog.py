import cv2
import os
import numpy as np
from activities import utils
from flask import Flask, render_template, Response, request, redirect, url_for

app = Flask(__name__)

camera = cv2.VideoCapture(0)


def generate_frame():
    while True:
        success, frame = camera.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('imagesave.html')


@app.route('/video')
def video():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['POST'])
def capture():
    # Capture an image from the webcam
    ret, frame = camera.read()

    # Save the captured image to the "shots" directory
    if not os.path.exists('./shots'):
        os.makedirs('./shots')
    image_path = './shots/captured_image.jpg'
    cv2.imwrite(image_path, frame)

    return redirect(url_for('result'))


@app.route('/result')
def result():
    # Load the captured image from the ./shots directory
    image_path = './shots/captured_image.jpg'
    captured_image = cv2.imread(image_path)

    # Perform face recognition on the captured image
    result_image, recognized_name = detect_faces(captured_image)

    # Save the result image to a temporary file (e.g., "result.jpg")
    result_image_path = 'static/result.jpg'
    cv2.imwrite(result_image_path, result_image)

    return render_template('result.html', name=recognized_name, image=result_image_path)


@app.route('/work.html')
def work():
    return render_template('work.html')


def detect_faces(our_image):
    # Get the full path to the current directory
    current_directory = os.path.abspath(os.path.dirname(__file__))

    # Load the trained recognizer model
    # recognizer = cv2.face.FisherFaceRecognizer_create()
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer_file_path = os.path.join(current_directory, 'recognition_dataset/trainingData.yml')

    if os.path.exists(recognizer_file_path):
        recognizer.read(recognizer_file_path)

    # Load the cascade classifier for face detection
    face_cascade_file = 'recognition_dataset/haarcascade_frontalface_default.xml'
    face_cascade_path = os.path.join(current_directory, face_cascade_file)

    if os.path.exists(face_cascade_path):
        detector = cv2.CascadeClassifier(face_cascade_path)
        # print(f'Detector Path: {face_cascade_path} | Detector: {detector}')

    gray = cv2.cvtColor(our_image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    name = 'Unknown'
    for (x, y, w, h) in faces:
        # Drawing rectangle
        cv2.rectangle(our_image, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Recognize the face
        face_roi = gray[y:y + h, x:x + w]

        # Resizing
        face_roi = cv2.resize(face_roi, (170, 170))
        Id, uncertainity = recognizer.predict(face_roi)
        print(f'ID is: {Id} | Uncertainity is {uncertainity}')

        # Determine if the face belongs to a known person or is unknown
        if uncertainity < 70:  # You may need to adjust this threshold
            if (Id == 1 or Id == 2):
                name = 'Himanshu'
                cv2.putText(our_image, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, utils.GREEN, 2)

        else:
            cv2.putText(our_image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, utils.RED, 2)
        print(f"Name is: {name}")

    return our_image, name


if __name__ == "__main__":
    app.run(debug=False, extra_files=['./shots/captured_image.jpg', 'static/result.jpg'])

