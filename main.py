from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)

# Load the trained recognizer model
current_directory = os.path.abspath(os.path.dirname(__file__))
recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer_file_path = os.path.join(current_directory, r'face_recognition\trainingData.yml')

if os.path.exists(recognizer_file_path):
    recognizer.read(recognizer_file_path)

# Load the cascade classifier for face detection
face_cascade_file = r'face_recognition\haarcascade_frontalface_default.xml'
face_cascade_path = os.path.join(current_directory, face_cascade_file)

if os.path.exists(face_cascade_path):
    detector = cv2.CascadeClassifier(face_cascade_path)


@app.route('/')
def index():
    return render_template('imagesave.html')


@app.route('/upload', methods=['POST'])
def upload():
    image_data = request.form['image']
    image = decode_image_data(image_data)
    result_image, name = detect_faces(image)
    result_image_data = encode_image(result_image)
    return jsonify({'image': result_image_data, 'name': name})


def decode_image_data(image_data):
    image_data = image_data.replace('data:image/png;base64,', '')
    image_data = image_data.encode()
    image = np.frombuffer(base64.b64decode(image_data), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def encode_image(image):
    _, buffer = cv2.imencode('.png', image)
    encoded_image = base64.b64encode(buffer).decode()
    return 'data:image/png;base64,' + encoded_image


def detect_faces(our_image):
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
        Id, confidence = recognizer.predict(face_roi)

        # Determine if the face belongs to a known person or is unknown
        if confidence * 100 > 70:  # You may need to adjust this threshold
            if (Id == 1 or Id == 2):
                name = 'Himanshu'
                cv2.putText(our_image, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)

        else:
            cv2.putText(our_image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return our_image, name


if __name__ == '__main__':
    app.run(debug=True)
