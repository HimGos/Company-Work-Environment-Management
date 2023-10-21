import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os

# Get the full path to the current directory
current_directory = os.path.abspath(os.path.dirname(__file__))

# Load the trained recognizer model
recognizer = cv2.face.FisherFaceRecognizer_create()

recognizer_file_path = os.path.join(current_directory, 'trainingData.yml')

if os.path.exists(recognizer_file_path):
    recognizer.read(recognizer_file_path)
else:
    st.error("Error: 'trainingData.yml' file not found in the current directory.")

# Load the cascade classifier for face detection
face_cascade_file = 'haarcascade_frontalface_default.xml'
face_cascade_path = os.path.join(current_directory, face_cascade_file)

if os.path.exists(face_cascade_path):
    detector = cv2.CascadeClassifier(face_cascade_path)
else:
    st.error(f"Error: '{face_cascade_file}' not found in the current directory.")

def detect_faces(our_image):
    img = np.array(our_image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    name = 'Unknown'
    for (x, y, w, h) in faces:
        # Drawing rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Recognize the face
        face_roi = gray[y:y + h, x:x + w]

        # Resizing
        face_roi = cv2.resize(face_roi, (170, 170))
        Id, confidence = recognizer.predict(face_roi)

        # Determine if the face belongs to a known person or is unknown
        if confidence*100 > 70:  # You may need to adjust this threshold
            st.text(f'confidence: {confidence*100:.2f}% | ID: {Id}')
            if (Id == 1 or Id == 2):
                name = 'Himanshu'
                cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2)

        else:
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return img, name


def main():
    """Face Recognition App"""

    st.title("Streamlit Tutorial")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Company Work Env. Mangament</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    camera_state = st.camera_input('Take a pic')
    if st.button("Recognise"):
        our_image = Image.open(camera_state)
        result_frame, name = detect_faces(our_image)
        st.image(result_frame, channels="RGB", use_column_width=True)
        st.subheader(f'Welcome {name}')

    # if image_file is not None:
    #     our_image = Image.open(image_file)
    #     st.text("Original Image")
    #     st.image(our_image)
    #
    # if st.button("Recognise"):
    #     result_img, name = detect_faces(our_image)
    #     st.image(result_img)
    #     st.text(f'He is {name}')


if __name__ == '__main__':
    main()