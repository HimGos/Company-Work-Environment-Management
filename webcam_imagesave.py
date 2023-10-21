from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os

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

    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(debug=True)
