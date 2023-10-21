import cv2
import os

if not os.path.exists('dataset'):
    os.makedirs('dataset')

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

Id = input('enter your ID:')
sampleNum = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        file_path = 'dataset/User.' + Id + '.' + str(sampleNum) + '.jpg'
        print(file_path)
        # Saving the captured face in the dataset folder
        cv2.imwrite(file_path, gray[y:y+h, x:x+w])
        sampleNum = sampleNum + 1
        cv2.imshow('frame', img)

    # Break if the sample number is more than 20
    if sampleNum > 20:
        break

cam.release()
cv2.destroyAllWindows()
