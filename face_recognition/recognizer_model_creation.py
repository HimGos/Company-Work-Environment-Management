import cv2
import os
import numpy as np
from PIL import Image

# recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'


def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []

    desired_width = 170
    desired_height = 170

    for imagepath in imagePaths:
        # faceImg = Image.open(imagepath).convert('L')
        faceImg = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
        faceImg = cv2.resize(faceImg, (desired_width, desired_height))

        faceNp = np.array(faceImg, 'uint8')
        print(imagepath)
        ID = int(os.path.split(imagepath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow('training', faceNp)
        cv2.waitKey(10)
    return np.array(IDs), faces


Ids, faces = getImagesWithID(path)
recognizer.train(faces, Ids)
recognizer.save('trainingData.yml')
cv2.destroyAllWindows()