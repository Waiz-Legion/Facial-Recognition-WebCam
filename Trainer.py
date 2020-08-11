import cv2
import numpy as np
from PIL import Image
import os  # Path for face image database
path = '/root/PycharmProjects/Face/Dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("/root/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml");


def getimagesandlabels(path):
    imagepaths = [os.path.join(path, f) for f in os.listdir(path)]
    facesamples=[]
    ids = []
    for imagePath in imagepaths:
        PIL_img = Image.open(imagePath).convert('L')  # grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            facesamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return facesamples, ids

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getimagesandlabels(path)
recognizer.train(faces, np.array(ids))  # Save the model into trainer/trainer.yml
recognizer.write('/root/PycharmProjects/Face/Trainer/trainer.yml')
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
