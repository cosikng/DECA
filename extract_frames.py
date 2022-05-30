import cv2
import os

file = '2.mp4'
path = 'TestSamples/' + file[:-4]
os.makedirs(path, exist_ok=True)
cap = cv2.VideoCapture(file)
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Frames', frame)
    cv2.imwrite('%s/%05d.jpg' % (path, i), frame)
    # cv2.waitKey(0)
    i += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
