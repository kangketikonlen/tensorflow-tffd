import cv2
import os

path = os.getcwd()
filename = "risu.mp4"
upload_dir = os.path.join(path, "uploads/", filename)
output_dir = os.path.join(path, "outputs/risu/")
capture = cv2.VideoCapture(upload_dir)
frameNr = 0
while (True):
    success, frame = capture.read()

    if success:
        cv2.imwrite(f'{output_dir}frame_{frameNr}.jpg', frame)
    else:
        break
    frameNr = frameNr+1
capture.release()
