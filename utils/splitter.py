import cv2
import os

path = os.getcwd()
object_name = "risu"
filename = object_name+".mp4"
upload_dir = os.path.join(path, "uploads/", filename)
output_dir = os.path.join(path, "outputs/"+object_name+"/")
capture = cv2.VideoCapture(upload_dir)
frameNr = 0

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

while (True):
    success, frame = capture.read()

    if success:
        original = f'{output_dir}{object_name}_{frameNr}.jpg'
        cv2.imwrite(original, frame)
        resize = cv2.imread(original, cv2.IMREAD_UNCHANGED)
        scale_percent = 40  # percent of original size
        width = int(resize.shape[1] * scale_percent / 100)
        height = int(resize.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(resize, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(original, resized)
        print('File created :', original)
    else:
        break

    frameNr = frameNr+1
capture.release()
