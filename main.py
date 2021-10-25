import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Define class names and sort them alphabatically as this is how tf.keras remembers them
labels = ['moona', 'risu', 'gilang']
labels.sort()

dir_name = os.getcwd()
model = keras.models.load_model(dir_name+'/models/datasets.h5')

# Read the rose image
target = "risu1.jpg"
samples_path = os.path.join(dir_name, "utils", "samples", target)
img = cv2.imread(samples_path)

# Resize the image to the size you trained on.
imgr = cv2.resize(img, (224, 224))

# Convert image BGR TO RGB, since OpenCV works with BGR and tensorflow in RGB.
imgrgb = cv2.cvtColor(imgr, cv2.COLOR_BGR2RGB)

# Normalize the image to be in range 0-1 and then convert to a float array.
final_format = np.array([imgrgb]).astype('float64') / 255.0

# Perform the prediction
pred = model.predict(final_format)

# Get the index of top prediction
index = np.argmax(pred[0])

# Get the max probablity for that prediction
prob = np.max(pred[0])

# Get the name of the predicted class using the index
label = labels[index]

# Display the image and print the predicted class name with its confidence.
print("Predicted Flowers is : {} {:.2f}%".format(label, prob*100))
plt.imshow(img[:, :, ::-1])
plt.axis("off")
