import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from generator import model, train_data_generator, vald_data_generator, batch_size
from check import classes
from splitter import object_name

epochs = 5

# Start Training
history = model.fit(
    train_data_generator,  steps_per_epoch=train_data_generator.samples // batch_size, epochs=epochs, validation_data=vald_data_generator,
    validation_steps=vald_data_generator.samples // batch_size
)

# Use model.fit_generator() if using TF version &lt; 2.2

# Plot the accuracy and loss curves for both training and validation

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training loss')
plt.legend()

plt.show()

# Read the rose image
target = "moona1.jpg"
path = os.getcwd()
samples_path = os.path.join(path, "samples/", target)
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
label = classes[index]

# Display the image and print the predicted class name with its confidence.
print("Predicted : {} {:.2f}%".format(label, prob*100))
plt.imshow(img[:, :, ::-1])
plt.axis("off")

path = os.getcwd()
dir_name = os.path.dirname(path)
# Saving your model to disk allows you to use it later
model.save(dir_name+'/models/datasets.h5')

# Later on you can load your model this way
#model = load_model('Model/flowers.h5')
