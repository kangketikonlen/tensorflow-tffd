# Import Required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from check import classes, dir_path
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, GlobalAveragePooling2D, Activation
from tensorflow.keras.optimizers import Adam

# Set the batch size, width, height and the percentage of the validation split.
batch_size = 60
IMG_HEIGHT = 224
IMG_WIDTH = 224
split = 0.2

#  Setup the ImagedataGenerator for training, pass in any supported augmentation schemes, notice that we're also splitting the data with split argument.
datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=split,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect'
)

# Setup the ImagedataGenerator for validation, no augmentation is done, only rescaling is done, notice that we're also splitting the data with split argument.
datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=split)


# Data Generation for Training with a constant seed valued 40, notice that we are specifying the subset as 'training'
train_data_generator = datagen_train.flow_from_directory(
    batch_size=batch_size,
    directory=dir_path,
    shuffle=True,
    seed=40,
    subset='training',
    interpolation='bicubic',
    target_size=(IMG_HEIGHT, IMG_WIDTH)
)

# Data Generator for validation images with the same seed to make sure there is no data overlap, notice that we are specifying the subset as 'validation'
vald_data_generator = datagen_val.flow_from_directory(
    batch_size=batch_size,
    directory=dir_path,
    shuffle=True,
    seed=40,
    subset='validation',
    interpolation='bicubic',
    target_size=(IMG_HEIGHT, IMG_WIDTH)
)

# The "subset" variable tells the Imagedatagerator class which generator gets 80% and which gets 20% of the data

# Here we are creating a function for displaying images of flowers from the data generators


def display_images(data_generator, no=15):
    sample_training_images, labels = next(data_generator)

    plt.figure(figsize=[25, 25])

    # By default we're displaying 15 images, you can show more examples
    total_samples = sample_training_images[:no]

    cols = 5
    rows = np.floor(len(total_samples) / cols)

    for i, img in enumerate(total_samples, 1):

        plt.subplot(rows, cols, i)
        plt.imshow(img)

        # Converting One hot encoding labels to string labels and displaying it.
        class_name = classes[np.argmax(labels[i-1])]
        plt.title(class_name)
        plt.axis('off')


# Display Augmented Images
display_images(train_data_generator)

# First Reset the generators, since we used the first batch to display the images.
vald_data_generator.reset()
train_data_generator.reset()

# Here we are creating Sequential model also defing its layers
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Dropout(0.10),

    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(256, 3, padding='same', activation='relu'),
    MaxPooling2D(),

    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dropout(0.10),
    Dense(len(classes), activation='softmax')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy', metrics=['accuracy']
)

model.summary()
