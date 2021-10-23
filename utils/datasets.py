# Import Required libraries
import tensorflow as tf
from check import dir_path

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
    fill_mode='reflect')

# Setup the ImagedataGenerator for validation, no augmentation is done, only rescaling is done, notice that we're also splitting the data with split argument.
datagen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=split)


# Data Generation for Training with a constant seed valued 40, notice that we are specifying the subset as 'training'
train_data_generator = datagen_train.flow_from_directory(batch_size=batch_size,
                                                         directory=dir_path,
                                                         shuffle=True,
                                                         seed=40,
                                                         subset='training',
                                                         interpolation='bicubic',
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH))

# Data Generator for validation images with the same seed to make sure there is no data overlap, notice that we are specifying the subset as 'validation'
vald_data_generator = datagen_val.flow_from_directory(batch_size=batch_size,
                                                      directory=dir_path,
                                                      shuffle=True,
                                                      seed=40,
                                                      subset='validation',
                                                      interpolation='bicubic',
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH))

# The "subset" variable tells the Imagedatagerator class which generator gets 80% and which gets 20% of the data
