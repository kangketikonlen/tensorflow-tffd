# Import Required libraries
import os

# This is the directory path where all the class folders are
path = os.getcwd()
dir_path = os.path.join(path, "outputs/")

# Initialize classes list, this list will contain the names of our classes.
classes = []

# Iterate over the names of each class
for class_name in os.listdir(dir_path):

   # Get the full path of each class
    class_path = os.path.join(dir_path, class_name)

    # Check if the class is a directory/folder
    if os.path.isdir(class_path):

        # Get the number of images in each class and print them
        No_of_images = len(os.listdir(class_path))
        print("Found {} images of {}".format(No_of_images, class_name))

        # Also store the name of each class
        classes.append(class_name)

# Sort the list in alphabatical order and print it
classes.sort()
print(classes)
