import os
from generator import model
from splitter import object_name

path = os.getcwd()
dir_name = os.path.dirname(path)
# Saving your model to disk allows you to use it later
model.save(dir_name+'/'+object_name+'.h5')

# Later on you can load your model this way
#model = load_model('Model/flowers.h5')
