import tensorflow as tf
from keras import models
from keras.utils import plot_model
import pickle
import math
import numpy as np
from PIL import Image

path_d = {'train_pics': "H:\\Downloads\\data_object_image_2\\training\\image_2",
          'train_labels': "H:\\Downloads\\data_object_label_2\\training\\label_2",
          'Car': "H:\\Databases\\data_object_cars\\processed",
          'Van': "H:\\Databases\\data_object_vans",
          'Truck': "H:\\Databases\\data_object_trucks",
          'Pedestrian': "H:\\Databases\\data_object_pedestrians",
          'Person_sitting': "H:\\Databases\\data_object_sittingpeople",
          'Cyclist': "H:\\Databases\\data_object_cyclists",
          'Tram': "H:\\Databases\\data_object_trams",
          'Misc': "H:\\Databases\\data_object_misc",
          'DontCare': "H:\\Databases\\data_object_dontcare",
          'Background': "H:\\Databases\\data_object_backgrounds",
          'MainDatabase': "H:\\Databases\\processed_database",
          'Model': "H:\\Databases\\model_weights"}

# List containing pictures with labels
main_database = []

def load_databases(path):
    global main_database
    loadf = open(path, "rb")
    main_database = pickle.load(loadf)
    loadf.close()

#############
# CONFIGURED PARAMETERS

load_path = path_d['MainDatabase'] + "\\maindata_20171028_225055_6.txt"
model_load_path = path_d['Model'] + "\\model_20171028_224634.hdf5"
used_labels = ['Car', 'Background', 'Pedestrian', 'Van', 'Truck', 'Cyclist']

picture_shape = (64, 64)
picture_array_shape = (64, 64, 3)

train_split = 0.6
valid_split = 0.25
batch_size = 5


load_databases(load_path)

#############
# CALCULATED PARAMETERS

num_labels = len(used_labels)
nb_samples = len(main_database)
nb_train_samples = math.floor(nb_samples * train_split)
nb_valid_samples = math.floor(nb_samples * valid_split)
nb_test_samples = nb_samples - nb_valid_samples - nb_train_samples

test_database = main_database[nb_train_samples + nb_valid_samples :]

test_x = np.array([row[0].reshape(64,64,3) for row in test_database])
test_y = np.array([row[1] for row in test_database])

test_database = None

model = models.load_model(model_load_path)


prediction = model.predict(test_x[0:batch_size], batch_size=batch_size, verbose=1)

for i,item in enumerate(prediction):
    pred_str = ""
    for j in range(len(item)):
        pred_str += "%.2f %s," % (item[j], used_labels[j])
    print("Prediction: %.2f car, %.2f background." % (item[0], item[1]))
    label_index = np.where(np.asarray(test_y[i]) == 1)[0][0]
    print("Truth: %s\n" % (used_labels[label_index]))
    pic_array = test_x[i]
    img = Image.fromarray(pic_array.astype(np.uint8), 'RGB')
    img.show()


#plot_model(model, to_file='model.png')
