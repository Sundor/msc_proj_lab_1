import os
import glob
from PIL import Image
import numpy as np
import math
import datetime
import sys
from keras import models
import matplotlib.pyplot as plt
import matplotlib.patches as patches


path_d = {'MainDatabase': "H:\\Databases\\processed_database",
          'Model': "H:\\Databases\\model_weights",
          'Wholepics': "H:\\Downloads\\data_object_image_2\\testing\\image_2"}

def coord_pic_array(x,y, pic, window, step):
    ar_x = math.ceil((pic.size[0] - window) / step)
    ar_y = math.ceil((pic.size[1] - window) / step)
    return ar_x, ar_y

#CONFIG
cfg_path = path_d['Wholepics'] + "\\000095.png"
cfg_model_load_path = path_d['Model'] + "\\model_20171109_155748.hdf5"

used_labels = ['Car', 'Background', 'Pedestrian', 'Van', 'Truck', 'Cyclist']

cfg_debug = False
cfg_use_latest_model = True
cfg_model_pic_size = 64
cfg_window_size = 64
cfg_step = 32

# save time and date of start
start_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
start_time = datetime.datetime.now()


# MAIN
if cfg_use_latest_model:
    model_files = glob.glob(path_d['Model'] + "\\*.hdf5")
    model_load_path = max(model_files, key=os.path.getctime)
else:
    model_load_path = cfg_model_load_path

model = models.load_model(model_load_path)

pic = Image.open(cfg_path)
label_array = np.full((int((pic.size[1] - cfg_window_size + 1)/cfg_step),
                       int((pic.size[0] - cfg_window_size + 1)/cfg_step)),
                      fill_value=-1, dtype='int8') #imshow shows array transposed

fig,ax = plt.subplots(3)
ax[0].imshow(np.asarray(pic))

# Create a Rectangle patch
#rect = patches.Rectangle((10,20),64,64,linewidth=1,edgecolor='r',facecolor='none')
#ax.add_patch(rect)

total_steps = int((pic.size[0] - cfg_window_size + 1) / cfg_step * (pic.size[1] - cfg_window_size + 1) / cfg_step)
i = 0
for x in range(0,int((pic.size[0] - cfg_window_size + 1)/cfg_step)):
    for y in range(0, int((pic.size[1] - cfg_window_size + 1)/cfg_step)):
        i += 1

        pic_x, pic_y = int(x * cfg_step), int(y * cfg_step)

        bbox = [pic_x, pic_y, pic_x+cfg_window_size, pic_y+cfg_window_size] #left, top, right, bottom coordinates
        pic_part = pic.crop(bbox)
        pic_input = pic_part.resize((cfg_model_pic_size, cfg_model_pic_size), resample=5)
        input = np.array(pic_input.getdata(), dtype='uint8').reshape(1, 64, 64, 3)

        prediction = model.predict(input, batch_size=1, verbose=0)

        label_array[y, x] = np.argmax(prediction)
        sys.stdout.write("\rFinding objects: " + str(i) + " / " + str(total_steps) + " Prediction: " + used_labels[np.argmax(prediction)])

        if cfg_debug:
            ax[1].clear()
            ax[1].imshow(label_array, cmap='plasma')
            ax[2].clear()
            ax[2].imshow(input[0])

        plt.pause(0.1)

end_time = datetime.datetime.now()
time_elapsed = (end_time - start_time).total_seconds()
print("\nTime: " + str(time_elapsed))

ax[1].imshow(label_array, cmap='plasma')
plt.show(block=True)