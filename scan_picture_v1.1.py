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

#CONFIG
cfg_path = path_d['Wholepics'] + "\\000095.png"
#cfg_model_load_path = path_d['Model'] + "\\model_20171109_155748.hdf5"

used_labels = ['Car', 'Background', 'Pedestrian', 'Van', 'Truck', 'Cyclist']

cfg_debug = True
cfg_use_latest_model = True
cfg_model_pic_size = 64
cfg_window_size = 48
cfg_step = 24
cfg_step_tolerance = 5 # step can be lowered by this amount at most, to reduce number of cols and rows not scanned

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

# Create plot
fig,ax = plt.subplots(2)
ax[0].imshow(np.asarray(pic))

n = 0
scale = cfg_model_pic_size / cfg_window_size
pic_scaled = pic.resize((int(pic.size[0] * scale), int(pic.size[1] * scale)), resample=5)
step_scaled = cfg_step * scale

#fit step
step_tries = [(((pic_scaled.size[0]-cfg_window_size) % (step_scaled-st)) +
               ((pic_scaled.size[1]-cfg_window_size) % (step_scaled-st))) for st in range(cfg_step_tolerance)]
step_scaled -= int(np.argmin(step_tries))

total_steps = int((pic_scaled.size[0] - cfg_model_pic_size + 1) / step_scaled * (pic_scaled.size[1] - cfg_model_pic_size + 1) / step_scaled)

x_max = int((pic_scaled.size[0] - cfg_model_pic_size + 1))
y_max = int((pic_scaled.size[1] - cfg_model_pic_size + 1))
x_num = int(np.floor(x_max/step_scaled))
y_num = int(np.floor(y_max/step_scaled))
label_array = np.full((y_num, x_num), fill_value=-1, dtype='int8') #imshow shows array transposed

bbox_x = np.linspace(0, x_max, x_num, dtype='uint16')
bbox_y = np.linspace(0, y_max, y_num, dtype='uint16')

for j,y in enumerate(bbox_y):
    input_row = np.zeros((len(bbox_x),cfg_model_pic_size,cfg_model_pic_size, 3))

    for i,x in enumerate(bbox_x):
        n += 1

        bbox = [x, y, x + cfg_model_pic_size, y + cfg_model_pic_size]  # left, top, right, bottom coordinates
        pic_part = pic_scaled.crop(bbox)
        input_row[i] = np.array(pic_part.getdata(), dtype='uint8').reshape(cfg_model_pic_size, cfg_model_pic_size, 3)

    prediction = model.predict(input_row, batch_size=input_row.shape[0], verbose=0)

    for k,pred in enumerate(prediction):
        if np.max(pred)>0.6:
            label_array[j, k] = np.argmax(pred)

    sys.stdout.write("\rFinding objects: " + str(n) + " / " + str(total_steps))

    if cfg_debug:
        ax[0].clear()
        ax[0].imshow(np.asarray(pic))
        rect = patches.Rectangle((0, j * cfg_step), x_num*cfg_step+cfg_window_size, cfg_window_size, linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
        ax[1].clear()
        ax[1].imshow(label_array, cmap='plasma')
        plt.pause(0.1)


end_time = datetime.datetime.now()
time_elapsed = (end_time - start_time).total_seconds()
print("\nTime: " + str(time_elapsed))

ax[1].imshow(label_array, cmap='plasma')
plt.show(block=True)