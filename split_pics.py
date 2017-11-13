import os
from PIL import Image
import csv
import gc
import numpy as np
import random
import math
import time
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

num_car = 0
num_van = 0
num_truck = 0
num_dontcare = 0

path_d = {'train_pics': "H:\\Downloads\\data_object_image_2\\training\\image_2",\
          'train_labels': "H:\\Downloads\\data_object_label_2\\training\\label_2",\
          'Car': "H:\\Databases\\data_object_cars",\
          'Van': "H:\\Databases\\data_object_vans",\
          'Truck': "H:\\Databases\\data_object_trucks",\
          'Pedestrian': "H:\\Databases\\data_object_pedestrians",\
          'Person_sitting': "H:\\Databases\\data_object_sittingpeople",\
          'Cyclist': "H:\\Databases\\data_object_cyclists",\
          'Tram': "H:\\Databases\\data_object_trams",\
          'Misc': "H:\\Databases\\data_object_misc",\
          'DontCare': "H:\\Databases\\data_object_dontcare",\
          'Background': "H:\\Databases\\data_object_backgrounds"}

#function for processing every camera picture in a folder
def cut_pics(path_d):
    i = 0
    start_time = time.time()
    files = os.listdir(path_d['train_pics'])
    for file in files:
        if file[-3:] == "png":
            pic = Image.open(path_d['train_pics'] + "\\" + file)
            labels = csv.reader(open(path_d['train_labels'] + "\\" + file[:-4] + ".txt"), delimiter=' ')
            print("Cutting " + file)
            cut_pic(pic, labels, file[0:-4], path_d)
            i += 1
            pic.close
            # Partial processing for testing:
            if(i>20):
               break

    print("Runtime: " + str(int(time.time() - start_time)) + " sec.")

    # garbage collector
    gc.collect()


#function for adding pictures of objects to the database
def cut_pic(pic, labels, picname, path_d):
    num_d = {'Car': 0, 'Van': 0, 'Truck': 0, 'Pedestrian': 0, 'Person_sitting': 0, 'Cyclist': 0, 'Tram': 0, 'Misc': 0,
             'DontCare': 0}

    for row in labels:
        type = row[0]
        bbox = [int(float(x)) for x in row[4:8]] #left, top, right, bottom coordinates
        pic_part = pic.crop(bbox)
        num_d[type] += 1
        pic_part.save(path_d[type] + "\\" + type + "_" + picname + "_" + str(num_d[type]) + ".png", "PNG")

def cut_background(path_d, num_per_pic, size, num_to_create=0):
    num_pic = 0
    start_time = time.time()
    files = os.listdir(path_d['train_pics'])
    for file in files:
        if file[-3:] == "png":
            pic = Image.open(path_d['train_pics'] + "\\" + file)

            #create a martix with 1 element for each pixel
            label_map = np.zeros((pic.size[1], pic.size[0]), dtype=np.uint8)

            labels = csv.reader(open(path_d['train_labels'] + "\\" + file[:-4] + ".txt"), delimiter=' ')

            #that element is 1, if it is part of a label, 0 if not - a part of image is background, if it does not contain any part of labelled fields
            #created bg parts will be also marked with 1 values
            for row in labels:
                bbox = [int(float(x)) for x in row[4:8]]  # left, top, right, bottom coordinates
                label_map[bbox[1]:bbox[3], bbox[0]:bbox[2]] = np.full((bbox[3]-bbox[1], bbox[2]-bbox[0]), 255, dtype=np.uint8)
            label_map_pic = Image.fromarray(label_map)

            n_bg = 0
            while(n_bg < num_per_pic):
                bg_size = random.randint(math.ceil(size / 2),size * 2)
                xmin = random.randint(0,pic.size[0]-bg_size-1)
                ymin = random.randint(0, pic.size[1] - bg_size - 1)
                xmax = xmin + bg_size
                ymax = ymin + bg_size
                bbox_bg = (xmin,ymin,xmax,ymax)
                if 255 in label_map[ymin:ymax, xmin:xmax]:
                    sys.stdout.write(".")
                    next
                else:
                    bg_pic = pic.crop(bbox_bg)
                    bg_pic = bg_pic.resize((size, size), resample=5)
                    n_bg += 1
                    sys.stdout.write("\rBackground #"+ str(n_bg) +" found in picture " + file[0:-4] +
                                     "  ( " + str(num_pic*num_per_pic+n_bg) + " / " + str(num_to_create) + " )") # +": " + str(bbox_bg))
                    bg_pic.save(path_d['Background'] + "\\Background_" + file[0:-4] + "_" + str(n_bg) + ".png", "PNG")

                    if num_pic*num_per_pic+n_bg>=num_to_create:
                        pic.close
                        print("\nRuntime: " + str(int(time.time() - start_time)) + " sec.")
                        return
            pic.close

        num_pic += 1

    print("\nRuntime: " + str(int(time.time()-start_time)) + " sec.")


#create uniform pictures for training database
def pre_process(path, size, num_to_create=0):
    files = os.listdir(path)
    num = 0
    if num_to_create == 0:
        num_total = len(files)
    else:
        num_total = num_to_create

    for i,file in enumerate(files):
        sys.stdout.write("\rPreprocessing pictures: " + str(i) + " / " + str(num_total)) #debug
        type = file.split('_')[0]
        if file[-3:] == "png":
            pic = Image.open(path + "\\" + file)
            if pic.size[0]<(size/4) or pic.size[1]<(size/4):
                continue    # skip extremely small pictures
            #CALCULATE SIZE
            if pic.size[0] > pic.size[1]:
                # picture is broad
                height = size
                width = int((height / pic.size[1]) * pic.size[0])
            elif pic.size[0] < pic.size[1]:
                # picture is tall
                width = size
                height = int((width / pic.size[0]) * pic.size[1])
            else:
                width = size
                height = size

            pic_small = pic.resize((width,height), resample=5)

            #split picture into multiple square shaped parts, and save them
            if height == width:
                pic_small.save(path + "//processed//" + type + "_part_" + str(num) + ".png", "PNG")
                num += 1
                if num_to_create != 0 and num >= num_to_create:
                    break

            elif height < width:
                num_parts = math.ceil(width/size)
                deltaw = math.floor((width - size) / (num_parts - 1))
                for i in range(0, num_parts):
                    bbox = [(i*deltaw), 0, (size+i*deltaw), (size)] #left, top, right, bottom coordinates
                    pic_part = pic_small.crop(bbox)
                    pic_part.save(path + "//processed//" + type + "_part_" + str(num) + ".png", "PNG")
                    num += 1
                    if num_to_create!=0 and num >= num_to_create:
                        break

            else:
                num_parts = math.ceil(height / size)
                deltah = math.floor((height - size) / (num_parts - 1))
                for i in range(0, num_parts):
                    bbox = [0, (i * deltah), (size), (size + i * deltah)]  # left, top, right, bottom coordinates
                    pic_part = pic_small.crop(bbox)
                    pic_part.save(path + "//processed//" + type + "_part_" + str(num) + ".png", "PNG")
                    num += 1
                    if num_to_create!=0 and num >= num_to_create:
                        break

        if num_to_create!=0 and num >= num_to_create:
            break
    print("\n")





#MAIN------------------------------------------------------------------------
#cut_pics(path_d)
#cut_background(path_d, num_per_pic=8, size=64, num_to_create=60000)
pre_process(path_d['Car'], 64)
pre_process(path_d['Pedestrian'], 64)
#pre_process(path_d['Cyclist'], 64)
#pre_process(path_d['Person_sitting'], 64)
#pre_process(path_d['Truck'], 64)
pre_process(path_d['Van'], 64)