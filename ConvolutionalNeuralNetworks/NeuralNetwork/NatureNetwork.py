from os import environ
import os
from os import chdir
from keras.preprocessing.image import ImageDataGenerator


workdir = r"C:\Users\paulk\Desktop\newMlIm\data2\train"
chdir(workdir)

#define how the images are modified
trainDg = ImageDataGenerator(rescale=1.0/255, zoom_range=[1.0,1.25], width_shift_range=0.1,height_shift_range=0.1, fill_mode='reflect')