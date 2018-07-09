import numpy as np
import scipy.ndimage as sp
import os

workDir = os.chdir(path=r"C:\Users\paulk\OneDrive - University of Toronto\engsci 1t9\python_scripts\ML_Repo\CNN_Scripts")
image = sp.imread(fname="redEyeTreeFrog.jpg", mode="RGB")
print(image)
print(image.shape)
newImage = image[:,:,1]

print(newImage)


