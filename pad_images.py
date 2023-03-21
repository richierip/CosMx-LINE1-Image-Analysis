import numpy as np
import os
from PIL import Image
import math
import IPython

path = r"/home/peter/data2/CosMx Texture Images/"
target_path = r"/home/peter/data2/CosMx Texture Images (Padded)/"
max_image_length = 170
count=0
for root,dirs,files in os.walk(path):
    for file in files:
        count+=1
        if count%5000==0:
            print(count)
        im = Image.open(root+file)
        npim = np.array(im)
        x,y = npim.shape
        xt = (170-x)/2
        yt = (170-y)/2
        new_arr = np.pad(npim,((math.floor(xt),math.ceil(xt)),(math.floor(yt),math.ceil(yt))), mode='constant',constant_values=0)
        new_im = Image.fromarray(new_arr)
        new_im.save(target_path+file)
print("Done.")