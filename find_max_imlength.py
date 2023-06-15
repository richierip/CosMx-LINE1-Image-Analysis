import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


path = r"/home/peter/B10 CosMx Texture Images/"
gmax = 0
total_images = 0
total_length = 0
for root,dirs,files in os.walk(path):
    for file in reversed(files):
        total_images+=1
        if total_images % 5000 == 0:
            print(f"\nOn image {total_images}")
        try:
            im = Image.open(root+file)
        except:
            print(f"{file}")
            print("\nCan't open image for some reason\n")
        cmax = max(np.array(im).shape)
        total_length += cmax
        if cmax > gmax:
            print(f"\nNew max is {cmax}. From {file}")
            gmax = cmax
print(f"\nDone. {gmax}")
print(f"Average cell bounding box length (larger of two per cell) = {total_length/total_images}")
