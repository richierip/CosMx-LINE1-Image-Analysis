import numpy as np
import os
from PIL import Image
import math
import IPython
import matplotlib.pyplot as plt


def pad(path, target_path, maxL):
    count=0
    for root,dirs,files in os.walk(path):
        for file in files:
            count+=1
            if count%5000==0:
                print(count)
            im = Image.open(root+file)
            npim = np.array(im)
            x,y = npim.shape
            xt = (maxL-x)/2
            yt = (maxL-y)/2
            new_arr = np.pad(npim,((math.floor(xt),math.ceil(xt)),(math.floor(yt),math.ceil(yt))), mode='constant',constant_values=0)
            new_im = Image.fromarray(new_arr)
            # continue
            new_im.save(target_path+file)
    print("Done.")

def view_single_cell_image(root, fov, cid):
    im = Image.open(f"{root}{fov}_{cid}_nuclearDapi_16bit.png")
    npim = np.array(im)
    plt.imshow(npim)
    plt.show()

def correct_single_cell_image(root, fov, cid):
    im = Image.open(f"{root}{fov}_{cid}_nuclearDapi_16bit.png")
    npim = np.array(im)
    c,d,a,b = input("\nNumbers>").split(",")
    npim = npim[int(a):int(b),int(c):int(d)]
    new_im = Image.fromarray(npim)
    plt.imshow(npim)
    plt.show()
    new_im.save(f"{root}{fov}_{cid}_nuclearDapi_16bit.png")
    return True


def main():
    path = r"/home/peter/B10 CosMx Texture Images/"
    target_path = r"/home/peter/B10 CosMx Texture Images (Padded)/"
    max_image_length = 200
    fov = 9
    cid = 72
    # view_single_cell_image(path, fov,cid)
    # correct_single_cell_image(path, fov,cid)
    pad(path,target_path, max_image_length)

if __name__ == "__main__":
    main()