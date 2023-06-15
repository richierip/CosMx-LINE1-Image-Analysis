import os
import numpy as np
# import argparse
import cv2
from matplotlib import pyplot as plt
from generate import *
from math import ceil
from PIL import Image
import largestinteriorrectangle as lir
import copy
import scipy.fft as fft


## Get parser arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--image_path", required=True, type=str, help="path of image you want to quilt")
# parser.add_argument("-b", "--block_size", type=int, default=20, help="block size in pixels")
# parser.add_argument("-o", "--overlap", type=int, default=1.0/6, help="overlap size in pixels (defaults to 1/6th of block size)")
# parser.add_argument("-s", "--scale", type=float, default=4, help="Scaling w.r.t. to image size")
# parser.add_argument("-n", "--num_outputs", type=int, default=1, help="number of output textures required")
# parser.add_argument("-f", "--output_file", type=str, default="output.png", help="output file name")
# parser.add_argument("-p", "--plot", type=int, default=1, help="Show plots")
# parser.add_argument("-t", "--tolerance", type=float, default=0.1, help="Tolerance fraction")

# args = parser.parse_args()

def show_fourier(nuc,mask,cid,fov):
    if f"{fov}_{cid}" not in ['7_3978', '7_732','7_734','7_675','7_684','19_653','19_675','19_782','19_806']:
        return None
    print(f"\nCurrently showing{fov}_{cid}\n")
    img_f = fft.fft2(nuc)
    display_img = 20*np.log(np.abs(fft.fftshift(img_f)))
    display_img = display_img * (255/np.max(display_img))


    # cv2.imshow(f"FOURIER FOV={fov}, CID={cid}",display_img.astype(np.uint8))
    # cv2.moveWindow(f"FOURIER FOV={fov}, CID={cid}", 300,450)
    # cv2.imshow(f"SPATIAL FOV={fov}, CID={cid}",nuc.astype(np.uint8))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    nuc8 = (nuc/256).astype(np.uint8)
    print(f"datatype is {nuc8.dtype}, max is {np.max(nuc8)}")
    plt.imshow(nuc8,vmin=0,vmax=50)
    plt.title(f"\nSpatial {fov}_{cid}\n")
    plt.show()
    plt.imshow(display_img)
    plt.title(f"\nFourier{fov}_{cid}\n")
    plt.show()

if __name__ == "__main__":
    # Start the main loop here
    # path = r"C:\Users\prich\Downloads\clipped_nuc.png"
    CID=675
    FOV=7
    path = r"C:\Users\prich\Downloads\\" + f"{FOV}_{CID}" + "_nuclearDapi_16bit.png"
    block_size = 6
    scale = 3
    num_outputs = 1
    create_plot = True
    output_file = f"{FOV}_{CID}_synthesis.png"
    tolerance = 0.3
    save_to_file = True
    # overlap = 6
    print(f"Using plot {create_plot}")
    # Set overlap to 1/6th of block size
    # if overlap > 0:
    # 	overlap = int(block_size*args.overlap)
    # else:
    overlap = int(2)

    # Get all blocks
    # image = cv2.imread(path) /256
    image = Image.open(path)
    image = np.asarray(image)
    print(f'Image type is {image.dtype}, shape is {image.shape} and maxint is {np.max(image)}')

    mask = copy.copy(image.astype(np.bool8))
    # mask[mask>1]=1
    rect = lir.lir(mask)
    ys = rect[0]
    ye = rect[0]+rect[2]
    xs = rect[1]
    xe = rect[1] +rect[3]
    print(image.shape)
    print(f'{xs}:{xe},{ys}:{ye}')

    subset = image[xs:xe,ys:ye]
    subset8 = (subset/256).astype(np.uint8)
    plt.imshow(subset8,vmin=0,vmax=50)
    plt.show()
    image = subset
    # import IPython
    # IPython.embed()
    # plt.imshow(image)
    # plt.show()
    # image = (image/256)
    # print(f'Image type is {image.dtype} and maxint is {np.max(image)}')
    # plt.imshow(image)
    # plt.show()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0
    print("Image size: ({}, {})".format(*image.shape[:2]))
    H, W = image.shape[:2]
    # outH, outW = int(scale*H), int(scale*W)
    outH, outW = int(120), int(120)

    for i in range(num_outputs):
        textureMap = generateTextureMap(image, block_size, overlap, outH, outW, tolerance)
        if create_plot:
            nuc8 = (textureMap/256).astype(np.uint8)
            plt.imshow(nuc8, vmin=0,vmax=50)
            plt.show()
            plt.imshow((image/256).astype(np.uint8), vmin=0,vmax=50)
            plt.show()

    show_fourier(textureMap,None,CID,FOV)
    # Save
    textureMap = (255*textureMap).astype(np.uint8)
    textureMap = cv2.cvtColor(textureMap, cv2.COLOR_RGB2BGR)
    if save_to_file:
        cv2.imwrite(output_file, textureMap)
        print("Saved output to {}".format(output_file))

