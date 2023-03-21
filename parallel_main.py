'''
Compare nuclear intensity vs LINE1 transcripts per cell, run correlation / regression
    This file compiles the data for plotting
Peter Richieri 
MGH Ting Lab 
11/7/22
'''
import numpy as np
import tifffile # for cell masks
from matplotlib import image # for DAPI .jpg
import pandas as pd
import time
import os
import copy
import math
import matplotlib.pyplot as plt
from skimage.draw import ellipse # For roundness metrics
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
import feret_diameter
from skimage.feature import graycomatrix,graycoprops # Creating texture metrics
import pyfeats
from kymatio.numpy import Scattering2D
import matplotlib.pyplot as plt # checking out fft
import numpy
# import cv2
NUM_IMAGES_TO_SHOW = 10 # Was used for manual fourier transformed image viewing
import scipy.fft as fft
import sys
numpy.set_printoptions(threshold=sys.maxsize)

# Texture features take a long time, so these modules
#   add concurrency
from multiprocessing import Process, Lock, Pipe
from multiprocessing.connection import wait
DO_PARALLEL = False
ASSEMBLE = True
WRITE = True
SAVE_CELL_IMAGES = False
# SAVE_SYNTHETIC_IMAGES = r"/home/peter/data2/CosMx Quilted Texture lir=15b=10o=5/"
SAVE_SYNTHETIC_IMAGES = False
MAKE_NEW_TEXTURE = False
# USE_IMAGES_IN_FOLDER = False
USE_IMAGES_IN_FOLDER = '/home/peter/data2/CosMx Texture Images (Padded)/'
PATH_EXTENSION = 'nuclearDapi_16bit.png'
from PIL import Image
import texture_synthesis
# CELL_MASK_DATA = r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\RawData\CellLabels\CellLabels_F004.tif"
# COMPOSITE_IMAGE = r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\RawData\CellComposite\CellComposite_F004.jpg"

METADATA = os.path.normpath (r"../RawData/C4_R5042_S1_metadata_file.csv")
FOV_POSITIONS = os.path.normpath (r"../RawData/C4_R5042_S1_fov_positions_file.csv")
TRANSCRIPTS = os.path.normpath (r"../RawData/C4_R5042_S1_exprMat_file.csv")
CELLTYPING = os.path.normpath (r"../C4_napari/C4_napari/slide_C4_R5042_S1_Napari_metadata.csv")
MIN_DAPI_INTENSITY_THRESHOLD = 15
MIN_DAPI_AREA = 30 # In pixels
MIN_PANCK_THRESHOLD = 900
GLCM_DISTANCE = [2]
DOWNSAMPLE = int(math.pow(2,6)-1)
REMOVE_GLCM_ZEROES = False
FREQ1 = .1
FREQ2 = .05
# FOV_GLOBAL_X = int(-4972.22222222222)# FOV_GLOBAL_Y = int(144450)
# RESULTS_FILE = os.path.normpath (f"../2.22.23_synthetic_b=8_o=3_t.3.csv")
RESULTS_FILE = os.path.normpath (f"../3.21.23_padded_geometric_features_only.csv")



''' Get cell data as numpy array, convert to binary, and return. '''
def read_mask(path_to_label_mask, path_to_compartment_mask):
    print(f'Reading cell masks',end="   ")
    labels = tifffile.imread(path_to_label_mask)
    compartments = tifffile.imread(path_to_compartment_mask) # I think this might only read in the top channel anyway...

    # Nuclear compartment pixels have a 1 value. 0 is background, 2 is something else, 3 is cytoplasm
    #   so replace 2 and 3 with 0.
    compartments[compartments>1] = 0
    # Have to flip is around to be [x,y] instead of [y,x]

    return np.transpose(compartments*labels,(1,0))

''' Data only exists as composite .jpg. DAPI signal should be the B channel of the RGB data. '''
def extract_dapi_signal_composite(path_to_file):
    print(f'Reading composite',end="   ")
    raw_data = image.imread(path_to_file)
    # Also have to flip is around to be [x,y] instead of [y,x]
    return np.transpose(raw_data[:,:,2],(1,0))

def extract_dapi_signal_tif(path_to_file):
    print("Reading tif",end="   ")
    all_bands = tifffile.imread(path_to_file)
    return np.transpose(all_bands[4,:,:],(1,0)) # DAPI is the last channel in this data


''' Take the numpy array and remove any signal lower than the chosen threshold'''
# def drop_weak_signal(arr, lower_limit):
#     arr[arr < lower_limit] = 0
#     return arr

''' Read in metadata file and return data for the current fov'''
def read_metadata(path_to_file, fov = 4):
    meta = pd.read_csv(path_to_file)
    meta_fov = meta.loc[meta["fov"] == fov]
    return meta_fov

'''Subset numpy array to only data corresponding to a certain Cell_ID and return that data'''
def get_pixels_for_cell(nuclear_mask,dapi_only,fov, cell_id,metadata,fov_x,fov_y, max_local_y):

    row = metadata.loc[metadata["cell_ID"]==cell_id]
    local_center_x = int(row["CenterX_global_px"].values[0] - fov_x)
    local_center_y = int(max_local_y - (row["CenterY_global_px"].values[0] - fov_y))
    width = row["Width"].values[0]
    height = row["Height"].values[0]

    if USE_IMAGES_IN_FOLDER:
        try:
            im = Image.open(f'{USE_IMAGES_IN_FOLDER}{fov}_{cell_id}_{PATH_EXTENSION}')
            data = np.asarray(im)
            return data, None, (local_center_x, local_center_y)
        except:
            return None,None, (local_center_x, local_center_y)
            pass # Image was never created (cell area too small?)
    
    # Have to get individual windows out of nuclear mask AND dapi image
    #   For nuclear mask, need to extract pixels where the value equals the cell ID
    single_cell_mask = nuclear_mask[local_center_x-(width//2):local_center_x+(width//2),local_center_y-(height//2):local_center_y+(height//2)]
    # print(f'My unique vals are  {np.unique(single_cell_mask)}')
    
    # !!! IMPORTANT!!! Arrays are mutable, don't screw up the original array. Make a copy
    single_cell_mask_copy = copy.copy(single_cell_mask)
    single_cell_mask_copy[single_cell_mask_copy != int(cell_id)] = 0 # not sure about type here...might as well cast it
    single_cell_mask_copy[single_cell_mask_copy > 0] = 1 # now have 1s where this cell's nucleus is

    single_cell_dapi = dapi_only[local_center_x-(width//2):local_center_x+(width//2),local_center_y-(height//2):local_center_y+(height//2)]
    # if cell_id == 20 or cell_id==23 or cell_id==31 or cell_id==32 or cell_id==1119:
    #     print(row.loc[:,["cell_ID","Area","CenterX_local_px","CenterY_local_px","Width","Height"]])
    #     print(f"X and Y {local_center_x} and {local_center_y}")
    return single_cell_mask_copy * single_cell_dapi, single_cell_mask_copy, (local_center_x,local_center_y)

''' Return a dictionary of the mean DAPI value for all cells
    Key: a cell ID          value: Mean of DAPI counts per pixel for that cell'''
def mean_for_all_cells(nuclear_dapi,cell_id, cell_lookup, fov, lock):
    
    # dump zeroes, then check length( i.e. testing the number of pixels in the nuclear mask)
    current = np.ndarray.flatten(nuclear_dapi)
    current = current[current !=0]
    if len(current)<MIN_DAPI_AREA:
        if len(current) >0: 
            # lock.acquire()
            print(f'FOV {fov} CID {cell_id} nucleus is only {len(current)} pixels, dropping this one.')
            # lock.release()
        return cell_lookup
    
    ## Checking a specific case
    # if cell_id == 1661 or cell_id == 1663:
    #     print(f'Nucleus of CID {cell_id} is  {len(current)} pixels')
    cell_lookup[f"{fov}_{cell_id}"] += [np.mean(current),len(current)]

    # other_params[f"{fov}_{cell_id}"] += [len(current)] # Adding area
    # if cell_id >2250:   
    #     print(f"\nCID {cell_id} shape is {current.shape}, max is {np.max(current)}, min is {np.min(current)}\n{current} mean is {cell_lookup[str(fov)+'_'+str(cell_id)]}")
    return cell_lookup#, other_params

def get_coordinate_conversions(fov):
    meta = pd.read_csv(FOV_POSITIONS)
    meta_fov = meta.loc[meta["fov"] == fov]
    return int(meta_fov["x_global_px"].values[0]), int(meta_fov["y_global_px"].values[0])

def compute_geometric_features(nuclear_dapi, cell_id, fov, cell_dict):

    binary_dapi = copy.copy(nuclear_dapi).astype(np.uint8)
    binary_dapi[binary_dapi > 0] = 1
    # print(f"\nComputing roundness...")
    label_img = label(binary_dapi)
    # regions = regionprops(label_img)

    # fig, ax = plt.subplots()
    # ax.imshow(binary_dapi, cmap=plt.cm.gray)

    # for props in regions:
    #     y0, x0 = props.centroid
    #     orientation = props.orientation
    #     x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
    #     y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
    #     x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
    #     y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

    #     ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    #     ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    #     ax.plot(x0, y0, '.g', markersize=15)

    #     minr, minc, maxr, maxc = props.bbox
    #     bx = (minc, maxc, maxc, minc, minc)
    #     by = (minr, minr, maxr, maxr, minr)
    #     ax.plot(bx, by, '-b', linewidth=2.5)

    # ax.axis((0, 600, 600, 0))
    # plt.show()

    selected_features = ("label", "area","area_convex", "area_filled","axis_major_length","axis_minor_length", 
                "eccentricity","feret_diameter_max","perimeter","solidity", "extent", "euler_number",
                "moments_hu","moments_weighted_hu")
    df = regionprops_table(label_img, nuclear_dapi, 
                           properties = selected_features)
    area = df['area']
    pos = np.where(area == max(area))[0][0]
    strlabel = int(df['label'][pos])
    a = df['axis_major_length'][pos] / 2; b = df['axis_minor_length'][pos] / 2
    ellipse_area = math.pi*a*b
    # Using one of Ramanujan's formulas https://www.mathsisfun.com/geometry/ellipse-perimeter.html
    ellipse_perimeter = math.pi*(3*(a+b) - math.sqrt((3*a +b) * (a+3*b)))

    # skimage doesn't have a utility to get minimum feret length built in yet, so I'm using this
    #   code found on their github pages. Just grabbing minimum feret length, even though it does
    #   both, so that max feret length is reproducible with skimage. The numbers seemed to be slightly different.
    min_feret = feret_diameter.get_min_max_feret_from_labelim(label_img)[strlabel][0]

    hu_moments = [] ; weighted_hu_moments = []
    for i in range(7):
        hu_moments.append(df['moments_hu-'+str(i)][pos])
    for i in range(7):
        weighted_hu_moments.append(df['moments_weighted_hu-'+str(i)][pos])

    cell_dict[f"{fov}_{cell_id}"] += [df['area'][pos],df['area_convex'][pos],df['area_filled'][pos],df['axis_major_length'][pos],
                                      df['axis_minor_length'][pos], ellipse_area, ellipse_perimeter,
                                      df['eccentricity'][pos],df['feret_diameter_max'][pos],min_feret,
                                      df['perimeter'][pos],df['solidity'][pos],df['extent'][pos],df['euler_number'][pos]]  
    cell_dict[f"{fov}_{cell_id}"] += [np.array_str(np.array(hu_moments)),
                                      np.array_str(np.array(weighted_hu_moments))]
    return cell_dict

def add_glcm_metrics(nuclear_dapi, cell_id,fov, other_params, distance_range,angles_range):

    def create_comatrix(img,distances,angles, max_level = int(math.pow(2,16))):
        downsampled_max_int = DOWNSAMPLE
        img = img / max_level * downsampled_max_int # downsample to n bit
        img = img.astype(np.uint8)
        # print(f'Checking shape of array before computing texture:\n {nuclear_dapi.shape}')
        # print(f'Type is {img.dtype}')
        return graycomatrix(img,distances=distances,angles=angles,levels = downsampled_max_int+1, symmetric=False)
    glcm = create_comatrix(nuclear_dapi,distance_range,angles_range)
    # print(f'glcm length is {len(glcm)} shape is {glcm.shape}. cell shape is {nuclear_dapi.shape}')
    
    # Average across distances and angles, and the remove interactions with the background
    # graycoprops seems to do this, but I'll just do it first to make it easier to
    #   remove the 0 gray level interactions (array is always 2D here)
    if glcm.shape[2] ==1 and glcm.shape[3] ==1:
        # avoid taking the mean if there are only 2 meaningful dimensions already
        #   i.e., only one distance and one angle chosen
        if REMOVE_GLCM_ZEROES:
            glcm[0,:,0,0] = 0 
            glcm[:,0,0,0] = 0 
        glcm_reduced = glcm
    else:     
        glcm_reduced = np.mean(glcm, axis= (2,3))
        if REMOVE_GLCM_ZEROES:
            glcm_reduced[0,:] = 0
            glcm_reduced[:,0] = 0
        # Have to expand the dimensions again since graycoprops expects a 4D array
        glcm_reduced = np.expand_dims(glcm_reduced, axis = (2,3))


    correlation = graycoprops(glcm_reduced, 'correlation')[0,0]
    dissimilarity = graycoprops(glcm_reduced, 'dissimilarity')[0,0]
    homogeneity = graycoprops(glcm_reduced, 'homogeneity')[0,0]
    ASM = graycoprops(glcm_reduced, 'ASM')[0,0]
    energy = graycoprops(glcm_reduced, 'energy')[0,0]
    contrast = graycoprops(glcm_reduced, 'contrast')[0,0]

    # other_params[f"{fov}_{cell_id}_texture-correlation"] = correlation
    # other_params[f"{fov}_{cell_id}_texture-dissimilarity"] = dissimilarity
    # other_params[f"{fov}_{cell_id}_texture-homogeneity"] = homogeneity
    # other_params[f"{fov}_{cell_id}_texture-ASM"] = ASM
    # other_params[f"{fov}_{cell_id}_texture-energy"] = energy
    # other_params[f"{fov}_{cell_id}_texture-contrast"] = contrast
    other_params[f"{fov}_{cell_id}"] += [correlation,dissimilarity,homogeneity,ASM,energy,contrast]
    return other_params

def add_gabor_metrics(nuclear_dapi, nuclear_mask, cell_id, fov, cell_dict):
    spectrograms, features, labels = pyfeats.gt_features(nuclear_dapi,nuclear_mask, deg=4, freq=[FREQ1, FREQ2])

    f1_means = []
    f1_std = []
    f2_means = []
    f2_std = []
    for pos, l in enumerate(labels):
        if l.endswith(f'freq_{FREQ1}_mean'): 
            f1_means.append(features[pos])
        elif l.endswith(f'freq_{FREQ2}_mean'):
            f2_means.append(features[pos])
        elif l.endswith(f'freq_{FREQ1}_std'):
            f1_std.append(features[pos])
        else:
            f2_std.append(features[pos])
    # other_params[f"{fov}_{cell_id}_gabor{FREQ1}_mean"] = np.mean(f1_means)
    # other_params[f"{fov}_{cell_id}_gabor{FREQ1}_std"] = np.mean(f1_std)
    # other_params[f"{fov}_{cell_id}_gabor{FREQ2}_mean"] = np.mean(f2_means)
    # other_params[f"{fov}_{cell_id}_gabor{FREQ2}_std"] = np.mean(f2_std)
    cell_dict[f"{fov}_{cell_id}"] += [np.mean(f1_means),np.mean(f1_std),np.mean(f2_means),np.mean(f2_std)]
    return cell_dict

def add_wavelet_metrics(nuclear_dapi, nuclear_mask, cell_id, fov, cell_dict):
    features, labels = pyfeats.wp_features(nuclear_dapi, nuclear_mask, wavelet='cof1', maxlevel=3)

def wavelet_scattering(nuclear_dapi, cell_id, fov, cell_dict):
    scattering = Scattering2D(J=2, shape=nuclear_dapi.shape)

    coefficients = scattering(nuclear_dapi.astype(np.float32))
    feat1 = np.mean(coefficients,axis=(1,2))
    feat2 = np.mean(coefficients,axis=(0,1))
    feat3 = np.mean(coefficients,axis=(0,2))

    cell_dict[f"{fov}_{cell_id}"] += [np.array_str(feat1),np.array_str(feat2),np.array_str(feat3)]
    # Here's how you get the information back
    #f2re = np.fromstring(f2str.strip('[]'), sep= ' ', dtype = np.float32)
    return cell_dict

def plot_frequency_domain(f_series):
    f1,f2 = f_series.shape
    plt.imshow(abs(f_series))
    plt.show()
    exit()

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
    exit()
    global NUM_IMAGES_TO_SHOW
    NUM_IMAGES_TO_SHOW -=1
    if NUM_IMAGES_TO_SHOW ==0:
        exit()

def fourier_stats(nuclear_dapi, nuclear_mask, cell_id, fov, other_params):
    F,features,labels = pyfeats.fps(nuclear_dapi,nuclear_mask)

    print(f"My features are {features}")
    print(f"My labels are {labels}")
    print(f"F is {F}")
    print(f"Dapi spatial domain shape: {nuclear_dapi.shape}")
    print(f"Frequency domain shape: {F.shape}")
    plot_frequency_domain(F)
    
def add_counts_for_fov(cell_dictionary, fov, mask_tuple, composite_path, lock):
    cell_mask_path = mask_tuple[0]; compartment_mask_path = mask_tuple[1]
    
    global_nuclear_mask = read_mask(cell_mask_path, compartment_mask_path)
    dapi_only = extract_dapi_signal_tif(composite_path)
    max_X = dapi_only.shape[0]; max_Y = dapi_only.shape[1]
    print(f"\ndapi shape is {dapi_only.shape}, max is {np.max(dapi_only)}, min is {np.min(dapi_only)}")

    # viewer.add_image(dapi_only,name='DAPI')
    
    # Now make quantitative counts for each DAPI pixel
    # dapi_cells = drop_weak_signal(cell_mask * dapi_only,MIN_DAPI_INTENSITY_THRESHOLD)
    metadata = read_metadata(METADATA, fov)
    fov_global_X, fov_global_Y = get_coordinate_conversions(fov)

    # Texture generator input
    # distances_range = [1,3,5,7,9]
    distances_range = GLCM_DISTANCE
    angle_step = np.pi/2
    angle_end = np.pi * 2
    angles_range = np.arange(0,angle_end,angle_step)
    # angles_range = [0]
    # scattering_features1, scattering_features2, scattering_features3 = [],[],[]

    for cell_id in metadata["cell_ID"]:
        if cell_id %100 ==0:
            # lock.acquire()
            print(f'On fov {fov}, cell {cell_id}')    
            # lock.release    
        # print(f"My inputs are CID {cell_id} ,fov {fov}")
        nuclear_dapi, nuclear_mask, coords = get_pixels_for_cell(global_nuclear_mask,dapi_only,fov, cell_id,metadata,fov_global_X,fov_global_Y,max_Y)
        if nuclear_dapi is None:
            print(f"fov {fov}, cell {cell_id} probably not in image library")
            # Texture can't be synthesized for this cell, move on
            continue
        
        cell_dictionary[f"{fov}_{cell_id}"] = [f"{fov}_{cell_id}",str(fov),str(cell_id),coords[0],coords[1]] # Add XY to make it easier to find cells of interest
        # other_params[f"{fov}_{cell_id}"] = other_params[f"{fov}_{cell_id}"].append(coords[1])
        # compute_geometric_features(dapi_cells)
        # other_params = add_gabor_metrics(nuclear_dapi,nuclear_mask, cell_id, fov, other_params)
        if SAVE_CELL_IMAGES:
            try:
                if np.any(nuclear_dapi):
                    im = Image.fromarray(nuclear_dapi)
                    im.save(SAVE_CELL_IMAGES+f"{fov}_{cell_id}_nuclearDapi_16bit.png")
            except:
                print(f"There was a problem saving fov {fov} CID {cell_id}")
        record_cell = True
        try:
            if np.any(nuclear_dapi):       
                if MAKE_NEW_TEXTURE:
                    nuclear_dapi = texture_synthesis.synthesize_texture(nuclear_dapi,fov,cell_id, SAVE_SYNTHETIC_IMAGES, block_size = 10)
                    nuclear_mask = None
                    if nuclear_dapi is None:
                        print(f"Cell {fov}_{cell_id} does not have an interior rectangle bigger than the required minimum square of 15")
                        record_cell = False
                cell_dictionary = compute_geometric_features(nuclear_dapi,cell_id,fov,cell_dictionary)
                # cell_dictionary = wavelet_scattering(nuclear_dapi,cell_id,fov,cell_dictionary)
                # cell_dictionary = add_glcm_metrics(nuclear_dapi,cell_id, fov, cell_dictionary, distances_range,angles_range)
                # cell_dictionary = add_gabor_metrics(nuclear_dapi,nuclear_mask, cell_id, fov, cell_dictionary)
                # show_fourier(nuclear_dapi, nuclear_mask, cell_id, fov)
                # fourier_stats(nuclear_dapi,nuclear_mask, cell_id, fov, other_params)
            else:
                # print(f'Empty list passed to texture creation code for cell {cell_id} in {fov}')
                pass
        except Exception as e:
            print(f'some other error occurred when trying to calculate texture for CID {cell_id} in fov {fov}')
            print(f'\n {e}')
            # exit()
        if record_cell:
            cell_dictionary = mean_for_all_cells(nuclear_dapi, cell_id, cell_dictionary,fov, lock)
    return cell_dictionary
    # return mean_for_all_cells(global_nuclear_mask,dapi_only,cell_dictionary,other_params, metadata,fov, fov_global_X, fov_global_Y, max_Y)

def add_columns(transcripts_path, meta_path, celltyping_path, cell_df):
    transcripts = pd.read_csv(transcripts_path)
    # Need this to get PanCK data
    metadata = pd.read_csv(meta_path) 
    typing = pd.read_csv(celltyping_path)

    cell_df.insert(5,"Line1_ORF1", "")
    cell_df.insert(6,"Line1_ORF2", "")
    cell_df.insert(7,"Line1_Combined", "")
    cell_df.insert(8,"Mean PanCK","")
    cell_df.insert(9,"Cancer?","")
    cell_df.insert(10,"Cell Width","")
    cell_df.insert(11,"Cell Height","")
    cell_df.insert(12,"Entire cell area","")
    cell_df.insert(13,"Diversity","")
    cell_df.insert(14,"Total transcript counts","")
    cell_df.insert(15,"Clustering","")
    cell_df.insert(16,"Cell type","")

    # cell_df.insert(17,"Correlation","")
    # cell_df.insert(18,"Dissimilarity","")
    # cell_df.insert(19,"Homogeneity","")
    # cell_df.insert(20,"ASM","")
    # cell_df.insert(21,"energy","")
    # cell_df.insert(22,"contrast","")
    for index,cell in cell_df.iterrows():
        cell_tx = transcripts[(transcripts["fov"] == int(cell["fov"])) & (transcripts["cell_ID"]== int(cell["local_cid"]))]
        cell_meta = metadata.loc[(metadata["fov"] == int(cell["fov"])) & (metadata["cell_ID"]== int(cell["local_cid"]))]
        # import IPython
        # IPython.embed()
        try:
            cell_typing = typing.loc[typing["cell_ID"]== 'c_4_'+ cell["fov"]+'_'+cell["local_cid"]]
        except:
            cell_typing = None
        panck = cell_meta["Mean.PanCK"].values[0]
        orf1 = cell_tx["LINE1_ORF1"].values[0]
        orf2 = cell_tx["LINE1_ORF2"].values[0]
        cell_df.at[index,"Line1_ORF1"] = orf1
        cell_df.at[index,"Line1_ORF2"] = orf2
        cell_df.at[index,"Line1_Combined"] = orf1 + orf2
        cell_df.at[index,"Mean PanCK"] = panck

        cell_df.at[index,"Cell Width"] = cell_meta["Width"].values[0]
        cell_df.at[index,"Cell Height"] = cell_meta["Height"].values[0]

        # Typing
        if not cell_typing.empty:
            cell_df.at[index,"Entire cell area"] = cell_typing["Area"].values[0]
            cell_df.at[index,"Diversity"] = cell_typing["Diversity"].values[0]
            cell_df.at[index,"Total transcript counts"] = cell_typing["totalcounts"].values[0]
            cell_df.at[index,"Clustering"] = cell_typing["nb_clus"].values[0]
            cell_df.at[index,"Cell type"] = cell_typing["updatedCellTypes"].values[0]
            # Add binary column for cancer
            if cell_typing["updatedCellTypes"].values[0] == 'cancer':
                cell_df.at[index,"Cancer?"] = "Cancer"
            else:
                cell_df.at[index,"Cancer?"] = "Not Cancer"
        
    return cell_df
        
def assemble_df(cell_dict):  
    '''It's absolutely critical to double check these colum names. The information is added back here,
        and has not been entirely kept track of until this point. Positions of the columns in the cell dict could change 
        depending on what funcitons were called, and if those functions were altered.'''
    df = pd.DataFrame.from_dict(cell_dict, orient="index", columns = ["global_cid","fov","local_cid", "Local X","Local Y", # Identifying information
                                "Nuclear area","Nuclear area convex", "Nuclear area filled","Major axis","Minor axis", # Geometric feats
                                 "Bounding ellipse area","Bounding ellipse perimeter", "Eccentricity","Feret diameter max",# Geometric feats
                                "Feret diameter min","Perimeter","Solidity", "Extent", "Euler number", #Geometric feats
                                "Hu moments","Weighted Hu moments", # Geometric feats
                                #"Wavelet Scattering Vector One", "Wavelet Scattering Vector Two", "Wavelet Scattering Vector Three", # Wavelet scattering
                                "DAPI Intensity Mean","DAPI Area (px)"])  # DAPI analysis
    
    # gabor and glcm
    # df = pd.DataFrame.from_dict(cell_dict, orient="index", columns = ["global_cid","fov","local_cid", "Local X","Local Y",
    #                             "Texture-correlation","Texture-dissimilarity","Texture-homogeneity","Texture-ASM","Texture-energy","Texture-contrast",
    #                             f'Gabor f{FREQ1} mean',f'Gabor f{FREQ1} std',
    #                             f'Gabor f{FREQ2} mean',f'Gabor f{FREQ2} std',
    #                             "DAPI Intensity Mean","DAPI Area (px)"])

    # # df = pd.DataFrame(columns=['global_cid', 'fov', 'local_cid','Local X','Local Y','DAPI Intensity Mean','DAPI Area (px)',
    # #     'Texture-correlation','Texture-dissimilarity','Texture-homogeneity','Texture-ASM','Texture-energy','Texture-contrast'])
    # df = pd.DataFrame(columns=['global_cid', 'fov', 'local_cid','Local X','Local Y','DAPI Intensity Mean','DAPI Area (px)',
    #     f'Gabor f{FREQ1} mean',f'Gabor f{FREQ1} std',f'Gabor f{FREQ2} mean',f'Gabor f{FREQ2} std'])
    # for cid in cell_dict.keys():
    #     cell_mean = cell_dict[cid]
    #     both = cid.split("_")
    #     # d = {'global_cid': cid, "fov":both[0],"local_cid":both[1],'Local X':other_params[f'{cid}_localX'],'Local Y':other_params[f'{cid}_localY'],
    #     #     'DAPI Intensity Mean':cell_mean, "DAPI Area (px)": other_params[cid+"_area"],
    #     #     'Texture-correlation':other_params[cid+"_texture-correlation"],'Texture-dissimilarity':other_params[cid+"_texture-dissimilarity"],
    #     #     'Texture-homogeneity':other_params[cid+"_texture-homogeneity"],'Texture-ASM':other_params[cid+"_texture-ASM"],
    #     #     'Texture-energy':other_params[cid+"_texture-energy"],'Texture-contrast':other_params[cid+"_texture-contrast"]}
    #     d = {'global_cid': cid, "fov":both[0],"local_cid":both[1],'Local X':other_params[f'{cid}_localX'],'Local Y':other_params[f'{cid}_localY'],
    #         'DAPI Intensity Mean':cell_mean, "DAPI Area (px)": other_params[cid+"_area"]}#,
    #         # f'Gabor f{FREQ1} mean':other_params[cid+f"_gabor{FREQ1}_mean"],f'Gabor f{FREQ1} std':other_params[cid+f"_gabor{FREQ1}_std"],
    #         # f'Gabor f{FREQ2} mean':other_params[cid+f"_gabor{FREQ2}_mean"],f'Gabor f{FREQ2} std':other_params[cid+f"_gabor{FREQ2}_std"]}
    #     df = pd.concat([df,pd.DataFrame(data=d,index=[1])], ignore_index=True)
    return df

def dump_csv(df):
    df.to_csv(RESULTS_FILE, index=False)
    return None

def process_fov(tif, root, pipe_out = None, lock = None, fov_selection = None):

    lap_start = time.time()
    fov = int(tif.split("_")[-2].lstrip("[F0]"))
    if fov_selection is not None:
        if fov not in fov_selection:
            return {}
    cell_mask = os.path.normpath(os.path.join(root,"../CellLabels/CellLabels_" + tif.split("_")[-2] + ".tif"))
    compartment_mask = os.path.normpath(os.path.join(root,"../CompartmentLabels/CompartmentLabels_" + tif.split("_")[-2] + ".tif"))
    composite_path = os.path.normpath(os.path.join(root,tif))
    # lock.acquire()
    print(f"Compartment path is {compartment_mask} and mask is {cell_mask}")
    print(f"\nWorking on FOV {fov}...",end="   ")
    # lock.release()
    dapi_means = add_counts_for_fov({}, fov, (cell_mask,compartment_mask), composite_path, lock)
    lap_end = time.time() 
    # lock.acquire()
    print(f"Finished with FOV {fov} in {lap_end-lap_start} seconds. Have data for {len(dapi_means)} nuclei")
    # lock.release()
    if DO_PARALLEL:
        pipe_out.send(dapi_means)
        pipe_out.close()
    else:
        return dapi_means

def main():
    # print(f"max value for dapi means is {max(dapi_means.values())}")
    dapi_means={}
    start = time.time()
    readers=[]
    lock = Lock()
    "../RawData/MultichannelImages/20220405_130717_S1_C902_P99_N99_F001_Z004.TIF"
    for root,dirs,files in os.walk(os.path.normpath("../RawData/MultichannelImages")):
        # print(files)
        for tif in files:  #[2:8]:
            if DO_PARALLEL is True: 
                print(f'tif is {tif}')
                r, w = Pipe(duplex=False)
                readers.append(r)
                p = Process(target = process_fov,args=(tif, root, w, lock))
                p.start()
                w.close()   
                # if fov ==1: break # for testing
            else:
                fov_dict = process_fov(tif,root, fov_selection=[21,19])
                dapi_means.update(fov_dict)

    # wait until everything is done.
    # As data comes in, add it to the appropriate dictionary
    if DO_PARALLEL:
        while readers:
            for r in wait(readers):
                try:
                    fov_dict = r.recv()
                    dapi_means.update(fov_dict)
                except EOFError:
                    readers.remove(r)

    end = time.time()
    print(f"\nAll FOVs runtime: {end-start} seconds.")

    if ASSEMBLE:
        start = time.time()
        print(f"\nAssembling data for {len(dapi_means.keys())} cells... ", end="   ")
        output = add_columns(TRANSCRIPTS,METADATA,CELLTYPING, assemble_df(dapi_means))
        end = time.time()
        print(f"Completed in {end-start} seconds.")
    if WRITE:
        print(f"\n Writing to {RESULTS_FILE} ... ", end='   ')
        dump_csv(output)
    print("Analysis completed.")

if __name__ == "__main__":
    main()