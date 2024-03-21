'''
Compare nuclear intensity vs LINE1 transcripts per cell, run correlation / regression
    This file compiles the data for plotting
Peter Richieri 
MGH Ting Lab 
11/7/22
'''

#-------------------------- Imports --------------------- #
import numpy as np
import tifffile # for cell masks
from matplotlib import image # for DAPI .jpg
import pandas as pd
import time
import os
from re import compile, match
import copy
import math
import matplotlib.pyplot as plt
from skimage.draw import ellipse # For roundness metrics
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.morphology import remove_small_holes, binary_erosion
from skimage.feature import graycomatrix,graycoprops # Creating texture metrics
import feret_diameter
import pyfeats
from kymatio.numpy import Scattering2D
import matplotlib.pyplot as plt # checking out fft
import numpy
# import cv2
NUM_IMAGES_TO_SHOW = 10 # Was used for manual fourier transformed image viewing
import scipy.fft as fft
import sys
numpy.set_printoptions(threshold=sys.maxsize)
import IPython
from PIL import Image
import texture_synthesis

# Texture features take a long time, so these modules
#   add concurrency
from multiprocessing import Process, Lock, Pipe
from multiprocessing.connection import wait

#---------------------- Global variables -----------------------#
MIN_DAPI_INTENSITY_THRESHOLD = 15
MIN_DAPI_AREA = 30 # In pixels
MIN_PANCK_THRESHOLD = 900
GLCM_DISTANCE = [1]
DOWNSAMPLE = int(math.pow(2,6)-1)
REMOVE_GLCM_ZEROES = True
FREQ1 = .1
FREQ2 = .05
EROSION = True
PIXEL_DISTANCE_HIST = False
#Runs: C4_R5042_S1, D10_R5042_S2, B10_R1171_S2, 
    # Run5573_TMA1, Run5573_TMA28, Run5573_TMA31, Run5584_TMA32
RUN_SELECTION =  ["Run5573_TMA28"] # Only used if NOT running in parallel
FOV_SELECTION = [19]
RESULTS_FILE = os.path.normpath (f"../5.8.23_test.csv")#allRuns_fromScratch_CAFerosion_GLCM+geom.csv")
DO_PARALLEL = False
ASSEMBLE = False
WRITE = False
SAVE_CELL_IMAGES = False #r"/home/peter/TMA CosMx Texture Images/"
SAVE_SYNTHETIC_IMAGES = False #r"/home/peter/data2/CosMx Quilted Texture lir=15b=10o=5/"
MAKE_NEW_TEXTURE = False
USE_IMAGES_IN_FOLDER = False #r"/home/peter/TMA CosMx Texture Images/"#'/home/peter/C4 CosMx Texture Images (Padded)/' # False
PATH_EXTENSION = 'nuclearDapi_16bit.png'

##------------------ C4
# METADATA = os.path.normpath (r"../All Runs/C4_R5042_S1/C4_R5042_S1_metadata_file.csv")
# FOV_POSITIONS = os.path.normpath (r"../All Runs/C4_R5042_S1/C4_R5042_S1_fov_positions_file.csv")
# TRANSCRIPTS = os.path.normpath (r"../All Runs/C4_R5042_S1/C4_R5042_S1_exprMat_file.csv")
# CELLTYPING = os.path.normpath (r"../All Runs/C4_R5042_S1/slide_C4_R5042_S1_Napari_metadata.csv")
#---------------- D10
# METADATA = os.path.normpath (r"../All Runs/D10_R5042_S2/D10_R5042_S2_metadata_file.csv")
# FOV_POSITIONS = os.path.normpath (r"../All Runs/D10_R5042_S2/D10_R5042_S2_fov_positions_file.csv")
# TRANSCRIPTS = os.path.normpath (r"../All Runs/D10_R5042_S2/D10_R5042_S2_exprMat_file.csv")
# CELLTYPING = os.path.normpath (r"../All Runs/D10_R5042_S2/slide_D10_R5042_S2_Napari_metadata.csv")
#------------------ B10
# METADATA = os.path.normpath (r"../All Runs/B10_R1171_S2/B10_R1171_S2_metadata_file.csv")
# FOV_POSITIONS = os.path.normpath (r"../All Runs/B10_R1171_S2/B10_R1171_S2_fov_positions_file.csv")
# TRANSCRIPTS = os.path.normpath (r"../All Runs/B10_R1171_S2/B10_R1171_S2_exprMat_file.csv")
# CELLTYPING = os.path.normpath (r"../All Runs/B10_R1171_S2/slide_B10_R1171_S2_Napari_metadata.csv")

# this is created by merging the cell typing from 'insitutype 15 clusts refined.RDS' with 'annot.RDS'
#   call this column 'updatedCellTyping'
# Can be found in the shared Nanostring Box under "16. Addon data/R files complete data/"
ALL_META = pd.read_csv(r"../All Runs/all_runs_metadata.csv")
ALL_META['local_id'] = ALL_META.cell_ID.str.split('_').str[-1]
ALL_META['global_id'] = ALL_META.cell_ID.str.split('_').str[-2] + '_' + ALL_META.cell_ID.str.split('_').str[-1]
ALL_META_cancer = ALL_META.loc[(ALL_META['updatedCellTyping'] == 'cancer') &
                               (ALL_META['tissue'] == 'PDAC_D10'),['global_id','updatedCellTyping']].copy()
ALL_META_CAF = ALL_META.loc[(ALL_META['updatedCellTyping'] == 'CAF') &
                               (ALL_META['tissue'] == 'PDAC_D10'),['global_id','updatedCellTyping']].copy()
ALL_META_T = ALL_META.loc[(ALL_META['updatedCellTyping'] == 'T.cell') &
                               (ALL_META['tissue'] == 'PDAC_D10'),['global_id','updatedCellTyping']].copy()
ALL_META_macrophage = ALL_META.loc[(ALL_META['updatedCellTyping'] == 'macrophage') &
                               (ALL_META['tissue'] == 'PDAC_D10'),['global_id','updatedCellTyping']].copy()
ALL_META_Endothelial = ALL_META.loc[(ALL_META['updatedCellTyping'] == 'Endothelial') &
                               (ALL_META['tissue'] == 'PDAC_D10'),['global_id','updatedCellTyping']].copy()
ALL_META_Vascular = ALL_META.loc[(ALL_META['updatedCellTyping'] == 'Vascular.smooth.muscle') &
                               (ALL_META['tissue'] == 'PDAC_D10'),['global_id','updatedCellTyping']].copy()


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

COUNTS = {}
CANCER_COUNTS = {}
CAF_COUNTS = {}
ENDO_COUNTS = {}
MACROPHAGE_COUNTS = {}
VASCULAR_COUNTS = {}
T_COUNTS = {}
def generate_pixel_intensity_histogram(intensity_im, label_im, name):
    distance = 1
    while(True):
        # print(f"\nDistance is {distance}")
        ero = binary_erosion(label_im)
        if not np.any(ero):
            break
        rim = np.logical_and(label_im, ~ero)
        rim_intensity = np.ndarray.flatten(rim*intensity_im)
        rim_intensity = rim_intensity[rim_intensity !=0]
        # try:
        #     ctype = ALL_META_SELECT.loc[(ALL_META_SELECT["tissue"]==f"PDAC_{name.split('_')[0]}") & (ALL_META_SELECT['fov']==int(name.split("_")[-2])) & 
        #                         (ALL_META_SELECT['local_id']==name.split("_")[-1]), ['updatedCellTyping']].values[0][0]
        # except KeyError:
        #     break
        # IPython.embed()

        if name.split('_')[-2]+ '_' + name.split('_')[-1] in list(ALL_META_cancer['global_id']):
            try:
                CANCER_COUNTS[distance] = np.concatenate([CANCER_COUNTS[distance],rim_intensity])
            except:
                CANCER_COUNTS[distance] = rim_intensity
        elif name.split('_')[-2]+ '_' + name.split('_')[-1] in list(ALL_META_CAF['global_id']):
            try:
                CAF_COUNTS[distance] = np.concatenate([CAF_COUNTS[distance],rim_intensity])
            except:
                CAF_COUNTS[distance] = rim_intensity
        elif name.split('_')[-2]+ '_' + name.split('_')[-1] in list(ALL_META_T['global_id']):
            try:
                T_COUNTS[distance] = np.concatenate([T_COUNTS[distance],rim_intensity])
            except:
                T_COUNTS[distance] = rim_intensity
        elif name.split('_')[-2]+ '_' + name.split('_')[-1] in list(ALL_META_macrophage['global_id']):
            try:
                MACROPHAGE_COUNTS[distance] = np.concatenate([MACROPHAGE_COUNTS[distance],rim_intensity])
            except:
                MACROPHAGE_COUNTS[distance] = rim_intensity
        elif name.split('_')[-2]+ '_' + name.split('_')[-1] in list(ALL_META_Endothelial['global_id']):
            try:
                ENDO_COUNTS[distance] = np.concatenate([ENDO_COUNTS[distance],rim_intensity])
            except:
                ENDO_COUNTS[distance] = rim_intensity
        elif name.split('_')[-2]+ '_' + name.split('_')[-1] in list(ALL_META_Vascular['global_id']):
            try:
                VASCULAR_COUNTS[distance] = np.concatenate([VASCULAR_COUNTS[distance],rim_intensity])
            except:
                VASCULAR_COUNTS[distance] = rim_intensity
        try:
            COUNTS[distance] = np.concatenate([COUNTS[distance],rim_intensity])
        except:
            COUNTS[distance] = rim_intensity
        distance += 1
        label_im = ero
    # IPython.embed()

def plot_pixel_intensity_histogram():
    min_pixel_count = 500
    df = pd.DataFrame([["All","1", np.mean(COUNTS[1]),COUNTS[1].shape[0],np.std(COUNTS[1])]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
    for dis in list(COUNTS.keys())[1:]:
        if COUNTS[dis].shape[0] < min_pixel_count: break
        new_row = pd.DataFrame([["All",str(dis), np.mean(COUNTS[dis]),COUNTS[dis].shape[0],np.std(COUNTS[dis])]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
        df = pd.concat([df,new_row])

    df1 = pd.DataFrame([["Cancer","1", np.mean(CANCER_COUNTS[1]),CANCER_COUNTS[1].shape[0],np.std(CANCER_COUNTS[1])]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
    for dis in list(CANCER_COUNTS.keys())[1:]:
        if CANCER_COUNTS[dis].shape[0] < min_pixel_count: break
        new_row = pd.DataFrame([["Cancer",str(dis), np.mean(CANCER_COUNTS[dis]),CANCER_COUNTS[dis].shape[0],np.std(CANCER_COUNTS[dis])]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
        df1 = pd.concat([df1,new_row])

    df2 = pd.DataFrame([["CAF","1", np.mean(CAF_COUNTS[1]),CAF_COUNTS[1].shape[0],np.std(CAF_COUNTS[1])]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
    for dis in list(CAF_COUNTS.keys())[1:]:
        if CAF_COUNTS[dis].shape[0] < min_pixel_count: break
        new_row = pd.DataFrame([["CAF",str(dis), np.mean(CAF_COUNTS[dis]),CAF_COUNTS[dis].shape[0],np.std(CAF_COUNTS[dis])]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
        df2 = pd.concat([df2,new_row])

    df3 = pd.DataFrame([["T.cell","1", np.mean(T_COUNTS[1]),T_COUNTS[1].shape[0],np.std(T_COUNTS[1])]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
    for dis in list(T_COUNTS.keys())[1:]:
        if T_COUNTS[dis].shape[0] < min_pixel_count: break
        new_row = pd.DataFrame([["T.cell",str(dis), np.mean(T_COUNTS[dis]),T_COUNTS[dis].shape[0],np.std(T_COUNTS[dis])]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
        df3 = pd.concat([df3,new_row])

    df4 = pd.DataFrame([["Macrophage","1", np.mean(MACROPHAGE_COUNTS[1]),MACROPHAGE_COUNTS[1].shape[0],np.std(MACROPHAGE_COUNTS[1])]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
    for dis in list(MACROPHAGE_COUNTS.keys())[1:]:
        if MACROPHAGE_COUNTS[dis].shape[0] < min_pixel_count: break
        new_row = pd.DataFrame([["Macrophage",str(dis), np.mean(MACROPHAGE_COUNTS[dis]),MACROPHAGE_COUNTS[dis].shape[0],np.std(MACROPHAGE_COUNTS[dis])]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
        df4 = pd.concat([df4,new_row])

    df5 = pd.DataFrame([["Endothelial","1", np.mean(ENDO_COUNTS[1]),ENDO_COUNTS[1].shape[0],np.std(ENDO_COUNTS[1])]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
    for dis in list(ENDO_COUNTS.keys())[1:]:
        if ENDO_COUNTS[dis].shape[0] < min_pixel_count: break
        new_row = pd.DataFrame([["Endothelial",str(dis), np.mean(ENDO_COUNTS[dis]),ENDO_COUNTS[dis].shape[0],np.std(ENDO_COUNTS[dis])]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
        df5 = pd.concat([df5,new_row])

    a = np.sum(COUNTS[1])
    al = len(COUNTS[1])
    b = np.sum(CANCER_COUNTS[1])
    bl = len(CANCER_COUNTS[1])
    c = a - b
    cl = al - bl
    m = c / cl
    df6 = pd.DataFrame([["Not Cancer","1", m,cl,0]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
    for dis in list(ENDO_COUNTS.keys())[1:]:
        a = np.sum(COUNTS[dis])
        al = len(COUNTS[dis])
        b = np.sum(CANCER_COUNTS[dis])
        bl = len(CANCER_COUNTS[dis])
        c = a - b
        cl = al - bl
        m = c / cl
        if cl < min_pixel_count: break
        new_row = pd.DataFrame([["Not Cancer",str(dis), m,cl,0]], columns=["Typing","Distance_from_border(px)", "Mean intensity", "Pixel_count", "std"])
        df6 = pd.concat([df6,new_row])

    df = pd.concat([df,df1,df2,df3,df4,df5, df6]).reset_index(drop=True)
    import seaborn as sns
    # IPython.embed()
    p1 = sns.lineplot(data = df, x = "Distance_from_border(px)", y="Mean intensity", hue = "Typing").set(title=f"Average pixel intensity as a function of distance from nuclear envelope")
    plt.show()

    dfco = df[df["Typing"].isin(["Cancer", "Not Cancer", "All"])].copy()
    dfco["upper"] = dfco["Mean intensity"] + dfco['std']
    dfco["lower"] = dfco["Mean intensity"] - dfco['std']

    # g = sns.FacetGrid(dfco).set(title=f"Average pixel intensity as a function of distance from nuclear envelope, Cancer vs Normal")
    p2 = sns.lineplot(data = dfco, x = "Distance_from_border(px)", errorbar=None,y="Mean intensity", hue = "Typing").set(title=f"Average pixel intensity as a function of distance from nuclear envelope, Cancer vs Normal")
    # g.map(sns.lineplot, "Distance_from_border(px)","Mean intensity", hue = "Typing", errorbar=None)
    # g.fill_between(dfco["Distance_from_border(px)"], dfco.lower, dfco.upper, alpha=0.2, hue = "Typing")
    # IPython.embed()
    plt.show()
    # IPython.embed()
    p3 = sns.barplot(data = df, x = "Distance_from_border(px)", y="Pixel_count", hue="Typing", 
                    hue_order=["All","Cancer", "CAF","T.cell","Macrophage","Endothelial"]).set(title=f"Counts of pixel distances from nuclear envelope")
    plt.show()
    IPython.embed()

'''Subset numpy array to only data corresponding to a certain Cell_ID and return that data'''
def get_pixels_for_cell(nuclear_mask,dapi_only,run_name, fov, cell_id,metadata,fov_x,fov_y, max_local_y):
    def erode_with_rim(raw_im):
        intensity = copy.copy(raw_im) # !!! very very important to deep copy
        # Label image, and only consider the largest area (cell of interest)
        binary_dapi = intensity.astype(np.uint8)
        binary_dapi[binary_dapi > 0] = 1
        label_img = label(binary_dapi)
        df = regionprops_table(label_img.astype(np.uint8), intensity, properties = ("label", "area", "axis_minor_length"))
        area = df['area']
        pos = np.where(area == max(area))[0][0]
        main_label = int(df['label'][pos])
        label_img[label_img != main_label] = 0
        # Fill any holes, and create two masks using erosion: inner portion, and outer ring
        # The erosion here will be tied to the minor axis length. The footprint parameter of 
        #   binary_erosion is a little unclear, but the implementation here of two arrays is for speed
        #   purposes (according to a skimage walkthrough). 
        filled = remove_small_holes(label_img.astype(np.bool8))
        # if int(cell_id) == 120:
        if len(np.unique(filled)) == 1:
            # hole filling has failed... just return intensity image
            return (intensity, None, None)
        if PIXEL_DISTANCE_HIST:
            generate_pixel_intensity_histogram(intensity*filled,filled, f"{run_name}_{fov}_{cell_id}")
        # if int(cell_id) == 120:
        #     IPython.embed()
        # val= int(df["axis_minor_length"][pos] /3)
        val = 12
        ero = binary_erosion(filled, footprint=[(np.ones((val, 1)), 1), (np.ones((1, val)), 1)])
        val = 12
        ero2 = binary_erosion(ero, footprint=[(np.ones((val, 1)), 1), (np.ones((1, val)), 1)])
        # rim = np.logical_and(filled, ~ero)
        rim = np.logical_and(ero, ~ero2)
        # if int(cell_id) == 120:
        #     IPython.embed()
        return (intensity, intensity*ero2, intensity*rim)
    row = metadata.loc[metadata["cell_ID"]==cell_id]
    local_center_x = int(row["CenterX_global_px"].values[0] - fov_x)
    local_center_y = int(max_local_y - (row["CenterY_global_px"].values[0] - fov_y))
    width = row["Width"].values[0]
    height = row["Height"].values[0]
    # Coords were sometimes out of bounds of the source array (bad width/height numbers)
    xmin = max(local_center_x-(width//2),0)
    xmax = min(local_center_x+(width//2),dapi_only.shape[0])
    ymin = max(local_center_y-(height//2),0)
    ymax = min(local_center_y+(height//2),dapi_only.shape[1])

    if USE_IMAGES_IN_FOLDER:
        try:    
            #TODO change this to use the run name as the parent folder
            im = Image.open(f'{USE_IMAGES_IN_FOLDER}{run_name}_{fov}_{cell_id}_{PATH_EXTENSION}')
            single_cell_dapi = np.asarray(im)
            if not EROSION:
                return (single_cell_dapi, None, None), None, (local_center_x, local_center_y)
            erosion_tuple = erode_with_rim(single_cell_dapi)
            single_cell_mask_copy = None
        except:
            return (None,None,None),None, (local_center_x, local_center_y)
            pass # Image was never created (cell area too small?)
    
    else:
        # Have to get individual windows out of nuclear mask AND dapi image
        #   For nuclear mask, need to extract pixels where the value equals the cell ID
        single_cell_mask = nuclear_mask[xmin:xmax,ymin:ymax]
        # print(f'My unique vals are  {np.unique(single_cell_mask)}')
        
        # !!! IMPORTANT!!! Arrays are mutable, don't screw up the original array. Make a copy
        single_cell_mask_copy = copy.copy(single_cell_mask)
        single_cell_mask_copy[single_cell_mask_copy != int(cell_id)] = 0 # not sure about type here...might as well cast it
        single_cell_mask_copy[single_cell_mask_copy > 0] = 1 # now have 1s where this cell's nucleus is

        single_cell_dapi = dapi_only[xmin:xmax,ymin:ymax]
        # if cell_id == 174:
        #     IPython.embed()
        if not EROSION: # don't add images for inner and outer cell regions
            return (single_cell_mask_copy * single_cell_dapi, None, None), single_cell_mask_copy, (local_center_x, local_center_y)
        sc = single_cell_mask_copy * single_cell_dapi
        try:
            erosion_tuple = erode_with_rim(sc)
        except:
            print(f"Problem with erosion for {run_name}_fov-{fov}_cid-{cell_id}")
            erosion_tuple = (sc,None,None)
    # if int(cell_id) > 150:
    #     return (data), None, (local_center_x, local_center_y)

    return erosion_tuple, single_cell_mask_copy, (local_center_x, local_center_y)

''' Return a dictionary of the mean DAPI value for all cells
    Key: a cell ID          value: Mean of DAPI counts per pixel for that cell'''
def mean_for_all_cells(image_data_tuple,run_name, fov, cell_id, cell_lookup, lock):
    nuclear_dapi = image_data_tuple[0]
    # dump zeroes, then check length( i.e. testing the number of pixels in the nuclear mask)
    current = np.ndarray.flatten(nuclear_dapi)
    current = current[current !=0]
    if len(current)<MIN_DAPI_AREA:
        if len(current) >0: 
            # lock.acquire()
            print(f'Run {run_name} | FOV {fov} | CID {cell_id} nucleus is only {len(current)} pixels, dropping this one.')
            # lock.release()
        cell_lookup[f"{run_name}_{fov}_{cell_id}"] += ["Too small","Too small","Too small","Too small","Too small","Too small"]
        return cell_lookup
    
    ## Checking a specific case
    # if cell_id == 1661 or cell_id == 1663:
    #     print(f'Nucleus of CID {cell_id} is  {len(current)} pixels')
    cell_lookup[f"{run_name}_{fov}_{cell_id}"] += [np.mean(current),len(current)] # Full nuclear mask intensity and area

    if EROSION:
        inner_dapi = image_data_tuple[1] ; outer_dapi = image_data_tuple[2]
        if inner_dapi is not None and np.any(inner_dapi):
            current = np.ndarray.flatten(inner_dapi)
            current = current[current !=0]
            if np.any(current):
                mean = np.mean(current)
            else:
                mean = 0
            cell_lookup[f"{run_name}_{fov}_{cell_id}"] += [mean,len(current)] # inner mask only intensity and area
        else:
            cell_lookup[f"{run_name}_{fov}_{cell_id}"] += ["Too small","Too small"] # inner mask only intensity and area
        if outer_dapi is not None and np.any(outer_dapi):
            current = np.ndarray.flatten(outer_dapi)
            current = current[current !=0]
            if np.any(current):
                mean = np.mean(current)
            else:
                mean = 0
            cell_lookup[f"{run_name}_{fov}_{cell_id}"] += [mean,len(current)] # outer mask only intensity and area
        else:
            cell_lookup[f"{run_name}_{fov}_{cell_id}"] += ["Too small","Too small"] # outer mask only intensity and area
    else:
        # preserve oerdering of columns even if no data
        cell_lookup[f"{run_name}_{fov}_{cell_id}"] += ["None", "None", "None", "None"] 

    # other_params[f"{fov}_{cell_id}"] += [len(current)] # Adding area
    # if cell_id >2250:   
    #     print(f"\nCID {cell_id} shape is {current.shape}, max is {np.max(current)}, min is {np.min(current)}\n{current} mean is {cell_lookup[str(fov)+'_'+str(cell_id)]}")
    return cell_lookup#, other_params

def get_coordinate_conversions(fov, fov_positions):
    meta = pd.read_csv(fov_positions)
    meta_fov = meta.loc[meta["fov"] == fov]
    return int(meta_fov["x_global_px"].values[0]), int(meta_fov["y_global_px"].values[0])

def compute_geometric_features(nuclear_dapi, run_name, fov,cell_id, cell_dict):

    binary_dapi = copy.copy(nuclear_dapi).astype(np.uint8)
    binary_dapi[binary_dapi > 0] = 1
    # print(f"\nComputing roundness...")
    label_img = label(binary_dapi)
    
    # if cell_id == 848:
    #     regions = regionprops(label_img)
    #     fig, ax = plt.subplots()
    #     ax.imshow(binary_dapi, cmap=plt.cm.gray)

    #     for props in regions:
    #         y0, x0 = props.centroid
    #         orientation = props.orientation
    #         x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
    #         y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
    #         x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
    #         y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

    #         ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    #         ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    #         ax.plot(x0, y0, '.g', markersize=15)

    #         minr, minc, maxr, maxc = props.bbox
    #         bx = (minc, maxc, maxc, minc, minc)
    #         by = (minr, minr, maxr, maxr, minr)
    #         ax.plot(bx, by, '-b', linewidth=2.5)

    #     ax.axis((0, 600, 600, 0))
    #     plt.show()

    selected_features = ("label", "area","area_convex", "area_filled","axis_major_length","axis_minor_length", 
                "eccentricity","feret_diameter_max","perimeter","solidity", "extent", "euler_number",
                "moments_hu","moments_weighted_hu", "image_convex")
    df = regionprops_table(label_img, nuclear_dapi, 
                           properties = selected_features)
    area = df['area']
    pos = np.where(area == max(area))[0][0]
    main_label = int(df['label'][pos])
    a = df['axis_major_length'][pos] / 2; b = df['axis_minor_length'][pos] / 2
    ellipse_area = math.pi*a*b
    # Using one of Ramanujan's formulas https://www.mathsisfun.com/geometry/ellipse-perimeter.html
    ellipse_perimeter = math.pi*(3*(a+b) - math.sqrt((3*a +b) * (a+3*b)))

    # skimage doesn't have a utility to get minimum feret length built in yet, so I'm using this
    #   code found on their github pages. Just grabbing minimum feret length, even though it does
    #   both, so that max feret length is reproducible with skimage. The numbers seemed to be slightly different.
    label_img[label_img != main_label] = 0
    min_feret = feret_diameter.get_min_max_feret_from_labelim(label_img)[main_label][0]
    s = df["solidity"][pos]
    if s > 0.89:
        im = df['image_convex'][pos]
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(nuclear_dapi)
        axarr[1].imshow(im)
        f.suptitle(f'{run_name} -- FOV: {fov} -- ID: {cell_id} -- Solidity: {s}', fontsize=12)
        plt.show()
    hu_moments = [] ; weighted_hu_moments = []
    for i in range(7):
        hu_moments.append(df['moments_hu-'+str(i)][pos])
    for i in range(7):
        weighted_hu_moments.append(df['moments_weighted_hu-'+str(i)][pos])
    
    # if cell_id == 848:
    #     IPython.embed()

    cell_dict[f"{run_name}_{fov}_{cell_id}"] += [df['area'][pos],df['area_convex'][pos],df['area_filled'][pos],df['axis_major_length'][pos],
                                      df['axis_minor_length'][pos], ellipse_area, ellipse_perimeter,
                                      df['eccentricity'][pos],df['feret_diameter_max'][pos],min_feret,
                                      df['perimeter'][pos],df['solidity'][pos],df['extent'][pos],df['euler_number'][pos]]  
    cell_dict[f"{run_name}_{fov}_{cell_id}"] += [np.array_str(np.array(hu_moments)),
                                      np.array_str(np.array(weighted_hu_moments))]
    return cell_dict

def add_glcm_metrics(nuclear_dapi, run_name, fov,cell_id, other_params, distance_range,angles_range):

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
    other_params[f"{run_name}_{fov}_{cell_id}"] += [correlation,dissimilarity,homogeneity,ASM,energy,contrast]
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

def wavelet_scattering(image_data_tuple, cell_id, fov, cell_dict):
    nuclear_dapi = image_data_tuple[0]
    if EROSION:
        inner_dapi = image_data_tuple[1] ; outer_dapi = image_data_tuple[2]
        scattering_inner = Scattering2D(J=2, shape=inner_dapi.shape)
        coefficients_inner = scattering_inner(inner_dapi.astype(np.float32))
        scattering_outer = Scattering2D(J=2, shape=outer_dapi.shape)
        coefficients_outer = scattering_outer(outer_dapi.astype(np.float32))
        feat2 = np.mean(coefficients_inner,axis=(1,2))
        feat3 = np.mean(coefficients_outer,axis=(1,2))

    scattering = Scattering2D(J=2, shape=nuclear_dapi.shape)
    coefficients = scattering(nuclear_dapi.astype(np.float32))
    feat1 = np.mean(coefficients,axis=(1,2))
    # feat2 = np.mean(coefficients,axis=(0,1))
    # feat3 = np.mean(coefficients,axis=(0,2))
    if EROSION:
        cell_dict[f"{fov}_{cell_id}"] += [np.array_str(feat1),np.array_str(feat2),np.array_str(feat3)]
    else:
        cell_dict[f"{fov}_{cell_id}"] += [np.array_str(coefficients),"None","None"]
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
    
def add_counts_for_fov(cell_dictionary,run_name, fov, mask_tuple, composite_path, metadata_path, fov_positions, lock):
    cell_mask_path = mask_tuple[0]; compartment_mask_path = mask_tuple[1]
    
    global_nuclear_mask = read_mask(cell_mask_path, compartment_mask_path)
    dapi_only = extract_dapi_signal_tif(composite_path)
    max_X = dapi_only.shape[0]; max_Y = dapi_only.shape[1]
    print(f"\ndapi shape is {dapi_only.shape}, max is {np.max(dapi_only)}, min is {np.min(dapi_only)}")

    # viewer.add_image(dapi_only,name='DAPI')
    
    # Now make quantitative counts for each DAPI pixel
    # dapi_cells = drop_weak_signal(cell_mask * dapi_only,MIN_DAPI_INTENSITY_THRESHOLD)
    metadata = read_metadata(metadata_path, fov)
    fov_global_X, fov_global_Y = get_coordinate_conversions(fov, fov_positions)

    # Texture generator input
    # distances_range = [1,3,5,7,9]
    distances_range = GLCM_DISTANCE
    angle_step = np.pi/2
    angle_end = np.pi * 2
    angles_range = np.arange(0,angle_end,angle_step)
    # angles_range = [0]
    # scattering_features1, scattering_features2, scattering_features3 = [],[],[]
    # for cell_id in metadata["cell_ID"].iloc[::-1]: # reverses CID order
    for cell_id in metadata["cell_ID"]: #normal
        # print(f"examining cell {cell_id}")
        if cell_id %100 ==0:
            # lock.acquire()
            print(f'On {run_name} fov {fov}, cell {cell_id}')    
            # lock.release    
        # print(f"My inputs are CID {cell_id} ,fov {fov}")
        image_data_tuple, nuclear_mask, coords = get_pixels_for_cell(global_nuclear_mask, dapi_only,run_name,fov,
                                                                      cell_id,metadata,fov_global_X,fov_global_Y,max_Y)
        try:
            nuclear_dapi = image_data_tuple[0]
            # print('extracting from image tuple')
            # IPython.embed()
        except:
            print('PROBLEM extracting from image tuple')
            IPython.embed()
        if nuclear_dapi is None:
            print(f"Skipping {run_name} fov {fov}, cell {cell_id}. Either: not in image library, has no DAPI, or no nuclear compartment.")
            # Texture can't be synthesized for this cell, move on
            continue
        if EROSION:
            inner_dapi = image_data_tuple[1] ; outer_dapi = image_data_tuple[2]
        
        cell_dictionary[f"{run_name}_{fov}_{cell_id}"] = [f"{run_name}_{fov}_{cell_id}",run_name,str(fov),str(cell_id),coords[0],coords[1]] # Add XY to make it easier to find cells of interest

        if SAVE_CELL_IMAGES: # Write an image for each nucleus to disk
            try:
                if np.any(nuclear_dapi):
                    im = Image.fromarray(nuclear_dapi)
                    im.save(SAVE_CELL_IMAGES+f"{run_name}_{fov}_{cell_id}_nuclearDapi_16bit.png")
            except:
                print(f"There was a problem saving run {run_name} | fov {fov} | CID {cell_id}")
        record_cell = True
        try:
            if np.any(nuclear_dapi):       
                if MAKE_NEW_TEXTURE:
                    nuclear_dapi = texture_synthesis.synthesize_texture(nuclear_dapi,fov,cell_id, SAVE_SYNTHETIC_IMAGES, block_size = 10)
                    nuclear_mask = None
                    if nuclear_dapi is None:
                        print(f"Cell {run_name}_{fov}_{cell_id} does not have an interior rectangle bigger than the required minimum square of 15")
                        record_cell = False
                # print("\nGeometric")
                cell_dictionary = compute_geometric_features(nuclear_dapi,run_name, fov, cell_id,cell_dictionary)
                # print('\nWavelet')
                # cell_dictionary = wavelet_scattering(image_data_tuple,cell_id,fov,cell_dictionary)
                # print('\nGLCM')
                cell_dictionary = add_glcm_metrics(nuclear_dapi,run_name, fov, cell_id, cell_dictionary, distances_range,angles_range)
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
            cell_dictionary = mean_for_all_cells(image_data_tuple, run_name, fov, cell_id, cell_dictionary, lock)
    return cell_dictionary
    # return mean_for_all_cells(global_nuclear_mask,dapi_only,cell_dictionary,other_params, metadata,fov, fov_global_X, fov_global_Y, max_Y)

def add_columns(transcripts_path, meta_path, cell_df):
    transcripts = pd.read_csv(transcripts_path)
    # Need this to get PanCK data
    metadata = pd.read_csv(meta_path) 

    # Initialize new columns
    cell_df.insert(6,"Line1_ORF1_nuclear", "")
    cell_df.insert(7,"Line1_ORF1_cytoplasm", "")
    cell_df.insert(8,"Line1_ORF1", "")
    cell_df.insert(9,"Line1_ORF2_nuclear", "")
    cell_df.insert(10,"Line1_ORF2_cytoplasm", "")
    cell_df.insert(11,"Line1_ORF2", "")
    cell_df.insert(12,"Line1_combined_nuclear", "")
    cell_df.insert(13,"Line1_combined_cytoplasm", "")
    cell_df.insert(14,"Line1_combined", "")
    cell_df.insert(15,"Mean PanCK","")
    cell_df.insert(16,"Mean CD45","")
    cell_df.insert(17,"Mean CD3","")
    cell_df.insert(18,"Cell Width","")
    cell_df.insert(19,"Cell Height","")
    cell_df.insert(20,"Entire cell area","")
    cell_df.insert(21,"Diversity","No data")
    cell_df.insert(22,"Total transcript counts","No data")
    cell_df.insert(23,"Clustering","No data")
    cell_df.insert(24,"Cancer?","No data")
    cell_df.insert(25,"Cell type","No data")

    # Assign values to column in row for cell
    for index,cell in cell_df.iterrows():

        # Transcripts with compartment labels
        cell_tx = transcripts[(transcripts["fov"] == int(cell["fov"])) & (transcripts["cell_ID"]== int(cell["local_cid"]))]
        try:
            nuclear_orf1 = cell_tx.loc[(cell_tx["CellComp"]== "Nuclear")&(cell_tx["target"]== "LINE1_ORF1"), ["target_count"]].values[0][0]
        except IndexError:
            nuclear_orf1 = 0
        try:
            nuclear_orf2 = cell_tx.loc[(cell_tx["CellComp"]== "Nuclear")&(cell_tx["target"]== "LINE1_ORF2"), ["target_count"]].values[0][0]
        except IndexError:
            nuclear_orf2 = 0
        
        # Excludes Nuclear columns, i.e. grabs 'Cytoplasm' and 'Membrane' and adds them together. 
        try:
            cytoplasm_orf1 = int(cell_tx.loc[(cell_tx["CellComp"]!= "Nuclear")&(cell_tx["target"]== "LINE1_ORF1"), ["target_count"]].sum().values[0])
        except IndexError:
            cytoplasm_orf1 = 0
        try:
            cytoplasm_orf2 = int(cell_tx.loc[(cell_tx["CellComp"]!= "Nuclear")&(cell_tx["target"]== "LINE1_ORF2"), ["target_count"]].sum().values[0])
        except IndexError:
            cytoplasm_orf2 = 0

        # Assign values 
        cell_df.at[index,"Line1_ORF1_nuclear"] = nuclear_orf1
        cell_df.at[index,"Line1_ORF1_cytoplasm"] = cytoplasm_orf1
        cell_df.at[index,"Line1_ORF1"] = nuclear_orf1 + cytoplasm_orf1
        cell_df.at[index,"Line1_ORF2_nuclear"] = nuclear_orf2
        cell_df.at[index,"Line1_ORF2_cytoplasm"] = cytoplasm_orf2
        cell_df.at[index,"Line1_ORF2"] = nuclear_orf2 + cytoplasm_orf2
        cell_df.at[index,"Line1_combined_nuclear"] = nuclear_orf1 + nuclear_orf2
        cell_df.at[index,"Line1_combined_cytoplasm"] = cytoplasm_orf1 + cytoplasm_orf2
        cell_df.at[index,"Line1_combined"] = nuclear_orf1 + nuclear_orf2 + cytoplasm_orf1 + cytoplasm_orf2
        
        # IPython.embed()
        # Metadata sheet
        cell_meta = metadata.loc[(metadata["fov"] == int(cell["fov"])) & (metadata["cell_ID"]== int(cell["local_cid"]))]
        cell_df.at[index,"Mean PanCK"] = cell_meta["Mean.PanCK"].values[0]
        cell_df.at[index,"Mean CD45"] = cell_meta["Mean.CD45"].values[0]
        cell_df.at[index,"Mean CD3"] = cell_meta["Mean.CD3"].values[0]
        cell_df.at[index,"Cell Width"] = cell_meta["Width"].values[0]
        cell_df.at[index,"Cell Height"] = cell_meta["Height"].values[0]
        cell_df.at[index,"Entire cell area"] = cell_meta["Area"].values[0]


        # Typing
        try:
            # cell_typing = ALL_META.loc[(ALL_META['Run_Tissue_name'] == cell['Run name']) & (ALL_META['global_id'] == cell['global_cid'])]
            cell_typing = ALL_META.loc[(ALL_META['Run_Tissue_name'] == cell['Run name']) & (ALL_META['global_id'] == cell['fov']+'_'+cell['local_cid'])]
        except:
            cell_typing = pd.DataFrame()
        if not cell_typing.empty:
            cell_df.at[index,"Diversity"] = cell_typing["Diversity"].values[0]
            cell_df.at[index,"Total transcript counts"] = cell_typing["totalcounts"].values[0]
            cell_df.at[index,"Clustering"] = cell_typing["nb_clus"].values[0]
            cell_df.at[index,"Cell type"] = cell_typing["updatedCellTyping"].values[0]
            # Add binary column for cancer
            if cell_typing["updatedCellTyping"].values[0] == 'cancer':
                cell_df.at[index,"Cancer?"] = "Cancer"
            else:
                cell_df.at[index,"Cancer?"] = "Not Cancer"
        
    return cell_df
        
def assemble_df(cell_dict):  
    '''It's absolutely critical to double check these colum names. The information is added back here,
        and has not been entirely kept track of until this point. Positions of the columns in the cell dict could change 
        depending on what funcitons were called, and if those functions were altered.'''
        # Geometric, wavelet scattering, and glcm
    df = pd.DataFrame.from_dict(cell_dict, orient="index", columns = ["global_cid","Run name","fov","local_cid", "Local X","Local Y", # Identifying information
                                "Nuclear area","Nuclear area convex", "Nuclear area filled","Major axis","Minor axis", # Geometric feats
                                 "Bounding ellipse area","Bounding ellipse perimeter", "Eccentricity","Feret diameter max",# Geometric feats
                                "Feret diameter min","Perimeter","Solidity", "Extent", "Euler number", #Geometric feats
                                "Hu moments","Weighted Hu moments", # Geometric feats
                                # "Wavelet Scattering Full Nucleus", "Wavelet Scattering Outer Nucleus", "Wavelet Scattering Inner Nucleus", # Wavelet scattering
                                "Texture-correlation","Texture-dissimilarity","Texture-homogeneity","Texture-ASM","Texture-energy","Texture-contrast",#glcm
                                "Full nucleus intensity mean","Full nucleus area (px)","Inner nucleus intensity mean","Inner nucleus area (px)",
                                "Outer nucleus intensity mean","Outer nucleus area (px)"])  # DAPI analysis

    # Geometric and wavelet scatterings
    # df = pd.DataFrame.from_dict(cell_dict, orient="index", columns = ["global_cid","fov","local_cid", "Local X","Local Y", # Identifying information
    #                             "Nuclear area","Nuclear area convex", "Nuclear area filled","Major axis","Minor axis", # Geometric feats
    #                              "Bounding ellipse area","Bounding ellipse perimeter", "Eccentricity","Feret diameter max",# Geometric feats
    #                             "Feret diameter min","Perimeter","Solidity", "Extent", "Euler number", #Geometric feats
    #                             "Hu moments","Weighted Hu moments", # Geometric feats
    #                             #"Wavelet Scattering Vector One", "Wavelet Scattering Vector Two", "Wavelet Scattering Vector Three", # Wavelet scattering
    #                             "DAPI Intensity Mean","DAPI Area (px)"])  # DAPI analysis
    
        # glcm ONLY
    # df = pd.DataFrame.from_dict(cell_dict, orient="index", columns = ["global_cid","fov","local_cid", "Local X","Local Y",
    #                             "Texture-correlation","Texture-dissimilarity","Texture-homogeneity","Texture-ASM","Texture-energy","Texture-contrast",
    #                             "DAPI Intensity Mean","DAPI Area (px)"])

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

def process_fov(tif, root, pipe_out = None, lock = None, run_selection = None, fov_selection = None):

    lap_start = time.time()
    fov = int(tif.split("_")[-1].lstrip("[F0]").lower().rstrip(".tif"))
    run = tif.replace("_"+tif.split("_")[-1], "")

    # If caller has specified a run (and FOV) skip FOVS not in list
    if run_selection is not None:
        if run not in run_selection:
            if DO_PARALLEL:
                pipe_out.send(pd.DataFrame())
                pipe_out.close()
            else:
                return pd.DataFrame()
        if fov_selection is not None:
            if fov not in fov_selection:
                if DO_PARALLEL:
                    pipe_out.send(pd.DataFrame())
                    pipe_out.close()
                else:
                    return pd.DataFrame()

    cell_mask = os.path.normpath(os.path.join(root,"../CellLabels/CellLabels_" + tif.split("_")[-1].replace("TIF", "tif")))
    compartment_mask = os.path.normpath(os.path.join(root,"../CompartmentLabels/CompartmentLabels_" + tif.split("_")[-1].replace("TIF", "tif")))
    composite_path = os.path.normpath(os.path.join(root,tif))
    meta_path = os.path.normpath(os.path.join(root,"../"+run+"_metadata_file.csv"))
    fov_positions_path = os.path.normpath(os.path.join(root,"../"+run+"_fov_positions_file.csv"))
    transcripts_path = os.path.normpath(os.path.join(root,"../LINE1_by_Compartment.csv"))
    # lock.acquire()
    # print(f"Compartment path is {compartment_mask} and mask is {cell_mask}")
    print(f"\nWorking on run {run} : FOV {fov}...")
    # lock.release()
    try:
        fov_data = add_counts_for_fov({},run, fov, (cell_mask,compartment_mask), composite_path, meta_path, fov_positions_path, lock)
    except FileNotFoundError:
        print(f"{run} fov {fov} does not have a mask available to read. There will be no data from this FOV.")
        if DO_PARALLEL:
            pipe_out.send(pd.DataFrame())
            pipe_out.close()
            return pd.DataFrame()
        else:
            return pd.DataFrame()

    lap_end = time.time() 
    # lock.acquire()
    print(f"Finished with {run} FOV {fov} in {lap_end-lap_start} seconds. Have data for {len(fov_data)} nuclei")
    # lock.release()

    if ASSEMBLE:
        start = time.time()
        print(f"\nAssembling data for {len(fov_data.keys())} cells in {run}_FOV{fov}... ")
        df = add_columns(transcripts_path, meta_path, assemble_df(fov_data))
        end = time.time()
        print(f"\nCompleted data assembly for {run}_FOV{fov} in {end-start} seconds.")
    else:
        df = pd.DataFrame()
    if DO_PARALLEL:
        pipe_out.send(df)
        pipe_out.close()
    else:
        return df

def main():

    start = time.time()
    readers=[]
    lock = Lock()
    image_validator = compile(r"^(?!Cell.*|Compartment.*).*.(tif|TIF|Tif)")
    
    output = pd.DataFrame() # will populate with info, row per cell
    for root,dirs,files in os.walk(os.path.normpath("/home/peter/home_projects/CosMx/All Runs")):
        for tif in files:  #[2:8]:
            if image_validator.match(tif) is not None:
                if DO_PARALLEL is True: 
                    print(f'tif is {tif}')
                    r, w = Pipe(duplex=False)
                    readers.append(r)
                    p = Process(target = process_fov,args=(tif, root, w, lock))
                    p.start()
                    w.close()   
                    # if fov ==1: break # for testing
                else:
                    #Runs: C4_R5042_S1, D10_R5042_S2, B10_R1171_S2, 
                    # Run5573_TMA1, Run5573_TMA28, Run5573_TMA31, Run5584_TMA32
                    df = process_fov(tif,root, run_selection = RUN_SELECTION, fov_selection=FOV_SELECTION)
                    # IPython.embed()
                    output = pd.concat([output, df])
                    

    # wait until everything is done.
    # As data comes in, add it to the appropriate dictionary
    if DO_PARALLEL:
        while readers:
            for r in wait(readers):
                try:
                    df = r.recv()
                    output = pd.concat([output, df])
                except EOFError:
                    readers.remove(r)

    end = time.time()
    print(f"\nAll FOVs runtime: {end-start} seconds.")

    if WRITE:
        print(f"\n Writing to {RESULTS_FILE} ... ", end='   ')
        dump_csv(output)
    print("Analysis completed.")
    print("Plot the pixel histograms!!")
    IPython.embed()

if __name__ == "__main__":
    main()