'''
Extract nuclear signal for cells in a CosMx run and save images
Create some metadata on these cells and save in a table

Peter Richieri 
MGH Ting Lab 
3/21/24
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

import matplotlib.pyplot as plt # checking out fft
import numpy
# import cv2
NUM_IMAGES_TO_SHOW = 10 # Was used for manual fourier transformed image viewing
import scipy.fft as fft
import sys

import IPython
from PIL import Image
import texture_synthesis

# Texture features take a long time, so these modules
#   add concurrency
from multiprocessing import Process, Lock, Pipe
from multiprocessing.connection import wait

#---------------------- Global variables -----------------------#
# MIN_DAPI_INTENSITY_THRESHOLD = 15 # not used currently
MIN_DAPI_AREA = 30 # In pixels
# MIN_PANCK_THRESHOLD = 900 # Not used currently
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
# The variable below should be the save folder for individual cell images
SAVE_CELL_IMAGES = False #r"/home/peter/TMA CosMx Texture Images/"
SAVE_SYNTHETIC_IMAGES = False #r"/home/peter/data2/CosMx Quilted Texture lir=15b=10o=5/"
MAKE_NEW_TEXTURE = False
USE_IMAGES_IN_FOLDER = False #r"/home/peter/TMA CosMx Texture Images/"#'/home/peter/C4 CosMx Texture Images (Padded)/' # False
PATH_EXTENSION = 'nuclearDapi_16bit.png'



''' Get cell data as numpy array, convert to binary, and return. '''
def read_mask(path_to_label_mask, path_to_compartment_mask):
    print(f'Reading cell masks',end="   ")
    labels = tifffile.imread(path_to_label_mask)
    compartments = tifffile.imread(path_to_compartment_mask) # I think this might only read in the top channel anyway...
    # Nuclear compartment pixels have a 1 value. 0 is background, 2 is membrane, 3 is cytoplasm
    #   so replace 2 and 3 with 0.
    compartments[compartments>1] = 0
    # Have to flip is around to be [x,y] instead of [y,x]
    return np.transpose(compartments*labels,(1,0))

''' Data only exists as composite .jpg. DAPI signal should be the B channel of the RGB data. Deprecated '''
def extract_dapi_signal_composite(path_to_file):
    print(f'Reading composite',end="   ")
    raw_data = image.imread(path_to_file)
    # Also have to flip is around to be [x,y] instead of [y,x]
    return np.transpose(raw_data[:,:,2],(1,0))

'''Get DAPI image from a multichannel TIF. In CosMx data, this has been the last channel (4)'''
def extract_dapi_signal_tif(path_to_file):
    print("Reading tif",end="   ")
    all_bands = tifffile.imread(path_to_file)
    return np.transpose(all_bands[4,:,:],(1,0)) # DAPI is the last channel in this data

''' Read in metadata file and return data for the current fov'''
def read_metadata(path_to_file, fov = 4):
    meta = pd.read_csv(path_to_file)
    meta_fov = meta.loc[meta["fov"] == fov]
    return meta_fov

'''Subset numpy array to only data corresponding to a certain Cell_ID and return that data. Optionally do some other analysis'''
def get_pixels_for_cell(nuclear_mask,dapi_only,run_name, fov, cell_id,metadata,fov_x,fov_y, max_local_y):
    ''' Helper function to perform an erosion and return the inner and outer images'''
    
    # Get bounding box coords
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

    if USE_IMAGES_IN_FOLDER: # Save time if the image has been created in an earlier run
        try:    
            #TODO change this to use the run name as the parent folder
            im = Image.open(f'{USE_IMAGES_IN_FOLDER}{run_name}_{fov}_{cell_id}_{PATH_EXTENSION}')
            single_cell_dapi = np.asarray(im)
            return single_cell_dapi, None, (local_center_x, local_center_y)
        except:
            return None, None, (local_center_x, local_center_y)
            # Image was never created (cell area too small?)
    
    else: # The image doesn't exist so we have to make it from the full TIF
        # Have to get individual windows out of nuclear mask AND dapi image
        #   For nuclear mask, need to extract pixels where the value equals the cell ID
        single_cell_mask = nuclear_mask[xmin:xmax,ymin:ymax]
        # print(f'My unique vals are  {np.unique(single_cell_mask)}')
        
        # !!! IMPORTANT!!! Arrays are mutable, don't screw up the original array. Make a copy
        single_cell_mask_copy = copy.copy(single_cell_mask)
        single_cell_mask_copy[single_cell_mask_copy != int(cell_id)] = 0 # not sure about type here...might as well cast it
        single_cell_mask_copy[single_cell_mask_copy > 0] = 1 # now have 1s where this cell's nucleus is
        single_cell_dapi = dapi_only[xmin:xmax,ymin:ymax] # Now we have dapi signal for one cell
        return single_cell_mask_copy * single_cell_dapi, single_cell_mask_copy, (local_center_x, local_center_y)
    

''' Return a dictionary of the mean DAPI value for all cells
    Key: a cell ID          value: Mean of DAPI counts per pixel for that cell'''
def mean_for_all_cells(nuclear_dapi,run_name, fov, cell_id, cell_lookup, lock):
    # dump zeroes, then check length( i.e. testing the number of pixels in the nuclear mask)
    current = np.ndarray.flatten(nuclear_dapi)
    current = current[current !=0]
    if len(current)<MIN_DAPI_AREA:
        print(f'Run {run_name} | FOV {fov} | CID {cell_id} nucleus is only {len(current)} pixels, dropping this one.')
        return cell_lookup
    cell_lookup[f"{run_name}_{fov}_{cell_id}"].update({"Nuclear DAPI intensity mean":np.mean(current),"Nuclear Area":len(current)}) # Full nuclear mask intensity and area
    return cell_lookup

'''Use the translation table and pass the xy value of the origin of an FOV relative to the global coordinate system'''
def get_coordinate_conversions(fov, fov_positions):
    meta = pd.read_csv(fov_positions)
    meta_fov = meta.loc[meta["fov"] == fov]
    return int(meta_fov["x_global_px"].values[0]), int(meta_fov["y_global_px"].values[0])

''' Measure geometric features for this cell's nucleus and add the information to the dictionary'''
def compute_geometric_features(nuclear_dapi, run_name, fov,cell_id, cell_dict):

    binary_dapi = copy.copy(nuclear_dapi).astype(np.uint8)
    binary_dapi[binary_dapi > 0] = 1
    label_img = label(binary_dapi)

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


    cell_dict[f"{run_name}_{fov}_{cell_id}"].update({"Nuclear area":df['area'][pos], "Nuclear area convex":df['area_convex'][pos],
                "Nuclear area filled":df['area_filled'][pos],"Major axis":df['axis_major_length'][pos],
                "Minor axis":df['axis_minor_length'][pos],"Bounding ellipse area": ellipse_area,"Bounding ellipse perimeter": ellipse_perimeter,
                "Eccentricity":df['eccentricity'][pos],"Feret diameter max":df['feret_diameter_max'][pos],"Feret diameter max":min_feret,
                "Perimeter":df['perimeter'][pos],"Solidity":df['solidity'][pos],"Extent":df['extent'][pos],"Euler number":df['euler_number'][pos]})
    cell_dict[f"{run_name}_{fov}_{cell_id}"].update({"Hu moments":np.array_str(np.array(hu_moments)),
                                      "Weighted Hu moments":np.array_str(np.array(weighted_hu_moments)) })
    return cell_dict


def get_images_from_fov(cell_dictionary,run_name, fov, mask_tuple, composite_path, metadata_path, fov_positions, lock):
    cell_mask_path = mask_tuple[0]; compartment_mask_path = mask_tuple[1]
    
    global_nuclear_mask = read_mask(cell_mask_path, compartment_mask_path)
    dapi_only = extract_dapi_signal_tif(composite_path)
    max_X = dapi_only.shape[0]; max_Y = dapi_only.shape[1]
    
    # Now make quantitative counts for each DAPI pixel
    metadata = read_metadata(metadata_path, fov)
    fov_global_X, fov_global_Y = get_coordinate_conversions(fov, fov_positions)

    # for cell_id in metadata["cell_ID"].iloc[::-1]: # reverses CID order
    for cell_id in metadata["cell_ID"]:
        if cell_id %100 ==0:
            print(f'On {run_name} fov {fov}, cell {cell_id}')    
        nuclear_dapi, nuclear_mask, coords = get_pixels_for_cell(global_nuclear_mask, dapi_only,run_name,fov,
                                                                      cell_id,metadata,fov_global_X,fov_global_Y,max_Y)

        if nuclear_dapi is None:
            print(f"Skipping {run_name} fov {fov}, cell {cell_id}. Either: not in image library, has no DAPI, or no nuclear compartment.")
            # move on
            continue

        cell_dictionary[f"{run_name}_{fov}_{cell_id}"] = {'global_cid': f"{run_name}_{fov}_{cell_id}",
                                                          'Run name':run_name, 
                                                          'fov':str(fov), 'local_cid':str(cell_id),'Local X':coords[0],'Local Y':coords[1]} # Add XY to make it easier to find cells of interest

        if SAVE_CELL_IMAGES: # Write an image for each nucleus to disk
            try:
                if np.any(nuclear_dapi):
                    im = Image.fromarray(nuclear_dapi)
                    im.save(SAVE_CELL_IMAGES+f"{run_name}_{fov}_{cell_id}_nuclearDapi_16bit.png")
            except:
                print(f"There was a problem saving run {run_name} | fov {fov} | CID {cell_id}")

        cell_dictionary = mean_for_all_cells(nuclear_dapi, run_name, fov, cell_id, cell_dictionary, lock)
    return cell_dictionary
    

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
 
        # Metadata sheet
        cell_meta = metadata.loc[(metadata["fov"] == int(cell["fov"])) & (metadata["cell_ID"]== int(cell["local_cid"]))]
        cell_df.at[index,"Mean PanCK"] = cell_meta["Mean.PanCK"].values[0]
        cell_df.at[index,"Mean CD45"] = cell_meta["Mean.CD45"].values[0]
        cell_df.at[index,"Mean CD3"] = cell_meta["Mean.CD3"].values[0]
        cell_df.at[index,"Cell Width"] = cell_meta["Width"].values[0]
        cell_df.at[index,"Cell Height"] = cell_meta["Height"].values[0]
        cell_df.at[index,"Entire cell area"] = cell_meta["Area"].values[0]
        
    return cell_df
        
def assemble_df(cell_dict):  
    ''' Create pandas dataframe from the dictionary'''
    df = pd.DataFrame.from_dict(list(cell_dict.values()), orient="index", columns = list(cell_dict.keys()))  # DAPI analysis
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

    print(f"\nWorking on run {run} : FOV {fov}...")
    try:
        fov_data = get_images_from_fov({},run, fov, (cell_mask,compartment_mask), composite_path, meta_path, fov_positions_path, lock)
    except FileNotFoundError:
        print(f"{run} fov {fov} does not have a mask available to read. There will be no data from this FOV.")
        if DO_PARALLEL:
            pipe_out.send(pd.DataFrame())
            pipe_out.close()
            return pd.DataFrame()
        else:
            return pd.DataFrame()

    lap_end = time.time() 
    print(f"Finished with {run} FOV {fov} in {lap_end-lap_start} seconds. Have data for {len(fov_data)} nuclei")

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
        for tif in files:  
            if image_validator.match(tif) is not None:
                if DO_PARALLEL is True: 
                    print(f'tif is {tif}')
                    r, w = Pipe(duplex=False)
                    readers.append(r)
                    p = Process(target = process_fov,args=(tif, root, w, lock))
                    p.start()
                    w.close()   
                else:
                    df = process_fov(tif,root, run_selection = RUN_SELECTION, fov_selection=FOV_SELECTION)
                    output = pd.concat([output, df])

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
    IPython.embed()

if __name__ == "__main__":
    main()