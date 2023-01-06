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
# Use napari to debug
# import napari
# viewer = napari.Viewer(title = 'Gonna see')

# For roundness metrics

import math
import matplotlib.pyplot as plt
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.feature import graycomatrix,graycoprops # Creating texture metrics
import pyfeats
import matplotlib.pyplot as plt # checking out fft
import numpy
import sys


# CELL_MASK_DATA = r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\RawData\CellLabels\CellLabels_F004.tif"
# COMPOSITE_IMAGE = r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\RawData\CellComposite\CellComposite_F004.jpg"
METADATA = os.path.normpath(r"../RawData/C4_R5042_S1_metadata_file.csv")
FOV_POSITIONS = os.path.normpath(r"../RawData/C4_R5042_S1_fov_positions_file.csv")
TRANSCRIPTS = os.path.normpath(r"../RawData/C4_R5042_S1_exprMat_file.csv")
CELLTYPING = os.path.normpath(r"../C4_napari/C4_napari/slide_C4_R5042_S1_Napari_metadata.csv")
MIN_DAPI_INTENSITY_THRESHOLD = 15
MIN_DAPI_AREA = 30 # In pixels
MIN_PANCK_THRESHOLD = 900
GLCM_DISTANCE = [1]
DOWNSAMPLE = int(math.pow(2,6)-1)
REMOVE_GLCM_ZEROES = False
# FOV_GLOBAL_X = int(-4972.22222222222)
# FOV_GLOBAL_Y = int(144450)
RESULTS_FILE = os.path.normpath(r"../CosMx_C4_CellResults_GaborFeats32413.csv")

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
def get_pixels_for_cell(nuclear_mask,dapi_only,cell_id,metadata,fov_x,fov_y, max_local_y):

    row = metadata.loc[metadata["cell_ID"]==cell_id]
    local_center_x = int(row["CenterX_global_px"].values[0] - fov_x)
    local_center_y = int(max_local_y - (row["CenterY_global_px"].values[0] - fov_y))
    width = row["Width"].values[0]
    height = row["Height"].values[0]

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
def mean_for_all_cells(nuclear_dapi,cell_id, cell_lookup, other_params, fov):
    
    # dump zeroes, then check length( i.e. testing the number of pixels in the nuclear mask)
    current = np.ndarray.flatten(nuclear_dapi)
    current = current[current !=0]
    if len(current)<MIN_DAPI_AREA:
        if len(current) >0: print(f'Nucleus of CID {cell_id} is only {len(current)} pixels, dropping this one.')
        return cell_lookup, other_params
    
    ## Checking a specific case
    # if cell_id == 1661 or cell_id == 1663:
    #     print(f'Nucleus of CID {cell_id} is  {len(current)} pixels')
    cell_lookup[f"{fov}_{cell_id}"] = np.mean(current)

    other_params[f"{fov}_{cell_id}_area"] = len(current)
    # if cell_id >2250:   
    #     print(f"\nCID {cell_id} shape is {current.shape}, max is {np.max(current)}, min is {np.min(current)}\n{current} mean is {cell_lookup[str(fov)+'_'+str(cell_id)]}")
    return cell_lookup, other_params

def get_coordinate_conversions(fov):
    meta = pd.read_csv(FOV_POSITIONS)
    meta_fov = meta.loc[meta["fov"] == fov]
    return int(meta_fov["x_global_px"].values[0]), int(meta_fov["y_global_px"].values[0])

def compute_roundness(dapi_cells):
    print(f"\nComputing roundness...")
    label_img = label(dapi_cells)
    regions = regionprops(label_img)

    fig, ax = plt.subplots()
    ax.imshow(dapi_cells, cmap=plt.cm.gray)

    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)

    ax.axis((0, 600, 600, 0))
    plt.show()

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

    other_params[f"{fov}_{cell_id}_texture-correlation"] = correlation
    other_params[f"{fov}_{cell_id}_texture-dissimilarity"] = dissimilarity
    other_params[f"{fov}_{cell_id}_texture-homogeneity"] = homogeneity
    other_params[f"{fov}_{cell_id}_texture-ASM"] = ASM
    other_params[f"{fov}_{cell_id}_texture-energy"] = energy
    other_params[f"{fov}_{cell_id}_texture-contrast"] = contrast
    return other_params

def add_gabor_metrics(nuclear_dapi, nuclear_mask, cell_id, fov, other_params):
    spectrograms, features, labels = pyfeats.gt_features(nuclear_dapi,nuclear_mask, deg=4, freq=[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.08])

    f1_means = []
    f1_std = []
    f2_means = []
    f2_std = []
    for pos, l in enumerate(labels):
        if l.endswith('freq_0.05_mean'): 
            f1_means.append(features[pos])
        elif l.endswith('freq_0.4_mean'):
            f2_means.append(features[pos])
        elif l.endswith('freq_0.05_std'):
            f1_std.append(features[pos])
        else:
            f2_std.append(features[pos])
    other_params[f"{fov}_{cell_id}_gabor0.05_mean"] = np.mean(f1_means)
    other_params[f"{fov}_{cell_id}_gabor0.05_std"] = np.mean(f1_std)
    other_params[f"{fov}_{cell_id}_gabor0.4_mean"] = np.mean(f2_means)
    other_params[f"{fov}_{cell_id}_gabor0.4_std"] = np.mean(f2_std)

    img = plot_spectrogram(spectrograms,nuclear_dapi)
    return img
    # return other_params

def plot_spectrogram(f_series, t_series):
    # f1,f2 = f_series.shape
    f_combined = t_series
    # f_combined = abs(f_series[0])
    # f_combined *= np.max(t_series)/np.max(f_combined)
    # print(f_series[1:])
    for n in f_series[12:22]:
        n = np.asarray(n)
        n  = abs(n)

        n *= np.max(t_series)/(np.max(n) + 0.00001)

        f_combined = np.append(f_combined,n, axis=1)

    both = np.append(t_series,f_combined, axis=1)
    
    # f_series = abs(f_series)
    # f_series *= np.max(t_series)/np.max(f_series)
    # both = np.append(t_series,f_series, axis=1)
    # plt.imshow(t_series)
    # plt.show()
    # plt.imshow(abs(f_series))
    # plt.show()

    return both
    # plt.imshow(both)
    # plt.title('[0.5 , 0.4 , 0.3 , 0.2 , 0.1 , 0.08 , 0.05]')
    # plt.show()
    # exit()

def plot_frequency_domain(f_series, t_series):
    f1,f2 = f_series.shape
    plt.imshow(t_series)
    plt.show()
    plt.imshow(abs(f_series))
    plt.show()
    # exit()

def fourier_stats(nuclear_dapi, nuclear_mask, cell_id, fov, other_params):
    F,features,labels = pyfeats.fps(nuclear_dapi,nuclear_mask)

    print(f"My features are {features}")
    print(f"My labels are {labels}")
    print(f"F is {F}")
    print(f"Dapi spatial domain shape: {nuclear_dapi.shape}")
    print(f"Frequency domain shape: {F.shape}")
    plot_frequency_domain(F, nuclear_dapi)

def add_counts_for_fov(cell_dictionary, other_params, fov, mask_tuple, composite_path):
    cell_mask_path = mask_tuple[0]; compartment_mask_path = mask_tuple[1]
    global_nuclear_mask = read_mask(cell_mask_path, compartment_mask_path)
    # print(f"\ncell_mask shape is {global_nuclear_mask.shape}, max is {np.max(global_nuclear_mask)}, min is {np.min(global_nuclear_mask)}")
    # print(f'unique vals are {np.unique(global_nuclear_mask)}')
    # exit(0)

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

    images = []
    count = 0
    for cell_id in metadata["cell_ID"]:
        if cell_id not in [653,675,806,782]:
            continue
        else:
            print(f"\n LOOKING AT CELL {cell_id}")
        if cell_id %100 ==0:
            print(f'On cell {cell_id}')        
        # print(f"My inputs are CID {cell_id} ,fov {fov}")
        nuclear_dapi, nuclear_mask, coords = get_pixels_for_cell(global_nuclear_mask,dapi_only,cell_id,metadata,fov_global_X,fov_global_Y,max_Y)
        other_params[f"{fov}_{cell_id}_localX"] = coords[0]
        other_params[f"{fov}_{cell_id}_localY"] = coords[1]
        # compute_roundness(dapi_cells)
        try:
            if np.any(nuclear_dapi):
                # other_params = add_glcm_metrics(nuclear_dapi,cell_id, fov, other_params, distances_range,angles_range)

                # other_params = add_gabor_metrics(nuclear_dapi,nuclear_mask, cell_id, fov, other_params)
                img = add_gabor_metrics(nuclear_dapi,nuclear_mask, cell_id, fov, other_params)
                images.append(img)
                # fourier_stats(nuclear_dapi,nuclear_mask, cell_id, fov, other_params)
            else:
                # print(f'Empty list passed to texture creation code for cell {cell_id} in {fov}')
                pass
        except Exception as e:
            print(f'some other error occurred when trying to calculate texture for {cell_id} in {fov}')
            print(f'\n {e}')
            exit()
        cell_dictionary,other_params = mean_for_all_cells(nuclear_dapi, cell_id, cell_dictionary,other_params,fov)

        count +=1
        if count % 4 ==0:
            first = images[0] 
            first = [np.asarray(first)]
            for i in range(len(images)-1):
                next = np.asarray(images[i+1])
                first .append(next)
            print(f"\n\nalength {len(first)}")    

            fig = plt.figure()
            fig.suptitle('[Spatial, 1 , 0.9 , 0.8 , 0.7 , 0.6 , 0.5 , 0.4 , 0.3 , 0.2 , 0.1 , 0.08]')

            #subplot(r,c) provide the no. of rows and columns
            f, axarr = plt.subplots(4,1) 

            # use the created array to output your multiple images. In this case I have stacked 4 images vertically
            for i in range(1,len(axarr)+1):
                ax = fig.add_subplot(4,1,i)
                ax.imshow(first[i-1])
            plt.show()
            images = []

    return cell_dictionary,other_params
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
        
def assemble_df(cell_dict,other_params):
    # df = pd.DataFrame(columns=['global_cid', 'fov', 'local_cid','DAPI Intensity Mean','DAPI Area (px)',
    #     'Texture-correlation','Texture-dissimilarity','Texture-homogeneity','Texture-ASM','Texture-energy','Texture-contrast'])
    df = pd.DataFrame(columns=['global_cid', 'fov', 'local_cid','Local X','Local Y','DAPI Intensity Mean','DAPI Area (px)',
        'Gabor f0.05 mean','Gabor f0.05 std','Gabor f0.4 mean','Gabor f0.4 std'])
    for cid in cell_dict.keys():
        cell_mean = cell_dict[cid]
        both = cid.split("_")
        # d = {'global_cid': cid, "fov":both[0],"local_cid":both[1],
        #     'DAPI Intensity Mean':cell_mean, "DAPI Area (px)": other_params[cid+"_area"],
        #     'Texture-correlation':other_params[cid+"_texture-correlation"],'Texture-dissimilarity':other_params[cid+"_texture-dissimilarity"],
        #     'Texture-homogeneity':other_params[cid+"_texture-homogeneity"],'Texture-ASM':other_params[cid+"_texture-ASM"],
        #     'Texture-energy':other_params[cid+"_texture-energy"],'Texture-contrast':other_params[cid+"_texture-contrast"]}
        d = {'global_cid': cid, "fov":both[0],"local_cid":both[1],'Local X':other_params[f'{cid}_localX'],'Local Y':other_params[f'{cid}_localY'],
            'DAPI Intensity Mean':cell_mean, "DAPI Area (px)": other_params[cid+"_area"],
            'Gabor f0.05 mean':other_params[cid+"_gabor0.05_mean"],'Gabor f0.05 std':other_params[cid+"_gabor0.05_std"],
            'Gabor f0.4 mean':other_params[cid+"_gabor0.4_mean"],'Gabor f0.4 std':other_params[cid+"_gabor0.4_std"]}
        df = pd.concat([df,pd.DataFrame(data=d,index=[1])], ignore_index=True)
    return df

def dump_csv(df):
    df.to_csv(RESULTS_FILE, index=False)
    return None

def main():
    # print(f"max value for dapi means is {max(dapi_means.values())}")
    dapi_means={}
    other_params={}
    start = time.time()
    "..\RawData\MultichannelImages\20220405_130717_S1_C902_P99_N99_F001_Z004.TIF"
    for root,dirs,files in os.walk(os.path.normpath("../RawData/MultichannelImages")):
        # print(files)
        for tif in files:
            lap_start = time.time()
            fov = int(tif.split("_")[-2].lstrip("[F0]"))
            if fov !=19:
                continue
            else: print(f"\n FOV is 19")
            cell_mask = os.path.normpath(os.path.join(root,"../CellLabels/CellLabels_" + tif.split("_")[-2] + ".tif"))
            compartment_mask = os.path.normpath(os.path.join(root,"../CompartmentLabels/CompartmentLabels_" + tif.split("_")[-2] + ".tif"))
            composite_path = os.path.normpath(os.path.join(root,tif))
            print(f"Compartment path is {compartment_mask} and mask is {cell_mask}")
            print(f"\nWorking on FOV {fov}...",end="   ")
            dapi_means, other_params = add_counts_for_fov(dapi_means,other_params, fov, (cell_mask,compartment_mask), composite_path)
            lap_end = time.time() 
            print(f"Completed in {lap_end-lap_start} seconds. Have data for {len(dapi_means)} nuclei in fov {fov}")
            # if fov ==1: break # for testing
    end = time.time()
    print(f"\nAll FOVs runtime: {end-start} seconds.")

    start = time.time()
    print(f"\nAssembling data for {len(dapi_means.keys())} cells... ", end="   ")
    output = add_columns(TRANSCRIPTS,METADATA,CELLTYPING, assemble_df(dapi_means, other_params))
    end = time.time()
    print(f"Completed in {end-start} seconds.")
    print(f"\n Writing to {RESULTS_FILE} ... ", end='   ')
    dump_csv(output)
    print("Done.")

if __name__ == "__main__":
    main()