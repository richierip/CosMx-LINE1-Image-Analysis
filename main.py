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

# For roundness metrics

import math
import matplotlib.pyplot as plt
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate

# CELL_MASK_DATA = r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\RawData\CellLabels\CellLabels_F004.tif"
# COMPOSITE_IMAGE = r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\RawData\CellComposite\CellComposite_F004.jpg"
METADATA = r"..\RawData\C4_R5042_S1_metadata_file.csv"
FOV_POSITIONS = r"..\RawData\C4_R5042_S1_fov_positions_file.csv"
TRANSCRIPTS = r"..\RawData\C4_R5042_S1_exprMat_file.csv"
MIN_DAPI_INTENSITY_THRESHOLD = 15
MIN_DAPI_AREA = 30 # In pixels
MIN_PANCK_THRESHOLD = 900
# FOV_GLOBAL_X = int(-4972.22222222222)
# FOV_GLOBAL_Y = int(144450)
RESULTS_FILE = r"..\DAPI_Intensity_by_Cell.csv"

''' Get cell data as numpy array, convert to binary, and return. '''
def read_mask(path_to_file):
    print(f'Reading cell masks',end="   ")
    raw_data = tifffile.imread(path_to_file)
    # Have to flip is around to be [x,y] instead of [y,x]
    return np.transpose(np.clip(raw_data,0,1),(1,0))

''' Data only exists as composite .jpg. DAPI signal should be the B channel of the RGB data. '''
def extract_dapi_signal(path_to_file):
    print(f'Reading composite',end="   ")
    raw_data = image.imread(path_to_file)
    # Also have to flip is around to be [x,y] instead of [y,x]
    return np.transpose(raw_data[:,:,2],(1,0))

''' Take the numpy array and remove any signal lower than the chosen threshold'''
def drop_weak_signal(arr, lower_limit):
    arr[arr < lower_limit] = 0
    return arr

''' Read in metadata file and return data for the current fov'''
def read_metadata(path_to_file, fov = 4):
    meta = pd.read_csv(path_to_file)
    meta_fov = meta.loc[meta["fov"] == fov]
    return meta_fov

'''Subset numpy array to only data corresponding to a certain Cell_ID and return that data'''
def get_pixels_for_cell(cell_mask,cell_id,metadata,fov_x,fov_y, max_local_y):
    row = metadata.loc[metadata["cell_ID"]==cell_id]
    local_center_x = int(row["CenterX_global_px"].values[0] - fov_x)
    local_center_y = int(max_local_y - (row["CenterY_global_px"].values[0] - fov_y))
    width = row["Width"].values[0]
    height = row["Height"].values[0]
    # if cell_id == 778:
    #     print(row.loc[:,["cell_ID","Area","CenterX_local_px","CenterY_local_px","Width","Height"]])
    #     print(f"X and Y {local_center_x} and {local_center_y}")
    return cell_mask[local_center_x-(width//2):local_center_x+(width//2),local_center_y-(height//2):local_center_y+(height//2)]

''' Return a dictionary of the mean DAPI value for all cells
    Key: a cell ID          value: Mean of DAPI counts per pixel for that cell'''
def mean_for_all_cells(cell_data, cell_lookup, other_params, metadata, fov, fov_x,fov_y, max_local_y):
    for cell_id in metadata["cell_ID"]:
        # print(f"My inputs are CID {cell_id} ,fov {fov}, fovX {fov_x}, fovY {fov_y}")
        current = get_pixels_for_cell(cell_data,cell_id,metadata,fov_x,fov_y,max_local_y)
        
        # dump zeroes
        current = np.ndarray.flatten(current)
        current = current[current !=0]
        if len(current)<MIN_DAPI_AREA:
            continue
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

def add_counts_for_fov(cell_dictionary, other_params, fov, cell_mask_data, composite):
    cell_mask = read_mask(cell_mask_data)
    # print(f"\ncell_mask shape is {cell_mask.shape}, max is {np.max(cell_mask)}, min is {np.min(cell_mask)}")

    dapi_only = extract_dapi_signal(composite)
    max_X = dapi_only.shape[0]; max_Y = dapi_only.shape[1]
    # print(f"\ndapi shape is {dapi_only.shape}, max is {np.max(dapi_only)}, min is {np.min(dapi_only)}")

    # Now make quantitative counts for each DAPI pixel
    dapi_cells = drop_weak_signal(cell_mask * dapi_only,MIN_DAPI_INTENSITY_THRESHOLD)

    # compute_roundness(dapi_cells)

    metadata = read_metadata(METADATA, fov)
    fov_global_X, fov_global_Y = get_coordinate_conversions(fov)

    return mean_for_all_cells(dapi_cells,cell_dictionary,other_params, metadata,fov, fov_global_X, fov_global_Y, max_Y)

def add_transcripts(path_to_file, cell_df):
    transcripts = pd.read_csv(path_to_file)

    # Need this to get PanCK data
    metadata = pd.read_csv(METADATA) 

    cell_df.insert(5,"Line1_ORF1", "")
    cell_df.insert(6,"Line1_ORF2", "")
    cell_df.insert(7,"Line1_Combined", "")
    cell_df.insert(8,"Mean PanCK","")
    cell_df.insert(9,"Cancer?","")
    for index,cell in cell_df.iterrows():
        cell_tx = transcripts[(transcripts["fov"] == int(cell["fov"])) & (transcripts["cell_ID"]== int(cell["local_cid"]))]
        panck = metadata.loc[(metadata["fov"] == int(cell["fov"])) & (metadata["cell_ID"]== int(cell["local_cid"])),"Mean.PanCK"].values[0]
        orf1 = cell_tx["LINE1_ORF1"].values[0]
        orf2 = cell_tx["LINE1_ORF2"].values[0]
        cell_df.at[index,"Line1_ORF1"] = orf1
        cell_df.at[index,"Line1_ORF2"] = orf2
        cell_df.at[index,"Line1_Combined"] = orf1 + orf2
        cell_df.at[index,"Mean PanCK"] = panck
        if panck > MIN_PANCK_THRESHOLD:
            cell_df.at[index,"Cancer?"] = "Cancer"
        else:
            cell_df.at[index,"Cancer?"] = "Not Cancer"
    return cell_df
        
def assemble_df(cell_dict,other_params):
    df = pd.DataFrame(columns=['global_cid', 'fov', 'local_cid','DAPI Intensity Mean','DAPI Area (px)'])
    for cid in cell_dict.keys():
        cell_mean = cell_dict[cid]
        both = cid.split("_")
        d = {'global_cid': cid, "fov":both[0],"local_cid":both[1]
            ,'DAPI Intensity Mean':cell_mean, "DAPI Area (px)": other_params[cid+"_area"]}
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
    for root,dirs,files in os.walk("../RawData/CellComposite"):
        # print(files)
        for composite in files:
            cell_mask = os.path.normpath(os.path.join(root,"../CellLabels/CellLabels_" + composite.split("_")[1].rstrip(".jpg") + ".tif"))
            fov = int(composite.split("_")[1].rstrip(".jpg").lstrip("[F0]"))
            composite_path = os.path.normpath(os.path.join(root,composite))
            # print(f"Composite path is {composite_path} and mask is {cell_mask}")
            print(f"\nWorking on FOV {fov}...",end="   ")
            dapi_means, other_params = add_counts_for_fov(dapi_means,other_params, fov, cell_mask, os.path.join(root,composite))
            print("Done.")
            # if fov ==1: break # for testing
    end = time.time()
    print(f"\nTotal runtime: {end-start} seconds.")

    print(f"\nAssembling data for {len(dapi_means.keys())} cells... ", end="   ")
    output = add_transcripts(TRANSCRIPTS, assemble_df(dapi_means, other_params))
    print("Done.")
    print(f"\n Writing to {RESULTS_FILE} ... ", end='   ')
    dump_csv(output)
    print("Done.")

if __name__ == "__main__":
    main()