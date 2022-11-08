'''
Compare nuclear intensity vs LINE1 transcripts per cell, run correlation / regression

Peter Richieri 
MGH Ting Lab 
11/7/22
'''
import numpy as np
import tifffile # for cell masks
from matplotlib import image # for DAPI .jpg
import pandas as pd
import statistics

CELL_MASK_DATA = r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\RawData\CellLabels\CellLabels_F004.tif"
COMPOSITE_IMAGE = r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\RawData\CellComposite\CellComposite_F004.jpg"
METADATA = r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\RawData\C4_R5042_S1_metadata_file.csv"
FOV_GLOBAL_X = int(-4972.22222222222)
FOV_GLOBAL_Y = int(144450)

''' Get cell data as numpy array, convert to binary, and return. '''
def read_mask(path_to_file):
    with tifffile.Timer(f'\nReading cell masks\n'):
        raw_data = tifffile.imread(path_to_file)
        print('... completed in ', end='')
    # Have to flip is around to be [x,y] instead of [y,x]
    return np.transpose(np.clip(raw_data,0,1),(1,0))

''' Data only exists as composite .jpg. DAPI signal should be the B channel of the RGB data. '''
def extract_dapi_signal(path_to_file):
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

def read_transcripts(path_to_file):
    pass

'''Subset numpy array to only data corresponding to a certain Cell_ID and return that data'''
def get_pixels_for_cell(cell_mask,cell_id,metadata,fov_x,fov_y, max_local_y):
    row = metadata.loc[metadata["cell_ID"]==cell_id]
    local_center_x = int(row["CenterX_global_px"].values[0] - fov_x)
    local_center_y = int(max_local_y - (row["CenterY_global_px"].values[0] - fov_y))
    width = row["Width"].values[0]
    height = row["Height"].values[0]

    if cell_id == 3583:
        print(f"\nCID 3583 cX is {local_center_x} and cY is {local_center_y}\n")
    # if cell_id == 3585:
    #     print(row.loc[:,["cell_ID","Area","CenterX_local_px","Width","Height"]])
    return cell_mask[local_center_x-(width//2):local_center_x+(width//2),local_center_y-(height//2):local_center_y+(height//2)]

''' Return a dictionary of the mean DAPI value for all cells
    Key: a cell ID          value: Maan of DAPI counts per pixel for that cell'''
def mean_for_all_cells(cell_data, metadata, fov_x,fov_y, max_local_y):
    cell_lookup = {}
    for cell_id in metadata["cell_ID"]:
        current = get_pixels_for_cell(cell_data,cell_id,metadata,fov_x,fov_y,max_local_y)
        cell_lookup[cell_id] = np.mean(current)
        # if cell_id >3570:
        #     print(f"\nCID {cell_id} shape is {current.shape}, max is {np.max(current)}, min is {np.min(current)}\n{current} mean is {cell_lookup[cell_id]}")
    return cell_lookup

def main():
    cell_mask = read_mask(CELL_MASK_DATA)
    print(f"\ncell_mask shape is {cell_mask.shape}, max is {np.max(cell_mask)}, min is {np.min(cell_mask)}")

    dapi_only = extract_dapi_signal(COMPOSITE_IMAGE)
    max_X = dapi_only.shape[0]; max_Y = dapi_only.shape[1]
    print(f"\ndapi shape is {dapi_only.shape}, max is {np.max(dapi_only)}, min is {np.min(dapi_only)}")

    # Now make quantitative counts for each DAPI pixel
    dapi_cells = drop_weak_signal(cell_mask * dapi_only,15)
    metadata = read_metadata(METADATA)

    dapi_means = mean_for_all_cells(dapi_cells,metadata, FOV_GLOBAL_X, FOV_GLOBAL_Y, max_Y)
    print(f"max value for dapi means is {max(dapi_means.values())}")
    for id in range(3500,3585):
        print(dapi_means[id])

if __name__ == "__main__":
    main()