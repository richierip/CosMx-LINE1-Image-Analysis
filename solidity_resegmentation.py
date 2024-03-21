'''
    Project: Examine the segmentation on a CosMx FOV. Using CosMx data and a HALO segmentation image,
        create new counts tables. Examine the difference between the CK signal in non-cancer cells
'''


import numpy as np
import tifffile
import pandas as pd
from IPython import embed
import matplotlib.pyplot as plt 
from skimage.transform import rescale, resize
from skimage.measure import label, regionprops_table, regionprops
from skimage.segmentation import expand_labels, watershed
from skimage.morphology import binary_erosion
import copy
from math import pow
import os
import pathlib

MIN_PANCK_THRESHOLD = 900
CYTOPLASM_DILATION = 10 # pixels
# tx = pd.read_csv(r"C:\Users\prich\Desktop\Projects\MGH\CosMx_Data\RawData\C4_R5042_S1_tx_file.csv")
image_masks_path = os.path.normpath(r"N:\Imagers\ImageProcessing\Peter\CosMx Resegmentation\Resegmented_Masks_allRuns")
labels_path = os.path.normpath(r"N:\Imagers\ImageProcessing\Peter\CosMx Resegmentation\Resegmented_Labels_allRuns")


def convert(img, original_bit_depth, target_bit_depth, target_type):
    imin = 0
    imax = pow(2,original_bit_depth) -1
    target_type_max = pow(2,target_bit_depth) -1
    target_type_min = 0

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def shrink_cells_by_nuclear_size(cell_labels, nuc_labels, initial_expansion_dist):
    nl = nuc_labels.copy()
    nl[nl>0]=1
    connected = label(nl)
    cell_labels = cell_labels.copy()
    props_table = pd.DataFrame(regionprops_table(nuc_labels,
                 properties=["label","bbox","area","axis_minor_length"])).rename(columns={"bbox-0":"XMin", "bbox-1":"YMin","bbox-2":"XMax","bbox-3":"YMax"})
    for index,row in props_table.iterrows():
        label = row["label"]
        nuc_size = row["area"]
        ax = row["axis_minor_length"]
        xmin = row["XMin"]; xmax = row["XMax"]; ymin = row["YMin"]; ymax = row["YMax"]
        xend = nuc_labels.shape[1]; yend = nuc_labels.shape[0]
        offset = initial_expansion_dist+5
        cutout = cell_labels[max(0,ymin-offset):min(ymax+offset, yend), max(0,xmin-offset):max(xmax+offset, xend)].copy()
        cutout_initial = cutout.copy() # save for later
        cutout[cutout !=label] = 0
        
        # calc desired expansion amount (1/3rd minor axis). Values should range between 4 and 24
        val = ax // 3
        # Convert to erosion. Values will be capped. In effect, the pseudoexpansion will be between 3 and initial_expansion_dist
        val = min(initial_expansion_dist-3, initial_expansion_dist - val)
        val = max(val, 0) # ensure no negatives
        if val !=0:
            ero = binary_erosion(cutout.astype(np.bool8), footprint=[(np.ones((val, 1)), 1), (np.ones((1, val)), 1)]).astype(np.uint8)
            ero[ero !=0] = label
            cutout_initial[cutout_initial == label] = 0
            cutout_initial += ero
            cell_labels[ymin-offset:ymax+offset, xmin-offset:xmax+offset] = cutout_initial
        else:
            pass
    
    return cell_labels



run_dictionary = {} # "C4":[image1, image2 ... imageN], "D10": [image1, image2...]
for fileN in os.listdir(image_masks_path):
    run_name = fileN.split("_")[0]
    try:
        run_dictionary[run_name]
    except KeyError:
        run_dictionary[run_name] = [fileN]
        continue
    run_dictionary[run_name].append(fileN)

global_metadata = pd.DataFrame()
global_counts_table = pd.DataFrame()
fov_count = 0
for run in run_dictionary.keys():

    # Get transcript file
    tx_path = os.path.normpath(image_masks_path+f"/../C4/")
    for f in os.listdir(tx_path):
        if f.endswith("_tx_file.csv"): break
    print(f"Reading {run} transcript file ...")
    tx = pd.read_csv(os.path.normpath(tx_path +"/"+f))


    for FOV in range(1,len(run_dictionary[run])):
        FOV = f'0{FOV}' if FOV<10 else FOV
        print(f"\nProcessing {run} FOV {FOV}. Have done {fov_count}")
        fov_count+=1
    # FOV = "16" # 18 has highest split for .89
        

    
        nuclear = tifffile.imread(os.path.normpath(image_masks_path + f"/{run}_F0{FOV}_CosMxResegmentationMask.tif"))

        for fl in os.listdir(os.path.normpath(image_masks_path + f"/../{run}/MultichannelImages/")):
            if f"F0{FOV}" in fl: break

        tif = tifffile.imread(os.path.normpath(image_masks_path + f"/../{run}/MultichannelImages/{fl}"))
        tif = np.transpose(tif, (1,2,0)) # put channels at the end

        # old_labels_path = f"../RawData/CellLabels/CellLabels_F0{FOV}.tif"
        # old_labels = tifffile.imread(old_labels_path)
        # # Binarize halo mask - pixels over a detected nucleus will have a 1, and background will have a 0
        # im = im[:,:,2]
        # im[im<100] = 0
        # im[im>=100] = 1

        # nuclear = resize(im, old_labels.shape, anti_aliasing=True)
        # nuclear[nuclear !=0] = 1 # binarize again for some reason?? Pixels are float values before this.

        # label with cell
        nuc_labels = label(nuclear.astype(np.uint), connectivity=1) # diagonal touching doesn't count
        dilation_for_cytoplasm = expand_labels(nuc_labels, distance=CYTOPLASM_DILATION)

        tifffile.imwrite(f'{labels_path}/{run}_{FOV}_nuclearLabels.tif', nuc_labels)
        tifffile.imwrite(f'{labels_path}/{run}_{FOV}_{CYTOPLASM_DILATION}_cellLabels.tif', dilation_for_cytoplasm)
        # continue
        
        # Correct y coord issue
        ymax = tif.shape[0] # this is the y axis channel  #tx.loc[tx['fov'] == int(FOV), 'y_local_px'].max()
        # print(ymax)
        tx.loc[tx['fov']==int(FOV),['y_local_px']] = ymax - tx['y_local_px']
        # assign transcripts to cell ID
        new_tx = tx.copy()
        new_tx['cell_ID'] = dilation_for_cytoplasm[new_tx['y_local_px'].astype(int),new_tx['x_local_px'].astype(int)]

        # collapse and pivot into the counts matrix
        t = new_tx.loc[new_tx["fov"]==int(FOV)].groupby(by='cell_ID')['target'].value_counts().to_frame().rename(columns = 
                            {"target":"target_count"}).reset_index().pivot(index="cell_ID",
                                columns="target", values="target_count").fillna(0)
        # Locate and add rows for missing cells (cells with no transcripts will not appear in this table and mess up a merge later)
        missingnos = list(set(np.unique(dilation_for_cytoplasm)).difference(set(t.index))) # find cids missing
        line = pd.DataFrame(0, index = missingnos, columns = t.columns) # create a frame for the missing cids
        t = pd.concat([t,line])
        t = t.sort_index().reset_index().rename(columns={"index":"cell_ID"}) # create a column from the index
        t["fov"] = f"{FOV}" # Multiple FOVs in an image with duplicate cell_IDs, so must label FOV
        t["Run_ID"] = run
        # t.to_csv(f"../Resegmentation/FOV Counts Tables/C4_{FOV}_Cyto{CYTOPLASM_DILATION}_resegmentation_countsTable.csv", index=False)
        global_counts_table = pd.concat([global_counts_table, t])

        props = ["label", "area", "eccentricity","feret_diameter_max","intensity_mean", "intensity_max", "perimeter","solidity"]
        nuc_metadata = regionprops_table(nuc_labels,tif[:,:,4], properties = props)
        nuc_metadata = pd.DataFrame(nuc_metadata).rename(columns={"label":"cell_ID","area":"Nuclear area", "perimeter":"Nuclear perimeter",
                                                                "intensity_mean":"Mean DAPI","intensity_max":"Max DAPI"}) # only want DAPI intensity in the nucleus

        props = ["label", "area","centroid", "bbox", "intensity_mean", "intensity_max","perimeter"]
        metadata = regionprops_table(dilation_for_cytoplasm, tif, properties = props)

        cell_metadata = pd.DataFrame(metadata).rename(columns={"label":"cell_ID","centroid-0":"fovX","centroid-1":"fovY",
                                            "bbox-0":"XMin", "bbox-1":"YMin","bbox-2":"XMax","bbox-3":"YMax",
                                            "perimeter":"Cell perimeter","area":"Cell area",
                                            "intensity_mean-1":"Mean PanCK","intensity_max-1":"Max PanCK",
                                            "intensity_mean-2":"Mean CD45","intensity_max-2":"Max CD45",
                                            "intensity_mean-3":"Mean CD3","intensity_max-3":"Max CD3",
                                            "intensity_mean-4":"Mean Membrane","intensity_max-4":"Max Membrane"}).drop(columns=["intensity_mean-0", "intensity_max-0"])
        # add total counts (exclude CID and FOV, and drop cell_ID = 0, which is the background)
        cell_metadata["totalcounts"] = t.drop(columns=["cell_ID","fov","Run_ID"]).sum(axis = 1).astype(int)[1:].reset_index(drop=True)

        metadata = cell_metadata.merge(nuc_metadata, how="left",on="cell_ID") # add nuclear characteristics
        metadata["fov"] = f"{FOV}" # add FOV 
        metadata["Run_ID"] = run
        
        global_metadata = pd.concat([global_metadata, metadata])
        # metadata.to_csv(f"../Resegmentation/FOV Metadata/C4_{FOV}_Cyto{CYTOPLASM_DILATION}_metadata.csv", index=False)
    # embed()

outpath = os.path.normpath("N:\Imagers\ImageProcessing\Peter\CosMx Resegmentation\Results")
# Done. Write to file, and embed in case of problem
global_metadata.to_csv(f"{outpath}/allRuns_allFOV_cyto{CYTOPLASM_DILATION}_metadata.csv", index=False)
global_counts_table.to_csv(f"{outpath}/allRuns_allFOV_cyto{CYTOPLASM_DILATION}_countsTable.csv", index=False)

embed()
# exit()
