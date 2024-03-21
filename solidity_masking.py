''' Make some masks that show positions of Solidity High vs Low cells spatially
    in a field of view. A little more difficult would be adding this to Nanostring's napari viewer.'''



import tifffile
import pandas as pd
import IPython
import napari
from napari.types import ImageData
import math
import numpy as np
from copy import copy
from skimage.morphology import remove_small_holes, binary_erosion
from skimage.color import label2rgb
from PyQt5.QtWidgets import QLabel, QSlider, QHBoxLayout, QCheckBox, QComboBox
from PyQt5.QtCore import QObject, Qt
from magicgui import magicgui
import magicgui.widgets as widgets
import seaborn as sns
import matplotlib.pyplot as plt


THRESHOLD = 0.89
PLOT_FOVS = True


FOV = "16" # 18 has highest split for .89

labels_path = f"../RawData/CellLabels/CellLabels_F0{FOV}.tif"
compartments_path = f"../RawData/CompartmentLabels/CompartmentLabels_F0{FOV}.tif"
tif_path = f"../RawData/MultichannelImages/20220405_130717_S1_C902_P99_N99_F0{FOV}_Z004.TIF"  
max_pixel_value = math.pow(2,16)-1 # bit depth is 16
gamma_correct = np.vectorize(lambda x:x**0.5)
low = 0; high = max_pixel_value
color_range = high - low

solidity_reference = pd.read_csv(r"C4only_coords+solidity_forMask.csv")
df = solidity_reference.loc[solidity_reference['fov'] == int(FOV)].copy()
df = df.set_index('global_cid') # set index to unique identifier for later
low_s = df.loc[df['Solidity_high_low'] == "Solidity Low"]["local_cid"].to_list()
high_s = df.loc[df['Solidity_high_low'] == "Solidity High"]["local_cid"].to_list()

# get labels, make two masks, and merge
labels = tifffile.imread(labels_path)
compartments = tifffile.imread(compartments_path)
compartments[compartments>1]=0 # only keep nuclear
labels = labels * compartments

# find edges of cell nuclear masks
# ero = binary_erosion(labels)
# rims = np.logical_and(labels, ~ero)
# rims[rims>0] = 255

labels_low = copy(labels)
labels_low[np.where(np.isin(labels_low, high_s))] = 0
# Flip around binarized image and multiple for other array
labels_low[labels_low >0 ] = 1
labels_high = labels * (labels_low^1)
labels_high[labels_high >0 ] = 1
# adjust intensity for color
labels_high *= 250
labels_low *=200
labels_high = labels_high.astype(np.uint8)
labels_low = labels_low.astype(np.uint8)
# rims.astype(np.uint8)

tif = tifffile.imread(tif_path)
tif = tif.astype('float64')  # scale to 8 bit for napari?

panck = tif[1,::]
panck = (panck - low) / color_range
# panck = gamma_correct(panck)
panck = panck * 255.0
panck = panck.astype(np.uint8)

cd45 = tif[2,::]
cd45 = (cd45 - low) / color_range
# cd45 = gamma_correct(cd45)
cd45 = cd45 * 255.0
cd45 = cd45.astype(np.uint8)


cd3 = tif[3,::]
cd3 = (cd3 - low) / color_range
# cd3 = gamma_correct(cd3)
cd3 = cd3 * 255.0
cd3 = cd3.astype(np.uint8)

dapi = tif[4,::]
dapi = (dapi - low) / color_range
# dapi = gamma_correct(dapi)
dapi = dapi * 255.0
dapi = dapi.astype(np.uint8)

# dapi = np.clip(dapi,0,np.max(dapi))


# def assign_color(x):   
#     if x == 0:
#         return [0,0,0,0]
#     palette = [[238, 149, 68,255], [244, 207, 130,255], [159, 77, 49,255], [146, 201, 174,255], [193, 213, 225,255], [131, 192, 35,255],[137, 77, 155,255], [185, 154, 196,255], [174, 33, 54,255], [254, 216, 0,255], [113, 231, 213,255], [75, 154, 142,255]]
#     return palette[int(x)%12]

# t = np.vectorize(assign_color)(labels)
# divergent_labels = np.array([assign_color(x) for x in np.nditer(labels)]).reshape((*labels.shape,4))
# divergent_labels = label2rgb(labels, bg_label=0) * 255

resegmented_labels = tifffile.imread(f'../Resegmentation/Labels/C4_{FOV}_nuclearLabels.tif')
resegmented_cell_labels = tifffile.imread(f'../Resegmentation/Labels/C4_{FOV}_10_cellLabels.tif')

# resegmented_labels_divergent = np.array([assign_color(x) for x in np.nditer(resegmented_labels)]).reshape((*resegmented_labels.shape,4))
# resegmented_labels_divergent = label2rgb(resegmented_labels, bg_label=0) * 255
# resegmented_cell_labels_divergent = label2rgb(resegmented_cell_labels, bg_label=0) * 255


viewer = napari.Viewer(title=f"Solidity masking for run C4, FOV {FOV}")

# Movement fxns
@viewer.bind_key('Up')
def scroll_up(viewer):
    z,y,x = viewer.camera.center
    viewer.camera.center = (y-50,x)
@viewer.bind_key('Down')
def scroll_down(viewer):
    z,y,x = viewer.camera.center
    viewer.camera.center = (y+50,x)
@viewer.bind_key('Left')
def scroll_left(viewer):
    z,y,x = viewer.camera.center
    viewer.camera.center = (y,x-50)
@viewer.bind_key('Right')   
def scroll_right(viewer):
    z,y,x = viewer.camera.center
    viewer.camera.center = (y,x+50)
# On Macs, ctrl-arrow key is taken by something else.
@viewer.bind_key('Shift-Right')  
@viewer.bind_key('Shift-Up') 
@viewer.bind_key('Control-Right')  
@viewer.bind_key('Control-Up')   
def zoom_in(viewer):
    viewer.camera.zoom *= 1.3

# viewer.add_image(rims, name = "segmentation", colormap="gray",blending="additive")
low_layer = viewer.add_image(labels_low, name = "Low Solidity (<.89)", colormap="bop orange",blending="additive", opacity=0.5)
high_layer = viewer.add_image(labels_high, name = "High Solidity (>.89)", colormap="bop purple",blending="additive", opacity=0.5)
old_labels_layer = viewer.add_labels(labels.astype(np.uint8),name='ID overlay',blending='additive',opacity=0.5,visible=False )



#Get points
tx = pd.read_csv(r"C:\Users\prich\Desktop\Projects\MGH\CosMx\RawData\C4_R5042_S1_tx_file.csv")
l1_fov = tx.loc[(tx["fov"] == int(FOV)) & (tx["target"].isin(["LINE1_ORF1","LINE1_ORF2"]))].copy()
ymax = l1_fov['y_local_px'].max()
# IPython.embed()
# exit()
l1_fov['y_local_px'] = ymax - l1_fov['y_local_px']
orf1 = l1_fov.loc[l1_fov['target'] == 'LINE1_ORF1']
orf2 = l1_fov.loc[l1_fov['target'] == 'LINE1_ORF2']
orf1_pts = np.array(list(zip(orf1["y_local_px"].to_numpy(),orf1["x_local_px"].to_numpy())))
orf2_pts = np.array(list(zip(orf2["y_local_px"].to_numpy(),orf2["x_local_px"].to_numpy())))

# IPython.embed()
# exit()
orf1_layer = viewer.add_points(orf1_pts, name="orf1",opacity = 0.4)#, face_color='white', edge_color='white')
orf2_layer = viewer.add_points(orf2_pts, name="orf2", opacity = 0.4)#, face_color='white', edge_color='white')

@viewer.bind_key('h')
def hide(viewer):
    low_layer.visible = not low_layer.visible
    high_layer.visible = not high_layer.visible

@viewer.bind_key('Control-h')
def hide_pts(viewer):
    orf1_layer.visible = not orf1_layer.visible
    orf2_layer.visible = not orf2_layer.visible



new_labels_layer = viewer.add_labels(resegmented_labels.astype(np.uint8),name='NEW nuclei overlay',blending='additive',opacity=0.5,visible=False )
new_cell_labels_layer = viewer.add_labels(resegmented_cell_labels.astype(np.uint8),name='NEW cell overlay',blending='additive',opacity=0.5,visible=False )
new_cell_labels_layer = viewer.add_image(resegmented_cell_labels.astype(np.uint8),name='NEW cell overlay',blending='additive',opacity=0.5,visible=False )

viewer.add_image(resegmented_labels, name = 'Halo segmentation',colormap='gray', visible=False)
# viewer.add_image(membrane, name = "B2M/CD298", colormap="gray",blending="additive")
viewer.add_image(panck, name = "PanCK", colormap="green",blending="additive")
# viewer.add_image(cd45, name = "CD45", colormap="red",blending="additive", visible=False)
# viewer.add_image(cd3, name = "CD3", colormap="yellow",blending="additive",visible=False)
viewer.add_image(dapi, name = "DAPI", colormap="gray",blending="additive")


@viewer.bind_key('Ctrl-Shift-h')
def toggle_masks(viewer):
    old_labels_layer.visible = not old_labels_layer.visible
    new_labels_layer.visible = not new_labels_layer.visible

@magicgui(auto_call=True,
        solidity={"widget_type": "FloatSlider", "max":1.0, "min":0.50},
        layout = 'horizontal')
def adjust_threshold(solidity: float = 0.89) -> ImageData: 
    global THRESHOLD
    THRESHOLD = solidity
    df["Solidity_high_low"] = np.where(df['Solidity'] > solidity, "Solidity High","Solidity Low")
    high_s = df.loc[df['Solidity_high_low'] == "Solidity High"]["local_cid"].to_list()
    labels_low = copy(labels)
    labels_low[np.where(np.isin(labels_low, high_s))] = 0
    labels_low[labels_low >0 ] = 1
    # Flip around binarized image and multiple for other array
    labels_high = labels * (labels_low^1)

    labels_high[labels_high >0 ] = 1    
    # adjust intensity for color
    labels_high *= 250
    labels_low *=200
    labels_high = labels_high.astype(np.uint8)
    labels_low = labels_low.astype(np.uint8)
    low_layer.data = labels_low.astype(np.uint8)
    low_layer.name = f'Low Solidity (<{solidity})'
    high_layer.data = labels_high.astype(np.uint8)
    high_layer.name = f'High Solidity (>{solidity})'


readout = QLabel("Cell solidity:")
readout.setStyleSheet('font-size: 14pt')

@viewer.mouse_move_callbacks.append
def show_solidity(image_layer, event):
    try:
        coords = tuple(np.round(event.position).astype(int))
        cid = labels[coords]
        if cid == 0 or coords[0]<0 or coords[1] <0:
            readout.setText("No cell")
            return None # mouse over image background
        s = df.loc[f'C4_R5042_S1_{str(int(FOV))}_{cid}']['Solidity']
        new = f'Cell {cid} solidity: {np.round(s,4)}'
        readout.setText(new)
    except IndexError:
        readout.setText("No cell")
        return None # Mouse is out of bounds


def switch_mode(ptype):
    global PLOT_FOVS
    PLOT_FOVS = ptype

plot_type = QComboBox()
plot_type.addItems(["Current FOV only", "All FOVs", "Facet all FOVs"])
plot_type.activated.connect(lambda: switch_mode(plot_type.currentText()))


w1 = widgets.PushButton(text='Make Plot')
@w1.changed.connect
def make_plot(auto_call = True, value = {}):
    ctypes = ["CAF", "cancer"]#,"T.cell","macrophage"]
    pal ={"Solidity High": "#8074b2", "Solidity Low": "#ffac68"}

    if PLOT_FOVS != 'Current FOV only':
        solidity_reference["Solidity_high_low"] = np.where(solidity_reference['Solidity'] > THRESHOLD, "Solidity High","Solidity Low")
        cdf = solidity_reference.loc[solidity_reference['Cell type'].isin(ctypes)].copy()
    else:
        cdf = df.loc[df['Cell type'].isin(ctypes)].copy()

    vc = cdf[["Solidity_high_low",'Cell type']].value_counts()
    high_cc = vc['Solidity High']['cancer']
    low_cc = vc['Solidity Low']['cancer']
    high_cf = vc['Solidity High']['CAF']
    low_cf = vc['Solidity Low']['CAF']

    if PLOT_FOVS == 'Facet all FOVs':
        o = solidity_reference.loc[solidity_reference["Cancer?"] == "Cancer"].groupby(by = 'fov')["Cancer?"].count().sort_values(ascending=False).index.tolist()
        g = sns.FacetGrid(cdf, col = "fov", col_wrap=5, col_order=o)
        g.map_dataframe(sns.barplot,x= "Cell type",y= "Line1_combined", hue ='Solidity_high_low', 
                        order = ["cancer", "CAF"], hue_order= ["Solidity High",'Solidity Low'], palette=pal)
        g.add_legend()
        cap = f"LINE1 per cell faceted by FOVs, threshold of {THRESHOLD} | {high_cc} high, {low_cc} low cancer cells | {high_cf} high, {low_cf} low CAFs"
        g.fig.suptitle(cap)
        plt.show()
    else:
        if PLOT_FOVS == "All FOVs":
            cap = f"Avg LINE1 per cell in ALL FOVs, threshold of {THRESHOLD} | {high_cc} high, {low_cc} low cancer cells | {high_cf} high, {low_cf} low CAFs"
        else: # only current FOVs
            cap = f"Avg LINE1 per cell in FOV {int(FOV)}, threshold of {THRESHOLD} | {high_cc} high, {low_cc} low cancer cells | {high_cf} high, {low_cf} low CAFs"
        b = sns.barplot(data = cdf, x = 'Cell type', y = 'Line1_combined', order = ["cancer", "CAF"],
                        hue ='Solidity_high_low',hue_order= ["Solidity High",'Solidity Low'], palette=pal)
        b.set(title=cap)
        plt.show()


viewer.window.add_dock_widget([readout, adjust_threshold, plot_type, w1], area = 'bottom')


napari.run()
 
# IPython.embed()