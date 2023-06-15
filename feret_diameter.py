# Volker Hilsenstein
# BSD-3 license

import numpy as np
import skimage.morphology
from rotating_calipers import min_max_feret

def get_min_max_feret_from_labelim(label_im, labels=None):
    """ given a label image, calculate the oriented 
    bounding box of each connected component with 
    label in labels. If labels is None, all labels > 0
    will be analyzed.
    Parameters:
        label_im: numpy array with labelled connected components (integer)
    Output:
        obbs: dictionary of oriented bounding boxes. The dictionary 
        keys correspond to the respective labels
    """
    if labels is None:
        labels = set(np.unique(label_im)) - {0}
    results = {}
    for label in labels:
        results[label] = get_min_max_feret_from_mask(label_im == label)
    return results

def get_min_max_feret_from_mask(mask_im):
    """ given a binary mask, calculate the minimum and maximum
    feret diameter of the foreground object. This is done
    by calculating the outline of the object, transform
    the pixel coordinates of the outline into a list of
    points and then calling 
    Parameters:
        mask_im: binary numpy array
    """
    eroded = skimage.morphology.erosion(mask_im)
    outline = mask_im ^ eroded
    boundary_points = np.argwhere(outline > 0)
    # convert numpy array to a list of (x,y) tuple points
    boundary_point_list = list(map(list, list(boundary_points)))
    return min_max_feret(boundary_point_list)