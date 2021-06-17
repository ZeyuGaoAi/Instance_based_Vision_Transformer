'''
Author: your name
Date: 2020-11-22 22:04:16
LastEditTime: 2020-11-22 22:14:28
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /grading/postprocessing.py
'''

from skimage.morphology import watershed
import numpy as np
from skimage.measure import label
from skimage.morphology import reconstruction, dilation, erosion, disk, diamond, square
from skimage import img_as_ubyte
import cv2 
import os 

def PrepareProb(img, convertuint8=True, inverse=True):
    """
    Prepares the prob image for post-processing, it can convert from
    float -> to uint8 and it can inverse it if needed.
    """
    if convertuint8:
        img = img_as_ubyte(img)
    if inverse:
        img = 255 - img
    return img


def HreconstructionErosion(prob_img, h): # h值越大，概率图被腐蚀的越严重
    """
    Performs a H minimma reconstruction via an erosion method.
    """

    def making_top_mask(x, lamb=h):
        return min(255, x + lamb)

    f = np.vectorize(making_top_mask)
    shift_prob_img = f(prob_img)

    seed = shift_prob_img
    mask = prob_img
    recons = reconstruction(
        seed.copy(), mask.copy(), method='erosion').astype(np.dtype('ubyte'))
    return recons


def find_maxima(img, convertuint8=False, inverse=False, mask=None):
    """
    Finds all local maxima from 2D image.
    """
    img = PrepareProb(img, convertuint8=convertuint8, inverse=inverse)
    recons = HreconstructionErosion(img, 1)
    if mask is None:
        return recons - img
    else:
        res = recons - img
        res[mask==0] = 0
        return res


def GetContours(img):
    """
    Returns only the contours of the image.
    The image has to be a binary image 
    """
    img[img > 0] = 1
    return dilation(img, disk(2)) - erosion(img, disk(2))


def generate_wsl(ws):
    """
    Generates watershed line that correspond to areas of
    touching objects.
    """
    se = square(3)
    ero = ws.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se)
    ero[ws == 0] = 0

    grad = dilation(ws, se) - ero
    grad[ws == 0] = 0
    grad[grad > 0] = 255
    grad = grad.astype(np.uint8)
    return grad


def DynamicWatershedAlias(p_img, lamb, p_thresh = 0.5):
    """
    Applies our dynamic watershed to 2D prob/dist image.
    """ 
    b_img = (p_img > p_thresh) + 0
    Probs_inv = PrepareProb(p_img)

    Hrecons = HreconstructionErosion(Probs_inv, lamb) # 7
    markers_Probs_inv = find_maxima(Hrecons, mask = b_img)
    markers_Probs_inv = label(markers_Probs_inv)

    ws_labels = watershed(Hrecons, markers_Probs_inv, mask=b_img)
    
    arrange_label = ArrangeLabel(ws_labels)
    wsl = generate_wsl(arrange_label)
    arrange_label[wsl > 0] = 0

    return arrange_label


def ArrangeLabel(mat):
    """
    Arrange label image as to effectively put background to 0.
    """
    val, counts = np.unique(mat, return_counts=True)
    background_val = val[np.argmax(counts)]
    mat = label(mat, background = background_val)
    if np.min(mat) < 0:
        mat += np.min(mat)
        mat = ArrangeLabel(mat)
    return mat


def PostProcess(prob_image, param=7, thresh = 0.5): # 分割程度，腐蚀程度
    """
    param:  控制marker的大小
    Perform DynamicWatershedAlias with some default parameters.
    """
    segmentation_mask = DynamicWatershedAlias(prob_image, param, thresh)
    return segmentation_mask


def gaussianmap2binary(positive_map, binary_map, param1=100, param2=30):
    positive_mask = positive_map
    Probs_inv = PrepareProb(positive_mask)
    Hrecons = HreconstructionErosion(Probs_inv, param1)
    b_img = binary_map 
    b_img[b_img>param2] = 255 #最优值30
    b_img[b_img!=255] = 0 

    markers_Probs_inv = label(find_maxima(Hrecons, mask = b_img))

    ws_labels = watershed(Probs_inv, markers_Probs_inv, mask=b_img)

    arrange_label = ArrangeLabel(ws_labels)
    wsl = generate_wsl(arrange_label)
    arrange_label[wsl > 0] = 0

    for i in range(1,np.max(np.unique(arrange_label))+1):
        if(np.sum(arrange_label==i) <= 10):
            arrange_label[arrange_label==i] = 0

    return arrange_label
