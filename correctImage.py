from imutils.perspective import four_point_transform
from skimage.exposure import is_low_contrast
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from skimage.io import imread, imshow
from imutils.paths import list_images
from skimage import exposure
from scipy.ndimage import interpolation as inter
import numpy as np
from rembg import remove
import imutils
import cv2
import sys
from sys import argv
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def find_color_card(path_to_image):

    image = cv2.imread(path_to_image)
    # load the ArUCo dictionary, grab the ArUCo parameters, and
    # detect the markers in the input image
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    # try to extract the coordinates of the color correction card
    try:
        # otherwise, we've found the four ArUco markers, so we can
        # continue by flattening the ArUco IDs list
        ids = ids.flatten()
        # extract the top-left marker
        i = np.squeeze(np.where(ids == 923))
        topLeft = np.squeeze(corners[i])[0]
        # extract the top-right marker
        i = np.squeeze(np.where(ids == 1001))
        topRight = np.squeeze(corners[i])[1]
        # extract the bottom-right marker
        i = np.squeeze(np.where(ids == 241))
        bottomRight = np.squeeze(corners[i])[2]
        # extract the bottom-left marker
        i = np.squeeze(np.where(ids == 1007))
        bottomLeft = np.squeeze(corners[i])[3]
    # we could not find color correction card, so gracefully return
    except:
        return None

    # build our list of reference points and apply a perspective
    # transform to obtain a top-down, bird’s-eye view of the color
    # matching card
    cardCoords = np.array([topLeft, topRight, bottomRight, bottomLeft])
    card = four_point_transform(image, cardCoords)
    # return the color matching card to the calling function
    return card

def bg_remove3(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)
    return output_path


def match_histograms_mod(inputCard, referenceCard, fullImage):
    """
        Return modified full image, by using histogram equalizatin on input and
         reference cards and applying that transformation on fullImage.
    """
    if inputCard.ndim != referenceCard.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.')
    matched = np.empty(fullImage.shape, dtype=fullImage.dtype)
    for channel in range(inputCard.shape[-1]):
        matched_channel = _match_cumulative_cdf_mod(inputCard[..., channel], referenceCard[..., channel],
                                                    fullImage[..., channel])
        matched[..., channel] = matched_channel
    return matched


def _match_cumulative_cdf_mod(source, template, full):
    """
    Return modified full image array so that the cumulative density function of
    source array matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

    # Here we compute values which the channel RGB value of full image will be modified to.
    interpb = []
    for i in range(0, 256):
        interpb.append(-1)

    # first compute which values in src image transform to and mark those values.

    for i in range(0, len(interp_a_values)):
        frm = src_values[i]
        to = interp_a_values[i]
        interpb[frm] = to

    # some of the pixel values might not be there in interp_a_values, interpolate those values using their
    # previous and next neighbours
    prev_value = -1
    prev_index = -1
    for i in range(0, 256):
        if interpb[i] == -1:
            next_index = -1
            next_value = -1
            for j in range(i + 1, 256):
                if interpb[j] >= 0:
                    next_value = interpb[j]
                    next_index = j
            if prev_index < 0:
                interpb[i] = (i + 1) * next_value / (next_index + 1)
            elif next_index < 0:
                interpb[i] = prev_value + ((255 - prev_value) * (i - prev_index) / (255 - prev_index))
            else:
                interpb[i] = prev_value + (i - prev_index) * (next_value - prev_value) / (next_index - prev_index)
        else:
            prev_value = interpb[i]
            prev_index = i

    # finally transform pixel values in full image using interpb interpolation values.
    wid = full.shape[1]
    hei = full.shape[0]
    ret2 = np.zeros((hei, wid))
    for i in range(0, hei):
        for j in range(0, wid):
            ret2[i][j] = interpb[full[i][j]]
    return ret2





if __name__ == '__main__':

    number_Fridge = '2'
    path_to_image_fromFridge = './imageSetfromFridges/fromFridge_2_112923in1209_origin.jpg'
    path_to_roomCard_fromFridges = './roomCard_forCalibrate/roomCard_' + number_Fridge + '.jpg'

    refCard = cv2.imread('refCard.jpg')

    if len(sys.argv)==1:
        roomCard = cv2.imread(path_to_roomCard_fromFridges)
    else:
        number_Fridge = sys.argv[1]
        path_to_image_fromFridge = sys.argv[2]
        path_to_roomCard_fromFridges = './roomCard_forCalibrate/roomCard_' + number_Fridge + '.jpg'
        roomCard = cv2.imread(path_to_roomCard_fromFridges)

    output_image = 'removeBG_fromFridgeImage.jpg'
    input_image_path = bg_remove3(path_to_image_fromFridge, output_image)
    input_image = cv2.imread(input_image_path)
    cv2.imshow("input image", input_image)
    cv2.waitKey(0)

    result_image = match_histograms_mod(roomCard, refCard, input_image)
    outFileName = './outputImages/correctedImage_fromFridge_' + number_Fridge + '.jpg'
    cv2.imwrite(outFileName, result_image)
    os.remove(output_image)
    cv2.imshow("corrected input image with colorCard", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
