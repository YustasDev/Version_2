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


if __name__ == '__main__':



    number_Fridge = '3'
    path_to_image_fromFridge = './cards/rotateTest1.jpg'
    refCard = cv2.imread('refCard.jpg')

    if len(sys.argv)==1:
        roomCard = find_color_card(path_to_image_fromFridge)

    else:
        number_Fridge = sys.argv[1]
        path_to_image_fromFridge = sys.argv[2]
        roomCard = find_color_card(path_to_image_fromFridge)

    # cv2.imshow("roomCard", roomCard)
    # cv2.waitKey(0)

    height, width, channels = refCard.shape
    dsize = (width, height)
    roomCard = cv2.resize(roomCard, dsize)

    path_to_roomCard = './roomCard_forCalibrate/roomCard_' + number_Fridge + '.jpg'
    cv2.imwrite(path_to_roomCard, roomCard)






