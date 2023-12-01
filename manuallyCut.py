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



def correct_skew(image, delta=0.5, limit=45):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
            borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected


def bg_remove3(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)
    return output_path


def findingTempl_byLeastSquaresMethod(fullImage_path, imageTemplate_path):

    image = cv2.imread(fullImage_path)
    templ = cv2.imread(imageTemplate_path)

    # https://github.com/neemiasbsilva/object-detection-opencv/blob/master/template-matching.ipynb
    # methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED',
    #            'cv2.TM_CCORR', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED',
    #            'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    # for m in methods:

    m = 'cv2.TM_SQDIFF_NORMED'
    image_copy = image.copy()
    method = eval(m)
    res = cv2.matchTemplate(image, templ, method)
    min_val, max_val, min_location, max_location = cv2.minMaxLoc(res)
    top_left = min_location
    height, width, channels = templ.shape
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(image_copy, top_left, bottom_right, (0, 0, 255), 2)
    # cv2.imshow(m, image_copy)
    # cv2.waitKey(0)

    cut_color_card = image[top_left[1]:(top_left[1] + height), top_left[0]:(top_left[0] + width)]
    # cv2.imshow("cut_color_card", cut_color_card)
    # cv2.waitKey(0)

    return cut_color_card



if __name__ == '__main__':

    number_Fridge = '5'
    path_to_image_fromFridge = './imageSetfromFridges/fromFridge_5_112723in1146_origin.jpg'
    path_to_temple_roomCard_fromFridges = './manuallyCutCards/templeRoomCard_fromFridge_5.jpg'
    path_to_fullImage_withCrudeRoomCard = './imageSetfromFridges/fullImage_fromFridge_5.jpg'
    path_to_roomCard_fromFridges = './roomCard_forCalibrate/roomCard_' + number_Fridge + '.jpg'

    refCard = cv2.imread('refCard.jpg')

    if len(sys.argv) == 1:
        crudRoomCard = findingTempl_byLeastSquaresMethod(path_to_fullImage_withCrudeRoomCard,
                                                         path_to_temple_roomCard_fromFridges)
    else:
        number_Fridge = sys.argv[1]
        path_to_image_fromFridge = sys.argv[2]
        path_to_fullImage_withCrudeRoomCard = sys.argv[3]
        path_to_temple_roomCard_fromFridges = './manuallyCutCards/templeRoomCard_fromFridge_' + number_Fridge + '.jpg'
        crudRoomCard = findingTempl_byLeastSquaresMethod(path_to_fullImage_withCrudeRoomCard,
                                                         path_to_temple_roomCard_fromFridges)

    angle, crudRoomCard = correct_skew(crudRoomCard)

    height, width, channels = refCard.shape
    dsize = (width, height)
    roomCard = cv2.resize(crudRoomCard, dsize)


    print('angle: ', angle)
    cv2.imshow("RoomCard", roomCard)
    cv2.waitKey(0)

    cv2.imwrite(path_to_roomCard_fromFridges, roomCard)




