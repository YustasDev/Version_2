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



#https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
def find_color_card(image):
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


#https://pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/
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

# https://pyimagesearch.com/2021/01/25/detecting-low-contrast-images-with-opencv-scikit-image-and-python/
def low_contrast_image_processing(imagePath):
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=450)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur the image slightly (to reduce high frequency noise) and perform edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)

    # initialize the text and color to indicate that the input image
    # is *not* low contrast
    text = "Low contrast: No"
    color = (0, 255, 0)

    # check to see if the image is low contrast
    if is_low_contrast(gray, fraction_threshold=0.35):
        # update the text and color
        text = "Low contrast: Yes"
        color = (0, 0, 255)

    # otherwise, the image is *not* low contrast, so we can continue processing it
    else:
        # find contours in the edge map and find the largest one,
        # which we'll assume is the outline of our color correction card
        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        # draw the largest contour on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    # draw the text on the output image
    cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 2)
    # show the output image and edge map
    cv2.imshow("Image", image)
    cv2.imshow("Edge", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#https://docs.opencv.org/3.0-beta/modules/imgproc/doc/object_detection.html
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


#https://github.com/danielgatis/rembg
def bg_remove3(input_path, output_path):
    with open(input_path, 'rb') as i:
        with open(output_path, 'wb') as o:
            input = i.read()
            output = remove(input)
            o.write(output)
    return output_path

def perspective_transformation(input_file):

    img = cv2.imread(input_file)
    rows, cols, ch = img.shape
    pts1 = np.float32([[60, 10], [787, 21], [54, 1104], [841, 1077]])
    pts2 = np.float32([[0, 0], [841, 0], [0, 1104], [841, 1104]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (850, 1105))

    correctedFile = 'correctedImg.jpg'
    cv2.imwrite(correctedFile, dst)
    #cv2.imshow('corrected', dst)
    return correctedFile



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


def whitepatch_balancing(image, from_row, from_column, row_width, column_width):
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].imshow(image)
    ax[0].add_patch(Circle((from_column, from_row),
                              linewidth=3,
                              edgecolor='r', facecolor='none'));

    ax[0].set_title('Original image')
    image_patch = image[from_row:from_row+row_width, from_column:from_column+column_width]
    image_max = (image*1.0 / image_patch.max(axis=(0, 1))).clip(0, 1)
    ax[1].imshow(image_max);
    ax[1].set_title('Whitebalanced Image with MAX value of white')
    plt.show()
    return image_max







if __name__ == '__main__':


    """    option #1 - photos of Color Cards of good quality   """
    # reading a reference color card (perfect quality)
    pre_ref_card = cv2.imread('./ref2.jpg')

    # reading a room(fridge) color card (good quality)
    pre_room_card = cv2.imread('./cards/A4DFF7EA-B4A1-401C-92DE-5F0958B541E5.jpg')

    # resize the reference and input images
    ref_card = imutils.resize(pre_ref_card, width=600)
    skew_room_card = imutils.resize(pre_room_card, width=600)

    # pre-treatment in case of angular inclination  - by default, we assume that the picture is skew
    angle, room_card = correct_skew(skew_room_card)
    print('angle: ', angle)
    # cv2.imshow("room_card correction", room_card)
    # cv2.waitKey(0)


    #find the color matching card in each image
    print("[INFO] finding color matching cards...")
    refCard = find_color_card(ref_card)
    roomCard = find_color_card(room_card)

    # if necessary, you can save the working version of the "roomCard" and "refCard"
    # cv2.imwrite('path/to/roomCard', roomCard)

    #checking recognizing color cards
    if refCard is None or roomCard is None:
        print("[ERROR] color cards are not recognized")
        # if roomcard is not found, but we know exactly its size in pixels in the photo from the fridge,
        # we can find it using the method ==> findingTempl_byLeastSquaresMethod(fullImage_path, imageTemplate_path)
        # where: fullImage_path - path to foto from fridge, imageTemplate_path - path to "refCard" with the dimensions
        # in pixels exactly corresponding to the size of the "roomCard" in the photo from the fridge


    # otherwise, the "roomCard" will have to be done manually
    """    option #2 - photos of Color Cards of bad quality
       you can just skip this part if the "roomCard" is of good quality  """

    roomCard = cv2.imread('./ColorCard_fromFridges/colorCard_fridge_3.jpg')

    #Since I used a graphic editor to cut and save a "room card" from photos in
    # the fridge, I don't know exactly what pixel size the cut-out image has.
    # Therefore, it is necessary to adjust its size in accordance with the size of the "refCard"
    height, width, channels = refCard.shape
    dsize = (width, height)
    roomCard = cv2.resize(roomCard, dsize)

    """     End option #2       """

    # before correcting the color of the fruit, remove the background in the image
    original_image_path_from_fridge = './imageSetfromFridges/fromFridge_3_011223in1206_origin.jpg'
    output_image = 'removeBG_fromFridge.jpg'

    input_image_path = bg_remove3(original_image_path_from_fridge, output_image)
    input_image = cv2.imread(input_image_path)
    # cv2.imshow("original input image", input_image)
    # cv2.waitKey(0)


    # we'll get the corrected image of fruits
    result_image = match_histograms_mod(roomCard, refCard, input_image)
    outFileName = './correctedBananaImages/imageOut011223_1206.jpg'
    cv2.imwrite(outFileName, result_image)
    os.remove(output_image)
    cv2.imshow("corrected input image with colorCard", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # input_image = imread('./imageSetfromFridges/fromFridge_3_011223in1206_origin.jpg')
    # whitebalanceCorrected = whitepatch_balancing(input_image, 73, 1084, 10, 10)



