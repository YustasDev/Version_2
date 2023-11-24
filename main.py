from imutils.perspective import four_point_transform
from skimage.exposure import is_low_contrast
from imutils.paths import list_images
from skimage import exposure
import numpy as np
import argparse
import imutils
import cv2
import sys

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




if __name__ == '__main__':


    #low_contrast_image_processing('/home/progforce/Banana/Version_2/card_fridge1.jpg')
    #low_contrast_image_processing('/home/progforce/Banana/Version_2/fridge_1.jpg')

    image = cv2.imread('/home/progforce/Banana/Version_2/fridge_1.jpg')
    templ = cv2.imread('/home/progforce/Banana/Version_2/cardTemplate1.jpg')

    # =============== finding a color card in a photo from the fridge =============>

    #https: // github.com / neemiasbsilva / object - detection - opencv / blob / master / template - matching.ipynb
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

    cut_color_card = image[top_left[1]:(top_left[1] + height), top_left[0]:(top_left[0] + width) ]
    # cv2.imshow("cut_color_card", cut_color_card)
    # cv2.waitKey(0)



    ref_image = cv2.imread('/home/progforce/Banana/Version_2/ref2.jpg')
    input_image = cut_color_card
    cv2.imshow("cut_color_card (for correction)", cut_color_card)
    cv2.waitKey(0)


    # resize the reference and input images
    ref = imutils.resize(ref_image, width=600)
    image = imutils.resize(input_image, width=600)

    # find the color matching card in each image
    print("[INFO] finding color matching cards...")
    refCard = find_color_card(ref)



    # Instead of ...
    #imageCard = find_color_card(image)
    imageCard = cv2.imread('/home/progforce/Banana/Version_2/needForCorrect_fridge_1.jpg')
    height, width, channels = refCard.shape
    dsize = (width, height)
    imageCard = cv2.resize(imageCard, dsize)

    # if the color matching card is not found in either the reference
    # image or the input image, gracefully exit
    if refCard is None or imageCard is None:
        print("[ERROR] could not find color matching card in both images")
        sys.exit(0)

    # ================= getting the corrected image =============================================================>

    input_image = cv2.imread('/home/progforce/Banana/yellow_banana.jpg')
    cv2.imshow("original input image", input_image)
    cv2.waitKey(0)


    result_image = match_histograms_mod(imageCard, refCard, input_image)
    cv2.imwrite('outBananas.jpg', result_image)


    cv2.imshow("corrected input image", result_image)
    cv2.waitKey(0)

