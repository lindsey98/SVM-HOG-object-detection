from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import numpy as np
import cv2
import config as cfg

# define the sliding window:
def sliding_window(image, stepSize, windowSize):# image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):# this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int"), probs[pick].astype("float")

def pred_logosense(model, shot_path):
    # Test the trained classifier on an image below!
    scale = 0
    detections = []

    # read the image you want to detect the object in:
    img = cv2.imread(shot_path)
    # Try it with image resized if the image is too big
    img = cv2.resize(img,(cfg.screenshot_resize[0], cfg.screenshot_resize[1])) # can change the size to default by commenting this code out our put in a random number

    # defining the size of the sliding window (has to be, same as the size of the image in the training data)
    (winW, winH)= (cfg.logo_size[0], cfg.logo_size[1])
    windowSize=(winW,winH)
    downscale=1.5

    # Apply sliding window:
    for resized in pyramid_gaussian(img, downscale=1.5, max_layer=2): # loop over each layer of the image that you take!
        # loop over the sliding window for each layer of the pyramid
        for (x,y,window) in sliding_window(resized, stepSize=cfg.sliding_window_step, windowSize=(winW,winH)):
            # if the window does not meet our desired window size, ignore it!
            if window.shape[0] != winH or window.shape[1] != winW: # ensure the sliding window has met the minimum size requirement
                continue
            fds = hog(window, cfg.hog_param['orientation'],
                        cfg.hog_param['pixel_per_cell'], cfg.hog_param['cell_per_block'], block_norm='L2')  # extract HOG features from the window captured
            fds = fds.reshape(1, -1) # reshape the image to make a silouhette of hog
            pred = model.predict(fds) # use the SVM model to make a prediction on the HOG features extracted from the window

            if pred == 1:
                if model.decision_function(fds) > 0.95:  # threshold to be 0.95
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                       int(windowSize[0]*(downscale**scale)), # create a list of all the predictions found
                                       int(windowSize[1]*(downscale**scale))))
        scale+=1

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) # do nms on the detected bounding boxes
    sc = [score for (x, y, score, w, h) in detections]
#     print("detection confidence score: ", sc)
    sc = np.array(sc)
    pick, pick_sc = non_max_suppression(rects, probs = sc, overlapThresh = 0.01)

    ## project bounding box on to the image
    if len(pick):
        for i, (xA, yA, xB, yB) in enumerate(pick):
            cv2.rectangle(img, (xA, yA), (xB, yB), (255,0,0), 2)
            cv2.putText(img,str(round(pick_sc[0][i], 2)), (xA, yA),
                        cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 2)
            
    return pick_sc, pick, img