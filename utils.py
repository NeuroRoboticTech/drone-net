"""
Utils for conversion
"""
import os
import logging
import numpy as np
import cv2
import scipy.misc as misc
import math

def setupLogging(prefix):
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/{1}.log".format('.', prefix))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(level=logging.INFO)
    logging.info("starting up")

def rotateImg(img, angle, mask_in=None):
    if angle == 0:
        return img, mask_in

    # grab the dimensions of the image
    (h, w) = img.shape[:2]

    max_dim = int(max(h, w) * 2.0)
    max_dim_2 = int(max_dim/2.0)

    # Get a blank array the max paste size
    if len(img.shape) > 2:
        buffer_roi = np.zeros([max_dim, max_dim, img.shape[2]], dtype=np.uint8)
        buffer_roi_mask = np.zeros([max_dim, max_dim], dtype=np.uint8)
    else:
        buffer_roi = np.zeros([max_dim, max_dim], dtype=np.uint8)
        buffer_roi_mask = np.zeros([max_dim, max_dim], dtype=np.uint8)

    center_rotate_roi = int(max_dim / 2.0)
    paste_left = int(img.shape[1] / 2.0)
    paste_right = img.shape[1] - paste_left
    paste_top = int(img.shape[0] / 2.0)
    paste_bottom = img.shape[0] - paste_top

    # Copy the image into the center of this
    buffer_roi[(center_rotate_roi - paste_top):(center_rotate_roi + paste_bottom),
               (center_rotate_roi - paste_left):(center_rotate_roi + paste_right)] = img
    if mask_in is not None:
        buffer_roi_mask[(center_rotate_roi - paste_top):(center_rotate_roi + paste_bottom),
                        (center_rotate_roi - paste_left):(center_rotate_roi + paste_right)] = mask_in

    # showAndWait('buffer_roi', buffer_roi)

    rotated = misc.imrotate(buffer_roi, angle)

    if len(img.shape) > 2:
        paste_grey = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    else:
        paste_grey = rotated

    # showAndWait('paste_grey', paste_grey)
    # cv2.imwrite('/media/dcofer/Ubuntu_Data/drone_images/paste_grey.png', paste_grey)

    ret, rotated_mask = cv2.threshold(paste_grey, 5, 255, cv2.THRESH_BINARY)

    # showAndWait('mask', rotated_mask)
    # cv2.imwrite('/media/dcofer/Ubuntu_Data/drone_images/rotated_mask.png', rotated_mask)

    where = np.array(np.where(rotated_mask))
    # np.savetxt('/media/dcofer/Ubuntu_Data/drone_images/fuckhead.csv', np.transpose(where))

    x1, y1 = np.amin(where, axis=1)
    x2, y2 = np.amax(where, axis=1)

    out_image = rotated[x1:x2, y1:y2]
    out_mask = rotated_mask[x1:x2, y1:y2]

    # showAndWait('out_image', out_image)
    # cv2.imwrite('/media/dcofer/Ubuntu_Data/drone_images/out_image.png', out_image)

    # return the rotated image
    return out_image, out_mask

def generateMask(img):
    if len(img.shape) == 3:
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_grey = img

    ret, mask = cv2.threshold(img_grey, 5, 255, cv2.THRESH_BINARY)
    return mask

def showAndWait(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

def findFilesOfType(input_dir, endings):
    # Get the xml files in the directory
    files = os.listdir(input_dir)

    out_files = []
    for file in files:
        for ext in endings:
            if file.endswith(ext):
                out_files.append(input_dir + '/' + file)
                break

    ret_files = sorted(set(out_files))
    # print img_files

    return ret_files

def writeFileList(list, filename):
    with open(filename, 'w') as f:
        for item in list:
            #logging.debug(item)
            f.write("%s\n" % item)


def saveDetectNetLabelFile(label, list, filename):
    with open(filename, 'w') as f:
        for item in list:
            f.write("{} 0.0 0 0.0 {} {} {} {} 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n".format(label, item[0],
                                                                                    item[1], item[2],
                                                                                    item[3]))
def loadYoloLabels(label_file):

    label_data = []
    with open(label_file) as reader:
        line = reader.readline()
        labels = line.split(' ')

        width_2 = float(labels[3]) / 2.0
        height_2 = float(labels[4]) / 2.0

        left = float(labels[1]) - width_2
        top = float(labels[2]) - height_2
        right = left + float(labels[3])
        bottom = top + float(labels[4])

        new_labels = [left, top, right, bottom]

        label_data.append(new_labels)

    return label_data

def rotate(origin, point, angle_deg):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    angle = math.radians(angle_deg)

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

