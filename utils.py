"""
Utils for conversion
"""
import os
import logging
import numpy as np
import cv2
import scipy.misc as misc
import math
from PIL import Image
from geometry import *


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

    # Get a blank array the max paste size
    if len(img.shape) > 2:
        buffer_roi = np.zeros([max_dim, max_dim, img.shape[2]], dtype=np.uint8)
    else:
        buffer_roi = np.zeros([max_dim, max_dim], dtype=np.uint8)

    if mask_in is not None:
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
    if mask_in is not None:
        rotated_mask = misc.imrotate(buffer_roi_mask, angle)

    if len(img.shape) > 2:
        paste_grey = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    else:
        paste_grey = rotated

    # showAndWait('paste_grey', paste_grey)
    # cv2.imwrite('/media/dcofer/Ubuntu_Data/drone_images/paste_grey.png', paste_grey)

    ret, rotated_mask_img = cv2.threshold(paste_grey, 5, 255, cv2.THRESH_BINARY)

    # showAndWait('mask', rotated_mask)
    # cv2.imwrite('/media/dcofer/Ubuntu_Data/drone_images/rotated_mask.png', rotated_mask)

    where = np.array(np.where(rotated_mask_img))
    # np.savetxt('/media/dcofer/Ubuntu_Data/drone_images/fuckhead.csv', np.transpose(where))

    x1, y1 = np.amin(where, axis=1)
    x2, y2 = np.amax(where, axis=1)

    out_image = rotated[x1:x2, y1:y2]
    if mask_in is not None:
        out_mask = rotated_mask[x1:x2, y1:y2]
        ret, out_mask = cv2.threshold(out_mask, 3, 255, cv2.THRESH_BINARY)
    else:
        out_mask = None

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
        for l in list:
            x_max = l['x'] + l['width']
            y_max = l['y'] + l['height']

            f.write("{} 0.0 0 0.0 {} {} {} {} 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n".format(label, l['x'],
                                                                                    l['y'], x_max, y_max))


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


def saveYoloLabelFile(label, list, filename, img_width, img_height):
    with open(filename, 'w') as f:
        for l in list:
            x_center = (l['x'] + float(l['width'])/2.0) / float(img_width)
            y_center = (l['y'] + float(l['height'])/2.0) / float(img_height)
            width = float(l['width']) / float(img_width)
            height = float(l['height']) / float(img_height)

            f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(label, x_center, y_center, width, height))


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


def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)


def color_map(N=256, normalized=False):
    cmap = []
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap.append(r)
        cmap.append(g)
        cmap.append(b)

    return cmap


def quantizetopalette(silf, palette, dither=False):
    """Convert an RGB or L mode image to use a given P image's palette."""

    silf.load()

    # use palette from reference image
    palette.load()
    if palette.mode != "P":
        raise ValueError("bad mode for palette image")
    if silf.mode != "RGB" and silf.mode != "L":
        raise ValueError(
            "only RGB or L mode images can be quantized to a palette"
            )
    im = silf.im.convert("P", 1 if dither else 0, palette.im)
    # the 0 above means turn OFF dithering
    return silf._makeself(im)


def savePascalColorMap(file_name):
    cm = color_map()

    # print cm
    cm_file = open(file_name, "w")
    color_idx = 0
    for c in cm:
        cm_file.write(str(c))

        color_idx = color_idx + 1
        if color_idx >= 3:
            color_idx = 0
            cm_file.write("\n")
        else:
            cm_file.write(" ")

    cm_file.close()


def saveIndexImage(file_name, img):
    cmap = color_map()

    rgb_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # pil_im = Image.open("/media/dcofer/Ubuntu_Data/train_data/orig_labels/P1040599_0_0.png")
    pil_im = Image.fromarray(rgb_im)

    palimage = Image.new('P', pil_im.size)
    palimage.putpalette(cmap)
    newimage = quantizetopalette(pil_im, palimage, dither=False)

    newimage.save(file_name)
    # print("Saved mask file: " + file_name)

def drawLabels(img_in, labels):

    img = img_in.copy()

    for l in labels:
        x_max = l['x'] + l['width']
        y_max = l['y'] + l['height']

        top_left = (int(l['x']), int(l['y']))
        top_right = (int(x_max), int(l['y']))
        bottom_right = (int(x_max), int(y_max))
        bottom_left =(int(l['x']), int(y_max))

        img = cv2.line(img, top_left, top_right, color=(0, 0, 255), thickness=3)
        img = cv2.line(img, top_right, bottom_right, color=(0, 0, 255), thickness=3)
        img = cv2.line(img, bottom_right, bottom_left, color=(0, 0, 255), thickness=3)
        img = cv2.line(img, bottom_left, top_left, color=(0, 0, 255), thickness=3)

    return img


def getYoloCoords(obj, img_width, img_height):
    coords = obj['relative_coordinates']
    conf = float(obj['confidence'])

    width = coords['width'] * img_width
    height = coords['height'] * img_height

    x_center = (coords['center_x'] * img_width)
    y_center = (coords['center_y'] * img_height)

    x_min = x_center - int(width / 2.0)
    y_min = y_center - int(height / 2.0)

    x_max = x_center + int(width / 2.0)
    y_max = y_center + int(height / 2.0)

    top_left = (int(x_min), int(y_min))
    top_right = (int(x_max), int(y_min))
    bottom_right = (int(x_max), int(y_max))
    bottom_left = (int(x_min), int(y_max))

    return top_left, top_right, bottom_left, bottom_right, width, height, conf


def drawYoloObjectLabels(img_in, labels):

    img = img_in.copy()

    for l in labels:
        logging.info(l)

        top_left, top_right, bottom_left, bottom_right, width, height, conf = getYoloCoords(l,
                                                                                            img_in.shape[1],
                                                                                            img_in.shape[0])
        if conf > 0.25:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        img = cv2.line(img, top_left, top_right, color=color, thickness=3)
        img = cv2.line(img, top_right, bottom_right, color=color, thickness=3)
        img = cv2.line(img, bottom_right, bottom_left, color=color, thickness=3)
        img = cv2.line(img, bottom_left, top_left, color=color, thickness=3)

    return img


def overlapsYolo(annotations, yolo_labels, img_width, img_height):

    overlaps_count = 0
    for a in annotations:
        a_rect = Rect(a['x'], a['y'], a['width'], a['height'])

        for y in yolo_labels:
            top_left, top_right, bottom_left, bottom_right, width, height, conf = getYoloCoords(y,
                                                                                                img_width,
                                                                                                img_height)

            y_rect = Rect(top_left[0], top_left[1], width, height)

            if y_rect.overlaps(a_rect):
                overlaps_count += 1
                break

    if overlaps_count >= len(annotations):
        return True
    else:
        return False


def randomFlipImage(img_in, flip_horizontal=True, flip_vertical=True,
                    horiz_perc=50, vert_perc=10):
    if flip_horizontal:
        flip_val = np.random.randint(0, 100)
        if flip_val < horiz_perc:
            logging.info("  flip_val: {}. Flipping image horizontal.".format(flip_val))
            flipped_canvas_img = np.fliplr(img_in)
        else:
            logging.info("  flip_val: {}. Leaving canvas  horizontal unflipped".format(flip_val))
            flipped_canvas_img = img_in

    if flip_vertical:
        flip_val = np.random.randint(0, 100)
        if flip_val < vert_perc:
            logging.info("  flip_val: {}. Flipping image vertical.".format(flip_val))
            flipped_canvas_img = np.flipud(flipped_canvas_img)
        else:
            logging.info("  flip_val: {}. Leaving canvas unflipped vertical".format(flip_val))
            flipped_canvas_img = flipped_canvas_img

    return flipped_canvas_img


def flipLabels(labels, paste_dim, vertical=False):

    new_labels = []
    for l in labels:
        # new_labels.append([paste_width - l[2], l[1], paste_width - l[0], l[3]])

        new_l = l.copy()

        if not vertical:
            new_l['x'] = paste_dim - (l['x'] + l['width'])
        else:
            new_l['y'] = paste_dim - (l['y'] + l['height'])

        new_labels.append(new_l)

        #new_labels.append([paste_width-l[2], l[1], paste_width-l[0], l[3]])

    printLabelDims(new_labels)

    return new_labels

def adjustLabels(labels, x, y):

    new_labels = []
    for l in labels:
        new_l = l.copy()
        new_l['x'] = x + l['x']
        new_l['y'] = y + l['y']
        new_labels.append(new_l)

    printLabelDims(new_labels)

    return new_labels

def scaleLabels(labels, ratio):

    new_labels = []
    for l in labels:
        new_l = l.copy()
        new_l['x'] = l['x'] * ratio
        new_l['y'] = l['y'] * ratio
        new_l['width'] = l['width'] * ratio
        new_l['height'] = l['height'] * ratio
        new_labels.append(new_l)

    printLabelDims(new_labels)

    return new_labels

def printLabelDims(labels):

    if len(labels) > 0:
        min_area = 99999999999
        for l in labels:
            logging.info("    x: {0:.2f}, y: {0:.2f}, w: {0:.2f}, h: {0:.2f}".format(l['x'], l['y'], l['width'], l['height']))
            area = l['width'] * l['height']
            if area < min_area:
                min_area = area

        return min_area
    else:
        return 0.0