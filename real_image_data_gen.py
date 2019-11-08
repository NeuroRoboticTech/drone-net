"""
Generates training data for DetectNet or image segmentation from the drone-net real images.
"""
import sys
import os
import random
import logging
import shutil
import time
import math
import json

import numpy as np
import cv2
import scipy.misc as misc
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import  img_as_ubyte
import argparse
import utils


def processArgs():
    """ Processes command line arguments     """
    parser = argparse.ArgumentParser(description='Generates detectnet simulated data.')
    parser.add_argument('--canvas_image_dir', type=str, required=True,
                        help='Data dir containing canvas images.')
    parser.add_argument('--paste_image_dir', type=str, required=True,
                        help='Data dir containing paste images.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='dir where data will be generated.')
    parser.add_argument('--paste_label_json', type=str, required=True,
                        help='paste label json file.')
    parser.add_argument('--min_paste_label_area', type=int, default=250,
                        help='minimum size of smallest dimension of pasted image.')
    parser.add_argument('--max_paste_rotation', type=int, default=80,
                        help='maximum rotation that can be randomly added to pasted image.')
    parser.add_argument('--max_canvas_rotation', type=int, default=5,
                        help='maximum rotation that can be randomly added to canvas image.')
    parser.add_argument('--final_img_width', type=int, default=608,
                        help='height of the final produced image.')
    parser.add_argument('--final_img_height', type=int, default=608,
                        help='width of the final produced image.')
    parser.add_argument('--max_canvas_images', type=int, default=-1,
                        help='If set to non-negative value it will only get that number of canvas images.')

    args, unknown = parser.parse_known_args()
    return args

class RealImageDataGen ():

    def __init__(self, args):
        """
        Creates a generic detectnet data generator class
        :param args: arguments for generator.
        """
        logging.info("creating image dataset generator ...")

        self.canvas_image_dir = args.canvas_image_dir
        self.paste_image_dir = args.paste_image_dir
        self.paste_label_json = args.paste_label_json
        self.save_dir = args.save_dir
        self.min_paste_label_area = args.min_paste_label_area
        self.final_img_width = args.final_img_width
        self.final_img_height = args.final_img_height
        self.max_paste_rotation = args.max_paste_rotation
        self.max_canvas_rotation = args.max_canvas_rotation
        self.max_canvas_images = args.max_canvas_images

        self.root_train_img_dir = ""

        self.train_img_dir = ""
        self.train_label_dir = ""

        self.paste_image_idx = 0

        self.all_paste_files_used_count = 0

        self.generate_masks = False

        self.force_scale = -1.0  # When -1 it means use default image label size.
        self.blur_thresh = 10
        self.bright_thresh = 10
        self.bright_max = 50
        self.contrast_max = 0.1
        self.blur_max = 5

        self.canvas_img_files = []
        self.paste_labels = []

        self.file_prefix = "real"

    def initialize(self):
        """
        Intialize a generic detectnet data generator class. It finds the filenames for canvas and paste images, and
        labels, and splits them into train and validation spilts.
        """
        logging.info("Initializing image dataset generator ...")

        if not os.path.exists(self.canvas_image_dir):
            raise RuntimeError("Canvas image directory does not exist: {}".format(self.canvas_image_dir))

        if not os.path.exists(self.paste_image_dir):
            raise RuntimeError("Paste image directory does not exist: {}".format(self.paste_image_dir))

        if not os.path.exists(self.paste_label_json):
            raise RuntimeError("Paste label json file does not exist: {}".format(self.paste_label_json))

        with open(self.paste_label_json, "r") as read_file:
            self.paste_labels = json.load(read_file)

        if len(self.paste_labels) <= 0:
            raise RuntimeError("Paste labels were empty.")

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.train_img_dir = self.save_dir + "/images"

        if os.path.exists(self.train_img_dir):
            shutil.rmtree(self.train_img_dir)
            # Give the OS a little time to actually make the new directory. Was running into errors
            # where creating the html folder inside this folder would periodically error out.
            time.sleep(0.1)

        os.mkdir(self.train_img_dir)

        self.train_label_dir = self.save_dir + "/labels"

        if os.path.exists(self.train_label_dir):
            shutil.rmtree(self.train_label_dir)
            # Give the OS a little time to actually make the new directory. Was running into errors
            # where creating the html folder inside this folder would periodically error out.
            time.sleep(0.1)

        os.mkdir(self.train_label_dir)

        self.canvas_img_files = utils.findFilesOfType(self.canvas_image_dir, ['png', 'jpg', 'jpeg'])

        if len(self.canvas_img_files) <= 0:
            raise RuntimeError("No canvas image files were found")

        if self.max_canvas_images > 0:
            if self.max_canvas_images > len(self.canvas_img_files):
                raise RuntimeError("Number of canvas images is less than max count: {} > {}".format(
                    len(self.canvas_img_files), self.max_canvas_images))

            self.canvas_img_files = self.canvas_img_files[:self.max_canvas_images]

        np.random.shuffle(self.canvas_img_files)
        np.random.shuffle(self.paste_labels)


    def loadPasteImage(self, filename, cut_height=0):
        """ Loads a paste image and mask.

        :param filename: path to paste image file
        :return: paste image and mask
        """
        # paste_img_in = misc.imread(filename)
        paste_img_in = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        # utils.showAndWait('paste_img_in', paste_img_in)

        if paste_img_in.shape[2] < 4:
            paste_img = paste_img_in
            ret, paste_img_mask = cv2.threshold(paste_img, 2, 255, cv2.THRESH_BINARY)

            paste_img_mask = paste_img_mask[:, :, 0]
        elif paste_img_in.shape[2] == 4:
            paste_img = paste_img_in[:, :, 0:3]
            paste_img_mask = paste_img_in[:, :, 3]

            ret, paste_img_mask = cv2.threshold(paste_img_mask, 100, 255, cv2.THRESH_BINARY)

            # Now black out everything on the image that is not in the mask
            paste_img = cv2.bitwise_and(paste_img, paste_img, mask=paste_img_mask)
        else:
            raise RuntimeError("Invalid number of paste bitmap layers for {}".format(filename))

        # utils.showAndWait('paste_img', paste_img)
        # utils.showAndWait('paste_img_mask', paste_img_mask)

        if cut_height > 0:
            paste_img = paste_img[:-cut_height, :, :]

        # utils.showAndWait('paste_img', paste_img)

        return paste_img, paste_img_mask

    def incrementNextPastedImageIndex(self, paste_labels):
        """ Gets the index of the next paste image so we go through them all.

        :param paste_labels: list of paste image files.
        :return: index of next image
        """

        self.paste_image_idx += 1
        if self.paste_image_idx >= len(paste_labels):
            self.paste_image_idx = 0
            self.all_paste_files_used_count += 1

            if self.force_scale <=0:
                self.force_scale = 1.0

            self.force_scale = self.force_scale * 0.75

            np.random.shuffle(self.canvas_img_files)
            np.random.shuffle(self.paste_labels)

            if self.all_paste_files_used_count == 1:
                self.blur_thresh = 70
                self.bright_thresh = 70
                self.bright_max = 50
                self.contrast_max = 0.08
                self.blur_max = 5
            elif self.all_paste_files_used_count == 2:
                self.blur_thresh = 80
                self.bright_thresh = 80
                self.bright_max = 40
                self.contrast_max = 0.05
                self.blur_max = 4
            elif self.all_paste_files_used_count == 3:
                self.blur_thresh = 90
                self.bright_thresh = 90
                self.bright_max = 30
                self.contrast_max = 0.03
                self.blur_max = 3
            elif self.all_paste_files_used_count == 3 or \
                 self.all_paste_files_used_count == 4 or \
                 self.all_paste_files_used_count == 5:
                self.blur_thresh = 90
                self.bright_thresh = 90
                self.bright_max = 30
                self.contrast_max = 0.03
                self.blur_max = 2
            elif self.all_paste_files_used_count == 6:
                self.blur_thresh = 95
                self.bright_thresh = 95
                self.bright_max = 70
                self.contrast_max = 0.2
                self.blur_max = 5
                self.force_scale = 1.0
            elif self.all_paste_files_used_count == 7:
                self.blur_thresh = 95
                self.bright_thresh = 95
                self.bright_max = 80
                self.contrast_max = 0.15
                self.blur_max = 6
            elif self.all_paste_files_used_count == 12:
                self.force_scale = 1.0
                self.all_paste_files_used_count = 0

        return self.paste_image_idx

    def getNonoverlappingPastePos(self, last_x, paste_width, paste_height,
                                  canvas_width, canvas_height,
                                  max_paste_x, max_paste_y):
        """ gets random paste positions until it finds one that will not overlap with any already existing paste images.

        :param paste_width: width
        :param paste_height: height
        :param canvas_width: width
        :param canvas_height: height
        :param labels: list of previous labels
        :return: new x,y positions for paste
        """

        if (last_x + paste_width) > canvas_width:
            return -1, -1, last_x

        if paste_height > canvas_height:
            return -1, -1, last_x

        range_y = max((canvas_height - paste_height), max_paste_y)
        range_x = min((canvas_width - (last_x + paste_width)), max_paste_x)

        if range_x <= 0 or range_y <= 0:
            return -1, -1, last_x

        x = np.random.randint(0, range_x)
        y = np.random.randint(0, range_y)

        paste_x = last_x + x
        last_x = paste_x + paste_width

        return paste_x, y, last_x

    def roatedPasteImageBy90(self, paste_img, labels):
        flip_val = np.random.randint(0, 100)
        if flip_val < 50:
            logging.info("Rotating by -90.")
            rotate_deg = -90
        else:
            logging.info("Rotating by 90.")
            rotate_deg = 90

        return self.rotatePasteImage(paste_img, labels, rotate_deg)

    def rotatePasteImageRandom(self, paste_img, labels, rotate_deg):
        if rotate_deg == 0:
            rotate_deg = self.getForcedRandomRotationValue(self.max_paste_rotation)

        logging.info("  rotate_deg: {}.".format(rotate_deg))

        return self.rotatePasteImage(paste_img, labels, rotate_deg)

    def rotatePasteImage(self, img, labels, rotate_deg=0):

        # drawn_img = self.drawLabels(img.copy(), labels)
        # utils.showAndWait('drawn_img', drawn_img)

        rotated, mask = utils.rotateImg(img, rotate_deg)
        # rotated = misc.imrotate(img, rotate_deg)
        # utils.showAndWait('rotated', rotated)

        center_x = img.shape[1] / 2.0
        center_y = img.shape[0] / 2.0

        new_center_x = rotated.shape[1] / 2.0
        new_center_y = rotated.shape[0] / 2.0

        new_labels = []
        for l in labels:
            w = l['width']  # * 0.85
            h = l['height']  # * 0.85
            tl = [l['x']-center_x, (l['y']-center_y)]
            tr = [l['x']+w-center_x, (l['y']-center_y)]
            bl = [l['x']-center_x, (l['y']+h-center_y)]
            br = [l['x']+w-center_x, (l['y']+h-center_y)]
            top_left = utils.rotate([0.0, 0.0], tl, -rotate_deg)
            top_right = utils.rotate([0.0, 0.0], tr, -rotate_deg)
            bottom_left = utils.rotate([0.0, 0.0], bl, -rotate_deg)
            bottom_right = utils.rotate([0.0, 0.0], br, -rotate_deg)

            tl2 = [(top_left[0] + new_center_x), (top_left[1] + new_center_y)]
            tr2 = [(top_right[0] + new_center_x), (top_right[1] + new_center_y)]
            bl2 = [(bottom_left[0] + new_center_x), (bottom_left[1] + new_center_y)]
            br2 = [(bottom_right[0] + new_center_x), (bottom_right[1] + new_center_y)]

            # rotated = cv2.line(rotated,
            #                    (int(tl2[0]), int(tl2[1])),
            #                    (int(tr2[0]), int(tr2[1])),
            #                    color=(0, 0, 255), thickness=3)
            # rotated = cv2.line(rotated,
            #                    (int(tr2[0]), int(tr2[1])),
            #                    (int(br2[0]), int(br2[1])),
            #                    color=(0, 0, 255), thickness=3)
            # rotated = cv2.line(rotated,
            #                    (int(br2[0]), int(br2[1])),
            #                    (int(bl2[0]), int(bl2[1])),
            #                    color=(0, 0, 255), thickness=3)
            # rotated = cv2.line(rotated,
            #                    (int(bl2[0]), int(bl2[1])),
            #                    (int(tl2[0]), int(tl2[1])),
            #                    color=(0, 0, 255), thickness=3)
            # utils.showAndWait('rotated', rotated)

            min_x = min(tl2[0], tr2[0], bl2[0], br2[0])
            max_x = max(tl2[0], tr2[0], bl2[0], br2[0])
            min_y = min(tl2[1], tr2[1], bl2[1], br2[1])
            max_y = max(tl2[1], tr2[1], bl2[1], br2[1])

            new_l = l.copy()
            new_l['x'] = min_x
            new_l['y'] = min_y
            new_l['width'] = (max_x - min_x)
            new_l['height'] = (max_y - min_y)

            new_labels.append(new_l)
            self.printLabelDims(new_labels)

        return rotated, new_labels

    def flipLabels(self, labels, paste_dim, vertical=False):

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

        self.printLabelDims(new_labels)

        return new_labels

    def adjustLabels(self, labels, x, y):

        new_labels = []
        for l in labels:
            new_l = l.copy()
            new_l['x'] = x + l['x']
            new_l['y'] = y + l['y']
            new_labels.append(new_l)

        self.printLabelDims(new_labels)

        return new_labels

    def scaleLabels(self, labels, ratio):

        new_labels = []
        for l in labels:
            new_l = l.copy()
            new_l['x'] = l['x'] * ratio
            new_l['y'] = l['y'] * ratio
            new_l['width'] = l['width'] * ratio
            new_l['height'] = l['height'] * ratio
            new_labels.append(new_l)

        self.printLabelDims(new_labels)

        return new_labels

    def printLabelDims(self, labels):

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


    def loadMaskImage(self, paste_img_file, cut_height=0):

        # remove the jpg extension and add on correct label ending
        mask_file = paste_img_file[:-4] + '_label.png'

        if not os.path.exists(mask_file):
            raise RuntimeError("mask_file does not exist: {}. Exiting".format(mask_file))

        mask_img_in = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)

        if cut_height > 0:
            mask_img_in = mask_img_in[:-cut_height, :, :]

        # Find all places where pixels are red and white.
        red_mask = cv2.inRange(mask_img_in, np.array([0, 0, 253]), np.array([5, 5, 255]))
        white_mask = cv2.inRange(mask_img_in, np.array([253, 253, 253]), np.array([255, 255, 255]))

        # Find all places where it is not 0.
        red_where = np.array(np.where(red_mask))
        white_where = np.array(np.where(white_mask))

        # Create a new image of same size with only one bit plane and all zeros
        label_img = np.zeros([mask_img_in.shape[0], mask_img_in.shape[1]], dtype=np.uint8)

        # Set all red and white places to 1 and 255
        label_img[red_where[0,:], red_where[1,:]] = 1
        if len(white_where) > 0:
            label_img[white_where[0,:], white_where[1,:]] = 255

        if mask_img_in.shape[0] != label_img.shape[0] or \
           mask_img_in.shape[1] != label_img.shape[1]:
            raise RuntimeError("Mismatch between train and label out shapes. {}".format(paste_img_file))

        return label_img

    def addPastedImages(self, canvas_img_file, canvas_img, paste_labels,
                        save_img_dir, save_label_dir, canvas_idx, tile_idx,
                        out_labels):
        """
        Adds paste images to the canvas file and saves it and the labels.
        :param canvas_img_file: canvas image filename to split
        :param canvas_img: canvas image to split
        :param paste_labels: paste image files.
        :param width_multiple: multiple of width to final image size.
        :param height_multiple: multiple of height to final image size.
        :param canvas_idx: canvas index.
        :param tile_idx: tile index.
        """

        canvas_width = canvas_img.shape[1]
        canvas_height = canvas_img.shape[0]
        all_labels = []

        if self.generate_masks:
            canvas_mask_img = np.zeros([canvas_height, canvas_width], dtype=np.uint8)
        else:
            canvas_mask_img = None

        if canvas_height != self.final_img_height or canvas_width != self.final_img_width:
            logging.error("The canvas height for a paste add does not match the final image dimensions. Skipping image.")
            return

        max_paste_x = 100
        max_paste_y = 100
        last_x = 0

        labels = []

        done = False
        while not done:
            try:
                paste_img_file_idx = self.paste_image_idx
                # paste_img_file  = '/media/dcofer/Ubuntu_Data/drone-net/images/03012017-dji-phantom-flying-sky-260nw-554568589.jpg'
                paste_label = paste_labels[paste_img_file_idx]
                paste_img_file = self.paste_image_dir + '/' + paste_label['filename']

                logging.info("  Pasting in {}".format(paste_img_file))
                logging.info("    Paste Image Idx {}".format(paste_img_file_idx))
                paste_img, paste_mask = self.loadPasteImage(paste_img_file, cut_height=20)
                # utils.showAndWait('paste_img', paste_img)

                if self.generate_masks:
                    mask_img = self.loadMaskImage(paste_img_file, cut_height=20)
                else:
                    mask_img = None

                # Try to load the label file for this image.
                orig_filename = os.path.basename(paste_img_file)
                orig_basename = os.path.splitext(orig_filename)[0]
                labels = paste_label['annotations']

                paste_width = paste_img.shape[1]
                paste_height = paste_img.shape[0]
                logging.info("    paste_width: {}".format(paste_width))
                logging.info("    paste_height: {}".format(paste_height))
                min_area = self.printLabelDims(labels)

                # If the paste height is greater than the canvas height then rotate by 90 or -90 degrees.
                if paste_height > canvas_height:
                    logging.info("paste height to large. flipping over.")
                    paste_img, labels = self.roatedPasteImageBy90(paste_img, labels)
                    width_temp = paste_width
                    paste_width = paste_height
                    paste_height = width_temp

                # randomly change brightness and contract of foreground drone
                bright_rand = np.random.randint(0, 100)
                if bright_rand < self.bright_thresh:
                    logging.info("    bright_rand: {}. Adjusting brightness/contrast.".format(bright_rand))

                    # utils.showAndWait('before bright', paste_img)

                    # Check the overal brightness and darkness of the image. If it is too bright then only darken
                    # if it is too dark then only lighten
                    light_mask = cv2.inRange(paste_img, np.array([210, 210, 210]), np.array([255, 255, 255]))
                    dark_mask = cv2.inRange(paste_img, np.array([0, 0, 0]), np.array([45, 45, 45]))

                    # utils.showAndWait('paste_img bright', paste_img)
                    # utils.showAndWait('light_mask bright', light_mask)
                    # utils.showAndWait('dark_mask bright', dark_mask)

                    where_light = np.array(np.where(light_mask))
                    where_dark = np.array(np.where(dark_mask))

                    contrast_val = np.random.normal(1.0, self.contrast_max)
                    if len(where_light[0]) > ((paste_height * paste_width) * 0.65):
                        logging.info("image is overly bright. Only allowing dimming. bright val: {}".format(len(where_light[0])))
                        bright_val = np.random.randint(-self.bright_max, 0)
                        contrast_val = 1 - abs(1 - contrast_val)
                    elif len(where_dark[0]) > ((paste_height * paste_width) * 0.65):
                        logging.info("image is overly dim. Only allowing brightning. bright val: {}".format(len(where_dark[0])))
                        bright_val = np.random.randint(0, self.bright_max)
                        contrast_val = 1 + abs(1 - contrast_val)
                    else:
                        bright_val = np.random.randint(-self.bright_max, self.bright_max)

                    if contrast_val < 0.5:
                        contrast_val = 0.7
                    if contrast_val > 1.5:
                        contrast_val = 1.3;
                    logging.info("    bright_val: {}".format(bright_val))
                    logging.info("    contrast_val: {}".format(contrast_val))
                    paste_img = cv2.convertScaleAbs(paste_img, alpha=contrast_val, beta=bright_val)

                    # utils.showAndWait('after bright', paste_img)
                else:
                    logging.info("    bright_rand: {}. Leaving image brightness/contrast alone".format(bright_rand))


                # Now randomly add blur
                blur_val = np.random.randint(0, 100)
                if blur_val < self.blur_thresh:
                    logging.info("    blur_val: {}. bluring image.".format(blur_val))

                    blur_kernel = np.random.randint(1, self.blur_max)
                    logging.info("    blur_kernel: {}".format(blur_kernel))
                    if blur_kernel > 0:
                        # blured_roi = cv2.GaussianBlur(merged_roi, (blur_val, blur_val), 0)
                        paste_img = cv2.blur(paste_img, (blur_kernel, blur_kernel))
                else:
                    logging.info("    blur_val: {}. Leaving image un-blurred".format(blur_val))

                # Scale the image
                logging.info("min area: {}".format(min_area))
                if paste_width >= canvas_width * 0.98:
                    force_scale = (canvas_width * 0.9) / paste_width
                elif paste_height >= canvas_height *.98:
                    force_scale = (canvas_height * 0.9) / paste_height
                else:
                    force_scale = self.force_scale

                # if min_area < 500:
                #     min_area = min_area

                forced_min_area = min_area * force_scale
                logging.info("forced_min_area: {}, force_scale: {}".format(forced_min_area, force_scale))
                if force_scale > 0:
                    if forced_min_area > self.min_paste_label_area:
                        paste_width = int(paste_width * force_scale)
                        paste_height = int(paste_height * force_scale)

                        paste_img = cv2.resize(paste_img, dsize=(paste_width, paste_height),
                                               interpolation=cv2.INTER_AREA)
                        if self.generate_masks:
                            mask_img = cv2.resize(mask_img, dsize=(paste_width, paste_height),
                                                  interpolation=cv2.INTER_AREA)
                        labels = self.scaleLabels(labels, force_scale)
                    elif forced_min_area < self.min_paste_label_area:
                        logging.info("forced min area below min area: {} < {}".format(forced_min_area,
                                                                                      self.min_paste_label_area))
                        logging.info("resetting scale to 1.0")
                        force_scale = 1.0

                if force_scale > 0:
                    paste_img, labels = self.rotatePasteImageRandom(paste_img, labels, rotate_deg=0)

                    paste_width = paste_img.shape[1]
                    paste_height = paste_img.shape[0]

                paste_x, paste_y, last_x = self.getNonoverlappingPastePos(last_x, paste_width, paste_height,
                                                                          canvas_width, canvas_height,
                                                                          max_paste_x, max_paste_y)
                if paste_x >= 0 and paste_y >= 0:
                    logging.info("    paste_x: {}".format(paste_x))
                    logging.info("    paste_y: {}".format(paste_y))
                    logging.info("    last_x: {}".format(last_x))

                    flip_horiz_val = np.random.randint(0, 100)
                    if flip_horiz_val < 50:
                        logging.info("    flip_horiz_val: {}. Flipping image horizontal.".format(flip_horiz_val))
                        paste_img = np.fliplr(paste_img)
                        if mask_img is not None:
                            mask_img = np.fliplr(mask_img)
                        labels = self.flipLabels(labels, paste_width, vertical=False)

                        # paste_img = utils.drawLabels(paste_img, labels)
                        # utils.showAndWait('paste_img', paste_img)
                    else:
                        logging.info("    flip_horiz_val: {}. Leaving image horizontal unflipped".format(flip_horiz_val))

                    # flip_vert_val = np.random.randint(0, 100)
                    # if flip_vert_val < 50:
                    #     logging.info("    flip_vert_val: {}. Flipping image vertical.".format(flip_vert_val))
                    #     paste_img = np.flipud(paste_img)
                    #     if mask_img is not None:
                    #         mask_img = np.flipud(mask_img)
                    #     labels = self.flipLabels(labels, paste_height, vertical=True)
                    #
                    #     paste_img = utils.drawLabels(paste_img, labels)
                    #     utils.showAndWait('paste_img', paste_img)
                    # else:
                    #     logging.info("    flip_vert_val: {}. Leaving image vertical unflipped".format(flip_vert_val))

                    where = np.array(np.where(paste_img))

                    where_x = where[1] + paste_x
                    where_y = where[0] + paste_y

                    canvas_img[where_y, where_x] = paste_img[where[0], where[1]]

                    # if not canvas_mask_img is None and not mask_img is None:
                    #     canvas_mask_img[paste_y:(paste_y + paste_height),
                    #                     paste_x:(paste_x + paste_width)] = mask_img

                    # utils.showAndWait('canvas_img', canvas_img)

                    labels = self.adjustLabels(labels, paste_x, paste_y)

                    all_labels.extend(labels)

                    self.incrementNextPastedImageIndex(paste_labels)
                else:
                    done = True
                    logging.info("Paste image was too big, skipping to go to next one.")
            except:
                done = True
                logging.exception("There was an exception. Skipping image.")

        # utils.showAndWait('canvas_img', canvas_img)

        if len(labels) <= 0:
            logging.warning("Labels for image were blank. Skipping.")
            return

        canvas_img = np.array(canvas_img)
        # canvas_img = utils.drawLabels(canvas_img, all_labels)

        # utils.showAndWait('canvas_img', canvas_img)

        save_img_filename = '{}_{}_{}.jpg'.format(self.file_prefix, canvas_idx, tile_idx)
        save_img_file = save_img_dir + '/{}'.format(save_img_filename)
        logging.info("saving image: {}".format(save_img_file))
        cv2.imwrite(save_img_file, canvas_img)
        #misc.imsave(save_file, canvas_img)

        save_label_file = save_label_dir + '/{}_{}_{}.txt'.format(self.file_prefix, canvas_idx, tile_idx)
        logging.info("saving lable: {}".format(save_label_file))
        # utils.saveDetectNetLabelFile('Car', labels, save_label_file)
        utils.saveYoloLabelFile(0, labels, save_label_file, img_width=canvas_width, img_height=canvas_height)

        if self.generate_masks and canvas_mask_img is not None:
            save_mask_file = save_label_dir + '/{}'.format(save_img_filename)
            logging.info("saving mask: {}".format(save_mask_file))
            #utils.saveIndexImage(save_mask_file, canvas_mask_img)
            cv2.imwrite(save_mask_file, canvas_mask_img)


        json_label = {"class": "image",
                      "filename": save_img_filename,
                      "annotations": all_labels}
        out_labels.append(json_label)


    def getForcedRandomRotationValue(self, max_rotation):

        flip_val = np.random.randint(0, 100)
        if flip_val < 50:
            rotate = int(np.random.uniform(2, max_rotation))
        else:
            rotate = int(np.random.uniform(-2, -max_rotation))
        return rotate

    def rotateCanvasImage(self, canvas_img, rotate_deg=0):
        if rotate_deg == 0:
            rotate_deg = self.getForcedRandomRotationValue(self.max_canvas_rotation)

        logging.info("  rotate_deg: {}.".format(rotate_deg))
        rotated_canvas_img, rotated_canvas_mask = utils.rotateImg(canvas_img, rotate_deg,
                                                                  mask_in=None)
        # utils.showAndWait('rotated_paste_img', rotated_canvas_img)

        if rotated_canvas_img.shape[0] != self.final_img_height or \
                rotated_canvas_img.shape[1] != self.final_img_width:
            left_idx = int(rotated_canvas_img.shape[1] / 2.0 - self.final_img_width / 2.0)
            right_idx = left_idx + self.final_img_width
            top_idx = int(rotated_canvas_img.shape[0] / 2.0 - self.final_img_height / 2.0)
            bottom_idx = top_idx + self.final_img_height
            rotated_canvas_img = rotated_canvas_img[top_idx:bottom_idx,
                                                    left_idx:right_idx]

        return rotated_canvas_img

    def splitCanvasIntoTiles(self, canvas_img_file, canvas_img, paste_labels,
                             save_img_dir, save_label_dir,
                             width_multiple, height_multiple,
                             canvas_idx, out_labels):
        """
        Splits the canvas image into multiple image tiles and adds pasted images to it.
        :param canvas_img_file: canvas image filename to split
        :param canvas_img: canvas image to split
        :param paste_labels: paste image files.
        :param width_multiple: multiple of width to final image size.
        :param height_multiple: multiple of height to final image size.
        :param canvas_idx: canvas index.
        """

        # Go through and get tile count for width
        width_tiles = int(math.ceil(width_multiple))
        height_tiles = int(math.ceil(height_multiple))

        if width_tiles < 1:
            width_tiles = 1
        if height_tiles < 1:
            height_tiles = 1

        canvas_width = canvas_img.shape[1]
        canvas_height = canvas_img.shape[0]

        tile_width = self.final_img_width
        tile_height = self.final_img_height
        rotate_deg = 0

        tile_idx = 1
        for width_idx in range(0, width_tiles):
            for height_idx in range(0, height_tiles):
                cut_x = width_idx * tile_width
                cut_y = height_idx * tile_height

                if cut_x + tile_width > canvas_width:
                    cut_x = canvas_width - tile_width
                    rotate_deg = self.getForcedRandomRotationValue(self.max_canvas_rotation)

                if cut_y + tile_height > canvas_height:
                    cut_y = canvas_height - tile_height
                    rotate_deg = self.getForcedRandomRotationValue(self.max_canvas_rotation)

                cut_canvas_img = canvas_img[cut_y:(cut_y+tile_height), cut_x:(cut_x+tile_width)].copy()

                flipped_canvas_img = utils.randomFlipImage(cut_canvas_img, vert_perc=50)
                if rotate_deg != 0:
                    rotated_canvas_img = self.rotateCanvasImage(flipped_canvas_img, rotate_deg)

                    # This fills in any black spots from rotation with pixels from the original flipped image.
                    where = np.array(np.where(rotated_canvas_img))

                    flipped_canvas_img[where[0], where[1]] = rotated_canvas_img[where[0], where[1]]

                    rotated_canvas_img = flipped_canvas_img
                else:
                    rotated_canvas_img = flipped_canvas_img

                self.addPastedImages(canvas_img_file, rotated_canvas_img, paste_labels,
                                     save_img_dir, save_label_dir, canvas_idx, tile_idx, out_labels)
                tile_idx += 1

        return tile_idx

    def generateImages(self, canvas_img_files, paste_labels, save_img_dir, save_label_dir):
        """
        Intialize a generic detectnet data generator class. It finds the filenames for canvas and paste images, and
        labels, and splits them into train and validation spilts.
        :param canvas_img_files: canvas image files
        :param paste_labels: paste image files.
        :param save_img_dir: save image directory.
        :param save_label_dir: save label directory.
        """

        out_labels = []

        # Go through each canvas image and generate a set of images from it depending on its size.
        canvas_idx = 1
        for canvas_img_file in canvas_img_files:
            # canvas_img_file = '/media/dcofer/Ubuntu_Data/drone_images/landscapes/vlcsnap-2018-12-21-1.png'
            # canvas_img_orig = misc.imread(canvas_img_file)
            canvas_img_orig = cv2.imread(canvas_img_file)
            # utils.showAndWait('canvas_img_orig', canvas_img_orig)

            logging.info("Processing file {}. Shape {}".format(canvas_img_file, canvas_img_orig.shape))

            width_multiple = float(canvas_img_orig.shape[1])/self.final_img_width
            height_multiple = float(canvas_img_orig.shape[0])/self.final_img_height

            # If one of the dimensions are less than our final values then just use this image once as is.
            if width_multiple < 1 or height_multiple < 1:
                ratio = float(self.final_img_width) / self.final_img_height

                # Use the dimension that is smallest and scale the image up so it is greater than final image height
                if width_multiple < height_multiple:
                    new_height = int(canvas_img_orig.shape[0] * ratio)
                    if new_height < self.final_img_height:
                        new_height = self.final_img_height
                    canvas_img = resize(canvas_img_orig, [new_height, self.final_img_width])
                else:
                    new_width = int(canvas_img_orig.shape[1] * ratio)
                    if new_width < self.final_img_width:
                        new_width = self.final_img_width
                    canvas_img = resize(canvas_img_orig, [self.final_img_height, new_width])
            else:
                canvas_img = canvas_img_orig

            canvas_img = img_as_ubyte(canvas_img.copy())
            # utils.showAndWait('canvas_img', canvas_img)
            # cv2.imwrite('/media/dcofer/Ubuntu_Data/drone_images/canvas_img.png', canvas_img)

            # Now recompute the multiple after potential resizing
            width_multiple = float(canvas_img.shape[1])/self.final_img_width
            height_multiple = float(canvas_img.shape[0])/self.final_img_height

            tile_idx = self.splitCanvasIntoTiles(canvas_img_file, canvas_img, paste_labels,
                                                 save_img_dir, save_label_dir,
                                                 width_multiple, height_multiple,
                                                 canvas_idx, out_labels)

            # Now resize the entire image into the final size and add paste images.
            whole_canvas_img = img_as_ubyte(resize(canvas_img_orig, [self.final_img_height, self.final_img_width]))

            # utils.showAndWait('whole_canvas_img', whole_canvas_img)
            flipped_canvas_img = np.fliplr(whole_canvas_img)
            # utils.showAndWait('flipped_canvas_img', flipped_canvas_img)
            rotated_canvas_img = self.rotateCanvasImage(flipped_canvas_img)

            self.addPastedImages(canvas_img_file, rotated_canvas_img, paste_labels, save_img_dir,
                                 save_label_dir, canvas_idx, tile_idx+1, out_labels)
            canvas_idx += 1

            logging.info("Canvas Idx: {}".format(canvas_idx))

        json_txt = json.dumps(out_labels)
        out_file = save_img_dir + "/real_output_labels.json"
        with open(out_file, 'w') as f:
            f.write(json_txt)

    def generate(self):
        """
        Intialize a generic detectnet data generator class. It finds the filenames for canvas and paste images, and
        labels, and splits them into train and validation spilts.
        """
        logging.info("Initializing image dataset generator ...")

        logging.info("Generating training images.")
        self.generateImages(self.canvas_img_files,
                            self.paste_labels,
                            self.train_img_dir,
                            self.train_label_dir)

        logging.info("Finished generating images.")

if __name__ == "__main__":
    np.random.seed(long(time.time()))

    utils.setupLogging('real_image_gen')

    args = processArgs()
    gen = RealImageDataGen(args)

    gen.initialize()

    gen.generate()
