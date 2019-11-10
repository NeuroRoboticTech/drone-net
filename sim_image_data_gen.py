"""
Generates sim data from landscapes and drone images. Generates for DetectNet and image segmentation.
"""
import sys
import os
import random
import logging
import shutil
import time
import math
from geometry import *
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
    parser.add_argument('--paste_label_dir', type=str, required=True,
                        help='Data dir that contains paste labels.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Data dir where sim images will be generated.')
    parser.add_argument('--max_paste_rotation', type=int, default=60,
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


class SimImageDataGen():

    def __init__(self, args):
        """
        Creates a generic detectnet data generator class
        :param args: arguments for generator.
        """
        logging.info("creating image dataset generator ...")

        self.canvas_image_dir = args.canvas_image_dir
        self.paste_image_dir = args.paste_image_dir
        self.paste_label_dir = args.paste_label_dir
        self.save_dir = args.save_dir
        self.final_img_width = args.final_img_width
        self.final_img_height = args.final_img_height
        self.min_paste_dim_size = 30
        self.max_paste_dim_size = int(args.final_img_width * 0.9)
        self.max_paste_rotation = args.max_paste_rotation
        self.max_canvas_rotation = args.max_canvas_rotation
        self.min_pasted_per_canvas = 0
        self.max_pasted_per_canvas = 3
        self.max_canvas_images = args.max_canvas_images

        self.canvas_img_files = []

        self.train_img_dir = ""
        self.train_label_dir = ""

        self.canvas_paste_links = []
        self.paste_img_files = []

        self.paste_image_idx = 0
        self.all_paste_files_used_count = 0

        # Initially use no blur or bright/contrast. Add that on next go around.
        self.blur_thresh = 0
        self.bright_thresh = 0
        self.bright_max = 25
        self.contrast_max = 0.02
        self.blur_max = 4

        self.file_prefix = "sim"

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

        if not os.path.exists(self.paste_label_dir):
            raise RuntimeError("Paste label directory does not exist: {}".format(self.paste_label_dir))

        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            # Give the OS a little time to actually make the new directory. Was running into errors
            # where creating the html folder inside this folder would periodically error out.
            time.sleep(0.1)

        # Now create the other directories we will need.
        os.mkdir(self.save_dir)

        self.train_img_dir = self.save_dir + "/images"
        os.mkdir(self.train_img_dir)

        self.train_label_dir = self.save_dir + "/labels"
        os.mkdir(self.train_label_dir)

        self.canvas_img_files = utils.findFilesOfType(self.canvas_image_dir, ['png', 'jpg', 'jpeg'])
        self.paste_img_files = utils.findFilesOfType(self.paste_image_dir, ['png', 'jpg', 'jpeg'])

        if len(self.canvas_img_files) <= 0:
            raise RuntimeError("No canvas image files were found")

        if len(self.paste_img_files) <= 0:
            raise RuntimeError("No pate image files were found")

        if self.max_canvas_images > 0:
            if self.max_canvas_images > len(self.canvas_img_files):
                raise RuntimeError("Number of canvas images is less than max count: {} > {}".format(
                    len(self.canvas_img_files), self.max_canvas_images))

            self.canvas_img_files = self.canvas_img_files[:self.max_canvas_images]

        np.random.shuffle(self.canvas_img_files)
        np.random.shuffle(self.paste_img_files)

    def loadPasteImage(self, filename):
        """ Loads a paste image and mask.

        :param filename: path to paste image file
        :return: paste image and mask
        """
        # paste_img_in = misc.imread(filename)
        paste_img_in = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        # showAndWait('paste_img_in', paste_img_in)

        if paste_img_in.shape[2] < 4:
            paste_img = paste_img_in
        elif paste_img_in.shape[2] == 4:
            paste_img = paste_img_in[:, :, 0:3]
        else:
            raise RuntimeError("Invalid number of paste bitmap layers for {}".format(filename))

        # showAndWait('paste_img', paste_img)
        # showAndWait('paste_img_mask', paste_img_mask)

        return paste_img

    def incrementNextPastedImageIndex(self, paste_img_files):
        """ Gets the index of the next paste image so we go through them all.

        :param paste_img_files: list of paste image files.
        :return: index of next image
        """

        self.paste_image_idx += 1
        if self.paste_image_idx >= len(paste_img_files):
            self.paste_image_idx = 0
            self.all_paste_files_used_count += 1

            self.max_paste_dim_size = self.max_paste_dim_size * 0.65

            np.random.shuffle(self.canvas_img_files)
            np.random.shuffle(self.paste_img_files)

            if self.all_paste_files_used_count == 1:
                self.blur_thresh = 70
                self.bright_thresh = 70
                self.bright_max = 50
                self.contrast_max = 0.08
                self.blur_max = 5
                self.max_pasted_per_canvas = 3
            elif self.all_paste_files_used_count == 2:
                self.blur_thresh = 80
                self.bright_thresh = 80
                self.bright_max = 40
                self.contrast_max = 0.05
                self.blur_max = 3
                self.max_pasted_per_canvas = 4
            elif self.all_paste_files_used_count == 3 or \
                 self.all_paste_files_used_count == 4:
                self.blur_thresh = 90
                self.bright_thresh = 90
                self.bright_max = 30
                self.contrast_max = 0.03
                self.blur_max = 3
                self.max_pasted_per_canvas = 4
            elif self.all_paste_files_used_count == 5:
                self.blur_thresh = 100
                self.bright_thresh = 100
                self.bright_max = 70
                self.contrast_max = 0.2
                self.blur_max = 5
                self.max_pasted_per_canvas = 2
                self.max_paste_dim_size = int(self.final_img_width * 0.8)
            elif self.all_paste_files_used_count == 6:
                self.blur_thresh = 100
                self.bright_thresh = 100
                self.bright_max = 80
                self.contrast_max = 0.15
                self.blur_max = 6
                self.max_pasted_per_canvas = 3
            elif self.all_paste_files_used_count == 11:
                self.max_paste_dim_size = int(self.final_img_width * 0.8)
                self.all_paste_files_used_count = 0

        return self.paste_image_idx

    def calcNewPasteDims(self, paste_width, paste_height):
        """ Calculates the new dimensions of the paste image

        :param paste_width: width
        :param paste_height: height
        :return: new width and height
        """

        if self.min_paste_dim_size >= self.max_paste_dim_size:
            logging.warning("Mismatch in paste dim sizes")

        paste_rand_height = np.random.randint(self.min_paste_dim_size, self.max_paste_dim_size)
        logging.info("  paste_rand_dim: {}".format(paste_rand_height))

        # Ensure it can never try to blow up a paste image, only reduce it.
        if paste_rand_height > paste_height:
            paste_rand_height = paste_height

        new_paste_height = paste_rand_height
        new_paste_width = int(paste_rand_height * (float(paste_width) / paste_height))

        if new_paste_width > self.max_paste_dim_size:
            new_paste_width = paste_rand_height
            new_paste_height = int(paste_rand_height * (float(paste_height) / paste_width))

        if new_paste_width < self.min_paste_dim_size:
            new_paste_width = self.min_paste_dim_size
            new_paste_height = int(new_paste_width * (float(paste_width) / paste_height))

        return new_paste_width, new_paste_height

    def overlapping(self, x, y, width, height, labels):
        """ Test if the specified coordinates overlaps any existing labels

        :param x: x pos
        :param y: y pos
        :param width: width
        :param height: height
        :param labels: existing labels
        :return: True if overlaps, otherwise False.
        """

        test_rect = Rect(x, y, width, height)
        logging.debug("test: ({}, {}), ({}, {})".format(x, y,
                                                       x + width,
                                                       y + height))
        logging.debug("test_rect: {}".format(str(test_rect)))

        for label in labels:
            label_x = label['x']
            label_y = label['y']
            label_width = label['width']
            label_height = label['height']
            logging.debug("label: ({}, {}), ({}, {})".format(label_x, label_y,
                                                            label_x + label_width,
                                                            label_y + label_height))

            label_rect = Rect(label_x, label_y, label_width, label_height)
            logging.debug("label_rect: {}".format(str(label_rect)))

            if label_rect.overlaps(test_rect):
                return True

        return False

    def getNonoverlappingPastePos(self, paste_width, paste_height,
                                  canvas_width, canvas_height, labels):
        """ gets random paste positions until it finds one that will not overlap with any already existing paste images.

        :param paste_width: width
        :param paste_height: height
        :param canvas_width: width
        :param canvas_height: height
        :param labels: list of previous labels
        :return: new x,y positions for paste
        """

        logging.info("** Testing overlap.")
        logging.info("")

        overlapping = True
        paste_x = 0
        paste_y = 0
        count = 0

        if paste_width >= canvas_width or paste_height >= canvas_height:
            logging.info("paste dim greater than canvas: ".format(canvas_width - paste_width, canvas_height - paste_height))
            logging.info("  canvas: ({}, {}) ".format(canvas_width, canvas_height))
            logging.info("  paste:  ({}, {}) ".format(paste_width, paste_height))

            return -1, -1

        while overlapping and count < 15:
            logging.info("paste_x dims: ({}, {})".format(canvas_width - paste_width, canvas_height - paste_height))
            paste_x = np.random.randint(0, canvas_width - paste_width)
            paste_y = np.random.randint(0, canvas_height - paste_height)

            logging.info("paste_coords: ({}, {})".format(paste_x, paste_y))
            if not self.overlapping(paste_x, paste_y, paste_width, paste_height, labels):
                overlapping = False

            count += 1
            logging.info("overlap count: {}".format(count))

        if overlapping:
            paste_x = -1
            paste_y = -1

        logging.info("")

        return paste_x, paste_y

    def addPastedImages(self, canvas_img_file, canvas_img, paste_img_files,
                        paste_label_dir, save_img_dir, save_label_dir, canvas_idx,
                        tile_idx, out_labels):
        """
        Adds paste images to the canvas file and saves it and the labels.
        :param canvas_img_file: canvas image filename to split
        :param canvas_img: canvas image to split
        :param paste_img_files: paste image files.
        :param width_multiple: multiple of width to final image size.
        :param height_multiple: multiple of height to final image size.
        :param canvas_idx: canvas index.
        :param tile_idx: tile index.
        """

        canvas_width = canvas_img.shape[1]
        canvas_height = canvas_img.shape[0]

        num_pastes = np.random.randint(self.min_pasted_per_canvas, self.max_pasted_per_canvas)
        labels = []

        if canvas_height != self.final_img_height or canvas_width != self.final_img_width:
            logging.error("The canvas height for a paste add does not match the final image dimensions. Skipping image.")
            return

        canvas_label = np.zeros([canvas_height, canvas_width, 3], dtype=np.uint8)

        logging.info("num_pastes: {}".format(num_pastes))

        for past_idx in range(num_pastes):
            paste_img_file_idx = self.paste_image_idx
            # paste_img_file  = '/media/dcofer/Ubuntu_Data/drone_images/drones/111.png'
            paste_img_file = paste_img_files[paste_img_file_idx]
            paste_filename = os.path.basename(paste_img_file)
            paste_basename = os.path.splitext(paste_filename)[0]

            logging.info("  Pasting in {}".format(paste_img_file))
            logging.info("    Paste Image Idx {}".format(paste_img_file_idx))
            paste_img = self.loadPasteImage(paste_img_file)
            # utils.showAndWait('paste_img', paste_img)

            paste_label_file = paste_label_dir + '/' + paste_basename + '_label.png'
            paste_label_img = cv2.imread(paste_label_file, cv2.IMREAD_UNCHANGED)
            paste_label_img = cv2.cvtColor(paste_label_img, cv2.COLOR_BGR2GRAY)
            # utils.showAndWait('paste_label_img', paste_label_img)

            paste_width = paste_img.shape[1]
            paste_height = paste_img.shape[0]
            logging.info("    paste_width: {}".format(paste_width))
            logging.info("    paste_height: {}".format(paste_height))

            if paste_label_img.shape[0] != paste_height or paste_label_img.shape[1] != paste_width:
                raise RuntimeError("Paste label dims do not match paste image.")

            new_paste_width, new_paste_height = self.calcNewPasteDims(paste_width, paste_height)

            logging.info("    new_paste_width: {}".format(new_paste_width))
            logging.info("    new_paste_height: {}".format(new_paste_height))

            if paste_width != new_paste_width or paste_height != new_paste_height:
                sized_paste_img = cv2.resize(paste_img, dsize=(new_paste_width, new_paste_height),
                                             interpolation=cv2.INTER_AREA)
                sized_paste_mask = cv2.resize(paste_label_img, dsize=(new_paste_width, new_paste_height),
                                              interpolation=cv2.INTER_AREA)
            else:
                sized_paste_img = paste_img
                sized_paste_mask = paste_label_img

            # utils.showAndWait('sized_paste_img', sized_paste_img)
            # utils.showAndWait('sized_paste_mask', sized_paste_mask)

            flip_val = np.random.randint(0, 100)
            if flip_val < 50:
                logging.info("    flip_val: {}. Flipping image.".format(flip_val))
                flipped_paste_img = np.fliplr(sized_paste_img)
                flipped_paste_mask = np.fliplr(sized_paste_mask)
            else:
                logging.info("    flip_val: {}. Leaving image unflipped".format(flip_val))
                flipped_paste_img = sized_paste_img
                flipped_paste_mask = sized_paste_mask

            rotate_deg = int(np.random.uniform(-self.max_paste_rotation, self.max_paste_rotation))
            logging.info("    rotate_deg: {}.".format(rotate_deg))
            rotated_paste_img, rotated_paste_mask  = utils.rotateImg(flipped_paste_img, rotate_deg,
                                                                     mask_in=flipped_paste_mask)
            # utils.showAndWait('rotated_paste_img', rotated_paste_img)
            # utils.showAndWait('rotated_paste_mask', rotated_paste_mask)

            paste_width = rotated_paste_img.shape[1]
            paste_height = rotated_paste_img.shape[0]

            paste_x, paste_y = self.getNonoverlappingPastePos(paste_width, paste_height,
                                                              canvas_width, canvas_height, labels)
            if paste_x < 0 or paste_y < 0:
                break

            # paste_x = 1081
            # paste_y = 266

            logging.info("    paste_x: {}".format(paste_x))
            logging.info("    paste_y: {}".format(paste_y))

            # Get canvas ROI
            canvas_roi = canvas_img[paste_y:(paste_y + paste_height),
                                    paste_x:(paste_x + paste_width)]
            #canvas_roi = np.zeros([paste_height, paste_width, 3], dtype=np.uint8)
            # showAndWait('canvas_roi', canvas_roi)

            # Regnerate a new mask because the one that was put through all the processing is not
            # intact anymore. This was causing weird artifacting.
            ret, mask = cv2.threshold(rotated_paste_mask, 5, 255, cv2.THRESH_BINARY)
            # mask = new_mask[:, :, 0]
            mask_inv = cv2.bitwise_not(mask)
            # utils.showAndWait('mask', mask)

            # Black out the are of the mask.
            background_roi = cv2.bitwise_and(canvas_roi, canvas_roi, mask=mask_inv)
            # showAndWait('background_roi', background_roi)
            # cv2.imwrite('/media/dcofer/Ubuntu_Data/drone_images/canvas_ros.png', background_roi)

            # Now randomly change brightness and contract of foreground drone
            bright_rand = np.random.randint(0, 100)
            if bright_rand < self.blur_thresh:
                logging.info("    bright_rand: {}. Adjusting brightness/contrast.".format(bright_rand))

                bright_val = np.random.randint(-self.bright_max, self.bright_max)
                contrast_val = np.random.normal(1.0, self.contrast_max)
                if contrast_val < 0.5:
                    contrast_val = 0.7
                if contrast_val > 1.5:
                    contrast_val = 1.3;
                logging.info("    bright_val: {}".format(bright_val))
                logging.info("    contrast_val: {}".format(contrast_val))
                bright_foreground_img = cv2.convertScaleAbs(rotated_paste_img, alpha=contrast_val, beta=bright_val)
                # utils.showAndWait('rotated_paste_img', rotated_paste_img)
                # utils.showAndWait('bright_foreground_img', bright_foreground_img)
            else:
                logging.info("    bright_rand: {}. Leaving image brightness/contrast alone".format(bright_rand))
                bright_foreground_img = rotated_paste_img

            # Now take only region of paste image that is not black
            foreground_roi = cv2.bitwise_and(bright_foreground_img, bright_foreground_img, mask=mask)
            # showAndWait('foreground_roi', foreground_roi)
            # cv2.imwrite('/media/dcofer/Ubuntu_Data/drone_images/foreground_roi.png', foreground_roi)

            # Put them together
            merged_roi = cv2.add(background_roi, foreground_roi)
            # showAndWait('merged_roi', merged_roi)
            # cv2.imwrite('/home/mfp/drone-net/test/merged_roi.png', merged_roi)


            # Now randomly add blur
            blur_rand = np.random.randint(0, 100)
            if blur_rand < self.blur_thresh:
                logging.info("    blur_rand: {}. bluring image.".format(blur_rand))

                blur_val = np.random.randint(1, self.blur_max)
                logging.info("    blur_val: {}".format(blur_val))
                if blur_val > 0:
                    # blured_roi = cv2.GaussianBlur(merged_roi, (blur_val, blur_val), 0)
                    blured_roi = cv2.blur(merged_roi, (blur_val, blur_val))
                else:
                    blured_roi = merged_roi
                # cv2.imwrite('/home/mfp/drone-net/test/blured_roi.png', blured_roi)
            else:
                logging.info("    blur_rand: {}. Leaving image un-blurred".format(blur_rand))
                blured_roi = merged_roi

            # Now put them back into the canvas
            canvas_img[paste_y:(paste_y + paste_height),
                       paste_x:(paste_x + paste_width)] = blured_roi

            # utils.showAndWait('canvas_img', canvas_img)

            # Now put the label into the canvas
            ret, label_mask = cv2.threshold(rotated_paste_mask, 5, 255, cv2.THRESH_BINARY)

            where_label = np.array(np.where(label_mask))

            canvas_label_roi = canvas_label[paste_y:(paste_y + label_mask.shape[0]),
                                            paste_x:(paste_x + label_mask.shape[1]), 2]
            canvas_label_roi[where_label[0], where_label[1]] = 255

            canvas_label[paste_y:(paste_y + label_mask.shape[0]),
                         paste_x:(paste_x + label_mask.shape[1]), 2] = canvas_label_roi

            # utils.showAndWait('canvas_label', canvas_label)

            self.canvas_paste_links.append([canvas_idx, tile_idx, canvas_img_file, paste_img_file,
                                            paste_x, paste_y, paste_width, paste_height])

            json_label = {"class": "rect",
                          "height": paste_height,
                          "width": paste_width,
                          "x": paste_x,
                          "y": paste_y}
            labels.append(json_label)

            self.incrementNextPastedImageIndex(paste_img_files)

        # canvas_img = utils.drawLabels(canvas_img, labels)
        # utils.showAndWait('canvas_img', canvas_img)

        save_img_file = save_img_dir + '/{}_{}_{}.jpg'.format(self.file_prefix, canvas_idx, tile_idx)
        cv2.imwrite(save_img_file, canvas_img)
        logging.info("saving image: {}".format(save_img_file))
        #misc.imsave(save_file, canvas_img)

        save_label_file = save_label_dir + '/{}_{}_{}.txt'.format(self.file_prefix, canvas_idx, tile_idx)
        utils.saveYoloLabelFile(0, labels, save_label_file, canvas_width, canvas_height)
        logging.info("saving lable: {}".format(save_label_file))
        #
        # save_label_file = save_label_dir + '/()_{}_{}_label.png'.format(self.file_prefix, canvas_idx, tile_idx)
        # cv2.imwrite(save_label_file, canvas_label)
        # logging.info("saving lable image: {}".format(save_label_file))

        json_label = {"class": "image",
                      "filename": save_img_file,
                      "annotations": labels}
        out_labels.append(json_label)

    def getForcedRandomRotationValue(self):

        flip_val = np.random.randint(0, 100)
        if flip_val < 50:
            rotate = int(np.random.uniform(2, self.max_canvas_rotation))
        else:
            rotate = int(np.random.uniform(-2, -self.max_canvas_rotation))
        return rotate

    def rotateCanvasImage(self, canvas_img, rotate_deg=0):
        if rotate_deg == 0:
            rotate_deg = self.getForcedRandomRotationValue()

        logging.info("  rotate_deg: {}.".format(rotate_deg))
        max_buff_dim = int(max(canvas_img.shape[0], canvas_img.shape[1]) * 2.0)
        rotated_canvas_img, rotated_canvas_mask = utils.rotateImg(canvas_img, rotate_deg,
                                                                  mask_in=None)
        # showAndWait('rotated_paste_img', rotated_canvas_img)

        if rotated_canvas_img.shape[0] != self.final_img_height or \
                rotated_canvas_img.shape[1] != self.final_img_width:
            left_idx = int(rotated_canvas_img.shape[1] / 2.0 - self.final_img_width / 2.0)
            right_idx = left_idx + self.final_img_width
            top_idx = int(rotated_canvas_img.shape[0] / 2.0 - self.final_img_height / 2.0)
            bottom_idx = top_idx + self.final_img_height
            rotated_canvas_img = rotated_canvas_img[top_idx:bottom_idx,
                                                    left_idx:right_idx]

        return rotated_canvas_img

    def splitCanvasIntoTiles(self, canvas_img_file, canvas_img, paste_img_files,
                             paste_label_dir, save_img_dir, save_label_dir,
                             width_multiple, height_multiple, canvas_idx, out_labels):
        """
        Splits the canvas image into multiple image tiles and adds pasted images to it.
        :param canvas_img_file: canvas image filename to split
        :param canvas_img: canvas image to split
        :param paste_img_files: paste image files.
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
                    rotate_deg = self.getForcedRandomRotationValue()

                if cut_y + tile_height > canvas_height:
                    cut_y = canvas_height - tile_height
                    rotate_deg = self.getForcedRandomRotationValue()

                cut_canvas_img = canvas_img[cut_y:(cut_y+tile_height), cut_x:(cut_x+tile_width)].copy()

                flipped_canvas_img = utils.randomFlipImage(cut_canvas_img)
                if rotate_deg != 0:
                    rotated_canvas_img = self.rotateCanvasImage(flipped_canvas_img, rotate_deg)

                    # This fills in any black spots from rotation with pixels from the original flipped image.
                    where = np.array(np.where(rotated_canvas_img))

                    flipped_canvas_img[where[0], where[1]] = rotated_canvas_img[where[0], where[1]]

                    rotated_canvas_img = flipped_canvas_img
                else:
                    rotated_canvas_img = flipped_canvas_img

                self.addPastedImages(canvas_img_file, rotated_canvas_img, paste_img_files,
                                     paste_label_dir, save_img_dir, save_label_dir, canvas_idx,
                                     tile_idx, out_labels)
                tile_idx += 1

        return tile_idx

    def generateImages(self, canvas_img_files, paste_img_files,
                       paste_label_dir, save_img_dir, save_label_dir):
        """
        Intialize a generic detectnet data generator class. It finds the filenames for canvas and paste images, and
        labels, and splits them into train and validation spilts.
        :param canvas_img_files: canvas image files
        :param paste_img_files: paste image files.
        :param save_img_dir: save image directory.
        :param save_label_dir: save label directory.
        """

        out_labels = []
        # Go through each canvas image and generate a set of images from it depending on its size.
        self.paste_image_idx = 0
        canvas_idx = 1
        for canvas_img_file in canvas_img_files:
            # canvas_img_file = '/media/dcofer/Ubuntu_Data/drone_images/landscapes/vlcsnap-2018-12-21-1.png'
            # canvas_img_orig = misc.imread(canvas_img_file)
            canvas_img_orig = cv2.imread(canvas_img_file)
            # showAndWait('canvas_img_orig', canvas_img_orig)

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
            # showAndWait('canvas_img', canvas_img)
            # cv2.imwrite('/media/dcofer/Ubuntu_Data/drone_images/canvas_img.png', canvas_img)

            # Now recompute the multiple after potential resizing
            width_multiple = float(canvas_img.shape[1])/self.final_img_width
            height_multiple = float(canvas_img.shape[0])/self.final_img_height

            tile_idx = self.splitCanvasIntoTiles(canvas_img_file, canvas_img, paste_img_files,
                                                 paste_label_dir, save_img_dir, save_label_dir,
                                                 width_multiple, height_multiple, canvas_idx,
                                                 out_labels)

            # Now resize the entire image into the final size and add paste images.
            whole_canvas_img = img_as_ubyte(resize(canvas_img_orig, [self.final_img_height, self.final_img_width]))

            # showAndWait('whole_canvas_img', whole_canvas_img)
            flipped_canvas_img = np.fliplr(whole_canvas_img)
            # showAndWait('flipped_canvas_img', flipped_canvas_img)
            rotated_canvas_img = self.rotateCanvasImage(flipped_canvas_img)

            # This fills in any black spots from rotation with pixels from the original flipped image.
            where = np.array(np.where(rotated_canvas_img))

            flipped_canvas_img[where[0], where[1]] = rotated_canvas_img[where[0], where[1]]

            rotated_canvas_img = flipped_canvas_img

            self.addPastedImages(canvas_img_file, rotated_canvas_img, paste_img_files,
                                 paste_label_dir, save_img_dir,
                                 save_label_dir, canvas_idx, tile_idx+1,
                                 out_labels)
            canvas_idx += 1

            logging.info("Canvas Idx: {}".format(canvas_idx))

        logging.info("writing json label files")
        json_txt = json.dumps(out_labels)
        out_file = save_img_dir + "/sim_output_labels.json"
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
                            self.paste_img_files,
                            self.paste_label_dir,
                            self.train_img_dir,
                            self.train_label_dir)


        logging.info("Finished generating images.")

if __name__ == "__main__":
    np.random.seed(long(time.time()))

    utils.setupLogging('sim_image_gen')

    args = processArgs()
    gen = SimImageDataGen(args)

    gen.initialize()

    gen.generate()
