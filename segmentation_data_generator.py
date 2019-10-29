"""
generates greyscale label image for use as annotation while doing semantic segmentation with digits.
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
    parser = argparse.ArgumentParser(description='Generates segmentation data.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data dir containing images and labels.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='dir where data will be generated.')

    args, unknown = parser.parse_known_args()
    return args

class SegmentDataGenerator ():

    def __init__(self, args):
        """
        Creates a generic detectnet data generator class
        :param args: arguments for generator.
        """
        logging.info("creating image dataset generator ...")

        self.data_dir = args.data_dir
        self.save_dir = args.save_dir

        self.img_out_dir = ""
        self.label_out_dir = ""

    def initialize(self):
        """
        Intialize a generic detectnet data generator class. It finds the filenames for canvas and paste images, and
        labels, and splits them into train and validation spilts.
        """
        logging.info("Initializing image dataset generator ...")

        if not os.path.exists(self.data_dir):
            raise RuntimeError("data image directory does not exist: {}".format(self.data_dir))

        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            # Give the OS a little time to actually make the new directory. Was running into errors
            # where creating the html folder inside this folder would periodically error out.
            time.sleep(0.1)

        # # Now create the other directories we will need.
        # os.mkdir(self.save_dir)
        #
        # self.img_out_dir = self.save_dir + "/images"
        # os.mkdir(self.img_out_dir)
        #
        # self.label_out_dir = self.save_dir + "/labels"
        # os.mkdir(self.label_out_dir)
        #
        self.label_files = utils.findFilesOfType(self.data_dir, ['_label.png'])

        if len(self.label_files) <= 0:
            raise RuntimeError("No label image files were found")

    def generateLabelFile(self, label_file, img_file, base_name, max_width, max_height):
        label_img_orig = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
        train_img_orig = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        if train_img_orig.shape != label_img_orig.shape:
            raise RuntimeError("Mismatch between train and label shapes. {}".format(label_file))

        # Find all places where pixels are red and white.
        red_mask = cv2.inRange(label_img_orig, np.array([0, 0, 253]), np.array([5, 5, 255]))
        white_mask = cv2.inRange(label_img_orig, np.array([253, 253, 253]), np.array([255, 255, 255]))

        # Find all places where it is not 0.
        red_where = np.array(np.where(red_mask))
        white_where = np.array(np.where(white_mask))

        # Create a new image of same size with only one bit plane and all zeros
        label_img = np.zeros([max_height, max_width], dtype=np.uint8)

        # Set all red and white places to 1 and 255
        label_img[red_where[0,:], red_where[1,:]] = 1
        if len(white_where) > 0:
            label_img[white_where[0,:], white_where[1,:]] = 255

        if train_img_orig.shape[0] != label_img.shape[0] or \
           train_img_orig.shape[1] != label_img.shape[1]:
            raise RuntimeError("Mismatch between train and label out shapes. {}".format(label_file))

        # Save the new label image.
        label_out = self.label_out_dir + '/' + base_name[:-6] + '.png'
        logging.info("saving label file {}".format(label_out))
        cv2.imwrite(label_out, label_img)

        # Now create a symlink to the original train image.
        img_out = self.img_out_dir + '/' + base_name[:-6] + '.jpg'
        logging.info("saving image file {}".format(img_out))
        os.symlink(img_file, img_out)

    def generate(self):

        # go through once and find max dims for all images.
        max_width = 0
        max_height = 0
        for label_file in self.label_files:
            img = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)

            if img.shape[0] > max_height:
                max_height = img.shape[0]

            if img.shape[1] > max_width:
                max_width = img.shape[1]


        for label_file in self.label_files:
            logging.info("processing {}".format(label_file))
            orig_filename = os.path.basename(label_file)
            orig_basename = os.path.splitext(orig_filename)[0]

            img_file = self.data_dir + orig_basename[:-6] + '.jpg'
            if os.path.exists(img_file):
                self.generateLabelFile(label_file, img_file, orig_basename, max_width, max_height)
            else:
                logging.warning("no matching image file for {}".format(label_file))


if __name__ == "__main__":
    utils.setupLogging('segment_data_gen')

    args = processArgs()
    gen = SegmentDataGenerator(args)

    gen.initialize()

    gen.generate()
