"""
Generates colored label files from drone paste images
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


class SimImageDataGen():

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

        self.label_files = []

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

        # Now create the other directories we will need.
        os.mkdir(self.save_dir)

        self.label_files =  utils.findFilesOfType(self.data_dir, ['png', 'jpg', 'jpeg'])

        if len(self.label_files) <= 0:
            raise RuntimeError("No label image files were found")

    def generateLabelFile(self, label_file, base_name):
        label_img_orig = cv2.imread(label_file)

        label_img_orig = cv2.cvtColor(label_img_orig, cv2.COLOR_BGR2GRAY)

        ret, label_img_orig = cv2.threshold(label_img_orig, 3, 255, cv2.THRESH_BINARY)

        label_img_new = np.zeros([label_img_orig.shape[0]+10, label_img_orig.shape[1]+10], dtype=np.uint8)

        label_img_new[5:(label_img_orig.shape[0]+5),
                      5:(label_img_orig.shape[1]+5)] = label_img_orig
        # utils.showAndWait('label_img_new', label_img_new)
        # cv2.imwrite(self.save_dir + '/label_img_new.png', label_img_new)

        label_img = np.zeros([label_img_new.shape[0], label_img_new.shape[1], 3], dtype=np.uint8)

        # Find contours in original image.
        (_, contours, _) = cv2.findContours(label_img_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask_color = [0, 0, 255]
        contour_idx = -1
        for cnt in contours:
            # area = cv2.contourArea(cnt)
            # print "contour " + str(contour_idx) + " area: " + str(area)
            # if area > 10.0:
            # print "drawing contour " + str(contour_idx)
            cv2.drawContours(label_img, contours, contour_idx, mask_color, -1)

        # utils.showAndWait('label_img', label_img)
        # cv2.imwrite(self.save_dir + '/label_img.png', label_img)

        final_mask = cv2.dilate(label_img, np.ones((5, 5), np.uint8))

        final_mask = final_mask[5:(final_mask.shape[0]-5),
                                5:(final_mask.shape[1]-5)]

        # utils.showAndWait('final_mask', final_mask)
        # cv2.imwrite(self.save_dir + '/final_mask.png', final_mask)

        # Save the new label image.
        label_out = self.save_dir + '/' + base_name + '_label_.png'
        logging.info("saving label file {}".format(label_out))
        cv2.imwrite(label_out, final_mask)

    def generate(self):

        for label_file in self.label_files:
            # label_file = '/media/dcofer/Ubuntu_Data/drone_images/drones/30_a.png'
            logging.info("processing {}".format(label_file))
            orig_filename = os.path.basename(label_file)
            orig_basename = os.path.splitext(orig_filename)[0]

            self.generateLabelFile(label_file, orig_basename)


if __name__ == "__main__":
    utils.setupLogging('segment_data_gen')

    args = processArgs()
    gen = SimImageDataGen(args)

    gen.initialize()

    gen.generate()
