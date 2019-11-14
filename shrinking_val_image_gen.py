"""
Takes val images and shrinks them down repeatedly for them to be tested for min resolution accuracy.
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
    parser = argparse.ArgumentParser(description='Generates set of small val iamges.')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Data dir images.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Data save dir images.')
    parser.add_argument('--base_img', type=str, required=True,
                        help='Base image to paste into.')
    parser.add_argument('--label_json', type=str, required=True,
                        help='label json file.')
    parser.add_argument('--min_label_area', type=int, default=50,
                        help='minimum size of smallest dimension of pasted image.')

    args, unknown = parser.parse_known_args()
    return args

class ShrinkingValImageGen ():

    def __init__(self, args):
        """
        Creates a generic detectnet data generator class
        :param args: arguments for generator.
        """
        logging.info("creating image dataset generator ...")

        self.image_dir = args.image_dir
        self.save_dir = args.save_dir
        self.base_img_file = args.base_img
        self.label_json = args.label_json
        self.min_label_area = args.min_label_area

        self.in_labels = []
        self.base_img = None

    def initialize(self):
        """
        Intialize a generic detectnet data generator class. It finds the filenames for canvas and paste images, and
        labels, and splits them into train and validation spilts.
        """
        logging.info("Initializing sacling app ...")

        if not os.path.exists(self.image_dir):
            raise RuntimeError("image directory does not exist: {}".format(self.image_dir))

        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            # Give the OS a little time to actually make the new directory. Was running into errors
            # where creating the html folder inside this folder would periodically error out.
            time.sleep(0.1)

        os.mkdir(self.save_dir)

        if not os.path.exists(self.label_json):
            raise RuntimeError("label json file does not exist: {}".format(self.label_json))

        if not os.path.exists(self.base_img_file):
            raise RuntimeError("base image file does not exist: {}".format(self.base_img_file))

        with open(self.label_json, "r") as read_file:
            self.in_labels = json.load(read_file)

        if len(self.in_labels) <= 0:
            raise RuntimeError("labels were empty.")

        # self.in_labels = self.in_labels[:10]

        logging.info("Loading base image {}".format(self.base_img_file))

        self.base_img = cv2.imread(self.base_img_file)

    def generate(self):
        """
        Intialize a generic detectnet data generator class. It finds the filenames for canvas and paste images, and
        labels, and splits them into train and validation spilts.
        """
        logging.info("Generating scaled images.")

        all_labels = {}
        files = []

        for file_label in self.in_labels:
            l = file_label['annotations']
            file = self.image_dir + '/' + file_label['filename']
            base_file = os.path.basename(file)
            base_name = os.path.splitext(base_file)[0]

            if os.path.exists(file):
                logging.info("Processing {}".format(file))

                file_labels = []

                img_in = cv2.imread(file)

                # Save out base 100% image
                scale_factor = 1.0
                new_base_file = base_name + '_{}.jpg'.format(int(100 * scale_factor))
                new_file = self.save_dir + '/' + new_base_file
                cv2.imwrite(new_file, img_in)
                files.append(new_file)
                logging.info("  saving image: {}".format(new_file))

                min_area = utils.printLabelDims(l)
                json_label = {"class": "image",
                              "filename": new_base_file,
                              "base_file": base_file,
                              "annotations": l,
                              "scale": scale_factor,
                              "min_area": min_area}
                # file_labels.insert(0, json_label)
                file_labels.append(json_label)

                # Now lets go through and shrink this down until it reaches the min value.
                done = False
                scale_factor = 0.9
                while not done:
                    new_height = int(img_in.shape[0] * scale_factor)
                    new_width = int(img_in.shape[1] * scale_factor)

                    new_img = img_as_ubyte(resize(img_in, [new_height, new_width]))
                    new_l = utils.scaleLabels(l, scale_factor)

                    min_area = utils.printLabelDims(new_l)
                    if min_area > self.min_label_area:
                        scale_img = self.base_img.copy()

                        offset_x = int((scale_img.shape[1] - new_img.shape[1])/2.0)
                        offset_y = int((scale_img.shape[0] - new_img.shape[0])/2.0)

                        # utils.showAndWait('base_img', base_img)
                        # utils.showAndWait('new_img', new_img)
                        scale_img[offset_y:(offset_y + new_img.shape[0]),
                                  offset_x:(offset_x + new_img.shape[1])] = new_img
                        # utils.showAndWait('base_img', scale_img)

                        new_base_file = base_name + '_{}.jpg'.format(int(100*scale_factor))
                        new_file = self.save_dir + '/' + new_base_file
                        cv2.imwrite(new_file, scale_img)
                        files.append(new_file)
                        logging.info("  saving image: {}".format(new_file))

                        adjusted = utils.adjustLabels(new_l, offset_x, offset_y)

                        json_label = {"class": "image",
                                      "filename": new_base_file,
                                      "base_file": base_file,
                                      "annotations": adjusted,
                                      "scale": scale_factor,
                                      "min_area": min_area}
                        # file_labels.insert(0, json_label)
                        file_labels.append(json_label)

                        scale_factor *= 0.9
                        logging.info("  reducing scale factor to {}".format(scale_factor))
                    else:
                        done = True
                        logging.info("  went below min area: {} < {}".format(min_area, self.min_label_area))

                json_label = {"filename": file,
                              "base_file": base_file,
                              "scaled": file_labels}

                all_labels[base_file] = json_label

        logging.info("Finished val images.")

        json_txt = json.dumps(all_labels)
        out_file = self.save_dir + "/../real_scaled_labels.json"
        with open(out_file, 'w') as f:
            f.write(json_txt)

        # Write out file list
        out_file = self.save_dir + "/../real_scaled_files.txt"
        with open(out_file, 'w') as f:
            for l in files:
                f.write("{}\n".format(l))


if __name__ == "__main__":
    np.random.seed(long(time.time()))

    utils.setupLogging('real_image_gen')

    args = processArgs()
    gen = ShrinkingValImageGen(args)

    gen.initialize()

    gen.generate()
