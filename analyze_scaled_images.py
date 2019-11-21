"""
Analyzes scaled images using the yolo output.
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
    parser.add_argument('--yolo_output_json', type=str, required=True,
                        help='output from the yolo NN.')
    parser.add_argument('--truth_json', type=str, required=True,
                        help='json with truth data.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Data save dir images.')

    args, unknown = parser.parse_known_args()
    return args

class AnalyzeScaledImages ():

    def __init__(self, args):
        """
        Creates a generic detectnet data generator class
        :param args: arguments for generator.
        """
        logging.info("creating image dataset generator ...")

        self.image_dir = args.image_dir
        self.yolo_output_json = args.yolo_output_json
        self.truth_json = args.truth_json
        self.save_dir = args.save_dir

        self.yolo_labels = []
        self.truth_labels = []

        self.img_width = 608
        self.img_height = 608

    def initialize(self):
        """
        Intialize a generic detectnet data generator class. It finds the filenames for canvas and paste images, and
        labels, and splits them into train and validation spilts.
        """
        logging.info("Initializing image dataset generator ...")

        if not os.path.exists(self.image_dir):
            raise RuntimeError("image directory does not exist: {}".format(self.image_dir))

        if not os.path.exists(self.yolo_output_json):
            raise RuntimeError("yolo output json does not exist: {}".format(self.yolo_output_json))

        if not os.path.exists(self.truth_json):
            raise RuntimeError("truth json file does not exist: {}".format(self.truth_json))

        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            # Give the OS a little time to actually make the new directory. Was running into errors
            # where creating the html folder inside this folder would periodically error out.
            time.sleep(0.1)

        os.mkdir(self.save_dir)

        with open(self.yolo_output_json, "r") as read_file:
            yolo_labels = json.load(read_file)

        if len(yolo_labels) <= 0:
            raise RuntimeError("yolo_output_json were empty.")

        with open(self.truth_json, "r") as read_file:
            self.truth_labels = json.load(read_file)

        if len(self.truth_labels) <= 0:
            raise RuntimeError("truth_labels were empty.")

        # Now go through and put the labels into a dictionary keyed by filename
        self.yolo_labels = {}
        for l in yolo_labels:
            base_file = os.path.basename(l['filename'])

            self.yolo_labels[base_file] = l.copy()

    def generate(self):
        """
        Intialize a generic detectnet data generator class. It finds the filenames for canvas and paste images, and
        labels, and splits them into train and validation spilts.
        """
        logging.info("analyzing scaled images ...")

        results = []

        min_areas = []

        for truth_key, truth_label in self.truth_labels.items():
            test_file = truth_label['filename']
            base_file = os.path.basename(test_file)
            base_name = os.path.splitext(base_file)[0]
            min_file_area = -1

            scaled_files = truth_label['scaled']
            for file_label in scaled_files:
                scaled_file = file_label['filename']
                scaled_base_file = file_label['base_file']
                annotations = file_label['annotations']
                scale = file_label['scale']
                min_area = file_label['min_area']

                img_file = self.image_dir + '/' + scaled_file
                if os.path.exists(img_file):
                    logging.info("Processing {}".format(scaled_file))

                    # Get the corresponding predicted labels
                    if scaled_file in self.yolo_labels.keys():
                        predicted = self.yolo_labels[scaled_file]
                        logging.info("found {} in predicted yolo labels.".format(scaled_base_file))

                        img_in = cv2.imread(img_file)

                        pred_objects = predicted['objects']
                        img_labeled = utils.drawYoloObjectLabels(img_in, pred_objects)

                        if utils.overlapsYolo(annotations, pred_objects, self.img_width, self.img_height) and \
                           (min_area < min_file_area or min_file_area < 0):
                            min_file_area = min_area

                        out_img_path = self.save_dir + "/" + scaled_file
                        cv2.imwrite(out_img_path, img_labeled)
                        # utils.showAndWait('img_labeled', img_labeled)
                    else:
                        logging.warning("{} was not found in yolo labels.".format(scaled_base_file))

            if min_file_area > 0:
                min_areas.append(min_area)

        logging.info("Finished val images.")

        avg_min_areas = np.average(min_areas)
        std_min_areas = np.std(min_areas)
        logging.info("Average min area detected: {} +- {} from {} images.".format(avg_min_areas,
                                                                                  std_min_areas,
                                                                                  len(min_areas)))

        # json_txt = json.dumps(all_labels)
        # out_file = self.save_dir + "/real_scaled_labels.json"
        # with open(out_file, 'w') as f:
        #     f.write(json_txt)


if __name__ == "__main__":
    np.random.seed(long(time.time()))

    utils.setupLogging('analyze_scaled_imgs')

    args = processArgs()
    gen = AnalyzeScaledImages(args)

    gen.initialize()

    gen.generate()
