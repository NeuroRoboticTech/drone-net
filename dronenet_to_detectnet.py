"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import sys
import os
import random
import logging
import shutil
import time
import math

import numpy as np
import cv2
import scipy.misc as misc
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import  img_as_ubyte
import argparse

def setupLogging():
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/{1}.log".format('.', 'dronene_to_detectnet'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    #logging.basicConfig(filename='/home/dcofer/detect_net_data_gen.log', level=logging.INFO)
    #logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    rootLogger.setLevel(level=logging.INFO)
    logging.info("starting up")

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


def loadLabels(label_file, height, width):

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


def writeFileList(list, filename):
    with open(filename, 'w') as f:
        for item in list:
            #logging.debug(item)
            f.write("%s\n" % item)


def saveLabelFile(label, list, filename):
    with open(filename, 'w') as f:
        for item in list:
            f.write("{} 0.0 0 0.0 {} {} {} {} 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n".format(label, item[0],
                                                                                    item[1], item[2],
                                                                                    item[3]))


def processArgs():
    """ Processes command line arguments     """
    parser = argparse.ArgumentParser(description='Generates detectnet simulated data.')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Data dir containing original images.')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Data dir containing original labels.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Data dir where train/val images will be generated.')
    parser.add_argument('--final_img_width', type=int, default=1248,
                        help='height of the final produced image.')
    parser.add_argument('--final_img_height', type=int, default=384,
                        help='width of the final produced image.')
    parser.add_argument('--percent_for_val', type=int, default=10,
                        help='percentage of images to use for val.')

    args, unknown = parser.parse_known_args()
    return args


class DroneNetToDetectNet ():

    def __init__(self, args):
        """
        Creates a generic detectnet data generator class
        :param args: arguments for generator.
        """
        logging.info("creating image dataset generator ...")

        self.image_dir = args.image_dir
        self.label_dir = args.label_dir
        self.save_dir = args.save_dir
        self.final_img_width = args.final_img_width
        self.final_img_height = args.final_img_height
        self.percent_for_val = args.percent_for_val / 100.0

        self.root_train_img_dir = ""
        self.root_val_img_dir = ""

        self.train_img_dir = ""
        self.train_label_dir = ""
        self.val_img_dir = ""
        self.val_label_dir = ""

        self.canvas_paste_links = []

        self.paste_image_idx = 0

        self.max_img_width = 0
        self.max_img_height = 0

    def initialize(self):
        """
        Intialize a generic detectnet data generator class. It finds the filenames for canvas and paste images, and
        labels, and splits them into train and validation spilts.
        """
        logging.info("Initializing image dataset generator ...")

        if not os.path.exists(self.image_dir):
            raise RuntimeError("image directory does not exist: {}".format(self.image_dir))

        if not os.path.exists(self.label_dir):
            raise RuntimeError("label directory does not exist: {}".format(self.label_dir))

        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
            # Give the OS a little time to actually make the new directory. Was running into errors
            # where creating the html folder inside this folder would periodically error out.
            time.sleep(0.1)

        # Now create the other directories we will need.
        os.mkdir(self.save_dir)
        os.mkdir(self.save_dir + "/training_data")
        os.mkdir(self.save_dir + "/training_data/train")
        os.mkdir(self.save_dir + "/training_data/val")

        self.train_img_dir = self.save_dir + "/training_data/train/image"
        os.mkdir(self.train_img_dir)

        self.train_label_dir = self.save_dir + "/training_data/train/label"
        os.mkdir(self.train_label_dir)

        self.val_img_dir = self.save_dir + "/training_data/val/image"
        os.mkdir(self.val_img_dir)

        self.val_label_dir = self.save_dir + "/training_data/val/label"
        os.mkdir(self.val_label_dir)

        img_files = findFilesOfType(self.image_dir, ['png', 'jpg', 'jpeg'])

        if len(img_files) <= 0:
            raise RuntimeError("No image files were found")

        np.random.shuffle(img_files)

        val_count = int(len(img_files) * self.percent_for_val)

        if val_count <= 0:
            val_count = 1

        self.train_img_files = img_files[:-val_count]
        self.val_img_files = img_files[(len(img_files)-val_count):]

        c_t_c = len(self.train_img_files)
        c_v_c = len(self.val_img_files)

        if len(img_files) != (c_t_c + c_v_c):
            raise RuntimeError("Mismatch in train/val image split.")

        filename = self.save_dir + "/train_images.csv"
        writeFileList(self.train_img_files, filename)

        filename = self.save_dir + "/val_images.csv"
        writeFileList(self.val_img_files, filename)

    def generateImages(self, img_files, save_img_dir, save_label_dir):
        """
        Pads the image so it is the correct size and then reworks the labels so they match KITTI format
        :param img_files: canvas image files
        :param save_img_dir: save image directory.
        :param save_label_dir: save label directory.
        """

        for img_file in img_files:
            # canvas_img_file = '/media/dcofer/Ubuntu_Data/drone_images/landscapes/vlcsnap-2018-12-21-1.png'
            # canvas_img_orig = misc.imread(canvas_img_file)
            img_orig = cv2.imread(img_file)
            # showAndWait('canvas_img_orig', canvas_img_orig)

            logging.info("Processing file {}. Shape {}".format(img_file, img_orig.shape))

            if img_orig.shape[0] > self.max_img_height:
                self.max_img_height = img_orig.shape[0]

            if img_orig.shape[1] > self.max_img_width:
                self.max_img_width = img_orig.shape[1]

            # Try to load the label file for this image.
            orig_filename = os.path.basename(img_file)
            orig_basename = os.path.splitext(orig_filename)[0]
            label_file = self.label_dir + '/' + orig_basename + '.txt'
            labels = loadLabels(label_file, img_orig.shape[0], img_orig.shape[1])

            # put the image in the upper left corner of the larger canvas so the label coordinates remain correct.
            buffer_img = np.zeros([self.final_img_height, self.final_img_width, img_orig.shape[2]], dtype=np.uint8)

            buffer_img[0:img_orig.shape[0],
                       0:img_orig.shape[1]] = img_orig

            save_img_file = save_img_dir + '/{}'.format(orig_filename)
            cv2.imwrite(save_img_file, buffer_img)
            logging.info("saving image: {}".format(save_img_file))

            # The redo the label data into KITTI format
            save_label_file = save_label_dir + '/{}.txt'.format(orig_basename)
            saveLabelFile('Car', labels, save_label_file)


    def generate(self):
        """
        Intialize a generic detectnet data generator class. It finds the filenames for canvas and paste images, and
        labels, and splits them into train and validation spilts.
        """
        logging.info("Initializing image dataset generator ...")

        logging.info("Generating training images.")
        self.generateImages(self.train_img_files,
                            self.train_img_dir,
                            self.train_label_dir)

        logging.info("Generating validation images.")
        self.generateImages(self.val_img_files,
                            self.val_img_dir,
                            self.val_label_dir)

        logging.info("Finished generating images.")
        logging.info("Max image width: {}".format(self.max_img_width))
        logging.info("Max image height: {}".format(self.max_img_height))

if __name__ == "__main__":
    setupLogging()

    args = processArgs()
    gen = DroneNetToDetectNet(args)

    gen.initialize()

    gen.generate()
