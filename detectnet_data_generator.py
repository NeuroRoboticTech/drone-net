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
    parser.add_argument('--min_paste_dim_size', type=int, default=300,
                        help='minimum size of any dimension of pasted image.')
    parser.add_argument('--max_paste_dim_size', type=int, default=800,
                        help='maximum size of any dimension of pasted image.')
    parser.add_argument('--max_paste_rotation', type=int, default=45,
                        help='maximum rotation that can be randomly added to pasted image.')
    parser.add_argument('--max_canvas_rotation', type=int, default=5,
                        help='maximum rotation that can be randomly added to canvas image.')
    parser.add_argument('--final_img_width', type=int, default=1248,
                        help='height of the final produced image.')
    parser.add_argument('--final_img_height', type=int, default=384,
                        help='width of the final produced image.')
    parser.add_argument('--percent_for_val', type=int, default=10,
                        help='percentage of images to use for val.')

    args, unknown = parser.parse_known_args()
    return args

class DetectNetDataGenerator ():

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
        self.min_paste_dim_size = args.min_paste_dim_size
        self.max_paste_dim_size = args.max_paste_dim_size
        self.final_img_width = args.final_img_width
        self.final_img_height = args.final_img_height
        self.max_paste_rotation = args.max_paste_rotation
        self.max_canvas_rotation = args.max_canvas_rotation
        self.percent_for_val = args.percent_for_val / 100.0

        self.root_train_img_dir = ""
        self.root_val_img_dir = ""

        self.train_img_dir = ""
        self.train_label_dir = ""
        self.val_img_dir = ""
        self.val_label_dir = ""

        self.paste_image_idx = 0

        self.all_paste_files_used = False

        self.generate_masks = True

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

        canvas_img_files = utils.findFilesOfType(self.canvas_image_dir, ['png', 'jpg', 'jpeg'])

        if len(canvas_img_files) <= 0:
            raise RuntimeError("No canvas image files were found")

        np.random.shuffle(canvas_img_files)
        np.random.shuffle(self.paste_labels)

        canvas_val_count = int(len(canvas_img_files) * self.percent_for_val)
        paste_val_count = int(len(self.paste_labels) * self.percent_for_val)

        if canvas_val_count <= 0:
            canvas_val_count = 1

        if paste_val_count <= 0:
            paste_val_count = 1

        self.canvas_train_img_files = canvas_img_files[:-canvas_val_count]
        self.canvas_val_img_files = canvas_img_files[(len(canvas_img_files)-canvas_val_count):]

        c_t_c = len(self.canvas_train_img_files)
        c_v_c = len(self.canvas_val_img_files)

        if len(canvas_img_files) != (c_t_c + c_v_c):
            raise RuntimeError("Mismatch in train/val canvas image split.")

        self.paste_train_img_files = self.paste_labels[:-paste_val_count]
        self.paste_val_img_files = self.paste_labels[(len(self.paste_labels)-paste_val_count):]

        p_t_c = len(self.paste_train_img_files)
        p_v_c = len(self.paste_val_img_files)

        if len(self.paste_labels) != (p_t_c + p_v_c):
            raise RuntimeError("Mismatch in train/val paste image split.")


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

    def incrementNextPastedImageIndex(self, paste_img_files):
        """ Gets the index of the next paste image so we go through them all.

        :param paste_img_files: list of paste image files.
        :return: index of next image
        """

        self.paste_image_idx += 1
        if self.paste_image_idx >= len(paste_img_files):
            self.paste_image_idx = 0
            self.all_paste_files_used = True

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

    def flipPasteImage(self, paste_img, labels):
        flip_val = np.random.randint(0, 100)
        if flip_val < 50:
            logging.info("Rotating by -90.")
            rotate_deg = -90
        else:
            logging.info("Rotating by 90.")
            rotate_deg = 90

        rotated_paste_img, = utils.rotateImg(paste_img, rotate_deg)

        new_labels = []
        for l in labels:
            new_right = utils.rotate([0.0, 0.0], l['x'], rotate_deg)
            new_top = utils.rotate([0.0, 0.0], l['y'], rotate_deg)
            new_left = utils.rotate([0.0, 0.0], l['x']+l['width'], rotate_deg)
            new_bottom = utils.rotate([0.0, 0.0], l['y']+l['height'], rotate_deg)

            new_l = l.copy()
            new_l['x'] = new_right
            new_l['y'] = new_top
            new_l['width'] = new_left - new_right
            new_l['height'] = new_bottom - new_top

            new_labels.append(new_l)
            self.printLabelDims(new_labels)

        return rotated_paste_img, new_labels

    def flipLabels(self, labels, paste_width):

        new_labels = []
        for l in labels:
            # new_labels.append([paste_width - l[2], l[1], paste_width - l[0], l[3]])

            new_l = l.copy()
            new_l['x'] = paste_width - (l['x'] + l['width'])
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

    def printLabelDims(self, labels):

        for l in labels:
            logging.info("    x: {0:.2f}, y: {0:.2f}, w: {0:.2f}, h: {0:.2f}".format(l['x'], l['y'], l['width'], l['height']))

    def drawLabels(self, img, labels):

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

    def addPastedImages(self, canvas_img_file, canvas_img, paste_img_files,
                        save_img_dir, save_label_dir, canvas_idx, tile_idx,
                        out_labels):
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
                # paste_img_file  = '/media/dcofer/Ubuntu_Data/drone-net/images_labeled/dji-mavic-pro-drone-riga-260nw-533974795.jpg'
                paste_label = paste_img_files[paste_img_file_idx]
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
                self.printLabelDims(labels)

                # If the paste height is greater than the canvas height then rotate by 90 or -90 degrees.
                if paste_height > canvas_height:
                    logging.info("paste height to large. flipping over.")
                    paste_img, labels = self.flipPasteImage(paste_img, labels)
                    width_temp = paste_width
                    paste_width = paste_height
                    paste_height = width_temp

                paste_x, paste_y, last_x = self.getNonoverlappingPastePos(last_x, paste_width, paste_height,
                                                                          canvas_width, canvas_height,
                                                                          max_paste_x, max_paste_y)
                if paste_x >= 0 and paste_y >= 0:
                    logging.info("    paste_x: {}".format(paste_x))
                    logging.info("    paste_y: {}".format(paste_y))
                    logging.info("    last_x: {}".format(last_x))

                    flip_val = np.random.randint(0, 100)
                    if flip_val < 50:
                        logging.info("    flip_val: {}. Flipping image.".format(flip_val))
                        paste_img = np.fliplr(paste_img)
                        if mask_img is not None:
                            mask_img = np.fliplr(mask_img)
                        labels = self.flipLabels(labels, paste_width)
                    else:
                        logging.info("    flip_val: {}. Leaving image unflipped".format(flip_val))

                    # Now put them back into the canvas
                    canvas_img[paste_y:(paste_y + paste_height),
                               paste_x:(paste_x + paste_width)] = paste_img

                    if not canvas_mask_img is None and not mask_img is None:
                        canvas_mask_img[paste_y:(paste_y + paste_height),
                                        paste_x:(paste_x + paste_width)] = mask_img

                    # utils.showAndWait('canvas_img', canvas_img)

                    labels = self.adjustLabels(labels, paste_x, paste_y)

                    all_labels.extend(labels)

                    self.incrementNextPastedImageIndex(paste_img_files)
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
        #canvas_img = self.drawLabels(canvas_img, all_labels)

        # utils.showAndWait('canvas_img', canvas_img)

        save_img_filename = '{}_{}.png'.format(canvas_idx, tile_idx)
        save_img_file = save_img_dir + '/{}'.format(save_img_filename)
        logging.info("saving image: {}".format(save_img_file))
        cv2.imwrite(save_img_file, canvas_img)
        #misc.imsave(save_file, canvas_img)

        save_label_file = save_label_dir + '/{}_{}.txt'.format(canvas_idx, tile_idx)
        logging.info("saving lable: {}".format(save_label_file))
        utils.saveDetectNetLabelFile('Car', labels, save_label_file)

        if self.generate_masks and canvas_mask_img is not None:
            save_mask_file = save_label_dir + '/{}'.format(save_img_filename)
            logging.info("saving mask: {}".format(save_mask_file))
            #utils.saveIndexImage(save_mask_file, canvas_mask_img)
            cv2.imwrite(save_mask_file, canvas_mask_img)


        json_label = {"class": "image",
                      "filename": save_img_filename,
                      "annotations": all_labels}
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

    def randomFlipImage(self, img_in):
        flip_val = np.random.randint(0, 100)
        if flip_val < 50:
            logging.info("  flip_val: {}. Flipping image.".format(flip_val))
            flipped_canvas_img = np.fliplr(img_in)
        else:
            logging.info("  flip_val: {}. Leaving canvas unflipped".format(flip_val))
            flipped_canvas_img = img_in

        return flipped_canvas_img

    def splitCanvasIntoTiles(self, canvas_img_file, canvas_img, paste_img_files,
                             save_img_dir, save_label_dir,
                             width_multiple, height_multiple,
                             canvas_idx, out_labels):
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

                flipped_canvas_img = self.randomFlipImage(cut_canvas_img)
                if rotate_deg != 0:
                    rotated_canvas_img = self.rotateCanvasImage(flipped_canvas_img, rotate_deg)
                else:
                    rotated_canvas_img = flipped_canvas_img

                self.addPastedImages(canvas_img_file, rotated_canvas_img, paste_img_files,
                                     save_img_dir, save_label_dir, canvas_idx, tile_idx, out_labels)
                tile_idx += 1

        return tile_idx

    def generateImages(self, canvas_img_files, paste_img_files, save_img_dir, save_label_dir):
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
        canvas_idx = 1
        while not self.all_paste_files_used:
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

                tile_idx = self.splitCanvasIntoTiles(canvas_img_file, canvas_img, paste_img_files,
                                                     save_img_dir, save_label_dir,
                                                     width_multiple, height_multiple,
                                                     canvas_idx, out_labels)

                # Now resize the entire image into the final size and add paste images.
                whole_canvas_img = img_as_ubyte(resize(canvas_img_orig, [self.final_img_height, self.final_img_width]))

                # utils.showAndWait('whole_canvas_img', whole_canvas_img)
                flipped_canvas_img = np.fliplr(whole_canvas_img)
                # utils.showAndWait('flipped_canvas_img', flipped_canvas_img)
                rotated_canvas_img = self.rotateCanvasImage(flipped_canvas_img)

                self.addPastedImages(canvas_img_file, rotated_canvas_img, paste_img_files, save_img_dir,
                                     save_label_dir, canvas_idx, tile_idx+1, out_labels)
                canvas_idx += 1

                if self.all_paste_files_used:
                    json_txt = json.dumps(out_labels)
                    out_file = save_img_dir + "/output_labels.json"
                    with open(out_file, 'w') as f:
                        f.write(json_txt)

                    break

    def generate(self):
        """
        Intialize a generic detectnet data generator class. It finds the filenames for canvas and paste images, and
        labels, and splits them into train and validation spilts.
        """
        logging.info("Initializing image dataset generator ...")

        logging.info("Generating training images.")
        self.generateImages(self.canvas_train_img_files,
                             self.paste_train_img_files,
                             self.train_img_dir,
                             self.train_label_dir)

        self.all_paste_files_used = False

        logging.info("Generating validation images.")
        self.generateImages(self.canvas_val_img_files,
                             self.paste_val_img_files,
                             self.val_img_dir,
                             self.val_label_dir)

        logging.info("Finished generating images.")

if __name__ == "__main__":
    utils.setupLogging('detectnet_data_gen')

    args = processArgs()
    gen = DetectNetDataGenerator(args)

    gen.initialize()

    gen.generate()
