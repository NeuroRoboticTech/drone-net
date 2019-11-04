"""
Converts yolo labels to sloth json format.
"""
import sys
import os
import shutil
import logging
import argparse
import utils
import json
import cv2
import time

def processArgs():
    """ Processes command line arguments     """
    parser = argparse.ArgumentParser(description='Converts label types from sloth to yolo.')
    parser.add_argument('--label_json', type=str, required=True,
                        help='paste lkabel json file.')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='Directory where labels will be generated.')

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    utils.setupLogging('sloth_to_yolo')

    args = processArgs()

    if not os.path.exists(args.label_json):
        raise RuntimeError("label json file does not exist: {}".format(args.label_json))

    with open(args.label_json, "r") as read_file:
        labels = json.load(read_file)

    if len(labels) <= 0:
        raise RuntimeError("labels were empty.")

    if os.path.exists(args.label_dir):
        shutil.rmtree(args.label_dir)
        # Give the OS a little time to actually make the new directory. Was running into errors
        # where creating the html folder inside this folder would periodically error out.
        time.sleep(0.1)

    os.mkdir(args.label_dir)

    for l in labels:
        img_file = l['filename']
        logging.info("processing {}".format(img_file))
        img_filename = os.path.basename(img_file)
        img_basename = os.path.splitext(img_filename)[0]
        label_file = args.label_dir + '/' + img_basename + '.txt'

        img = cv2.imread(img_file)
        img_height = img.shape[0]
        img_width = img.shape[1]

        annotations = l['annotations']
        with open(label_file, 'w') as f:
            for a in annotations:
                width = a['width']
                height = a['height']
                top = a['y']
                left = a['x']

                center_x = ((left + width / 2.0)) / img_width
                center_y = ((top + height / 2.0)) / img_height

                norm_width = width / img_width
                norm_height = height / img_height

                f.write("0 {0:.6f} {0:.6f} {0:.6f} {0:.6f}\n".format(center_x, center_y, norm_width, norm_height))
