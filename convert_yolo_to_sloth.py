"""
Converts yolo labels to sloth json format.
"""
import sys
import os
import logging
import argparse
import utils
import json
import cv2

def processArgs():
    """ Processes command line arguments     """
    parser = argparse.ArgumentParser(description='Converts label types from yolo to sloth and back.')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Data dir containing images.')
    parser.add_argument('--label_dir', type=str, required=True,
                        help='dir where labels are stored.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='sloth json file that will be generated.')

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    utils.setupLogging('yolo_to_sloth')

    args = processArgs()

    img_files = utils.findFilesOfType(args.image_dir, ['png', 'jpg', 'jpeg'])

    json_labels = []

    for img_file in img_files:
        orig_filename = os.path.basename(img_file)
        orig_basename = os.path.splitext(orig_filename)[0]
        label_file = args.label_dir + '/' + orig_basename + '.txt'

        labels = utils.loadYoloLabels(label_file)

        annotations = []
        for l in labels:
            json_label = {"class": "rect",
                          "height": (l[3]-l[1]),
                          "width": (l[2] - l[0]),
                          "x": l[0],
                          "y": l[1]}
            annotations.append(json_label)

        json_label = {"class": "image",
                      "filename": img_file,
                      "annotations": annotations}

        json_labels.append(json_label)

    json_txt = json.dumps(json_labels)

    with open(args.output_file, 'w') as f:
        f.write(json_txt)

