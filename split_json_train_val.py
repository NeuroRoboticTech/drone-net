"""
Converts yolo labels to sloth json format.
"""
import sys
import os
import logging
import argparse
import utils
import json
import numpy as np

def processArgs():
    """ Processes command line arguments     """
    parser = argparse.ArgumentParser(description='List of files to use for training.')
    parser.add_argument('--train_list', type=str, required=True,
                        help='List of files to use for training.')
    parser.add_argument('--val_list', type=str, required=True,
                        help='List of files to use for val.')
    parser.add_argument('--root_json', type=str, required=True,
                        help='dir where labels are stored.')
    parser.add_argument('--train_json', type=str, required=True,
                        help='json file to use for training.')
    parser.add_argument('--val_json', type=str, required=True,
                        help='json file to use for val.')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='directory where images are located.')

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    utils.setupLogging('split_json_train_val')

    args = processArgs()

    train_labels = []
    val_labels = []

    with open(args.root_json, "r") as read_file:
        labels = json.load(read_file)

    if len(labels) <= 0:
        raise RuntimeError("Labels were empty.")

    with open(args.train_list, "r") as read_file:
        train_files = read_file.readlines()

    if len(train_files) <= 0:
        raise RuntimeError("Train files were empty.")

    with open(args.val_list, "r") as read_file:
        val_files = read_file.readlines()

    if len(val_files) <= 0:
        raise RuntimeError("Val files were empty.")


    # Go through the labels and put them in the correct bin
    for l in labels:

        file = l['filename']
        orig_filename = str(os.path.basename(file))
        # orig_basename = os.path.splitext(orig_filename)[0] + '.jpg'

        if any(orig_filename in s for s in train_files):
            logging.info("Adding {} to the train json.".format(orig_filename))
            train_labels.append(l.copy())
        elif any(orig_filename in s for s in val_files):
            logging.info("Adding {} to the val json.".format(orig_filename))
            val_labels.append(l.copy())
        else:
            logging.info("file {} was not in either the train or val lists.".format(orig_filename))
            if os.path.exists(args.image_dir + '/' + file):
                flip_val = np.random.randint(0, 100)
                if flip_val < 50:
                    logging.info("  Adding to training list")
                    train_labels.append(l.copy())
                else:
                    logging.info("  Adding to val list")
                    val_labels.append(l.copy())
            else:
                logging.info("  File does not really exist so skipping it.")

    logging.info("Saving train json to: {}".format(args.train_json))
    json_txt = json.dumps(train_labels)
    with open(args.train_json, 'w') as f:
        f.write(json_txt)

    logging.info("Saving val json to: {}".format(args.train_json))
    json_txt = json.dumps(val_labels)
    with open(args.val_json, 'w') as f:
        f.write(json_txt)

    logging.info("Finished splitting labels.")