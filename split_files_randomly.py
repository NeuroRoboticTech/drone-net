"""
Randomly slipts files into a different folder. Used to split out val files from training files.
"""
import sys
import os
import logging
import argparse
import utils
import json
import numpy as np
import shutil
import time

def processArgs():
    """ Processes command line arguments     """
    parser = argparse.ArgumentParser(description='List of files to use for training.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory where data files are located.')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='directory where moved files will be located.')
    parser.add_argument('--count', type=str, required=True,
                        help='Number of files to move.')
    parser.add_argument('--extension', type=str, default='png',
                        help='Extension of files to move')

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    utils.setupLogging('split_json_train_val')

    args = processArgs()

    count = int(args.count)

    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
        # Give the OS a little time to actually make the new directory. Was running into errors
        # where creating the html folder inside this folder would periodically error out.
        time.sleep(0.1)

    # Now create the other directories we will need.
    os.mkdir(args.save_dir)

    files = utils.findFilesOfType(args.data_dir, [args.extension])

    if len(files) <= 0:
        raise RuntimeError("No data files were found")

    if len(files) < count:
        raise RuntimeError("Number of files is less than count to randomly move: {} < {}".format(len(files),
                                                                                                 args.count))

    np.random.shuffle(files)

    move_files = files[:count]

    for file in move_files:
        base_filename = os.path.basename(file)

        new_path = args.save_dir + '/' + base_filename
        shutil.move(file, new_path)

    logging.info("Finished splitting files.")