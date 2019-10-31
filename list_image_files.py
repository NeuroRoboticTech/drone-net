"""
Finds image files and writes them to a list.
"""
import os
import argparse
import utils


def processArgs():
    """ Processes command line arguments     """
    parser = argparse.ArgumentParser(description='Generates segmentation data.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data dir containing images and labels.')
    parser.add_argument('--save_file', type=str, required=True,
                        help='file where data will be generated.')

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    utils.setupLogging('segment_data_gen')

    args = processArgs()

    if not os.path.exists(args.data_dir):
        raise RuntimeError("data image directory does not exist: {}".format(args.data_dir))

    img_files = utils.findFilesOfType(args.data_dir, ['png', 'jpg', 'jpeg'])

    with open(args.save_file, 'w') as f:
        for l in img_files:
            f.write("{}\n".format(l))
