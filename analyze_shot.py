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
import cv2

def processArgs():
    """ Processes command line arguments     """
    parser = argparse.ArgumentParser(description='List of files to use for training.')
    parser.add_argument('--input_img', type=str, required=True,
                        help='input image.')

    args, unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    utils.setupLogging('analyze_shot')

    args = processArgs()

    if not os.path.exists(args.input_img):
        raise RuntimeError("input image does not exist: {}".format(args.input_img))

    img = cv2.imread(args.input_img)

    # Perform basic threshold to find white spots and view
    white_mask = cv2.inRange(img, np.array([190, 190, 190]), np.array([255, 255, 255]))
    # utils.showAndWait('white_mask', white_mask)

    kernel = np.ones((1, 1), np.uint8)
    img_erosion = cv2.erode(white_mask, kernel, iterations=3)
    # utils.showAndWait('img_erosion', img_erosion)

    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=3)
    # utils.showAndWait('img_dilation', img_dilation)

    # white_mask_inv = cv2.bitwise_not(img_dilation)
    # utils.showAndWait('white_mask_inv', white_mask_inv)

    # streamers = cv2.bitwise_and(img, img, mask=img_dilation)
    # utils.showAndWait('streamers', streamers)

    contour_img = cv2.cvtColor(img_dilation, cv2.COLOR_GRAY2RGB)

    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 4)
    # utils.showAndWait('contour_img', contour_img)

    where = np.array(np.where(contour_img))

    img[where[0], where[1]] = contour_img[where[0], where[1]]

    utils.showAndWait('streamers', img)

    # detector = cv2.SimpleBlobDetector_create()
    # keypoints = detector.detect(img2)
    # im_with_keypoints = cv2.drawKeypoints(img2, keypoints, np.array([]), (0, 0, 255),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # utils.showAndWait('streamers', im_with_keypoints)

    logging.info("Finished analysis.")