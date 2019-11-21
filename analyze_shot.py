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

frames_per_second = 24.0
res = '720p'

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(cap, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(cap, width, height)
    return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']


def processArgs():
    """ Processes command line arguments     """
    parser = argparse.ArgumentParser(description='List of files to use for training.')
    parser.add_argument('--input_video', type=str, required=True,
                        help='input video.')
    parser.add_argument('--output_video', type=str, required=True,
                        help='output video.')

    args, unknown = parser.parse_known_args()
    return args

def shotAnalysis(img):
    # Perform basic threshold to find white spots and view
    # top_blackout_range = 200
    # top_blackout = np.zeros((top_blackout_range, img.shape[1]), np.uint8)

    white_mask = cv2.inRange(img, np.array([230, 230, 230]), np.array([255, 255, 255]))
    # utils.showAndWait('white_mask', white_mask)

    # white_mask[0:top_blackout_range, :] = top_blackout
    # utils.showAndWait('white_mask_blackout', white_mask)

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

    # utils.showAndWait('streamers', img)

    # detector = cv2.SimpleBlobDetector_create()
    # keypoints = detector.detect(img2)
    # im_with_keypoints = cv2.drawKeypoints(img2, keypoints, np.array([]), (0, 0, 255),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # utils.showAndWait('streamers', im_with_keypoints)

    # return contour_img
    return img


if __name__ == "__main__":
    utils.setupLogging('analyze_shot')

    args = processArgs()

    if not os.path.exists(args.input_video):
        raise RuntimeError("input video does not exist: {}".format(args.input_video))

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(args.input_video)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        raise RuntimeError("video file not opened.")

    out_cap = cv2.VideoCapture(0)
    out = cv2.VideoWriter(args.output_video, get_video_type(args.output_video), 25, get_dims(out_cap, res))

    if (out.isOpened() == False):
        raise RuntimeError("output video file not opened.")

    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            new_frame = shotAnalysis(frame)

            # Display the resulting frame
            cv2.imshow('Shot Analysis', new_frame)

            out.write(new_frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    logging.info("Finished analysis.")