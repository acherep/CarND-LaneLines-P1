#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 22:24:09 2017

@author: cherep
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called "gray")
    you should call plt.imshow(gray, cmap="gray")"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from "vertices". The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending
    # on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=4):
    """
    the function averages/extrapolates the line segments you detect 
    to map out the full extent of the lane

    The lines are first separated based on their slope. 
    If slope is greater than .3, the line segments are assigned 
    to the right line of the lane
    if slope is smaller than -.3, the line segments are assigned 
    to the left line of the lane
    Slope between [-.3, .3] is excluded such that almost horizontal lines 
    do not influence the result
    Then average the position of each of the lines and extrapolate to the top 
    and bottom of the lane.

    In order to average and extrapolate the line, the reverse equation 
    of the line was used: x = ay + c
    This approach was chosen because of the known range of values y 
    y = numpy.arange(int(imshape[0] * 0.6), imshape[0]) that eventually 
    is drawn

    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    the lines semi-transparent because of the combination with 
    the weighted_img() function below
    """
    left_line_data = []         # data for the left line ([x,y])
    right_line_data = []        # data for the right line ([x,y])

    imshape = img.shape

    top_y = int(imshape[0] * 0.6)

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope > .3:
                right_line_data.append([x1, y1])
                right_line_data.append([x2, y2])
            if slope < -.3:
                left_line_data.append([x1, y1])
                left_line_data.append([x2, y2])

    left_line_x_array = [item[0] for item in left_line_data]
    left_line_y_array = [item[1] for item in left_line_data]
    right_line_x_array = [item[0] for item in right_line_data]
    right_line_y_array = [item[1] for item in right_line_data]
    # fitting the lines with the approximation and obtaining
    # the coefficients a and c from x = ay + c
    left_line_coefficients = np.polyfit(left_line_y_array,
                                        left_line_x_array, 1)
    right_line_coefficients = np.polyfit(right_line_y_array,
                                         right_line_x_array, 1)
    # extrapolating the lines
    left_line_x_values = np.polyval(left_line_coefficients, [
        top_y, imshape[0]]).astype(int)
    right_line_x_values = np.polyval(right_line_coefficients, [
        top_y, imshape[0]]).astype(int)
    # drawing the lines
    cv2.line(img, (left_line_x_values[0], top_y),
             (left_line_x_values[1], imshape[0]), color, thickness)
    cv2.line(img, (right_line_x_values[0], top_y),
             (right_line_x_values[1], imshape[0]), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.


def weighted_img(img, initial_img, α=0.8, β=1.5, λ=0.):
    """
    `img` is the output of the hough_lines(), an image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image):
    gray = grayscale(image)

    blur_gray = gaussian_blur(gray, 5)

    # the thresholds are chosen quite to be low in order to include lines
    # which have low gradient, especially in the challenge.mp4
    low_threshold = 33
    high_threshold = 100
    edges = canny(blur_gray, low_threshold, high_threshold)

    # calculation of the shape of vertices relative to the picture size
    imshape = image.shape
    # top left and right corners of the region have the offset of 5% from
    # the middle
    x_middle = imshape[1] / 2
    top_left_x = int(x_middle - x_middle * 0.05)
    top_right_x = int(x_middle + x_middle * 0.05)
    # y is between int(imshape[0] * 0.6) and imshape[0]
    top_y = int(imshape[0] * 0.6)

    vertices = np.array([[(0, imshape[0]), (top_left_x, top_y),
                          (top_right_x, top_y), (imshape[1], imshape[0])]],
                        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2                 # distance resolution in pixels of the Hough grid
    theta = np.pi / 180       # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 50
    min_line_length = 40    # minimum number of pixels making up a line
    max_line_gap = 25       # maximum gap in pixels between connectable line segments

    lines = hough_lines(masked_edges, rho, theta, threshold,
                        min_line_length, max_line_gap)

    weighted_image = weighted_img(lines, image)

    # we might need to save the pictures for finetuning parameters
    # save_image(image, gray, blur_gray, edges, masked_edges, lines,
    #           weighted_image)

    return weighted_image


def save_image(image, gray, blur_gray, edges, masked_edges, lines,
               weighted_image):
    """
    function saves the image after each transformation step for further 
    paramenter tuning for the Canny and hough space transforms 
    used in function process_image
    """

    global imageCounter

    plt.imsave("test_images_output/" +
               str(imageCounter) + "_0_original.png", image)
    plt.imsave("test_images_output/" + str(imageCounter) +
               "_1_gray.png", gray, cmap="gray")
    plt.imsave("test_images_output/" + str(imageCounter) +
               "_2_blur_gray.png", blur_gray, cmap="gray")
    plt.imsave("test_images_output/" +
               str(imageCounter) + "_3_canny.png", edges)
    plt.imsave("test_images_output/" + str(imageCounter) +
               "_4_region.png", masked_edges)
    plt.imsave("test_images_output/" +
               str(imageCounter) + "_5_hough.png", lines)
    plt.imsave("test_images_output/" + str(imageCounter) +
               "_6_weighed_image.png", weighted_image)
    imageCounter = imageCounter + 1


# used in function save_image to set the file name dynamically, e.g.,
# in order to avoid overwriting when videos are analyzed
imageCounter = 1

imageNames = os.listdir("test_images/")
for imageName in imageNames:
    image = mpimg.imread("test_images/" + imageName)
    weighted_image = process_image(image)
    plt.imsave("test_images_output/" + imageName, weighted_image, format="jpg")

videoNames = os.listdir("test_videos/")
for videoName in videoNames:
    # clip = VideoFileClip("test_videos/" + videoName).subclip(4.8, 6)
    clip = VideoFileClip("test_videos/" + videoName)
    white_clip = clip.fl_image(process_image)
    white_clip.write_videofile("test_videos_output/" + videoName, audio=False)
