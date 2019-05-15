import cv2
import numpy as np
import matplotlib
import random
import sys

if sys.stdin.isatty():
    matplotlib.use('agg') 

from matplotlib import pyplot as plt
from skimage.measure import compare_ssim

def int_to_char(x):
    return chr(65 + int(x % 26))


def float_to_char(x):
    return int_to_char(x*26)


def punish_white(a, b, max_diff=255):
    h,w,d = a.shape
    max_diff_color = np.full((3,), fill_value=max_diff, dtype=np.uint8)
    diff = abs(a-b)
    diff[np.where(np.logical_and(np.sum(a, axis=2) != 255*3, np.sum(b, axis=2) == 255*3))] = max_diff_color

    return sum(diff.flatten())/(max_diff*h*w*d)-1


def rms(a, b):
    """ Returns the Root Mean Square between two RGB images """
    pixel_count = a.shape[0] * a.shape[1]
    diffs = np.abs((a - b).flatten())
    values, idxs = np.histogram(diffs, bins=range(257))

    sum_of_squares = sum(value*(idx**2) for idx, value in zip(idxs, values))
    return np.sqrt(sum_of_squares / pixel_count)


def ssim_rgb(a, b):
    """ Returns the negative ssim between two rgb images """
    return -compare_ssim(a, b, multichannel=True)


def rgb_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def split_hsv(img_hsv):
    """ Returns a tuple with the components of an HSV image """
    h = img_hsv[:,:, 0]
    s = img_hsv[:,:, 1]
    v = img_hsv[:,:, 2]
    return h, s, v


def square_diff(a, b): 
    return sum((a-b).flatten()**2)**(1/2)


def normalized_pixel_diff(a, b, max_diff=255):
    h = a.shape[0]
    w = a.shape[1]
    d = 1
    if len(a.shape) == 3:
        d = a.shape[2]
    # normalized difference
    return sum(abs((a-b).flatten()))/(max_diff*h*w*d)-1


def create_weighted_hsv_ssim(weights=(0.15, 0.05, 0.8)):
    """ Returns a function for comparing RGB images, A and B. The idea is to 
    convert the images to HSV and do a weighted sum over grayscale SSIM, hue
    and saturation normalized difference. The result will be bound to [-1, 0]
    """
    assert sum(weights) > 0.99
    h, s, v = weights
    def weighted_hsv_ssim(a, b):
        a_hsv = split_hsv(rgb_to_hsv(a))
        b_hsv = split_hsv(rgb_to_hsv(b))
        return h * normalized_pixel_diff(a_hsv[0], b_hsv[0]) + \
               s * normalized_pixel_diff(a_hsv[1], b_hsv[1]) + \
               v * -compare_ssim(a_hsv[2], b_hsv[2])
    return weighted_hsv_ssim


def render_circles(image_shape, circles):
    """ Returns an image with all circles rendered in it
    Parameters
    ----------
    image_shape(array-like): The image dimensions, including the depth, which must be 3
    circles(array-like of Circle): A list of circle instances to be rendered

    Return
    ------
    (ndarray): 3D numpy array with white background overlayed by the rendered circles
    """
    from circle import Circle

    # Validate params
    assert len(image_shape) == 3 and image_shape[2] == 3
    assert all([isinstance(circle, Circle) for circle in circles])

    # Create a white image
    image = np.full(shape=image_shape, fill_value=255, dtype=np.uint8)

    # Add circles to it
    for circle in circles:
        circle.renderAtImage(image)
        
    return image


def prepare_canvas(image, figax=None):
    """ Plot image and returns the matplotlib's (Figure, AxesSubplot) pair 
    Parameters
    ----------
    image(array-like): Image to be passed to imshow
    figax(array-like of (Figure, AxesSubplot)): Optional canvas, if None a new one is created

    Return
    ------
    (array-like of (Figure, AxesSubplot)): Canvas elements in which the image got plotted
    """
    # Prepare canvas
    fig, ax = figax if figax is not None else plt.subplots(figsize=(6, 6))

    # Disable plot ticks and their labels
    ax.tick_params(axis='both', which='both', 
                bottom=False, left=False,
                labelbottom=False, labelleft=False)
    ax.imshow(image)
    fig.canvas.draw()
    return fig, ax

def plot_image(image, figax=None):
    """ Plot and show image """
    fig, ax = prepare_canvas(image, figax)
    plt.show()
    return fig, ax

def save_image(filename, image):
    """ Plot and save image """
    fig, ax = prepare_canvas(image)
    fig.savefig(filename)
