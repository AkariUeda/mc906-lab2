import cv2
import numpy as np
import random

from circle import Circle

class ImageFromCircles:
    def __init__(self, circles=None):
        self.circles = circles

    # TODO consider stop using OpenCV for this conversion
    def hsv_to_rgb(self, color):
        """ Returns a (r,g,b) converted from the HSV color """
        src = np.array([[color]], dtype=np.uint8)
        color = list(map(int, cv2.cvtColor(src, cv2.COLOR_HSV2RGB)[0][0]))
        return color

    def render(self, image_shape):
        # Create brand new image
        image = np.full(shape=image_shape, fill_value=255, dtype=np.uint8)

        # Add circles to it
        for circle in self.circles:
            circle.render(image)
            
        return image