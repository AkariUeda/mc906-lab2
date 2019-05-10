import cv2
import numpy as np
import random

from circle import Circle

class ImageFromCircles:
    def __init__(self, circles=None):
        self.circles = circles

    def render(self, image_shape):
        # Create brand new image
        image = np.full(shape=image_shape, fill_value=255, dtype=np.uint8)

        # Add circles to it
        for circle in self.circles:
            circle.render(image)
            
        return image