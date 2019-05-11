import cv2
import numpy as np
import random

FILL_CIRCLE = -1

class Circle:
    def __init__(self, left=None, top=None, radius=None, color=None, alpha=None, image=None):
        """ It's a circle

        left(float): distance to the left normalized between [0.0, 1.0]
        top(float): distance to the top normalized between [0.0, 1.0]
        radius(float): Circle radius
        color(array-like(h,s,v)): Color to be used on render coded as HSV
        alpha(float): Opacity level bound to [0.0, 1.0]
        """
        
        # If the circle parameters were not passed, generate random circle
        if left is None:
            self.left = random.uniform(0,1)
            self.top = random.uniform(0,1)
            self.radius = random.uniform(0.01,1)
            if image is not None:
                h, w, depth = image.shape
                x = int(w * self.left)
                y = int(h * self.top)
                self.color = [int(c) for c in image[y, x]]
            else:
                self.color = [random.uniform(0,255),random.uniform(0,255),random.uniform(0,255)]
            self.alpha = 0.8 #random.uniform(0,1)
        else:
            self.left = left
            self.top = top
            self.radius = radius
            self.color = color
            self.alpha = alpha
    
    def render(self, image):
        """ Update image with this circle in it """

        h, w, depth = image.shape
        x = int(w * self.left)
        y = int(h * self.top)
        
        radius = int(self.radius * h / 4)
        color = self.color
        alpha = self.alpha
        
        # render square
        # Bound to image limit to avoid index errors
        # bottom = min(h, y+radius)
        # right = min(w, x+radius)
        # section = image[y:bottom, x:right]
        # image[y:bottom, x:right] = alpha*section + (1-alpha)*np.tile(color, (*section.shape[:2], 1))

        # Clone image to use as a canvas
        circle = np.copy(image)
        # Plot circle in this canvas
        cv2.circle(circle, center=(x, y), radius=radius, color=color, thickness=FILL_CIRCLE)
        # Combine the original image with the 'circled' one considering the alpha
        cv2.addWeighted(src1=circle, alpha=alpha, src2=image, beta=1-alpha, gamma=0, dst=image)

if __name__ == "__main__":
    image = cv2.imread('mona.jpg')
    Circle(left=None, top=None, radius=None, color=None, alpha=None).render(image)
