import cv2
import numpy as np
import random

from utils import int_to_char, float_to_char

# OpenCV's constant used at `cv2.circle`
FILL_CIRCLE = -1

# If you want to allow random alpha values, use None
FIXED_ALPHA = 0.8


class Circle:
    def __init__(self, left=None, top=None, radius=None, color=None, alpha=FIXED_ALPHA, image=None):
        """ It's a circle

        left(float): distance to the left normalized between [0.0, 1.0]
        top(float): distance to the top normalized between [0.0, 1.0]
        radius(float): Circle radius
        color(array-like(h,s,v)): Color to be used on render coded as HSV
        alpha(float): Opacity level bound to [0.0, 1.0]
        """
        
        # If the circle parameters were not passed, generate random circle
        self.left = random.uniform(0,1) if left is None else left
        self.top = random.uniform(0,1) if top is None else top
        self.radius = random.uniform(0.01,0.2) if radius is None else radius
        self.alpha = random.uniform(0,1) if alpha is None else alpha
        self.used_heuristic = False

        if color is None:
            # If the image is set, use the color at the circle center
            if image is not None:
                h, w, depth = image.shape
                x = int(w * self.left)
                y = int(h * self.top)
                self.color = image[y, x]
                self.used_heuristic = True
            else:
                self.color = np.array([random.randint(0,255),random.randint(0,255),random.randint(0,255)], dtype=np.uint8)
        else:
            self.color = color
    
    def copy(self):
        return Circle(self.left, self.top, self.radius, self.color, self.alpha)

    def mutate(self, mutation_rate=0.2):
        """ Replace all attributes, using the weighted sum between this circle
        and a new random one (this*(1-mutation_rate) + random*mutation_rate)  """
        def add_weighted(a, b, w):
             return (1 - w) * a + w * b

        # Create a new random circle
        new = Circle()

        self.left = add_weighted(self.left, new.left, mutation_rate)
        self.top = add_weighted(self.top, new.top, mutation_rate)
        self.radius = add_weighted(self.radius, new.radius, mutation_rate)
        self.alpha = add_weighted(self.alpha, new.alpha, mutation_rate)
        self.color = np.array(add_weighted(self.color, new.color, mutation_rate), dtype=np.uint8)

    def renderAtImageAsSquare(self, image):
        """ Update image with a square using this circle attributes """

        # Validate params
        assert isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3 and image.dtype == np.uint8

        color = self.color
        alpha = self.alpha

        h, w, depth = image.shape
        
        radius = int(self.radius * h)
        x = max(0, int(w * self.left - radius))
        y = max(0, int(h * self.top - radius))

        # Bound dimensions to image limit to avoid index errors
        bottom, right = min(h, y+radius), min(w, x+radius)
        section = image[y:bottom, x:right]
        image[y:bottom, x:right] = alpha*section + (1-alpha)*np.tile(color, (*section.shape[:2], 1))

    def renderAtImage(self, image):
        """ Update image with this circle in it """

        # Validate params
        assert isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3 and image.dtype == np.uint8

        color = self.color
        alpha = self.alpha

        h, w, depth = image.shape

        radius = int(self.radius * h)
        x = int(w * self.left)
        y = int(h * self.top)

        # Clone image to use as a canvas
        circle = np.copy(image)
        # Plot circle in this canvas
        cv2.circle(circle, center=(x, y), radius=radius, color=color.tolist(), thickness=FILL_CIRCLE)
        # Combine the original image with the 'circled' one considering the alpha
        cv2.addWeighted(src1=circle, alpha=alpha, src2=image, beta=1-alpha, gamma=0, dst=image)

    def __str__(self):
        used_heuristic = 'H!' if self.used_heuristic else ''

        return used_heuristic + \
            float_to_char(self.radius) + \
            float_to_char(self.left) + \
            float_to_char(self.top) + \
            ''.join(int_to_char(color) for color in self.color)

    def to_dict(self):
        return {
            'radius': self.radius,
            'left': self.left,
            'top': self.top,
            'color': self.color.tolist(),
            'alpha': self.alpha
        }

    def __repr__(self):
        used_heuristic = '(copied)' if self.used_heuristic else ''

        return 'Left: {:.5f} Top: {:.5f} Radius: {:.5f} Alpha: {:.2f} Color{}: {}'.format(
            self.left, self.top, self.radius, self.alpha, used_heuristic, str(self.color))

if __name__ == "__main__":
    from utils import save_image

    image = np.full(shape=(100, 100, 3), fill_value=255, dtype=np.uint8)
    color = np.array([255, 0, 0], dtype=np.uint8)
    circle = Circle(left=0.5, top=0.5, radius=0.3, color=color, alpha=FIXED_ALPHA)
    circle.renderAtImage(image)
    circle.mutate(0.2)
    circle.renderAtImage(image)
    save_image('/tmp/image.png', image)