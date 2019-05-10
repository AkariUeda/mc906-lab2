import numpy as np

from circle import Circle
from image_from_circles import ImageFromCircles
from utils import rms
import matplotlib.pyplot as plt

class Individual:
    def __init__(self, size, circles=[]):
        """ A individual from the population, which represents an image.

        image(array-like(shape=(h, w, 3))): Reference image for computing the fitness
        circles(array-like of Circle): The second vertex of the triangle.
        """

        # Assert param types
        assert isinstance(size, int)
        assert all([isinstance(circle, Circle) for circle in circles])

        self.circles = []
        self.fitness = 0

        if circles:
            if len(circles) < size:
                print("Missing {} circles. Created them randomly".format(size - len(circles)))
            self.circles = circles

        for i in range(len(self.circles), size):
            self.circles.append(Circle())

    def fitness(self, image):
        # Assert image is an array-like with shape=(h, w, 3)
        assert isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3
        
        self.rendered = ImageFromCircles(circles=self.circles).render(image.shape)
        self.fitness = -rms(image, self.rendered)
        return self.fitness

    def plot_image(self):
        # Prepare canvas
        fig, ax = plt.subplots(figsize=(6, 6))

        # Disable plot ticks and their labels
        ax.tick_params(axis='both', which='both', 
                    bottom=False, left=False,
                    labelbottom=False, labelleft=False)
        ax.imshow(self.rendered)
        plt.show()

    def save_img(self, filename):
        # Prepare canvas
        fig, ax = plt.subplots(figsize=(6, 6))

        # Disable plot ticks and their labels
        ax.tick_params(axis='both', which='both', 
                    bottom=False, left=False,
                    labelbottom=False, labelleft=False)
        ax.imshow(self.rendered)
        fig.savefig(filename)

if __name__ == "__main__":
    Individual(size=1, circles=[Circle(left=None, top=None, radius=None, color=None, alpha=None)])