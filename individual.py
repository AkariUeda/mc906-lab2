import numpy as np

from circle import Circle
from image_from_circles import ImageFromCircles
from utils import rms, plot_image, save_image
from skimage.measure import compare_ssim

class Individual:
    def __init__(self, size, circles=[], image=None):
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
            self.circles = circles
            if len(circles) < size:
                print("Missing {} circles. Created them randomly".format(size - len(circles)))

        # Fill in missing circles
        for i in range(len(self.circles), size):
            self.circles.append(Circle(image=image))

    def __lt__(self, other):
        return (self.fitness < other.fitness)

    def __le__(self,other):
        return (self.fitness <= other.fitness)

    def __gt__(self,other):
        return (self.fitness > other.fitness)

    def __ge__(self,other):
        return (self.fitness >= other.fitness)

    def update_fitness(self, image):
        # Assert image is an array-like with shape=(h, w, 3)
        assert isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3
        
        self.rendered = ImageFromCircles(circles=self.circles).render(image.shape)
        # self.fitness = rms(image, self.rendered)
        self.fitness = -compare_ssim(image, self.rendered, multichannel=True)
        return self.fitness

    def plot_image(self):
        plot_image(self.rendered)

    def save_image(self, filename):
        save_image(filename, self.rendered)

if __name__ == "__main__":
    image = cv2.imread('mona.jpg')
    Individual(size=1, circles=[Circle()]).update_fitness(image)