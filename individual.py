import numpy as np
import random
import cv2
from circle import Circle
from image_from_circles import ImageFromCircles
from utils import rms, plot_image, save_image, dif_imgs
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

        self.individual_size = size
        self.circles = []
        self.fitness = 0

        if circles:
            self.circles = circles
            # if len(circles) < size:
            #     print("Missing {} circles. Created them randomly".format(size - len(circles)))

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
        # self.fitness = -compare_ssim(image, self.rendered, multichannel=True)
        self.fitness = dif_imgs(image, self.rendered)
        return self.fitness

    def mutate(self, mutation_rate, image):
        affected_genes = int(self.individual_size*mutation_rate)
        for gene in random.sample(range(self.individual_size), affected_genes):
            self.circles[gene] = Circle(image)
        if len(self.circles) < 300:
            self.circles.append(Circle())
            self.individual_size += 1

    def plot_image(self, figax=None):
        return plot_image(self.rendered, figax)

    def save_image(self, filename):
        save_image(filename, self.rendered)

    def __str__(self):
        to_char = lambda x: chr(65 + int(x % 26))
        from_float = lambda x: to_char(x*26)

        result = ''
        for c in self.circles:
            result += to_char(c.radius) + \
                    from_float(c.left) + from_float(c.top) + \
                    ''.join(to_char(color) for color in c.color) + '.'
        return result

if __name__ == "__main__":
    image = cv2.imread('mona.jpg')
    ind = Individual(size=1, circles=[Circle()])
    ind.update_fitness(image)
    ind.mutate(0.2)
