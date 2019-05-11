import numpy as np
import random
import inspect

from circle import Circle
from image_from_circles import ImageFromCircles
from utils import rms, plot_image, save_image
from skimage.measure import compare_ssim

class Individual:
    def __init__(self, size, circles=[], image=None, fitness_function='SSIM_RGB', max_size=300):
        """ A individual from the population, which represents an image.

        image(array-like(shape=(h, w, 3))): Reference image for computing the fitness
        circles(array-like of Circle): The second vertex of the triangle.
        """

        # Assert param types
        assert isinstance(size, int)
        assert all([isinstance(circle, Circle) for circle in circles])

        self.individual_size = size
        self.max_ind_size = max_size
        self.circles = []
        self.fitness = 0
        self.fitness_function = fitness_function

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

        # If self.fitness_function is an actual function
        if callable(self.fitness_function) and len(inspect.signature(self.fitness_function).parameters) == 2:
            self.fitness =  self.fitness_function(image, self.rendered)
        elif self.fitness_function == 'RMS':
            self.fitness = rms(image, self.rendered)
        else: # Use 'SSIM_RGB' by default
            self.fitness =  -compare_ssim(image, self.rendered, multichannel=True)
        
        return self.fitness

    def mutate(self, mutation_rate, image):
        affected_genes = int(self.individual_size*mutation_rate)
        for gene in random.sample(range(self.individual_size), affected_genes):
            self.circles[gene] = Circle(image=image)
        
        # If we are starting with 1 circle and increasing
        # the individual size:
        if self.individual_size < self.max_ind_size:
            self.circles.append(Circle(image=image))
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
