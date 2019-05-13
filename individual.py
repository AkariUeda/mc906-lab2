import numpy as np
import random
import inspect

from circle import Circle
from utils import rms, plot_image, save_image, ssim_rgb, create_weighted_hsv_ssim, render_circles
from skimage.measure import compare_ssim

def get_fitness_function(fitness_function):
    if fitness_function == 'RMS':
        return rms
    elif fitness_function == 'SSIM_HSV':
        return create_weighted_hsv_ssim()
    elif fitness_function == 'SSIM_RGB':
        return ssim_rgb
    else:
        raise ValueError('There is no fitness function named "{}"'.format(fitness_function))

class Individual:
    def __init__(
        self,
        image,
        ind_size,
        max_size=None,
        circles=[],
        good_genes=True,
        fitness_function=get_fitness_function('SSIM_RGB')):
        """ A individual from the population, which represents an image.

        image(ndarray): Reference image for computing the fitness with shape=(h, w, 3)
        ind_size(int): How many genes/circles compose this individual
        max_size(int): How many genes/circles this individual can reach through mutation
        circles(array-like of Circle): Optional circle array. If it got less 
                elements than ind_size, it will be filled with random circles
        good_genes(bool): Whether to use a heuristic to create new genes using 
        fitness_function(callable): A function expecting params (reference_image, rendered_image)
        """

        # Validate parameters
        assert isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3
        assert isinstance(ind_size, int)
        assert isinstance(max_size, int) or max_size is None
        assert all([isinstance(circle, Circle) for circle in circles])
        assert isinstance(good_genes, bool)
        assert Individual.is_valid_fitness_function(fitness_function) 

        self.image = image
        self.ind_size = ind_size
        self.max_size = max_size if max_size else ind_size
        self.good_genes = good_genes
        self.fitness_function = fitness_function
        self.circles = [c.copy() for c in circles]

        # Fill in missing circles
        for i in range(len(self.circles), ind_size):
            self.circles.append(Circle(image=image if good_genes else None))
        
        self.fitness = None
        self.rendered = None

    @staticmethod
    def is_valid_fitness_function(fitness_function):
        return callable(fitness_function) and len(inspect.signature(fitness_function).parameters) == 2

    @staticmethod
    def get_fitness_function(fitness_function):
        return get_fitness_function(fitness_function)

    def copy(self):
        ind = Individual(image=self.image, ind_size=self.ind_size, max_size=self.max_size, circles=[c.copy() for c in self.circles], good_genes=self.good_genes, fitness_function=self.fitness_function)
        ind.fitness = self.fitness
        ind.rendered = self.rendered
        return ind

    def __lt__(self, other):
        return (self.fitness < other.fitness)

    def __le__(self,other):
        return (self.fitness <= other.fitness)

    def __gt__(self,other):
        return (self.fitness > other.fitness)

    def __ge__(self,other):
        return (self.fitness >= other.fitness)

    def create_gene(self):
        """ Returns a new random circle. If `good_genes` is True, copy the center pixel color """
        return Circle(image=self.image if self.good_genes else None)

    def update_fitness(self):
        """ Update and returns `fitness` and the `rendered` attributes """
        image = self.image
        
        # Render image 
        self.rendered = render_circles(image.shape, self.circles)
        # Update fitness
        self.fitness =  self.fitness_function(image, self.rendered)
        
        return self.fitness, self.rendered

    def mutate(self, mutation_rate, expand_step=1, inner_mutation_rate=1):
        """ Replace some genes, proportional to the mutation rate, and, if
        possible, add more. The `good_genes` attribute will be considered if a
        new gene iscreated. If using `inner_mutation_rate < 0` all properties
        of the genewill proportionally be adjusted.
  
        Parameters
        ----------
        mutation_rate(float): How many genes should be mutated, bound to [0, 1]
        expand_step(int): How many genes should be added if there's still room (ind_size < max_size)
        inner_mutation_rate(float): How much each gene/circle's attribute should change, bound to (0,1]

        """

        assert 0 <= mutation_rate <= 1
        assert 0 < inner_mutation_rate <= 1

        # Sample randomly some genes and replace them
        affected_genes = int(self.ind_size * mutation_rate)
        for gene in random.sample(range(self.ind_size), affected_genes):
            self.circles[gene].mutate(inner_mutation_rate)
        
        # Expand as much as it's desired and possible
        for i in range(min(expand_step, self.max_size - self.ind_size)):
            self.circles.append(self.create_gene())
            self.ind_size += 1

        # Unset fitness and rendered until next evaluation
        self.fitness, self.rendered = None, None

    def plot_image(self, figax=None):
        if self.rendered is None:
            raise Exception('You must evaluate before plotting the solution')
        return plot_image(self.rendered, figax)

    def save_image(self, filename):
        if self.rendered is None:
            raise Exception('You must evaluate before plotting the solution')
        save_image(filename, self.rendered)

    def __str__(self):
        """ Represent the circles as letters """
        result = '.'.join([str(c) for c in self.circles])
        return result

if __name__ == "__main__":
    import cv2

    image = cv2.imread('mona.jpg')
    ind_size = 2
    ind = Individual(image=image, ind_size=ind_size, max_size=10, circles=[Circle()], 
                     good_genes=True, fitness_function=Individual.get_fitness_function('SSIM_HSV'))
    assert len(ind.circles) == ind_size

    fitness, img = ind.update_fitness()
    assert fitness == ind.fitness
    assert np.array_equal(img, ind.rendered)
    print('Should see two strings:', ind)
    save_image('mona-1.png', img)
    
    ind.mutate(mutation_rate=0.5, expand_step=1, inner_mutation_rate=0.2)
    ind.update_fitness()
    print('Should see three strings, one unchanged:', ind)
    save_image('mona-2.png', ind.rendered)
