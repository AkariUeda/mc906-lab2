import json
import random
import numpy as np

from os import path 
from individual import Individual


def evaluate(population):
    # calculate the fitness for the entire population
    for ind in population:
        ind.update_fitness()

    # sort population by fitness
    population.sort(key=lambda ind: ind.fitness)


class Evolve:
    def __init__(
        self,
        image,
        pop_size=10,
        initial_pop=[],
        ind_size=100,
        ind_size_max=100,
        good_genes=True,
        fitness_function=Individual.get_fitness_function('SSIM_RGB'),
        crossover_rate=0.5,
        newgen_parent_ratio=0.0,
        children_ratio=1,
        mutation_rate=0.2,
        inner_mutation_rate=1,
        unmutable_ratio=0,
        verbose=False):
        """ Try to replicate an image using circles

        image(ndarray): A 3D array representing a RGB image with shape(h, w, 3)
        pop_size(int): Number of individuals
        initial_pop(array-like of Individual): Initial individual array
        ind_size(int): Individual size
        ind_size_max(int): Allow individuals to grow during the mutation phase
        good_genes(bool): Whether to use a heuristic to create new genes using 
        fitness_function(string or callable): One of ['SSIM_RGB', 'RMS'] or a
                callable expecting params (reference_image, rendered_image)
        crossover_rate(float): How many individuals from the current 
                population should be used in crossover bound to [0, 1] 
        newgen_parent_ratio(float): How many individuals in the new generation 
                are allowed to be from the current one, bound to [0,1]
        children_ratio(float): How many individuals are created during 
                crossover proportional to the pop_size (must be >=1)
        mutation_rate(float): How many genes (circles) are renewed for each 
                individual during mutation, bound to [0,1]
        inner_mutation_rate(float): How much each gene's (circle) attribute 
                should change, bound to (0,1]
        unmutable_ratio(float): How many of the best individuals are not 
                mutated, bound to [0,1]
        verbose(bool): Whether to print progress messages
        """

        # Validate params
        assert isinstance(pop_size, int)
        assert all(isinstance(ind, Individual) for ind in initial_pop)
        assert isinstance(ind_size, int)
        assert isinstance(ind_size_max, int) and ind_size <= ind_size_max
        assert isinstance(good_genes, bool)
        assert Individual.is_valid_fitness_function(fitness_function), 'Invalid fitness function (%r)' % fitness_function
        assert 0 == crossover_rate or (0 < crossover_rate <= 1 and int(crossover_rate * pop_size) > 1 and ind_size >= 2)
        assert 0 <= newgen_parent_ratio <= 1  # Arguably, the cannonical way is 0 or 1
        assert 1 <= children_ratio
        assert 0 <= mutation_rate <= 1
        assert 0 <= inner_mutation_rate <= 1
        assert 0 <= unmutable_ratio <= 1

        # Objetive images
        self.original_image = image

        # Evolution manager properties
        self.pop_size = pop_size
        self.verbose = verbose

        # Individual properties
        self.ind_size = ind_size
        self.ind_size_max = ind_size_max
        self.good_genes = good_genes
        self.fitness_function = fitness_function

        # Crossover properties
        self.crossover_rate = crossover_rate
        self.newgen_parent_ratio = newgen_parent_ratio
        self.children_ratio = children_ratio

        # Mutation properties
        self.mutation_rate = mutation_rate
        self.inner_mutation_rate = inner_mutation_rate
        self.unmutable_ratio = unmutable_ratio
        
        # Create the population
        self.pop = []

        # Start by creating individuals with the initial population, if any were set
        for ind in initial_pop:
            self.pop.append(self.create_individual(circles=ind.circles))

        # Creates the rest of the population with random individuals
        for i in range(len(initial_pop), pop_size):
            self.pop.append(self.create_individual())

        self.solution = None

        # Initialize counter
        self.generation = 0

    def create_individual(self, circles=[]):
        """ Returns an individual using the configured attributes """
        return Individual(
                    image=self.original_image,
                    ind_size=self.ind_size, 
                    max_size=self.ind_size_max, 
                    fitness_function=self.fitness_function, 
                    good_genes=self.good_genes,
                    circles=circles
                )

    def evaluate(self):
        """ Update individuals' fitness and sort them """
        if self.verbose: print('Running evaluation')

        evaluate(self.pop)
        if self.solution is None or self.pop[0].fitness < self.solution.fitness:
            self.solution = self.pop[0].copy()

    def crossover(self):
        """ Update population combining individuals genes and resort them """
        if self.verbose: print('Running crossover')

        # New generation
        self.generation += 1
        children = []

        cross_parents = int(self.crossover_rate * self.pop_size)

        # If crossover_rate is 0, skip crossover 
        if self.crossover_rate == 0:
            return
    
        # Generates new individuals from crossover
        children_count = int(self.children_ratio * self.pop_size)
        for i in range(children_count):
            # Get two parents in the crossover_rate range
            parents = random.sample(self.pop[:cross_parents], 2)

            # Get the smallest parent size
            child_size = min(parents[0].ind_size, parents[1].ind_size)

            # Pick at least one gene (circle) from a different parents
            from_which = [0] * child_size
            idx = random.sample(range(child_size), random.randint(1, child_size-1))
            for j in idx:
                from_which[j] = 1
    
            circles = [parents[from_which[j]].circles[j] for j in range(child_size)]            

            # Append child to children population
            children.append(self.create_individual(circles=circles))

        # Update children fitness and sort them
        evaluate(children)

        parents = self.pop
        self.pop = []
        i, j = 0, 0
        max_parent_count = int(self.newgen_parent_ratio * self.pop_size)

        # Merge the best between parents and children respecting the configured propagable parent ratio
        while j < max_parent_count and i + j < self.pop_size:
            # Less is better
            if children[i] <= parents[j]:
                self.pop.append(children[i])
                i += 1
            else:
                self.pop.append(parents[j])
                j += 1

        # Fill the rest with children only
        while i + j < self.pop_size:
            self.pop.append(children[i])
            i += 1

        if self.pop[0].fitness < self.solution.fitness:
            self.solution = self.pop[0].copy()
        return parents, children

    def mutate(self, expand_step=1):
        """ Update some individuals fully replacing some genes or just adjust their attributes """
        if self.verbose: print('Running mutation')

        keep = int(self.unmutable_ratio*self.pop_size)

        for ind in self.pop[keep:]:
            ind.mutate(self.mutation_rate, expand_step=expand_step, inner_mutation_rate=self.inner_mutation_rate)

    def plot_image(self, figax=None):
        if self.solution is None:
            raise Exception('You must evaluate before plotting the solution')
        return self.solution.plot_image(figax)

    def save_image(self, filepath):
        if self.solution is None:
            raise Exception('You must evaluate before plotting the solution')
        filename = path.join(filepath, 'generation_{}.png'.format(self.generation))
        self.solution.save_image(filename)
        if self.verbose: print('{} saved'.format(filename))

    def save_json(self, filepath):
        filename = path.join(filepath, 'generation_{}.json'.format(self.generation))
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f)
        if self.verbose: print('{} saved'.format(filename))

    def __str__(self):
        result = ''
        for ind in self.pop:
            fit = str(ind.fitness*100)
            result += str(ind) + '{}...{}\n'.format(fit[:3], fit[-2:])
        return result

    def to_dict(self):
        return {
        'pop_size': self.pop_size,
        'ind_size': self.ind_size,
        'ind_size_max': self.ind_size_max,
        'good_genes': self.good_genes,
        'crossover_rate': self.crossover_rate,
        'newgen_parent_ratio': self.newgen_parent_ratio,
        'children_ratio': self.children_ratio,
        'mutation_rate': self.mutation_rate,
        'inner_mutation_rate': self.inner_mutation_rate,
        'unmutable_ratio': self.unmutable_ratio,
        'pop': [i.to_dict() for i in self.pop],
        'generation': self.generation
        }

    
if __name__ == "__main__":
    import cv2

    image = cv2.imread('mona.jpg')
    print('Creating Evolve instance...')
    evolve = Evolve(image,
                    pop_size=10,
                    initial_pop=[],
                    ind_size=2,
                    ind_size_max=3,
                    good_genes=True,
                    fitness_function=Individual.get_fitness_function('SSIM_RGB'),
                    crossover_rate=0.5,
                    newgen_parent_ratio=0.0,
                    children_ratio=1,
                    mutation_rate=0.2,
                    inner_mutation_rate=1,
                    unmutable_ratio=0)
    print('Running evaluation...')
    evolve.evaluate()
    print(evolve)
    print('Running crossover...')
    evolve.crossover()
    print(evolve)
    print('Running mutation...')
    evolve.mutate()
    evolve.evaluate()
    print(evolve)
    evolve.save_json('/tmp')


    