import random
import numpy as np
from individual import Individual

class Evolve:
    def __init__(self, image, pop_size=10, crossover_rate=0.5, individual_size=250, objective=0.3, mutation_rate=0.2):
        # Save params
        self.original_image = image
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.individual_size = individual_size
        self.objective = objective  # Not used

        # Creates a random initial population with `pop_size` members
        self.pop = [Individual(individual_size, image=image) for i in range(pop_size)]

        # Initialize counter
        self.generation = 0

    def crossover(self):
        # New generation
        self.generation += 1
        pop = []

        # Generates pop_size individuals from crossover
        for i in range(self.pop_size):
            # Get two parents i the top "crossover_rate" individuals
            parents = [random.randint(0,int(self.crossover_rate*self.pop_size)),
                        random.randint(0,int(self.crossover_rate*self.pop_size))]

            # choose from which parent we are going to pick each gene
            ind_circles = [random.randint(0,1) for j in range(self.individual_size)]
            circles = []
            for j in range(self.individual_size):
                indx = ind_circles[j]
                parent = self.pop[parents[indx]]
                circles.append(parent.circles[j])
            
            pop.append(Individual(self.individual_size, circles=circles))

        pop.sort(key=lambda ind: ind.fitness)

        self.evaluate(pop)
        parents = self.pop
        self.pop = []
        i, j = 0, 0
        # Merge the best between parents and children
        while i + j < self.pop_size:
            # Less is better
            if pop[i] < parents[j]:
                self.pop.append(pop[i])
                i += 1
            else:
                self.pop.append(parents[j])
                j += 1

    def mutate(self):
        for ind in self.pop:
            ind.update_fitness(self.original_image)

        # sort population by fitness
        self.pop.sort(key=lambda ind: ind.fitness)

        # replace the population fraction with new random individuals
        for idx in range(self.pop_size-1,int(self.pop_size*self.mutation_rate), -1):
            self.pop[idx] = Individual(self.individual_size)

    def evaluate(self, pop=None):
        population = pop or self.pop
        # calculate the fitness for the entire population
        for ind in population:
            ind.update_fitness(self.original_image)

        # sort population by fitness
        population.sort(key=lambda ind: ind.fitness)

        if pop is None:
            print('Gen:', self.generation, 'Fitness:', self.pop[0].fitness)


    def plot_image(self):
        self.pop[0].plot_image()

    def save_image(self):
        filename = './results/generation_{}.png'.format(self.generation)
        print(filename)
        self.pop[0].save_image(filename)


    
    
    
    