import random
import numpy as np
from individual import Individual


class Evolve:
    def __init__(self, image, pop_size=10, crossover_rate=0.5, max_ind_size=300, objective=0.3, mutation_rate=0.2, unmutable_ratio=0, initial_pop=[], fitness_function='SSIM_RGB'):
        # Save params
        self.original_image = image
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.unmutable_ratio = unmutable_ratio
        self.max_ind_size = max_ind_size
        self.fitness_function = fitness_function
        self.objective = objective  # Not used
        

        # Creates a random initial population with `pop_size` members
        self.pop = []
        # for ind in initial_pop:
        #     self.pop.append(Individual(0, image=image, circles=ind.circles, fitness_function=self.fitness_function))

        for i in range(len(initial_pop), pop_size):
            self.pop.append(Individual(max_ind_size, image=image, fitness_function=self.fitness_function, max_size=max_ind_size))

        # Initialize counter
        self.generation = 0

        self.solution = self.pop[0]

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
            child_size = min(self.pop[parents[0]].individual_size, self.pop[parents[1]].individual_size)
            ind_circles = [random.randint(0,1) for j in range(child_size)]
            circles = []
            for j in range(child_size):
                indx = ind_circles[j]
                parent = self.pop[parents[indx]]
                circles.append(parent.circles[j])
            
            pop.append(Individual(child_size, circles=circles, fitness_function=self.fitness_function, max_size=self.max_ind_size))

        # Update children fitness
        for ind in pop:
                ind.update_fitness(self.original_image)
        else:
            # sort population by fitness
            pop.sort(key=lambda ind: ind.fitness)
        children = pop

        # parents = self.pop
        # self.pop = []
        # i, j = 0, 0
        # # Merge the best between parents and children
        # while i + j < self.pop_size:
        #     # Less is better
        #     if pop[i] < parents[j]:
        #         self.pop.append(pop[i])
        #         i += 1
        #     else:
        #         self.pop.append(parents[j])
        #         j += 1
        # print('Kept {} parents'.format(j))
        self.pop = children
        if self.pop[0].fitness < self.solution.fitness:
            self.solution = self.pop[0]
        return parents, children

    def __str__(self):
        result = ''
        for ind in self.pop:
            fit = str(ind.fitness*100)
            result += str(ind) + '{}...{}\n'.format(fit[:3], fit[-2:])
        return result

    def mutate(self):
        keep = int(self.unmutable_ratio*self.pop_size)
        print(keep)
        # print('keep', keep, 'unmutated')
        for ind in self.pop[keep:]:
            ind.mutate(self.mutation_rate, self.original_image)
            ind.update_fitness(self.original_image)
        self.pop.sort(key=lambda ind: ind.fitness)

    def evaluate(self):
        if self.generation == 0:
            # calculate the fitness for the entire population
            for ind in self.pop:
                ind.update_fitness(self.original_image)
        else:
            # sort population by fitness
            self.pop.sort(key=lambda ind: ind.fitness)

    def plot_image(self, figax=None):
        return self.solution.plot_image(figax)

    def save_image(self):
        filename = './results/generation_{}.png'.format(self.generation)
        print(filename)
        self.pop[0].save_image(filename)


    
    
    
    