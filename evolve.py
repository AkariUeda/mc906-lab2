import random
import numpy as np
from individual import Individual

def evaluate(population, image):
    # calculate the fitness for the entire population
    for ind in population:
        ind.update_fitness(image)

    # sort population by fitness
    population.sort(key=lambda ind: ind.fitness)

class Evolve:
    def __init__(self, image, pop_size=10, crossover_rate=0.5, individual_size=250, objective=0.3, mutation_rate=0.2, unmutable_ratio=0, initial_pop=[]):
        # Save params
        self.original_image = image
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.unmutable_ratio = unmutable_ratio
        self.objective = objective  # Not used

        # Creates a random initial population with `pop_size` members
        self.pop = []
        for ind in initial_pop:
            self.pop.append(Individual(1, image=image, circles=ind.circles))

        for i in range(len(initial_pop), pop_size):
            self.pop.append(Individual(1, image=image))

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
            child_size = min(self.pop[parents[0]].individual_size, self.pop[parents[1]].individual_size)
            ind_circles = [random.randint(0,1) for j in range(child_size)]
            circles = []
            for j in range(child_size):
                indx = ind_circles[j]
                parent = self.pop[parents[indx]]
                circles.append(parent.circles[j])
            
            # circles = []
            # circles.extend(self.pop[parents[0]].circles[:self.pop[parents[0]].individual_size//2])
            # circles.extend(self.pop[parents[1]].circles[:self.pop[parents[1]].individual_size//2])

            pop.append(Individual(len(circles), circles=circles))

        # Update children fitness
        evaluate(pop, self.original_image)
        children = pop

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
        # print('Kept {} parents'.format(j))
        return parents, children

    def __str__(self):
        result = ''
        for ind in self.pop:
            fit = str(ind.fitness*100)
            result += str(ind) + '{}...{}\n'.format(fit[:3], fit[-2:])
        return result

    def mutate(self):
        evaluate(self.pop, self.original_image)
        keep = int(self.unmutable_ratio*self.pop_size)
        # print('keep', keep, 'unmutated')
        print(keep)
        for ind in self.pop[keep:]:
            ind.mutate(self.mutation_rate, self.original_image)

    def evaluate(self):
        evaluate(self.pop, self.original_image)

    def plot_image(self, figax=None):
        return self.pop[0].plot_image(figax)

    def save_image(self):
        filename = './results/generation_{}.png'.format(self.generation)
        print(filename)
        self.pop[0].save_image(filename)


    
    
    
    