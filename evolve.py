import random
import numpy as np
from individual import Individual

class Evolve:
    def __init__(self, image, pop_size=10, crossover_rate=0.5, individual_size=250, objective=0.3):
        self.original_image = image
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.individual_size = individual_size
        self.objective = objective
        self.generation = 0
        #Gera uma populacao aleatoria de tamanho pop_size
        self.pop = []
        for i in range(0,pop_size):
            self.pop.append(Individual(individual_size))
    
    def compare_fitness(self,a):
        return a.fitness


    def crossover(self):

        # New generation
        self.generation += 1
        pop = []

        # Generates pop_size individuals from crossover
        for i in range(self.pop_size):
            # Get two parents i the top "crossover_rate" individuals
            parent1 = random.randint(0,self.crossover_rate*self.pop_size)
            parent2 = random.randint(0,self.crossover_rate*self.pop_size)

            
            # choose from which parent we are going to pick each gene
            ind_circles = [random.randint(0,2) for j in range(self.individual_size)]
            circles = []
            for j in range(self.individual_size):
                if ind_circles[j] == 0:
                    circles.append(self.pop[parent1].circles[j])
                else:
                    circles.append(self.pop[parent2].circles[j])
            
            pop.append(Individual(self.individual_size, circles=circles))
        self.pop = pop

    def mutation(self):
        # pick 10% of the population individuals to mutate
        mi = [random.randint(0,self.pop_size) for i in range(int(0.1*self.pop_size))]
        
        # substitute these 10% with random individuals
        for idx in mi:
            self.pop[idx] = Individual(self.individual_size)
            
    def evaluate(self):
        # calculate the fitness for the entire population
        for ind in self.pop:
            ind.fitness = Individual.fitness(ind, self.original_image)

        #sort population by fitness
        self.pop = sorted(self.pop, key=self.compare_fitness)


    def save_img(self):
        filename = 'generation_' + str(self.generation) + '.png'
        print(filename)
        self.pop[0].save_img(filename)


    
    
    
    