import random

from individual import Individual

class Evolve:
    def __init__(self, image, pop_size=10, crossover_rate=0.5, individual_size=250):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.original_image = image
        self.pop = []
        self.individual_size = individual_size
        #Gera uma populacao aleatoria de tamanho pop_size
        for i in range(0,pop_size):
            self.pop.append(Individual(individual_size))
            
    def compare_fitness(a, b):
        if a.fitness(self.original_image) > b.fitness(self.original_image):
            return 1
        return -1
            
    def crossover(self):
        # Sort parents by fitness
        parents = sorted(pop, key=compare_fitness)
        
        # New generation
        self.pop = []

        # Generates pop_size individuals from crossover
        for i in range(self.pop_size):
            # Get two parents i the top "crossover_rate" individuals
            parent1 = random.randint(0,self.crossover_rate*self.pop_size)
            parent2 = random.randint(0,self.crossover_rate*self.pop_size)

            # choose from which parent we are going to pick each gene
            gene = [random.randint(0,2) for j in range(self.individual_size)]
            for j in range(0, self.individual_size):
                if gene[j] == 0:
                    self.pop.append(parent1[j])
                else:
                     self.pop.append(parent2[j])
            
            

    
    
    
    