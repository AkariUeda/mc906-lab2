import random

from individual import Individual

class Evolve:
    def __init__(self, image, pop_size=10, crossover_rate=0.5, individual_size=250):
        self.original_image = image
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.individual_size = individual_size
        #Gera uma populacao aleatoria de tamanho pop_size
        self.pop = []
        for i in range(0,pop_size):
            self.pop.append(Individual(individual_size))
    
    def compare_fitness(self,a):
        print(a)
        return a.fitness
    
    def evaluate(self):
        # calculate the fitness for the entire population
        for ind in self.pop:
            ind.fitness = ind.fitness(self.original_image)
        
        
        #sort population by fitness
        self.pop = sorted(pop, key=self.compare_fitness)
        plot_image(self.pop[0].rendered)
        
    def crossover(self):
        #copy parents
        parents = self.pop.copy()
        
        # New generation
        self.pop = []

        # Generates pop_size individuals from crossover
        for i in range(self.pop_size):
            # Get two parents i the top "crossover_rate" individuals
            parent1 = random.randint(0,self.crossover_rate*self.pop_size)
            parent2 = random.randint(0,self.crossover_rate*self.pop_size)

            # choose from which parent we are going to pick each gene
            gene = [random.randint(0,2) for j in range(self.Individual_size)]
            for j in range(0, self.Individual_size):
                if gene[j] == 0:
                    self.pop.append(parent1[j])
                else:
                     self.pop.append(parent2[j])
            
    def mutation(self):
        # pick 10% of the population individuals to mutate
        mi = [random.randint(0,self.pop_size) for i in range(0.1*self.pop_size)]
        
        # substitute these 10% with random individuals
        for idx in mi:
            self.pop[idx] = Individual(self.individual_size)
            
            

    
    
    
    