import cv2

from time import time
from evolve import Evolve
from individual import Individual

def first_experiment():
    image = cv2.imread('mona-lisa.jpg')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  # Convert color loaded by OpenCV

    evolve = Evolve(image,
                    pop_size=50,
                    initial_pop=[],
                    ind_size=100,
                    ind_size_max=250,
                    good_genes=True,
                    fitness_function=Individual.get_fitness_function('SSIM_RGB'),
                    crossover_rate=0.9,
                    use_interval=False, 
                    newgen_parent_ratio=1.0,
                    children_ratio=1,
                    mutation_rate=0.03,
                    inner_mutation_rate=1,
                    unmutable_ratio=0.5,
                    radius_range=(0.01, 0.1))

    evolve.evaluate()
    evolve.save_image('.')
    print('G:{:4d}\tF:{:.20f}\tT:{:.3f}'.format(evolve.generation, -evolve.solution.fitness, time()))

    last_best = evolve.solution.fitness
    last_enh = time()

    number_generations = 1000
    for i in range(1, number_generations):
        evolve.crossover()
        evolve.mutate(expand_step=1)
        evolve.evaluate()
        print('G:{:4d}\tF:{:.20f}\tT:{:.3f}'.format(evolve.generation, -evolve.solution.fitness, time()))
        
        if evolve.solution.fitness < last_best:
            last_best = evolve.solution.fitness
            last_enh = time()
        elif time() - last_enh > 30 * 60:
            print("No enhancement in the last 30 minutes")
            break

        if i % 10 == 0:
            evolve.save_image('.')


if __name__ == "__main__":
    # Read reference image
    first_experiment()
