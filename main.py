import cv2

from evolve import Evolve
from individual import Individual

def first_experiment():
    image = cv2.imread('mona-lisa.jpg')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  # Convert color loaded by OpenCV

    evolve = Evolve(image,
                    pop_size=10,
                    initial_pop=[],
                    ind_size=100,
                    ind_size_max=700,
                    good_genes=True,
                    fitness_function=Individual.get_fitness_function('SSIM_RGB'),
                    crossover_rate=0.9,
                    newgen_parent_ratio=0.0,
                    children_ratio=1,
                    mutation_rate=0.03,
                    inner_mutation_rate=1,
                    unmutable_ratio=0.5)

    evolve.evaluate()
    print(evolve)
    evolve.save_image('.')

    number_generations = 100
    for i in range(1, number_generations):
        evolve.crossover()
        evolve.mutate()
        evolve.evaluate()
        if i % 10 == 0:
            print(evolve)
            evolve.save_image('.')


if __name__ == "__main__":
    # Read reference image
    first_experiment()
