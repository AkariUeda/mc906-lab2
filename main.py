import cv2

from time import time
from evolve import Evolve
from individual import Individual

def experiment(name, pop_size=20, inner=1, usetime=True, use_interval=False, newgen_parent_ratio=1, mutation_rate=0.03, crossover_rate=0.9, use_roleta=False):
    image = cv2.imread('mona-lisa.jpg')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  # Convert color loaded by OpenCV

    evolve = Evolve(image,
                    pop_size=pop_size,
                    initial_pop=[],
                    ind_size=300,
                    ind_size_max=400,
                    good_genes=True,
                    fitness_function=Individual.get_fitness_function('SSIM_RGB'),
                    crossover_rate=crossover_rate,
                    use_interval=use_interval, 
                    newgen_parent_ratio=newgen_parent_ratio,
                    children_ratio=1,
                    mutation_rate=mutation_rate,
                    inner_mutation_rate=inner,
                    unmutable_ratio=0.5,
                    radius_range=(0.01, 0.1),
                    use_roleta=use_roleta)

    evolve.evaluate()
    evolve.save_image(name)
    evolve.save_json(name)
    print('G:{:4d}\tF:{:.20f}\tT:{:.3f}\tSOL_S:{}\tMAX_S:{}'.format(evolve.generation, -evolve.solution.fitness, time(), evolve.solution.ind_size, max(evolve.pop, key=lambda ind: ind.ind_size).ind_size))


    since_enh = 0
    last_best = evolve.solution.fitness
    last_enh = time()

    number_generations = 90
    for i in range(1, number_generations):
        evolve.crossover()
        evolve.mutate(expand_step=1)
        evolve.evaluate()
        print('G:{:4d}\tF:{:.20f}\tT:{:.3f}\tSOL_S:{}\tMAX_S:{}'.format(evolve.generation, -evolve.solution.fitness, time(), evolve.solution.ind_size, max(evolve.pop, key=lambda ind: ind.ind_size).ind_size))
        
        if evolve.solution.fitness < last_best:
            last_best = evolve.solution.fitness
            last_enh = time()
        else:
            if usetime and time() - last_enh > 30 * 60:
                print("No enhancement in the last 30 minutes")
                break
            elif not usetime and since_enh > 30:
                print("No enhancement in the last 30 generations")
                break

        if i % 10 == 0:
            evolve.save_image(name)
            evolve.save_json(name)

# Popsize
def experiment_1():
    experiment('./exp1', pop_size=20)

# Referencia
def experiment_2():
    experiment('./exp2', pop_size=10)

# Mutação
def experiment_3():
    experiment('./exp3', inner=0.1)

# Criterio da parada
def experiment_4():
    experiment('./exp4', usetime=False, inner=0.1)

# Crossover
def experiment_5():
    experiment('./exp5', use_interval=True, inner=0.1)

# Seleção
def experiment_6():
    experiment('./exp6', use_roleta=True, inner=0.1)

# Taxa de mutação
def experiment_7():
    experiment('./exp7', mutation_rate=0.2, inner=0.1)

# Taxa de crossover
def experiment_8():
    experiment('./exp8', crossover_rate=0.5, inner=0.1)

# Substituição
def experiment_9():
    experiment('./exp9', newgen_parent_ratio=0, inner=0.1)

if __name__ == "__main__":
    # Read reference image
    # experiment_1()
    # experiment_2()
    # experiment_3()
    # experiment_4()
    # experiment_5()
    # experiment_6()
    # experiment_7()
    # experiment_8()
    # experiment_9()
