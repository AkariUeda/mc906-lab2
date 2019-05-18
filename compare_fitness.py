import numpy as np
import sys
import matplotlib

matplotlib.use('agg')

from matplotlib import pyplot as plt


# ex: python compare.py exp1/experiment1.txt exp2/experiment2.txt

def get_fitness(filename):
    with open(filename) as f:
        return [float(a.split('F:')[1].split('\t')[0]) for a in f]

def plot_fitness(filename, ax, label, line):
    fitness = get_fitness(filename)
    ax.plot(range(len(fitness)), fitness, label=label, linestyle=line)


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1)
    plot_fitness(sys.argv[1], ax, sys.argv[2], '--')
    plot_fitness(sys.argv[3], ax, sys.argv[4], '-')
    ax.set_xlabel('Geração')
    ax.set_xticks(range(0, 90, 10))
    ax.set_xlim(0, 90)
    plt.legend()

    ax.set_ylabel('Fitness')
    #ax.set_yticks(np.linspace(0, 1, 20))
    #ax.set_ylim(0, 1)
    fig.savefig('result.png')

