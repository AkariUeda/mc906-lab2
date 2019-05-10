import cv2
import numpy as np

from evolve import Evolve

def plot_image(image):
    from matplotlib import pyplot as plt
    # Prepare canvas
    fig, ax = plt.subplots(figsize=(6, 6))

    # Disable plot ticks and their labels
    ax.tick_params(axis='both', which='both', 
                   bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    ax.imshow(image)
    plt.show()



if __name__ == "__main__":
    # Read reference image
    image = cv2.imread('mona.jpg')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  # Convert color loaded by OpenCV

    number_generations = 10

    generation = Evolve(image, pop_size=10, crossover_rate=0.5, individual_size=50)
    generation.evaluate()
    generation.save_img()

    for i in range(0, number_generations):
        generation.crossover()
        generation.mutation()
        generation.evaluate()
        generation.save_img()
