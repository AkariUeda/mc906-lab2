import cv2

from evolve import Evolve

if __name__ == "__main__":
    # Read reference image
    image = cv2.imread('chrome.jpg')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  # Convert color loaded by OpenCV

    number_generations = 100

    generation = Evolve(image, pop_size=16, crossover_rate=0.8, individual_size=100, mutation_rate=0.5)
    generation.evaluate()
    generation.save_image()

    for i in range(number_generations):
        generation.crossover()
        generation.mutate()
        generation.evaluate()
        generation.save_image()
