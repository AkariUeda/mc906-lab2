import cv2
import numpy as np

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

def rms(a, b):
    """ Return the Root Mean Square between two RGB images"""
    pixel_count = a.shape[0] * a.shape[1]
    diffs = np.abs((a - b).flatten())
    values, idxs = np.histogram(diffs, bins=range(257))

    sum_of_squares = sum(value*(idx**2) for idx, value in zip(idxs, values))
    return np.sqrt(sum_of_squares / pixel_count)

if __name__ == "__main__":
    # Read reference image
    image = cv2.imread('mona.jpg')
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  # Convert color loaded by OpenCV

    # Evolve(image, pop_size=10, crossover_rate=0.5, individual_size)
