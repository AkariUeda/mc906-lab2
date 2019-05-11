import numpy as np

from matplotlib import pyplot as plt

def rms(a, b):
        """ Return the Root Mean Square between two RGB images"""
        pixel_count = a.shape[0] * a.shape[1]
        diffs = np.abs((a - b).flatten())
        values, idxs = np.histogram(diffs, bins=range(257))

        sum_of_squares = sum(value*(idx**2) for idx, value in zip(idxs, values))
        return np.sqrt(sum_of_squares / pixel_count)

def prepare_canvas(image, figax=None):
    # Prepare canvas
    fig, ax = figax if figax is not None else plt.subplots(figsize=(6, 6))

    # Disable plot ticks and their labels
    ax.tick_params(axis='both', which='both', 
                bottom=False, left=False,
                labelbottom=False, labelleft=False)
    ax.imshow(image)
    fig.canvas.draw()
    return fig, ax

def plot_image(image, figax=None):
    fig, ax = prepare_canvas(image, figax)
    plt.show()
    return fig, ax

def save_image(filename, image):
    fig, ax = prepare_canvas(image)
    fig.savefig(filename)

def hsv_to_rgb(color):
    """ Returns a (r,g,b) converted from the HSV color """
    src = np.array([[color]], dtype=np.uint8)
    color = list(map(int, cv2.cvtColor(src, cv2.COLOR_HSV2RGB)[0][0]))
    return color