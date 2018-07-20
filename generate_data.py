import numpy as np
import matplotlib.pyplot as plt
import hypertools as hyp

def save_as_images_grid(f_name, batch_of_images, batch_size, grid_size):
    batch_of_images = np.reshape(batch_of_images, (batch_size, 28, 28))
    images_num = grid_size * grid_size
    images = batch_of_images[:images_num]

    rows = []
    for i in range(grid_size):
        rows.append(np.hstack(images[i * grid_size: (i+1) * grid_size]))
    image_grid = np.vstack(rows)

    plt.imsave("{}.png".format(f_name), image_grid,
               vmin=0, vmax=1, cmap='Greys_r')


def visualize_z_space(f_name, z, y):
    if z.shape[1] == 2:
        pass
    else:
        geo = hyp.plot(z, '.', hue=y.tolist(), save_path=f_name+".pdf") #, legend=True





