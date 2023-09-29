import os
import numpy as np
import einops as ein
from pydicom import dcmread
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale

class IndexTracker:
    def __init__(self, ax, X, side_view=False):
        aspect = "auto" if side_view else None
        vmin, vmax = np.min (X), np.max (X)

        self.side_view = side_view
        self.index = 0
        self.X = X
        self.ax = ax
        self.im = ax.imshow(self.X[:, :, self.index], cmap="gray", aspect=aspect, vmin=vmin, vmax=vmax)
        self.update()

    def on_scroll(self, event):
        #print(event.button, event.step)
        increment = 1 if event.button == 'up' else -1
        max_index = self.X.shape[-1] - 1
        self.index = np.clip(self.index + increment, 0, max_index)
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.index])
        self.ax.set_title(
            f'Use scroll wheel to navigate\nindex {self.index}')
        self.im.axes.figure.canvas.draw()


def plot_volume (volume):
    fig, ax = plt.subplots()
    tracker = IndexTracker(ax, volume)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()


class ScanVolume:
    def __load_image__ (self, img_path):
        img = [ img_path + img_name for img_name in os.listdir (img_path) ]
        img.sort()
        return [ dcmread (path) for path in img ]

    def __normalise__ (self, x):
        return (x - np.min (x)) / (np.max (x) - np.min (x))

    def __to_numpy__ (self, image, normalise=False):
        return ein.rearrange (np.stack ([ layer.pixel_array
                                          for layer in image ]),
                              "z x y -> x y z")
    def norm_img (self):
        return self.__normalise__ (self.img)

    def size (self):
        return self.img.shape

    def rescale (self, new_spacing):
        scale_factor = new_spacing / self.pixel_spacing
        self.img = rescale (self.img, 1/scale_factor)
        self.pixel_spacing = new_spacing

    def resize (self, new_size):
        scale_factor = np.array (self.size()) / np.array (new_size)
        self.img = resize (self.img, new_size)
        self.pixel_spacing = self.pixel_spacing * scale_factor

    #def expand (self, new_size):
    #    assert np.all (np.array (self.size()) < np.array (new_size))
    #    new_img = np.zeros (new_size)
    #    new_img = new_img
    #    self.img =

    def __init__(self, img_path):
        self.img_data = self.__load_image__ (img_path)
        self.img = self.__to_numpy__ (self.img_data)

        # Pixel spacing in the data doesn't contain z coordinate spacing, so I get it elsewhere
        self.pixel_spacing = self.img_data[0].PixelSpacing
        self.pixel_spacing.append (self.img_data[1].ImagePositionPatient[2])
        self.pixel_spacing = np.array (self.pixel_spacing)

        #self.img2 = resize (self.img2, self.img1.shape)

        #self.maxX = max (self.img1.shape[0], self.img2.shape[0])
        #self.maxY = max (self.img1.shape[1], self.img2.shape[1])
