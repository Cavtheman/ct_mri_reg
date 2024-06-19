import os
import numpy as np
import einops as ein
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pydicom import dcmread
from skimage.transform import resize, rescale
from mpl_toolkits.axes_grid1 import make_axes_locatable

class IndexTracker:
    def __init__(self, ax, X, side_view=False, rgb=False, labels=None):
        if side_view:
            X = np.flip (np.transpose(X, [2,1,0,3]), axis=0)

        vmin, vmax = np.min (X), np.max (X)

        self.index = 0
        self.X = X
        self.ax = ax
        self.rgb = rgb
        self.labels = labels
        self.im = ax.imshow(self.X[:, :, self.index], cmap="gray", vmin=vmin, vmax=vmax)


        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.colorbar = plt.colorbar(self.im, cax=cax)
        if labels is not None:
            colors = ["red", "green", "blue"]
            handles = [ mpatches.Patch (color=color, label=label) for color,label in zip(colors,labels) ]
            plt.legend(handles=handles, loc="center left")
        plt.tight_layout()
        self.update()

    def on_scroll(self, event):
        #print(event.button, event.step)
        increment = 1 if event.button == 'up' else -1
        if self.rgb:
            max_index = self.X.shape[-2] - 1
        else:
            max_index = self.X.shape[-1] - 1

        self.index = np.clip(self.index + increment, 0, max_index)
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.index])
        self.ax.set_title(f'Use scroll wheel to navigate\nindex {self.index}')
        self.im.axes.figure.canvas.draw()

def parse_transform (path):
    lines = [ line.split() for line in open (path) ]
    arr = np.array ([ [ float (x) for x in line[1:] ] for line in lines[15:23] ])
    origo = arr[:,:arr.shape[1]//2]
    transform = arr[:,arr.shape[1]//2:]
    return origo, transform

# Interactive plot showing slices of 3D volume
def plot_volume (volume, side_view=False, rgb=False, labels=None):
    fig, ax = plt.subplots(figsize=(8,8))
    tracker = IndexTracker(ax, volume, side_view=side_view, rgb=rgb, labels=labels)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    plt.show()

# Normalisation between 0 and 1
def normalise_array (arr):
    min_val = np.min (arr)
    max_val = np.max (arr)
    return (arr-min_val) / (max_val-min_val)

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

    def pad_to_size (self, new_size):
        x,y,z = self.size()
        pad_width1 = [(0, new_size[0]-x), (0, new_size[1]-y), (0,0)]
        pad_width2 = [(0,0), (0,0), (0, new_size[2]-z)]
        self.img = np.pad (self.img, pad_width=pad_width1, mode="edge")
        self.img = np.pad (self.img, pad_width=pad_width2, mode="constant")
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
