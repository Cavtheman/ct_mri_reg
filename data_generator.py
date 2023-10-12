import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
import random

from PIL import Image

class Dataset(data.Dataset):
    # Characterizes a dataset for PyTorch
    # Takes a bunch of arguments for data augmentation as well
    def __init__(self, ct_paths, t1_paths, translation=0, rotation=0, shear=0):
        assert (len (ct_paths) == len (t1_paths))

        self.ct_paths = ct_paths
        self.ct_paths.sort ()

        self.t1_paths = t1_paths
        self.t1_paths.sort ()

        self.mask_paths = mask_paths
        self.mask_paths.sort ()

        self.translation = translation
        self.rotation = rotation
        self.shear = shear

    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, index):
        # Generates one sample of data
        ct_path = self.ct_paths[index]
        t1_path = self.t1_paths[index]
        #mask_path = self.mask_paths[index]

        sitk_t1 = sitk.ReadImage(t1_path)
        #t1 = sitk.GetArrayFromImage(sitk_t1)

        sitk_ct = sitk.ReadImage(ct_path)
        #ct = sitk.GetArrayFromImage(sitk_ct)

        aug_ct = self.__augment_data__ (sitk_ct)

        return sitk_t1, sitk_ct, aug_ct

    def __augment_data__(self, data):
        # Rotates image randomly
        match self.rotation:
            case (lower, upper):
                rotation = random.randint(lower, upper)
        match self.shear:
            case (lower, upper):
                shear = random.uniform(lower, upper)
        match self.translate:
            case ((lower_x, upper_x), (lower_y, upper_y)):
                translate = (random.uniform (lower_x, upper_x), random.uniform (lower_y, upper_y))

        data = TF.affine(data, rotation, translate, 1, shear)
    #def __init__(self, list_IDs, data_path, label_path, hflip=None, vflip=None, angle=0, shear=0, brightness=1, pad=(19,0,0,0), contrast=1, use_channel=None):
    #    self.list_IDs = list_IDs
    #    self.data_path = data_path
    #    self.label_path = label_path
    #    self.hflip = hflip
    #    self.vflip = vflip
    #    self.angle = angle
    #    self.shear = shear
    #    self.brightness = brightness
    #    self.pad = pad
    #    self.contrast = contrast
    #    self.use_channel = use_channel
'''
    def __augment_data__(self, data, labels, hflip, vflip, angle, shear, brightness, pad, contrast, use_channel):

        # Flipping the image horizontally
        if hflip and random.random() > hflip:
            data = TF.hflip(data)
            labels = TF.hflip(labels)

        # Flipping the image vertically
        if vflip and random.random() > vflip:
            data = TF.vflip(data)
            labels = TF.vflip(labels)

        # Adjusts the brightness randomly
        if type(brightness) is tuple:
            brightness = random.uniform(brightness[0], brightness[1])

        # Rotates image randomly
        if type(angle) is tuple:
            angle = random.randint(angle[0], angle[1])

        # Shears image randomly
        if type(shear) is tuple:
            shear = random.uniform(shear[0], shear[1])

        if type(contrast) is tuple:
            contrast = random.uniform(contrast[0], contrast[1])

        data = TF.adjust_contrast(data, contrast)
        data = TF.adjust_brightness(data, brightness)

        data = TF.affine(data, angle, (0,0), 1, shear)
        labels = TF.affine(labels, angle, (0,0), 1, shear)

        data = F.pad(ToTensor()(data), pad)
        labels = F.pad(ToTensor()(labels), pad)

        if use_channel:
            data = torch.narrow(data, 0, use_channel, 1)

        return data, labels


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # Generates one sample of data
        ID = self.list_IDs[index]

        # Load data and get label
        data = Image.open(self.data_path.format(ID))
        labels = Image.open(self.label_path.format(ID))

        # Augment the data and labels randomly using given arguments
        data, labels = self.__augment_data__(data, labels, self.hflip, self.vflip, self.angle, self.shear, self.brightness, self.pad, self.contrast, self.use_channel)

        return data, labels
'''
