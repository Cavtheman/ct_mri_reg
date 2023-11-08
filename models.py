import torch
import numpy as np
import einops as ein
from data_generator import Dataset
from transformers import AutoFeatureExtractor, ResNetForImageClassification

def normalise (x):
    return (x - torch.min (x)) / (torch.max (x) - torch.min (x))

############################################################
#TEMP
############################################################
np.set_printoptions(suppress=True)
ct_path = 'data/synthrad/brain/1BA001/ct.nii.gz'
t1_path = 'data/synthrad/brain/1BA001/mr.nii.gz'
mask_path = 'data/synthrad/brain/1BA001/mask.nii.gz'
rotate = ((-np.pi/16,np.pi/16), (-np.pi/16,np.pi/16), (-np.pi/16,np.pi/16))
#rotate = ((0,0), (0,0), (0,0))
translate = ((0,0), (0,0), (0,0))
shear = ((0,0), (0,0), (0,0))
data = Dataset ([t1_path], [ct_path], rotate=rotate, shear=shear, translate=translate)

############################################################
#TEMP
############################################################

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
print (dir (model))
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")


for fixed, moving, aug_moving, transform in data:
    #plot_volume (moving)
    #plot_volume (aug_moving)
    #print (transform)
    #plot_volume (moving + aug_moving)
    aug_moving = normalise (aug_moving)
    aug_moving = ein.rearrange (aug_moving, "x y z -> x y z ()")
    print (aug_moving.size())
    print (torch.min (aug_moving))
    print (torch.max (aug_moving))
    inputs = feature_extractor(aug_moving[100], return_tensors="pt")
    print (inputs.size())
    break

#if __name__ == "__main__":
#    np.set_printoptions(suppress=True)
