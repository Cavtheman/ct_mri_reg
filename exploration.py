from utils import *
import torch
from data_generator import AugmentData, SynthradData
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    ct_path = 'data/synthrad/brain/1BA001/ct.nii.gz'
    t1_path = 'data/synthrad/brain/1BA001/mr.nii.gz'
    mask_path = 'data/synthrad/brain/1BA001/mask.nii.gz'

    #rotate = ((-0.1,0.1), (-0.1,0.1), (-0.1,0.1))
    rotate = ((-0.3,0.3), (-0.3,0.3), (-0.3,0.3))
    #rotate = ((0,0), (0,0), (0.5,0.5))
    #rotate = ((0,0), (0,0), (0,0))

    translate = ((-20,20), (-20,20), (-20,20))
    #translate = ((-5,5), (-5,5), (-5,5))
    #translate = ((20,20), (0,0), (0,0))
    #translate = ((0,0), (0,0), (0,0))

    shear = ((0,0), (0,0), (0,0))
    data = AugmentData ([ct_path], [t1_path], [mask_path], rotate=rotate, shear=shear, translate=translate)

    print (data.__max_size__())

    for fixed, fixed_mask, moving, moving_mask, aug_moving, transform, inv_transform in data:
        print (fixed.shape)
        print (moving.shape)
        print (aug_moving.shape)
        print (transform)
        print (inv_transform)

        #print (np.min (fixed))

        #plot_volume (aug_moving.numpy())
        plot_volume (fixed)
        plot_volume (fixed * fixed_mask)
        plot_volume (moving + aug_moving)
        plot_volume (fixed + moving)
        plot_volume (fixed + aug_moving)
        #plot_volume (moving + affine_transform(aug_moving, inv_transform, mode="nearest"))
        break

    #s = 50
    #test = torch.zeros ((s,s,s), dtype=torch.double)
    #test[5:-5,24:27,5:-5] = 1
    #test_aug, transform, inv_transform = data.__augment_image__(test)

    #print (transform)
    #print (inv_transform)
    #plot_volume (test_aug.numpy())
    #plot_volume (test.numpy())
    #plot_volume (test_aug)
    #plot_volume (affine_transform (test_aug, inv_transform, mode="nearest"))
    #fig, ax = plt.subplots (2, s)
    #for i in range(s):
        #ax[0,i].imshow (test[i], vmin=-1, vmax=1)
        #im = ax[1,i].imshow (test_aug[i], vmin=-1, vmax=1)


    #fig.colorbar (im)
    #fig.savefig ("test.png")
