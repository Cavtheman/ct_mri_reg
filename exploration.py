from utils import *
import torch
from data_generator import Dataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    ct_path = 'data/synthrad/brain/1BA001/ct.nii.gz'
    t1_path = 'data/synthrad/brain/1BA001/mr.nii.gz'
    mask_path = 'data/synthrad/brain/1BA001/mask.nii.gz'
    rotate = ((np.pi/16,np.pi/16), (np.pi/16,np.pi/16), (np.pi/16,np.pi/16))
    #rotate = ((0,0), (0,0), (0,0))
    translate = ((0,0), (0,0), (0,0))
    shear = ((0,0), (0,0), (0,0))
    data = Dataset ([t1_path], [ct_path], rotate=rotate, shear=shear, translate=translate)

    s = 10
    test = torch.zeros ((s,s,s), dtype=torch.double)
    test[1:-1,4,1:-1] = 1
    print (test)

    for fixed, moving, aug_moving, transform in data:
        #plot_volume (moving)
        #plot_volume (aug_moving)
        plot_volume (moving + aug_moving)

    #test_aug, transform = data.__augment_image__(test)
    #fig, ax = plt.subplots (2, s)
    #for i in range(s):
    #    ax[0,i].imshow (test[i], vmin=-1, vmax=1)
    #    im = ax[1,i].imshow (test_aug[i], vmin=-1, vmax=1)
    #fig.colorbar (im)
    #fig.savefig ("test.png")
