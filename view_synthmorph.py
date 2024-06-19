import numpy as np
import einops as ein
import nibabel as nib

from torch.utils import data
from data_generator import SynthradData
from utils import plot_volume, normalise_array

if __name__ == "__main__":
    num_workers = 2
    aug_path = "./aug_data/norm_rot0.2_trans20_shearNone/"
    #aug_path = "./aug_data/norm_rot0.2_trans20_shear0.1/"
    #aug_path = "./aug_data/rot0.2_trans20_shearNone/"
    #aug_path = "./aug_data/rot0.2_trans20_shear0.1/"

    ct_mr = nib.load ("ct_mr_moved.nii.gz")
    mr_mr = nib.load ("mr_mr_moved.nii.gz")
    ct_mr_transform = ct_mr.affine
    mr_mr_transform = mr_mr.affine
    ct_mr = ct_mr.get_fdata().squeeze()
    mr_mr = mr_mr.get_fdata().squeeze()

    print (ct_mr.shape)
    print (mr_mr.shape)

    aug_data = SynthradData (aug_path, include_original=True)
    dataloader = data.DataLoader(aug_data,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=num_workers)

    for fixed, fixed_mask, moving, moving_mask, orig_moving, target_transform, aug_transform in dataloader:
        #plot_volume (ct_mr)
        print (orig_moving.shape)
        print (orig_moving.squeeze().shape)
        print (moving.shape)
        #plot_volume (moving.squeeze().numpy() + mr_mr + orig_moving.squeeze().numpy())
        print (np.min (orig_moving.numpy()))
        print (np.max (orig_moving.numpy()))

        print (np.min (moving.numpy()))
        print (np.max (moving.numpy()))

        print (np.min (fixed.numpy()))
        print (np.max (fixed.numpy()))

        print (np.min (mr_mr))
        print (np.max (mr_mr))

        orig_moving = normalise_array (orig_moving.squeeze().numpy())
        moving = normalise_array (moving.squeeze().numpy())
        fixed = normalise_array (fixed.squeeze().numpy())
        mr_mr = normalise_array (mr_mr)
        ct_mr = normalise_array (ct_mr)

        #test = np.stack ([moving, fixed, mr_mr], axis=3)
        #plot_volume (ct_mr)
        plot_volume (np.stack ([orig_moving, mr_mr, moving], axis=3),
                     side_view=True,
                     rgb=True,
                     labels=("Fixed", "Moved using MR", "Moving"))

        plot_volume (np.stack ([orig_moving, ct_mr, moving], axis=3),
                     side_view=True,
                     rgb=True,
                     labels=("Fixed", "Moved using CT", "Moving"))

        #plot_volume (np.stack ([fixed * fixed_mask.squeeze().numpy(), moving * moving_mask.squeeze().numpy(), orig_moving * fixed_mask.squeeze().numpy()], axis=3), rgb=True)
        #plot_volume (moving - moving * moving_mask.squeeze().numpy())
        #break
