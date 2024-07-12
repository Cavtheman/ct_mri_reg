import numpy as np
import einops as ein
import nibabel as nib

from torch.utils import data
from data_generator import SynthradData
from utils import plot_volume, normalise_array

if __name__ == "__main__":
    test_path = "./test/"
    test_good_fixed = nib.load (test_path + "fixed/2.nii.gz").get_fdata().squeeze()
    test_good_moving = nib.load (test_path + "moving/2.nii.gz").get_fdata().squeeze()
    #test_good_moved = nib.load (test_path + "moved/2.nii.gz").get_fdata().squeeze()

    test_bad_fixed = nib.load (test_path + "fixed/6.nii.gz").get_fdata().squeeze()
    test_bad_moving = nib.load (test_path + "moving/6.nii.gz").get_fdata().squeeze()
    #test_bad_moved = nib.load (test_path + "moved/6.nii.gz").get_fdata().squeeze()

    plot_volume (np.stack ([test_good_fixed, test_good_moving, np.zeros (test_bad_fixed.shape)], axis=3),
                     side_view=False,
                     rgb=True,
                     labels=("Fixed", "Moving", ""))

    plot_volume (np.stack ([test_bad_fixed, test_bad_moving, np.zeros (test_bad_fixed.shape)], axis=3),
                     side_view=False,
                     rgb=True,
                     labels=("Fixed", "Moving", ""))

    num_workers = 2
    aug_path = "./aug_data/norm_rot0.2_trans20_shearNone/"
    #aug_path = "./aug_data/norm_rot0.2_trans20_shear0.1/"
    #aug_path = "./aug_data/rot0.2_trans20_shearNone/"
    #aug_path = "./aug_data/rot0.2_trans20_shear0.1/"

    #ct_mr = nib.load ("ct_mr_moved.nii.gz")
    #mr_mr = nib.load ("mr_mr_moved.nii.gz")
    ct_mr = nib.load (aug_path + "moved_ct_to_mr/ct/0.nii.gz")
    mr_mr = nib.load (aug_path + "moved_mr_to_mr/0.nii.gz")

    ct_mr_transform = ct_mr.affine
    mr_mr_transform = mr_mr.affine
    ct_mr = ct_mr.get_fdata().squeeze()
    mr_mr = mr_mr.get_fdata().squeeze()

    print (ct_mr.shape)
    print (mr_mr.shape)

    aug_data = SynthradData (aug_path)
    dataloader = data.DataLoader(aug_data,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=num_workers)

    for ct_fixed, mr_fixed, mask_fixed, ct_moving, mr_moving, mask_moving, target_transform, aug_transform in dataloader:
        #plot_volume (ct_mr)
        print (mr_fixed.shape)
        print (mr_fixed.squeeze().shape)
        print (mr_moving.shape)
        #plot_volume (mr_moving.squeeze().numpy() + mr_mr + mr_fixed.squeeze().numpy())
        print (np.min (mr_fixed.numpy()))
        print (np.max (mr_fixed.numpy()))

        print (np.min (mr_moving.numpy()))
        print (np.max (mr_moving.numpy()))

        print (np.min (ct_fixed.numpy()))
        print (np.max (ct_fixed.numpy()))

        print (np.min (mr_mr))
        print (np.max (mr_mr))

        mr_fixed = normalise_array (mr_fixed.squeeze().numpy())
        mr_moving = normalise_array (mr_moving.squeeze().numpy())
        ct_fixed = normalise_array (ct_fixed.squeeze().numpy())
        mr_mr = normalise_array (mr_mr)
        ct_mr = normalise_array (ct_mr)

        #test = np.stack ([mr_moving, ct_fixed, mr_mr], axis=3)
        #plot_volume (ct_mr)
        #plot_volume (np.stack ([mr_fixed, mr_mr, mr_moving], axis=3),
        #             side_view=False,
        #             rgb=True,
        #             labels=("Fixed", "Moved using MR", "Moving"))

        plot_volume (np.stack ([mr_mr], axis=3),
                     side_view=False,
                     rgb=True,
                     labels=("Fixed", "Moved using MR", "Moving"))

        #plot_volume (np.stack ([mr_fixed, ct_mr, mr_moving], axis=3),
        #             side_view=True,
        #             rgb=True,
        #             labels=("Fixed", "Moved using CT", "Moving"))

        #plot_volume (np.stack ([ct_fixed * mask_fixed.squeeze().numpy(), mr_moving * mask_moving.squeeze().numpy(), orig_moving * mask_fixed.squeeze().numpy()], axis=3), rgb=True)
        #plot_volume (moving - moving * moving_mask.squeeze().numpy())
        #break
