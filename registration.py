import os
import numpy as np
from utils import *
import torchvision.transforms as TF
#from data_generator import Dataset
import SimpleITK as sitk
from downloaddata import fetch_data as fdata

# this funciton should use moving image im_moving to compute the transformation matrix outTx
# that will align im_moving with im_fixed
def registration(im_fixed, im_moving):
    R = sitk.ImageRegistrationMethod()
    #R.SetMetricAsMeanSquares()
    R.SetInterpolator(sitk.sitkLinear)

    R.SetMetricAsMattesMutualInformation()
    R.SetOptimizerAsGradientDescent(4.0, .01, 200. )
    #R.SetInitialTransform(initial_transform, inPlace=False)
    #R.SetInitialTransform(sitk.BSplineTransform(im_fixed.GetDimension()))
    R.SetInitialTransform(sitk.AffineTransform)

    #R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

    set_non_rigid_transform (R, im_fixed)

    outTx = R.Execute(im_fixed, im_moving)
    return outTx # this is transformation between images

def find_transform (moving, fixed):
    n = moving.shape[0]
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    X = pad(moving)
    Y = pad(fixed)
    #Y = np.flip (Y, axis=0)
    print (X.shape)
    print (Y.shape)
    # Solve the least squares problem X * A = Y
    # to find our transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)

    transform = lambda x: unpad(np.dot(pad(x), A))
    #A = A.T
    #temp = A[3]
    #A[3] = np.array ([0,0,0,1])
    #A[:,3] = temp
    return transform (moving), A

def affine_transform (img, mat):
    affine = sitk.AffineTransform (3)
    affine.SetMatrix (mat)
    affine.Execute(img)


def __normalise__ (x):
    return (x - np.min (x)) / (np.max (x) - np.min (x))

def get_filenames (path, filetype):
    temp = [ [ path + folder + "/" + elem for elem in os.listdir (path + folder)
               if filetype in elem ]
             for folder in os.listdir (path) ]
    return [item for sublist in temp for item in sublist]

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    '''
    ct_path = 'data/synthrad/brain/1BA001/ct.nii.gz'
    t1_path = 'data/synthrad/brain/1BA001/mr.nii.gz'
    mask_path = 'data/synthrad/brain/1BA001/mask.nii.gz'

    sitk_t1 = sitk.ReadImage(t1_path)
    t1 = sitk.GetArrayFromImage(sitk_t1)

    sitk_ct = sitk.ReadImage(ct_path)
    ct = sitk.GetArrayFromImage(sitk_ct)

    sitk_mask = sitk.ReadImage(mask_path)
    mask = sitk.GetArrayFromImage(sitk_mask)

    # and access the numpy array:
    print (t1.shape)
    print (ct.shape)

    print (np.max (t1))
    print (np.min (t1))
    print (np.max (ct))
    print (np.min (ct))

    plot_volume (__normalise__ (ct) + __normalise__ (t1))
    plot_volume (mask)
    #plt.plot (t1)
    #plt.show ()
    '''
    #print (os.listdir ("data/synthrad/brain/"))
    for elem in get_filenames ("data/synthrad/brain/", "ct"):
        print (elem)

    for elem in get_filenames ("data/synthrad/brain/", "mr"):
        print (elem)

    for elem in get_filenames ("data/synthrad/brain/", "mask"):
        print (elem)


    '''
    #==================================
    import numpy as np
    import matplotlib.pyplot as plt
    #==================================
    # load image (4D) [X,Y,Z_slice,time]
    nii_img  = nib.load('data/synthrad/brain/1BA001/ct.nii.gz')
    nii_data = nii_img.get_fdata()
    print (nii_data.shape)

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    #fig.canvas.set_window_title('4D Nifti Image')
    fig.suptitle('4D_Nifti 10 slices 30 time Frames', fontsize=16)
    #-------------------------------------------------------------------------------
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()

    for slice in range(1):
        # if your data in 4D, otherwise remove this loop
        #for frame in range(number_of_frames):
        ax[frame, slice].imshow(nii_data[:,:,slice,frame],cmap='gray', interpolation=None)
        ax[frame, slice].set_title("layer {} / frame {}".format(slice, frame))
        ax[frame, slice].axis('off')

    plt.show()
    '''
    #####################################################################
    '''
    ct_path = "data/training/dicom_ct/"
    t1_path = "data/training/dicom_t1/"
    ct_vol = ScanVolume (ct_path)
    t1_vol = ScanVolume (t1_path)

    t1_vol.pad_to_size (ct_vol.size())
    print (ct_vol.size ())
    print (t1_vol.size ())
    #plot_volume (t1_vol.norm_img ())


    moving_image =  sitk.GetImageFromArray (ct_vol.img, sitk.sitkFloat32)
    fixed_image = sitk.GetImageFromArray (t1_vol.img, sitk.sitkFloat32)

    origin, target = parse_transform ("data/training/transformations/ct_T1.standard")

    print (ct_vol.pixel_spacing)
    print (origin.shape)

    result, transform = find_transform (origin, target)
    #transform = transform.T
    temp = transform[3]
    transform[:,3] = temp
    transform[3] = np.array ([0,0,0,1])
    print (transform)


    print (affine_transform (moving_image, transform))
    '''

    #outTx = registration (fixed_image, moving_image)
