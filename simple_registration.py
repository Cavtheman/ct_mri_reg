from utils import *
import SimpleITK as sitk
from scipy.ndimage import affine_transform

def set_non_rigid_transform(R, im_fixed):
    # this function should be called instead of R.SetInitialTransform
    mesh_size= [int(im_fixed.GetSize()[0] / 20), int(im_fixed.GetSize()[1] / 20)]
    initial_transform = sitk.BSplineTransformInitializer(image1 = im_fixed,
                                                         transformDomainMeshSize = mesh_size, order = 2)

    transformDomainMeshSize = [2] * im_fixed.GetDimension()
    tx = sitk.BSplineTransformInitializer(im_fixed,
                                          transformDomainMeshSize )
    R.SetInitialTransformAsBSpline(initial_transform,
                                   inPlace = True,
                                   scaleFactors=[1,2,5])

# this funciton should use moving image im_moving to compute the transformation matrix outTx
# that will align im_moving with im_fixed
def registration(im_fixed, im_moving):
    #def command_iteration(method) :
    #    print("{0:3} = {1:10.5f} : {2}".format(method.GetOptimizerIteration(),
    #                                           method.GetMetricValue(),
    #                                           method.GetOptimizerPosition()))

    R = sitk.ImageRegistrationMethod()
    #R.SetMetricAsMeanSquares()
    R.SetMetricAsMattesMutualInformation()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200 )
    R.SetInitialTransform(sitk.BSplineTransform(im_fixed.GetDimension()))
    #R.AddCommand( sitk.sitkIterationEvent, lambda: command_iteration(R) )

    set_non_rigid_transform (R, im_fixed)

    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(im_fixed, im_moving)
    return outTx # this is transformation between images

# this function applies transformation to im_to_transform using transformation outTx
# im_fixed is needed to know the output size
def transform_img (im_to_transform, im_fixed, outTx, compose=True):
    # .....
    #sitk.WriteTransform(outTx, "test_file")

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im_fixed);
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(outTx)

    out = resampler.Execute(im_to_transform)
    if compose:
        simg1 = sitk.Cast(sitk.RescaleIntensity(im_fixed), sitk.sitkUInt8)
        simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
        cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)

        return out, cimg
    else:
        return out

def register_image (img_fixed, img_moving):
    R = sitk.ImageRegistrationMethod ()
    R.SetMetricAsMattesMutualInformation ()
    R.SetOptimizerAsRegularStepGradientDescent(4.0, .01, 200 )

    initial_transform = sitk.CenteredTransformInitializer(
        img_fixed,
        img_moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    R.SetInitialTransform (sitk.Euler3DTransform(initial_transform))

    return initial_transform

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


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    ct_path = "data/training/dicom_ct/"
    t1_path = "data/training/dicom_t1/"
    ct_vol = ScanVolume (ct_path)
    t1_vol = ScanVolume (t1_path)
    #t1_vol.rescale (ct_vol.pixel_spacing)
    #t1_vol.resize (ct_vol.size())
    print (ct_vol.size())
    print (t1_vol.size())
    t1_vol.pad_to_size (ct_vol.size())
    print (t1_vol.size())
    #ct_vol.rescale (t1_vol.pixel_spacing)
    #ct_vol.resize (t1_vol.size())


    ct_img = sitk.GetImageFromArray (ct_vol.norm_img ())
    t1_img = sitk.GetImageFromArray (t1_vol.norm_img ())

    print (ct_vol.size ())
    print (t1_vol.size ())
    #print (dir (ct_vol.img_data[0]))
    #transform = register_image (ct_img, t1_img)
    #transform = registration (ct_img, t1_img)

    origin, target = parse_transform ("data/training/transformations/ct_T1.standard")

    print (ct_vol.pixel_spacing)
    print (origin.shape)

    result, transform = find_transform (origin, target)
    #transform = transform.T
    temp = transform[3]
    transform[:,3] = temp
    transform[3] = np.array ([0,0,0,1])
    print (transform)

    print (np.min (ct_vol.img))
    print (np.max (ct_vol.img))
    print (np.min (t1_vol.img))
    print (np.max (t1_vol.img))
    #plot_volume (ct_vol.img)
    transformed_ct = affine_transform (ct_vol.img,
                                       transform,
                                       mode="mirror",)
                                       #output_shape=t1_vol.size())
    print (np.min (transformed_ct))
    print (np.max (transformed_ct))
    print (transformed_ct.shape)
    transformed_ct = np.flip (transformed_ct, axis=2)
    transformed_ct = (transformed_ct - np.min (transformed_ct)) / (np.max (transformed_ct) - np.min (transformed_ct))

    plot_volume (transformed_ct)
    #plot_volume (transformed_ct + t1_vol.norm_img())
    #plot_volume (ct_vol.norm_img() + t1_vol.norm_img())
    #plot_volume (transformed_ct + t1_vol.norm_img())
    plot_volume (transformed_ct + ct_vol.norm_img())
    #print (origin)
    #print ()
    #print (result)
    #print (target)
    #print ()
    #print (transform)

    #out, composed_img = transform_img (ct_img, t1_img, transform)
    #print (out)
    #composed_img = ein.reduce (sitk.GetArrayFromImage (composed_img),
    #                           "x y z rgb -> x y z", "sum")
    #composed_img = composed_img / np.max (composed_img)
    #print (composed_img.shape)
    #print (np.max (composed_img))
    #print (np.min (composed_img))
    #print (transform)
    #transform = register_image (ct_img, t1_img)
    #print (transform)
    #plot_volume (ct_vol.norm_img () + t1_vol.norm_img ())
    #plot_volume (composed_img)
