from utils import *

if __name__ == "__main__":
    ct_path = "data/training/dicom_ct/"
    t1_path = "data/training/dicom_t1/"
    t1_r_path = "data/training/dicom_t1_rectified/"
    transform_path = "data/training/transformations/ct_T1.standard"
    #print (ct_scan[0].ImagerPixelSpacing)
    origo, transform = parse_transform (transform_path)
    print (origo)
    print (origo.shape)

    print (transform)
    print (transform.shape)

    ct_vol = ScanVolume (ct_path)
    t1_vol = ScanVolume (t1_path)

    #print (ct_vol.img_data[0].)
    print (ct_vol.pixel_spacing)
    print (ct_vol.size ())
    print (t1_vol.size ())

    #t1_vol.rescale (ct_vol.pixel_spacing)
    #print (t1_vol.size ())
    #print (t1_vol.pixel_spacing)

    t1_vol.resize (ct_vol.size ())
    #print (t1_vol.size ())
    #print (t1_vol.pixel_spacing)

    #plot_volume (ct_vol.norm_img ())
    #plot_volume (t1_vol.norm_img ())
    plot_volume (ct_vol.norm_img () + t1_vol.norm_img ())
    #plot_volume (pair.img2)
