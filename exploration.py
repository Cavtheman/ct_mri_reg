import os
import numpy as np
from utils import *
import einops as ein
from PIL import Image
from pydicom import dcmread
import matplotlib.pyplot as plt
from skimage.transform import resize
from pydicom.data import get_testdata_file


if __name__ == "__main__":
    ct_path = "data/training/dicom_ct/"
    t1_path = "data/training/dicom_t1/"
    side_view = False
    #print (ct_scan[0].ImagerPixelSpacing)

    ct_vol = ScanVolume (ct_path)
    t1_vol = ScanVolume (t1_path)

    print (dir (ct_vol.img_data[0]))
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
