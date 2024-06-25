import os
import sys

sys.path.insert(0, 'SynthSeg/')
from SynthSeg.predict import predict

# Much of the code in the function is taken and modified from https://github.com/BBillot/SynthSeg/blob/master/bibtex.bib
# Citation is in paper
def simple_predict (input_data, output_data, gt_folder=None):
    path_images = input_data
    path_segm = output_data
    if not os.path.exists(path_segm): os.makedirs(path_segm)

    path_model = 'SynthSeg/models/synthseg_1.0.h5'

    # but we also need to provide the path to the segmentation labels used during training
    path_segmentation_labels = 'SynthSeg/data/labels_classes_priors/synthseg_segmentation_labels.npy'
    # optionally we can give a numpy array with the names corresponding to the structures in path_segmentation_labels
    path_segmentation_names = 'SynthSeg/data/labels_classes_priors/synthseg_segmentation_names.npy'

    # We can now provide various parameters to control the preprocessing of the input.
    # First we can play with the size of the input. Remember that the size of input must be divisible by 2**n_levels, so the
    # input image will be automatically padded to the nearest shape divisible by 2**n_levels (this is just for processing,
    # the output will then be cropped to the original image size).
    # Alternatively, you can crop the input to a smaller shape for faster processing, or to make it fit on your GPU.
    cropping = 192
    # Finally, we finish preprocessing the input by resampling it to the resolution at which the network has been trained to
    # produce predictions. If the input image has a resolution outside the range [target_res-0.05, target_res+0.05], it will
    # automatically be resampled to target_res.
    target_res = 1.

    # Note that if the image is indeed resampled, you have the option to save the resampled image.
    path_resampled = './aug_segmentations/predicted_information/'
    if not os.path.exists(path_resampled): os.makedirs(path_resampled)

    # Second, we can smooth the probability maps produced by the network. This doesn't change much the results, but helps to
    # reduce high frequency noise in the obtained segmentations.
    sigma_smoothing = 0.5

    # Then we can operate some fancier version of biggest connected component, by regrouping structures within so-called
    # "topological classes". For each class we successively: 1) sum all the posteriors corresponding to the labels of this
    # class, 2) obtain a mask for this class by thresholding the summed posteriors by a low value (arbitrarily set to 0.1),
    # 3) keep the biggest connected component, and 4) individually apply the obtained mask to the posteriors of all the
    # labels for this class.
    # Example: (continuing the previous one)  generation_labels = [0, 24, 507, 2, 3, 4, 17, 25, 41, 42, 43, 53, 57]
    #                                             output_labels = [0,  0,  0,  2, 3, 4, 17,  2, 41, 42, 43, 53, 41]
    #                                       topological_classes = [0,  0,  0,  1, 1, 2,  3,  1,  4,  4,  5,  6,  7]
    # Here we regroup labels 2 and 3 in the same topological class, same for labels 41 and 42. The topological class of
    # unsegmented structures must be set to 0 (like for 24 and 507).
    topology_classes = 'SynthSeg/data/labels_classes_priors/synthseg_topological_classes.npy'
    # Finally, we can also operate a strict version of biggest connected component, to get rid of unwanted noisy label
    # patch that can sometimes occur in the background. If so, we do recommend to use the smoothing option described above.
    keep_biggest_component = True

    # Regarding the architecture of the network, we must provide the predict function with the same parameters as during
    # training.
    n_levels = 5
    nb_conv_per_level = 2
    conv_size = 3
    unet_feat_count = 24
    activation = 'elu'
    feat_multiplier = 2

    # Finally, we can set up an evaluation step after all images have been segmented.
    # In this purpose, we need to provide the path to the ground truth corresponding to the input image(s).
    # This is done by using the "gt_folder" parameter, which must have the same type as path_images (i.e., the path to a
    # single image or to a folder). If provided as a folder, ground truths must be sorted in the same order as images in
    # path_images.
    # Just set this to None if you do not want to run evaluation.
    #gt_folder = '/the/path/to/the/ground_truth/gt.nii.gz'
    #gt_folder = None

    # Dice scores will be computed and saved as a numpy array in the folder containing the segmentation(s).
    # This numpy array will be organised as follows: rows correspond to structures, and columns to subjects. Importantly,
    # rows are given in a sorted order.
    # Example: we segment 2 subjects, where output_labels = [0,  0,  0,  2, 3, 4, 17,  2, 41, 42, 43, 53, 41]
    #                             so sorted output_labels = [0, 2, 3, 4, 17, 41, 42, 43, 53]
    # dice = [[xxx, xxx],  # scores for label 0
    #         [xxx, xxx],  # scores for label 2
    #         [xxx, xxx],  # scores for label 3
    #         [xxx, xxx],  # scores for label 4
    #         [xxx, xxx],  # scores for label 17
    #         [xxx, xxx],  # scores for label 41
    #         [xxx, xxx],  # scores for label 42
    #         [xxx, xxx],  # scores for label 43
    #         [xxx, xxx]]  # scores for label 53
    #         /       \
        #   subject 1    subject 2
    #
    # Also we can compute different surface distances (Hausdorff, Hausdorff99, Hausdorff95 and mean surface distance). The
    # results will be saved in arrays similar to the Dice scores.
    compute_distances = True

    # All right, we're ready to make predictions !!
    predict(path_images,
            path_segm,
            path_model,
            path_segmentation_labels,
            path_resampled=path_resampled,
            cropping=cropping,
            target_res=target_res,
            topology_classes=topology_classes,
            sigma_smoothing=sigma_smoothing,
            keep_biggest_component=keep_biggest_component,
            n_levels=n_levels,
            nb_conv_per_level=nb_conv_per_level,
            conv_size=conv_size,
            unet_feat_count=unet_feat_count,
            feat_multiplier=feat_multiplier,
            activation=activation,
            gt_folder=gt_folder,
            compute_distances=compute_distances)

def predict_and_eval(fixed_folder, moved_folder, base_output, single=False):
    # Making "ground truth" predictions from the fixed images
    if single:
        simple_predict (fixed_folder + "0.nii.gz", base_output + "fixed/")
        simple_predict (moved_folder + "0.nii.gz", base_output + "moved/", gt_folder=(base_output + "fixed/0.nii.gz"))
    else:
        simple_predict (fixed_folder, base_output + "fixed/")
        simple_predict (moved_folder, base_output + "moved/", gt_folder=(base_output + "fixed/"))



if __name__ == "__main__":
    fixed_folder = "./aug_data/norm_rot0.2_trans20_shearNone/fixed/mr/"

    moved_folder = "./aug_data/norm_rot0.2_trans20_shearNone/moved/mr/"
    base_output = "./aug_segmentations/"

    predict_and_eval (fixed_folder, moved_folder, base_output, single=True)
