from synthseg_eval import predict_and_eval

if __name__ == "__main__":
    predict_and_eval (
        "./aug_data/test_norm_rot0.4_trans40_shearNone/fixed/mr/",
        "./aug_data/test_norm_rot0.4_trans40_shearNone/fixed/mr/",
        "id_seg_test/",
        single=True
    )
