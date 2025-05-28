import img_util
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utility
import feature_extract_ex

def abc_feature_extract(img, mask):
    """Extracts 3 features, asymmetry, border, and color, and returns a tuple of scores respectively."""
    """The bigger values, the 'worse'"""

    '''Asymmetry: the lower the number the more symmetrical'''
    asymmetry = feature_extract_ex.rotation_asymmetry(mask, 5)["average"]

    '''Border: not well defined, irregular, inconsistent ; higher score, less compact'''
    border = feature_extract_ex.get_compactness(mask, 2)["score"]

    '''Color: prescence of several colors in the same mole ; lower number, more uniformly colored'''
    color = feature_extract_ex.get_color_uniformity(utility.extract_lesion_mask(img, mask), mask)["score"]
    # Color alternatives
    # print(feature_extract_ex.get_multicolor_rate(test_img, test_mask, 2))
    # print(feature_extract_ex.significant_color_count(extracted_img,test_mask, significance=0.1))

    return (asymmetry, border, color)

# Test image feature extraction
if __name__ == "__main__":
    # test_img_path = "data/images/PAT_8_15_820.png" # pretty circular mask
    test_img_path = "data/images/PAT_156_241_538.png" # pretty circular mask
    # test_img_path = "data/images/PAT_15_23_240.png" # very circular mask
    # test_img_path = "data/images/PAT_38_1002_34.png" # not very circular mask

    # test_img_path = "data/images/PAT_29_40_561.png" # pretty uniformly brown lesion # score: 623.5938
    # test_img_path = "data/images/PAT_1202_722_827.png" # not so uniformly brown lesion # score: 905.31
    # test_img_path = "data/images/PAT_2114_4609_615.png" # DOESNT HAVE A MASK !!!
    # test_img_path = "data/images/PAT_27_38_240.png" # not so black/grey lesion # score: 1297
    # test_img_path = "data/images/PAT_1903_3791_14.png" # very uniform brown lesion

    test_img = img_util.readImageFile(test_img_path)[0]
    test_mask = img_util.readImageFile(utility.get_mask_path(test_img_path))[0]
    """Test mask is RGB..."""

    # print(abc_feature_extract(test_img, test_mask))
    print(feature_extract_ex.color_metrics(test_img, test_mask))

    extracted_img = utility.extract_lesion_mask(test_img,test_mask)
    plt.imshow(extracted_img)
    plt.show()