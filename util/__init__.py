import img_util
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lesion_mask import extract_mask, get_mask_path
import feature_extract_ex


# test_img_path = "./data/images/PAT_8_15_820.png" # pretty circular mask
# test_img_path = "./data/images/PAT_15_23_240.png" # very circular mask
# test_img_path = "./data/images/PAT_38_1002_34.png" # not very circular mask

# test_img_path = "./data/images/PAT_29_40_561.png" # pretty uniformly brown lesion # score: 623.5938
test_img_path = "./data/images/PAT_1202_722_827.png" # not so uniformly brown lesion # score: 905.31
# test_img_path = "./data/images/PAT_2114_4609_615.png" # DOESNT HAVE A MASK !!!
# test_img_path = "./data/images/PAT_27_38_240.png" # not so black/grey lesion # score: 1297

test_img = img_util.readImageFile(test_img_path)[0]
test_mask = img_util.readImageFile(get_mask_path(test_img_path))[0]

extracted_img = extract_mask(test_img_path)
plt.imshow(extracted_img)

# Asymmetry: the lower the number the more symmetrical
print(feature_extract_ex.rotation_asymmetry(test_mask, 5)["average"])

# plt.imshow(test_mask, cmap='gray')
# Border: not well defined, irregular, inconsistent ; higher score, less compact
print(feature_extract_ex.get_compactness(test_mask, 2)["score"])

# Color: prescence of several colors in the same mole
print(feature_extract_ex.get_multicolor_rate(test_img, test_mask, 2))

# color_uniformity = feature_extract_ex.get_color_uniformity(extracted_img, test_mask)
# print(color_uniformity["score"])
# avg_patch = np.ones(shape=test_img.shape, dtype=np.uint8)*np.uint8(average)
# plt.imshow(avg_patch)

print(feature_extract_ex.significant_color_count(extracted_img,test_mask))

plt.show()
