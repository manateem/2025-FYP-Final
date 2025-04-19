import img_util
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lesion_mask import extract_mask, get_mask_path
import feature_extract_ex

test_img_path = "./data/images/PAT_8_15_820.png" # pretty circular mask
# test_img_path = "./data/images/PAT_15_23_240.png" # very circular mask
# test_img_path = "./data/images/PAT_38_1002_34.png" # not very circular mask
test_mask = img_util.readImageFile(get_mask_path(test_img_path))[0]

extracted_img = extract_mask(test_img_path)

# Asymmetry: the lower the number the better
print(feature_extract_ex.rotation_asymmetry(test_mask, 5))
print(feature_extract_ex.asymmetry(test_mask))
# print(feature_extract_ex.get_asymmetry(test_mask))

# Border: 

# Color:


plt.imshow(extracted_img, cmap='gray')
plt.show()
