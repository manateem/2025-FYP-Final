import os
import random

images_path = "../data/images"
image_name_list = os.listdir("../data/images")
file_count = len(image_name_list)

random.seed(a=999, version=2)
test_data = random.sample((image_name_list), k=int(file_count*0.2))
training_data = [x for x in image_name_list if x not in test_data]

with open("../result/training_data.txt", 'w') as f:
    for i in training_data:
        f.write(i[:-4]+"\n") # [:-4] substring to remove .png

with open("../result/test_data.txt", 'w') as f:
    for i in test_data:
        f.write(i[:-4]+"\n") # [:-4] substring to remove .png