import os
import random
import pandas as pd

images_path = "data/images" # replace with metadata read instead of data folder read
image_name_list = os.listdir(images_path)
file_count = len(image_name_list)

random.seed(a=999, version=2)
testing_data = random.sample((image_name_list), k=int(file_count*0.2))
training_data = [x for x in image_name_list if x not in testing_data]

with open("../result/training_data.txt", 'w') as f:
    for i in training_data:
        f.write(i[:-4]+"\n") # [:-4] substring to remove .png

with open("../result/testing_data.txt", 'w') as f:
    for i in testing_data:
        f.write(i[:-4]+"\n") # [:-4] substring to remove .png

# data = pd.read_csv("/Users/matthew/Desktop/Uni/semester_2/ProjData/FinalProject/data/metadata.csv")
# print(len(data["lesion_id"].unique()))