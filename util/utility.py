"""utility.py replaces: lesion_mask.py, img_util.py, SaveModel.py, data_partition.py"""

import pandas as pd
import numpy as np
import cv2
import os
import joblib
import os
import matplotlib.pyplot as plt
import random

metadata_fp = "data/metadata.csv"
images_fp = "data/images/"
masks_fp = "data/masks/"
hairless_fp = "data/hairless/"

def get_lesion_ids(patient_id):
    "Given a patient id, return a list of all unique lesion id's for that patient. No 'PAT', just the number."
    data = pd.read_csv(metadata_fp)
    return [x for x in data[data['patient_id']==f"PAT_{patient_id}"]['lesion_id'].unique()]

def get_lesion_img(patient_id, lesion_id):
    """Returns an image, mask, and hairless image, will still work if a mask or hairless image doesn't exist."""
    data = pd.read_csv(metadata_fp)

    file_path = None
    for fp in os.listdir(images_fp):
        if not fp.find(f"PAT_{patient_id}_{lesion_id}") == -1:
            file_path = fp
            break

    if file_path == None:
        print(f"[ERROR] PAT_{patient_id}_{lesion_id} doesn't exist!")
        return None, None, None

    img = cv2.imread(images_fp+file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    isExist = False
    for fp in os.listdir(masks_fp):
        if not fp.find(file_path[:-4]+"_mask.png") == -1:
            mask = cv2.imread(masks_fp+file_path[:-4]+"_mask.png")
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            isExist = True
            break
    if not isExist:
        print(f"[ERROR] PAT_{patient_id}_{lesion_id}'s mask doesn't exist!")

    hairless = cv2.imread(hairless_fp+file_path[:-4]+"_hairless.png")
    if hairless == None:
        print(f"[WARNING] PAT_{patient_id}_{lesion_id}'s hairless doesn't exist!")
    else:
        hairless = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return (img, mask, hairless)

def extract_lesion_mask(img, mask):
    """Provided an image and a mask, return the extracted masked image."""
    a = img.copy()
    a[mask==0] = 0
    return a

def get_mask_path(path):
    return masks_fp+path[12:-4]+"_mask.png"

def saveModel(model, modelName, directory):
    if ".pkl" not in modelName:
        modelName += ".pkl"
    filepath = os.path.join(directory, modelName)
    if not os.path.exists(directory):
        os.makedirs(directory)
    joblib.dump(model, filepath)

class HairExtractor:
    def __init__(self, img):
        self.img = img
    
    def countWhitePercentage(self, threshold=240):
        """
        Counts the number of white pixels in a grayscale image.

        :param self.img: the image
        :param threshold: (int) Intensity threshold to consider a pixel as "white".
        
        :returns: Percentage of white pixels in the image.
        """
        # Load the image in grayscale
        # Create a mask of pixels above the threshold
        white_mask = self.img >= threshold

        # Count white pixels
        white_pixel_count = np.sum(white_mask)
        total_pixels = self.img.size
        white_percentage = (white_pixel_count / total_pixels)

        return white_percentage

    def getHair(self, kernel_size=25, threshold=10):
        # kernel for the morphological filtering
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # perform the hat transform on the grayscale image
        hat_img = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel) 
        
        
        # threshold the hair contours
        _, thresh = cv2.threshold(hat_img, threshold, 255, cv2.THRESH_BINARY)
        
        return thresh

    def amountOfHairFeature(self, black_threshold: int = 50) -> float:
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)

        _, thresh = cv2.threshold(img_gray, black_threshold, 255, cv2.THRESH_BINARY)

        count_black_pxls = np.sum(thresh == 0)

        return (count_black_pxls / thresh.size) * 10

class DataPartitioner:
    """Splits data into training and testing data in a list that can be used for processing"""
    def __init__(self, seed):
        data = pd.read_csv(metadata_fp)
        self.patient_list = [x for x in data['patient_id'].unique()]
        self.seed = seed

        # Partitions into "groups"/patients, hopefully useful for GroupKFold?
        random.seed(a=self.seed, version=2)
        self.testing_list = random.sample((self.patient_list), k=int(len(self.patient_list)*0.2))
        self.training_list = [x for x in self.patient_list if x not in self.testing_list]
    
    def get_training(self):
        return self.training_list

    def get_testing(self):
        return self.testing_list

    def export_lists(self, testing_fp="result/testing_data.txt", training_fp="result/training_data.txt"):
        with open(training_fp, 'w') as f:
            for i in self.training_list:
                f.write(i+"\n")

        with open(testing_fp, 'w') as f:
            for i in testing_data:
                f.write(i+"\n")

if __name__ == "__main__":
    # Hair Extractor Example
    image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    # First create a hairextractor obj
    hairExtObj = HairExtractor(img=image)
    # Perform methods on obj
    print(f"White %: {hairExtObj.countWhitePercentage()}")
    print(f"Hair Amount: {hairExtObj.amountOfHairFeature()}")

    # Data Partition Example
    dataPart = DataPartitioner(seed=999)
    print(dataPart.get_testing()[:10])
    print("...")
    print(dataPart.get_training()[:10])
    print("...")

    # Example
    pat_id = "705"
    ids = get_lesion_ids(pat_id)
    img, mask, hairless = get_lesion_img(pat_id, ids[0])
    extracted_img = extract_lesion_mask(img, mask)
    plt.imshow(extracted_img)
    plt.show()




# Code that is deprecated, but I decided to save anyway
def removeHair(img_org, img_gray, kernel_size=25, threshold=10, radius=3):
    # kernel for the morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    # perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting algorithm
    _, thresh = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    img_out = cv2.inpaint(img_org, thresh, radius, cv2.INPAINT_TELEA)

    return blackhat, thresh, img_out