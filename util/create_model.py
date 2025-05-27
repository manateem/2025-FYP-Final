from typing import Callable, List
import pandas as pd
import numpy as np
from inpaint_util import removeHair
from feature_extract_ex import asymmetry
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import GroupKFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from SaveModel import saveModel
from hairFeature import amountOfHairFeature
from constants import p
import random
import os
import cv2

MODEL_NAME = "model.pkl"
MODEL_DIR = p("testmodels/")


DF  = pd.read_csv(p("data/metadata.csv"))

TRAINING_IMAGES_DIR = p("data/images")
MASKS_DIR = p("data/masks")

# limit data in data frame only to images which have a mask
masked_images = pd.DataFrame(data = {
    "img_id": [x.replace("_mask.png", ".png") for x in os.listdir(MASKS_DIR)]
})

DF = pd.merge(DF, masked_images, on='img_id', how='inner')

# DELETE LATER: take only the first 20 items to speed up debugging
DF = DF.head(20)
print(DF)

groups = DF["patient_id"].unique()
DF['biopsed'] = DF['biopsed'].astype(int)
#metadata = metadata.dropna()
y_all = DF["biopsed"]


# Feature extraction
def extractFeaturesFromImage(record):
    """
    Takes a DataFrame row as an argument and extracts all image features.

    Pass this function to DF.apply() with axis=1

    TODO:
        - [x] Extract feature "hair"
        - [ ] Save a copy of the image after hair's been removed
        - [ ] Extract feature "asymmetry"
        - [ ] Extract feature "border"
        - [ ] Extract feature "color"
    
    REMEMBER:
        - [ ] delete directory "data/tmp" after the features have been extracted
    """
    image_path = os.path.join(TRAINING_IMAGES_DIR, record["img_id"])
    print(f"Opening image {image_path}...")

    # it's possible that the image is not in the training dataset -> skip it then
    if not os.path.isfile(image_path):
        print(f"{image_path}: no such file, continuing....")
        return record

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # hair feature
    record["feat_hair"] = amountOfHairFeature(image)

    # remove hair
    _, _, image = removeHair(image, image_gray)
    # load the image mask
    image_mask_filename = record["img_id"].replace(".png", "_mask.png")
    image_mask_path = os.path.join(MASKS_DIR, image_mask_filename)
    image_mask = cv2.imread(image_mask_path)

    # calculate asymmetry
    record["feat_asymmetry"] = asymmetry(image_mask)


    return record


def addFeatures(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame.apply(extractFeaturesFromImage, axis=1)


df = addFeatures(DF)
df.to_csv(p("data/dataset.csv"), index=False)

baseline_feats = [col for col in df.columns if col.startswith("feat_")]
print(baseline_feats)
x_all = df[baseline_feats]

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=42)
print(y_train)
gkf = GroupKFold(5)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Test Accuracy:", acc)
print("Confusion Matrix:\n", cm)
print(recall_score(y_test,y_pred))
# save the model
saveModel(model, MODEL_NAME, MODEL_DIR)





