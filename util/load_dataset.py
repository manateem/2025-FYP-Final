from typing import Callable, List
import pandas as pd
import numpy as np
from inpaint_util import removeHair
from hairFeature import amountOfHairFeature
from constants import p
import random
import os
import cv2
import feature_extract_ex as features

TRAINING_IMAGES_DIR = p("data/images")
MASKS_DIR = p("data/masks")


# Feature extraction
def extractFeaturesFromImage(record):
    """
    Takes a DataFrame row as an argument and extracts all image features.

    Pass this function to DF.apply() with axis=1

    TODO:
        - [x] Extract feature "hair"
        - [x] Save a copy of the image after hair's been removed
        - [x] Extract feature "asymmetry"
        - [x] Extract feature "border"
        - [x] Extract feature "color"
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
    record["feat_asymmetry"] = features.rotation_asymmetry(image_mask, 5)["average"]
    record["feat_border_irregularity"] = features.get_compactness(image_mask, 2)["score"]
    record["feat_color"] = features.get_multicolor_rate(image, image_mask, 2)


    return record


def addFeatures(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame.apply(extractFeaturesFromImage, axis=1)


def loadDataFrameWithFeatures(
        write_csv_to: str | None = "data/dataset.csv",
        truncate: int | None = None) -> pd.DataFrame:
    """
    Load a data frame from the metadata, with new
    columns for the data extracted from the image files.

    :param write_csv_to: If specified, save the data frame as
    a CSV file to the specified file. By default, the CSV is saved
    to data/dataset.csv.
    :param truncate: Use this for debugging. If specified, only
    the first `n` entries will be loaded. Leave as None during
    final model training.

    :returns: A pd.DataFrame with the data from the metadata.csv,
    and the features extracted from the images.
    """
    DF  = pd.read_csv(p("data/metadata.csv"))

    # limit data in data frame only to images which have a mask
    masked_images = pd.DataFrame(data = {
        "img_id": [x.replace("_mask.png", ".png") for x in os.listdir(MASKS_DIR)]
    })

    DF = pd.merge(DF, masked_images, on='img_id', how='inner')

    if isinstance(truncate, int):
        DF = DF.head(truncate)
    
    DF = addFeatures(DF)

    # write to a .csv file if specified
    if isinstance(write_csv_to, str):
        DF.to_csv(p(write_csv_to), index=False)

    return DF

