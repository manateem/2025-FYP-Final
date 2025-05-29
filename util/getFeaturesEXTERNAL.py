from typing import Callable, List
import pandas as pd
import numpy as np
from constants import p
import random
import os
import cv2
import feature_extract_ex as features
from utility import extract_lesion_mask

TRAINING_IMAGES_DIR = p("data/noHairEXTERNAL") #NOW THE IMAGES WITH NO HAIR
MASKS_DIR = p("data/masksEXTERNAL")
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

    #data\noHairEXTERNAL\ISIC_0024306.jpg
    image_path = os.path.join(TRAINING_IMAGES_DIR, f"{record['img_id']}.jpg")
    print(f"Opening image {image_path}...")


    # it's possible that the image is not in the training dataset -> skip it then
    if not os.path.isfile(image_path):
        print(f"{image_path}: no such file, continuing....")
        return record

    image = cv2.imread(image_path)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # hair feature
    #record["feat_hair"] = amountOfHairFeature(image_rgb)

    # remove hair
    #_, _, image = removeHair(image, image_gray)
    # load the image mask
    image_mask_filename = f"{record['img_id']}.png"
    image_mask_path = os.path.join(MASKS_DIR, image_mask_filename)
    image_maskG = cv2.imread(image_mask_path,cv2.IMREAD_GRAYSCALE)
    image_mask = cv2.imread(image_mask_path)

    # calculate asymmetry
    # try:
    #     record["feat_asymmetry"] = features.rotation_asymmetry(image_maskG,5)["average"]
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_asymmetry"] = float("nan")
    # try:
    #     record["feat_border_irregularity"] = features.get_compactness(image_mask,2)["score"]
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_border_irregularity"] = float("nan")
    # try:
    #     record["feat_multiColorRate"] = features.get_multicolor_rate(image, image_mask,1)
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_multiColorRate"] = float("nan")
    # try:
    #     record["feat_convexity"] = features.convexity_score(image_mask)
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_convexity"] = float("nan")
    # try:
    #     record["feat_avgColor"] = features.get_color_uniformity(image,image_mask)["average_color"]
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_avgColor"] = float("nan")
    # a = features.convexity_metrics(image_mask)
    # try:
    #     record["feat_convexVariance"] = a["variance"]
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_convexVariance"] = float("nan")
    # try:
    #     record["feat_convexMax"] = a["max"]
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_convexMax"] = float("nan")
    # try:
    #     record["feat_convexAverage"] = a["average"]
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_convexAverage"] = float("nan")
    
    # b = features.texture_analysis(image,image_mask)
    # try:
    #     record["feat_contrast"] = b["glcm_contrast"]
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_contrast"] = float("nan")
    # try:
    #     record["feat_energy"] = b["glcm_energy"]
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_energy"] = float("nan")
    # try:
    #     record["feat_homogeneity"] = b["glcm_homogeneity"]
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_homogeneity"] = float("nan")
    # try:
    #     c = features.get_color_uniformity(image,image_maskG)
    # except:
    #     record["feat_colorUniformity"] = float("nan")
    #     record["feat_averageColor"] = float("nan")
    #     return record
    # try:
    #     record["feat_colorUniformity"] = c["score"]
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_colorUniformity"] = float("nan")
    # try:
    #     record["feat_averageColor"] = c["average_color"]
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_averageColor"] = float("nan")
    # try:
    #     record["feat_averageRedness"] = features.get_avg_max_redness(image, image_maskG)
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     record["feat_averageRedness"] = float("nan")
    
    a = features.color_metrics(image, image_maskG)
    try:
        record["feat_maxBrightness"] = a["max_brightness"]
    except Exception as e:
        print(f"ERROR: {e}")
        record["feat_maxBrightness"] = float("nan")
    try:
        record["feat_minBrightness"] = a["min_brightness"]
    except Exception as e:
        print(f"ERROR: {e}")
        record["feat_minBrightness"] = float("nan")

    
    return record


def addFeatures(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame.apply(extractFeaturesFromImage, axis=1)


def loadDataFrameWithFeatures(
        write_csv_to: str | None = "result/featuresEXTERNAL.csv"
,
        truncate: int | None = None,
        start: int | None = None) -> pd.DataFrame:
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
    DF  = pd.read_csv(p("result/featuresEXTERNAL.csv"))

    # limit data in data frame only to images which have a mask
    # masked_images = pd.DataFrame(data = {
    #     "img_id": [x.replace("_mask.png", ".png") for x in os.listdir(MASKS_DIR)]
    # })

    #DF = pd.merge(DF, masked_images, on='img_id', how='inner')

    if start is not None or truncate is not None:
        start = start or 0  # default to 0 if None
        end = start + truncate if truncate is not None else None
        DF = DF.iloc[start:end]
    
    DF = addFeatures(DF)

    # write to a .csv file if specified
    if isinstance(write_csv_to, str):
        DF.to_csv(p(write_csv_to), index=False)

    return DF


if __name__ == "__main__":
    DF = loadDataFrameWithFeatures(write_csv_to="result/featuresEXTERNAL2.csv")
