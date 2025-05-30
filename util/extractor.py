"""The purpose of this python file, is to utilize existing feature extractors from feature_extract_ex.py to output valid data csv files."""
"""At every iteration of feature extraction, a copy will be saved as to prevent significant time waste."""
import feature_extract_ex as featurex
from constants import p
import utility
import os
import pandas as pd
import numpy as np
import progress_bar
import cv2
from time import sleep

DATA_DIR = p("data/")
IMAGES_DIR = p("data/images/")
MASKS_DIR = p("data/masks/")
NOHAIR_DIR = p("data/noHair/")
METADATA_FP = p("data/metadata.csv")

REL_OUTPUT_CSV_FP = "result/features.csv"
OUTPUT_CSV_FP = p(REL_OUTPUT_CSV_FP)

def hairExtractFeature(record, output_csv_df, forceHairFeature=False):
    """Unique to extracting hair and outputting a unique hairless image to data/noHair/"""
    shouldExtract = False
    try:
        record["feat_hair"] # assumes NaN also means its been checked before with a bad result
        shouldExtract = False
        print("[INFO] Feature value already exists!")
    except KeyError:
        shouldExtract = True
        # print("Value doesn't already exist!")
    
    for i in range(8):
        print("")
    print(f"Current feature extraction: Hair Amount")
    progress_bar.render(len(output_csv_df),record.name)
    
    # Set file path
    image_fp = os.path.join(IMAGES_DIR, record["img_id"])

    # Read the image
    if shouldExtract or forceHairFeature:
        print(f"[INFO] Reading image {image_fp}...")
        try:
            image = cv2.imread(image_fp)
        except Exception as e:
            print(f"[ERROR] Failed to read {image_fp}; Chosen behavior: exit(); Reason: {e}")
            exit()
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Read the mask
        image_mask_filename = record["img_id"].replace(".png", "_mask.png")
        image_mask_path = os.path.join(MASKS_DIR, image_mask_filename)
        image_mask = cv2.imread(image_mask_path)
        
        feature = None
        try:
            hair_extraction = featurex.HairExtractor(image_mask)
            feature = hair_extraction.amountOfHairFeature()
            record.at["feat_hair"] = feature
        except Exception as e:
            print(f"ERROR: {e}")
            record.at["feat_hair"] = float("nan")

    # Every 100 images, save a copy to prevent wasted work
    if int(record.name) % 100 == 0:
        output_csv_df.to_csv(REL_OUTPUT_CSV_FP, index=False)

    # Save the inpainted image
    if shouldExtract or forceHairFeature:
        _, _, img_out = utility.removeHair(image, image_gray)
        noHairOutputPath = os.path.join(NOHAIR_DIR, record["img_id"])
        if not os.path.isfile(noHairOutputPath):
            print(f"[INFO] Creating image {noHairOutputPath}...")
            cv2.imwrite(noHairOutputPath,img_out)
        else:
            print(f"[INFO] Image {noHairOutputPath} already exists!")

    return record

def extractFeature(record, options, output_csv_df):
    """Extract the feature, utilizing the new no hair images."""
    # Verify options
    if options.get("feat_name") == None: # should be a string
        print("Invalid input 1!")
        exit()
    if options.get("feature_fn") == None: # should be a function
        print("Invalid input 2!")
        exit()
    if options.get("arguments") == None: # should be a dictionary
        print("Invalid input 3!")
        exit()
    feat_name = options.get("feat_name")
    feature_fn = options.get("feature_fn")
    given_args = options.get("arguments")
    subscript = options.get("subscript")

    # Progress bar print
    for i in range(8):
        print("")
    print(f"Current feature extraction: {feat_name}")
    progress_bar.render(len(output_csv_df),record.name)
    # Set file path
    image_fp = os.path.join(NOHAIR_DIR, record["img_id"])

    # Checks if a value is already filled out
    try:
        record[feat_name] # assumes NaN also means its been checked before with a bad result
        shouldExtract = False
        print("[INFO] Feature value already exists!")
    except KeyError:
        shouldExtract = True
        # print("Value doesn't already exist!")

    # Read the image
    if shouldExtract:
        print(f"[INFO] Reading image {image_fp}...")
        try:
            image = cv2.imread(image_fp)
        except Exception as e:
            print(f"[ERROR] Failed to read {image_fp}; Chosen behavior: exit(); Reason: {e}")
            exit()

        # Read the mask
        image_mask_filename = record["img_id"].replace(".png", "_mask.png")
        image_mask_path = os.path.join(MASKS_DIR, image_mask_filename)
        image_mask = cv2.imread(image_mask_path)

        for i in range(len(given_args)):
            x = str(given_args[i])
            if x == "image_mask": 
                given_args[i] = image_mask
            if x == "image": 
                given_args[i] = image

        # Extract the feature (and skip if this field is already filled)
        feature = None
        try:
            print(f"WHAT I SHAPPENING: {subscript}")
            if subscript == None:
                record.at[feat_name] = feature_fn(*given_args)
            else:
                record.at[feat_name] = feature_fn(*given_args)[subscript]
        except Exception as e:
            print(f"[ERROR] {e}")
            record.at["feat_hair"] = float("nan")
            sleep(0.1)

    # Every 100 images, save a copy to prevent wasted work // doesn't work...
    if int(record.name) % 100 == 0:
        output_csv_df.to_csv(REL_OUTPUT_CSV_FP, index=False)

    return record

def saveDataFrame(df, fp):
    df.to_csv(fp, index=False)

def extractAllFeatures(forceHairFeature):
    # Create the final features file if not already existing
    if not os.path.isfile(OUTPUT_CSV_FP):
        metadata_df = pd.read_csv(METADATA_FP)
        # Remove from the df any lesions that don't have a mask
        masked_images = pd.DataFrame(data = {
            "img_id": [x.replace("_mask.png", ".png") for x in os.listdir(MASKS_DIR)]
        })
        print(f"[INFO] Features file created in data/ successfully.")
        metadata_df.to_csv(OUTPUT_CSV_FP, index=False)

    # Create hairless images dir if not already existing
    if not os.path.isdir(NOHAIR_DIR):
        try:
            os.mkdir(NOHAIR_DIR)
            print(f"[INFO] Directory '{NOHAIR_DIR}' created successfully.")
        except Exception as e:
            print(f"[ERROR] Unexpected behavior: {e}")
            exit()

    # First, we want to extract hair values, then append to our output csv.
    output_csv_df = pd.read_csv(OUTPUT_CSV_FP)
    print("...")
    sleep(0.5)
    print("...")
    sleep(0.5)
    print("...")
    sleep(0.5)
    output_csv_df = output_csv_df.apply(hairExtractFeature, args=(output_csv_df,forceHairFeature), axis=1)
    output_csv_df.to_csv(REL_OUTPUT_CSV_FP, index=False)

    # List of features that will be iterated through
    feature_list = [
        {
            "feat_name": "feat_asymmetry",
            "feature_fn": featurex.rotation_asymmetry,
            "arguments": ["image_mask", 5],
            "subscript": "average"
        },{
            "feat_name": "feat_border_irregularity",
            "feature_fn": featurex.get_compactness,
            "arguments": ["image_mask", 2],
            "subscript": "score"
        },{
            "feat_name": "feat_multiColorRate",
            "feature_fn": featurex.get_multicolor_rate,
            "arguments": ["image", "image_mask", 2],
        },{
            "feat_name": "feat_convexity",
            "feature_fn": featurex.convexity_score,
            "arguments": ["image_mask"],
        },{
            "feat_name": "feat_maxBrightness",
            "feature_fn": featurex.color_metrics,
            "arguments": ["image","image_mask"],
            "subscript": "max_brightness"
        },{
            "feat_name": "feat_minBrightness",
            "feature_fn": featurex.color_metrics,
            "arguments": ["image","image_mask"],
            "subscript": "min_brightness"
        },{
            "feat_name": "feat_convexVariance",
            "feature_fn": featurex.convexity_metrics,
            "arguments": ["image_mask"],
            "subscript": "variance"
        },{
            "feat_name": "feat_convexMax",
            "feature_fn": featurex.convexity_metrics,
            "arguments": ["image_mask"],
            "subscript": "max"
        },{
            "feat_name": "feat_convexAverage",
            "feature_fn": featurex.convexity_metrics,
            "arguments": ["image_mask"],
            "subscript": "average"
        },{
            "feat_name": "feat_contrast",
            "feature_fn": featurex.texture_analysis,
            "arguments": ["image","image_mask"],
            "subscript": "glcm_contrast"
        },{
            "feat_name": "feat_energy",
            "feature_fn": featurex.texture_analysis,
            "arguments": ["image","image_mask"],
            "subscript": "glcm_energy"
        },{
            "feat_name": "feat_homogeneity",
            "feature_fn": featurex.texture_analysis,
            "arguments": ["image","image_mask"],
            "subscript": "glcm_homogeneity"
        },{
            "feat_name": "feat_colorUniformity",
            "feature_fn": featurex.get_color_uniformity,
            "arguments": ["image","image_mask"],
            "subscript": "score"
        },{
            "feat_name": "feat_averageColor",
            "feature_fn": featurex.get_color_uniformity,
            "arguments": ["image","image_mask"],
            "subscript": "average_color"
        },{
            "feat_name": "get_avg_max_redness",
            "feature_fn": featurex.get_color_uniformity,
            "arguments": ["image","image_mask"],
            "subscript": None
        },
    ]
    # Then use these new noHair images to extract the rest of our features.
    for opt in feature_list:
        print("...")
        sleep(0.5)
        print("...")
        sleep(0.5)
        print("...")
        sleep(0.5)
        output_csv_df = output_csv_df.apply(extractFeature, args=(opt,output_csv_df), axis=1)
        output_csv_df.to_csv(REL_OUTPUT_CSV_FP, index=False)

if __name__ == "__main__":
    extractAllFeatures()