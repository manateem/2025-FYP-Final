"""The purpose of this python file, is to utilize existing feature extractors from feature_extract_ex.py to output valid data csv files."""
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
NOHAIR_DIR = p("data/noHairTEST/")
METADATA_FP = p("data/metadata.csv")

REL_OUTPUT_CSV_FP = "result/featuresTEST.csv"
OUTPUT_CSV_FP = p(REL_OUTPUT_CSV_FP)

def hairExtractFeature(record):
    """Unique to extracting hair and outputting a unique hairless image to data/noHair/"""
    shouldExtract = False
    try:
        record["feat_hair"] # assumes NaN also means its been checked before with a bad result
        shouldExtract = False
        print("[INFO] Feature value already exists!")
    except KeyError:
        shouldExtract = True
        # print("Value doesn't already exist!")
    
    progress_bar.render(len(output_csv_df),record.name)
    
    # Set file path
    image_fp = os.path.join(IMAGES_DIR, record["img_id"])

    # Read the image
    if shouldExtract:
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
    if shouldExtract:
        _, _, img_out = utility.removeHair(image, image_gray)
        noHairOutputPath = os.path.join(NOHAIR_DIR, record["img_id"])
        if not os.path.isfile(noHairOutputPath):
            print(f"[INFO] Creating image {noHairOutputPath}...")
            cv2.imwrite(noHairOutputPath,img_out)
        else:
            print(f"[INFO] Image {noHairOutputPath} already exists!")

    return record

def extractAllFeatures(record):
    """Extract the remaining features, utilizing the new no hair images."""
    # Progress bar print
    progress_bar.render(len(output_csv_df),record.name)
    # Set file path
    image_fp = os.path.join(NOHAIR_DIR, record["img_id"])

    # Checks if a value is already filled out
    try:
        record["feat_hair"] # assumes NaN also means its been checked before with a bad result
        shouldExtract = False
        print("[INFO] Feature value already exists!")
    except KeyError:
        shouldExtract = True
        # print("Value doesn't already exist!")

    # Read the image
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

    # Extract the hair feature (and skip if this field is already filled)
    if shouldExtract:
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

    return record

def saveDataFrame(df, fp):
    df.to_csv(fp, index=False)

if __name__ == "__main__":
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
    sleep(1)
    print("...")
    sleep(1)
    print("...")
    sleep(1)
    output_csv_df = output_csv_df.apply(hairExtractFeature, axis=1)
    output_csv_df.to_csv(REL_OUTPUT_CSV_FP, index=False)

    print("quick exit...")
    exit()
    # Then use these new noHair images to extract the rest of our features.
    print("...")
    sleep(1)
    print("...")
    sleep(1)
    print("...")
    sleep(1)
    output_csv_df = output_csv_df.apply(extractAllFeatures, axis=1)
    output_csv_df.to_csv(REL_OUTPUT_CSV_FP, index=False)