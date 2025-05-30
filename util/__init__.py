import img_util
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utility
import feature_extract_ex
import extractor
import train_model
import plotter

if __name__ == "__main__":
    # Creates a folder of inpainted hair images.
    # Extracts all features, and saves them to data/features.csv
    extractor.extractAllFeatures(forceHairFeature=True)
    # Trains (and tests) all 4 defined models and puts them in their respective folders in results/models/
    train_model.train_all_models()
    # Generate plots used for report
    plotter.generate_all_plots()    
