# Manateem Remarks
This is how our project structure looks like and how you should structure the data folder yourself.
```
2025-FYP-group/
├── data/                               # please structure your data folder as so
│   ├── images/
│   │   ├── PAT_8_15_820.png
│   │   ......
│   │   └── PAT_XXX_XXX_XXX.png
│   ├── masks/
│   │   ├── PAT_8_15_820_mask.png
│   │   ......
│   │   └── PAT_XXX_XXX_XXX_mask.png
│   ├── noHair/                         # this will be created for you
│   │   ├── PAT_8_15_820.png
│   │   ......
│   │   └── PAT_XXX_XXX_XXX.png
│   ├── metadata.csv
│ 
├── util/
│   ├── __init__.py             # starting place for TA's :)
│   ├── img_util.py             # basic image read and write functions
│   ├── data_binarizer.py       # changes true and false values to 0 and 1 for dataframes
│   ├── getFeaturesEXTERNAL.py  # extracts feature data specifically for external dataset
│   ├── GroupKfold.py           # used for performance analysis of features of megaclassifier
│   ├── plotter.py              # used for creation of plots for project
│   ├── predict.py              # used for analyzing model testing of external data
│   ├── progress_bar.py         # visual aid for ETA of feature extraction completion
│   ├── train_model.py          # model training
│   ├── feature_extract_ex.py   # code for all the features that we use
│   ├── utility.py              # a variety of useful utility files for processing
│   ├── plotter.py              # script for plot creation
│   ├── extractor.py            # extracts feature data from all images, 
│   |                           # allowing for reproducibility of our results
│   ├── visualize_models.ipynb  # used to visualize the results from the model after processing
│   ......
│   └── constants.py            # ensures that file paths are universal between machines
│ 
├── result/
│   ├── features_binarized.csv  # features for mega-classifier (metadata and image features)
│   ├── features.csv            # features for general classifier use
│   ├── featuresEXTERNAL.csv    # features of external data
│   ├── logbook.md              # logbook of major decisions, unfortunately not well used
│   ├── todo.md                 # to-do reminders for the group, unfortunately not too well used
│   ├── models                  # models, where you can find their .pkl file and results/metrics of each model
│   │   ├── 1_ABC_Classifiers
│   │   ├── 2_MegaClassifier
│   │   ├── 3_UltraClassifiers
│   │   ├── 4_finalModel
│   │   └── model_performance.md  # a summary of performance metrics of all models
│   ├── Plots                   # used for report
│   │   ├── ...
│   └── report.pdf      		# our report in PDF
│ 
├── main_demo.py		# demo script (reference setup, remove in your hand-in)
├── main_baseline.py	# complete script (baseline setup)
├── main_extended.py	# complete script (extended setup)
├── dataset.csv    		# all image file names, ground-truth labels, and chosen features
└── README.md
```
## Language and Definitions
Whenever we refer to a particular 'classifier', we refer mainly to the model using a certain set of features. This is why we have folders titled as 'classifiers' under the `models/` folder. Each 'classifier' is infact a group of 3 different trained classifiers, logistic regression, decision tree, and KNN.
The "ABC Classifier" is the group of classifiers utilizing only 3 features, aka our baseline
The "Mega-classifier" is the group of classifiers utilizing only 16 features, which is compatible with the external dataset used as it only uses image features, while still maximizing how many features it has.
The "Ultra-classifier" is the group of classifiers utilizing 33 features, including both metadata and image features, but is unfortunately incompatible with our external dataset.
The "final/reduced model" is the model (and group of classifiers) using 4 features that were selected after analyzing performance metrics from ONLY GroupKFold.

# Projects in Data Science (2025)
## Overview
### result/logbook.md
This is where we put big decisions we've made as a group to arrive at our final product, which we can then use as reminders to write up the final report.

### result/todo.md
This is where we try to split up this huge project into multiple tasks so that we can individually go and perform these tasks if they are not done yet.


## Final Assignment
#### Overview

This is a template repository for the final assignment of course "Projects in Data Science." You should refer to this repository in your final hand-in.

If using github.itu.dk, you need to download the repository and make your own. 

If you are using general Github, you can clone or fork the repository directly. If your usernames do not give sufficient hints as to who you are, you can tell the TAs how to match them. 

Your repository MUST be named 2025-FYP-groupXX where XX is your group number. 

Look at the slides of the previous two weeks for details of the hand-in. 



#### Python environment

Follow TA instructions when setting up the Python environment before running any code. Remember to export your Python library requirements by `pip freeze > requirements.txt` and attach it to the repo so we can evaluate your scripts.



#### File Hierarchy

The file hierarchy of your hand-in repo should be as follows:

```
2025-FYP/
├── data/               # unzip the dataset and put it here (remove in your hand-in)
│   ├── img_001.jpg
│   ......
│   └── img_XXX.jpg
│ 
├── util/
│   ├── __init__.py
│   ├── img_util.py     # basic image read and write functions
│   ├── inpaint.py      # image inpainting function
│   ├── feature_A.py    # code for feature A extraction
│   ├── feature_B.py    # code for feature B extraction
│   ├── feature_C.py    # code for feature C extraction
│   ......
│   └── classifier.py   # code for training, validating, and testing the classifier
│ 
├── result/
│   ├── result_baseline.csv      # your results on the baseline setup
│   ├── result_extended.csv      # your results on the extended setup
│   └── report.pdf      		 # your report in PDF
│ 
├── main_demo.py		# demo script (reference setup, remove in your hand-in)
├── main_baseline.py	# complete script (baseline setup)
├── main_extended.py	# complete script (extended setup)
├── dataset.csv    		# all image file names, ground-truth labels, and chosen features
└── README.md
```



**Notes:**

1. DO NOT upload your data (images) to Github.
2. When the same code block needs to be executed multiple times in the script, make it a custom function instead. All the custom functions and modules, such as image read and write, should be grouped into different files under the *"util"* subfolder, based on the task they are designed for. Do not put everything in a single Python file or copy-paste the same code block across the script.







