from typing import Callable, List
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import GroupKFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from SaveModel import saveModel
from hairFeature import amountOfHairFeature
from constants import p
import random

MODEL_NAME = "model.pkl"
MODEL_DIR = p("testmodels/")


DF  = pd.read_csv(p("data/metadata.csv"))

TRAINING_IMAGES_DIR = p("data/training")

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
        - [ ] Extract feature "hair"
        - [ ] Save old hairy image, override current file with the one with hair removed
        - [ ] Extract feature "asymmetry"
        - [ ] Extract feature "border"
        - [ ] Extract feature "color"
    """
    record["feat_test"] = random.random()

    return record


def addFeatures(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame.apply(extractFeaturesFromImage, axis=1)



# # add 
# def addFeatures(features: list[Callable[[str], float]] ) -> pd.DataFrame:
#     newDF = DF.copy()
#     for feature in features:
#         col_name = feature.__name__
#         newDF[f"feat_{col_name}"] = newDF["img_id"].apply(feature)
        
#     return newDF

# #testing
# def test(i):
#     return np.random.rand()
# functions = [test] # FUNCTIONS HERE

df = addFeatures(DF)
print(df)

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
print(MODEL_DIR)





