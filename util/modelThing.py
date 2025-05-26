from typing import Callable, List
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

metadata = pd.read_csv("C:/Users/hjalt/Desktop/Uni/projects data/metadata.csv")
imagepath = "C:/Users/hjalt/Desktop/Uni/projects data/images"
groups = metadata["patient_id"].unique()
metadata['biopsed'] = metadata['biopsed'].astype(int)
metadata = metadata.dropna()
y_all = metadata["biopsed"]

# add 
def addFeatures(features: list[Callable[[str], float]] ) -> pd.DataFrame:
    newDF = metadata.copy()
    for feature in features:
        col_name = feature.__name__
        newDF[f"feat_{col_name}"] = newDF["img_id"].apply(feature)
        
    return newDF
#testing
functions = []

df = addFeatures(functions)


baseline_feats = [col for col in df.columns if col.startswith("feat_")]
x_all = df[baseline_feats]
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.3, random_state=42)
gkf = GroupKFold(5)
model = KNeighborsClassifier(random_state=42,n_neighbors=5)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Test Accuracy:", acc)
print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    def filename_length(img_id: str) -> int:
        return len(img_id)

    def is_jpeg(img_id: str) -> int:
        return int(img_id.lower().endswith('.jpg') or img_id.lower().endswith('.jpeg'))

    def count_digits(img_id: str) -> int:
        return sum(char.isdigit() for char in img_id)
    features = [filename_length, is_jpeg, count_digits]
    a = addFeatures(features)
    print(a)





