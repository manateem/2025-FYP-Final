import joblib
from constants import p
import pandas as pd
import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


def load_model(path_to_model_pkl):
    return joblib.load(path_to_model_pkl)


if __name__ == "__main__":
    DF = pd.read_csv(p("result/featuresEXTERNAL.csv"))
    knn_model = load_model(p("result/models/1_ABC_Classifiers/KNN/KNN.pkl"))
    # DF["feat_asymmetry"] = pd.to_numeric(DF["feat_asymmetry"], errors="coerce")

    feature_columns = [
        "feat_asymmetry",
        "feat_border_irregularity",
        "feat_multiColorRate"
    ]

    X = DF[feature_columns]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #print(X)
    Y = DF["BIOPSED"]

    Y_predicted = knn_model.predict_proba(X)

    pprint.pprint(Y_predicted)
    pprint.pprint(Y)
    print()

