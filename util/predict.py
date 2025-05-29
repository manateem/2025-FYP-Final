import joblib
from constants import p
import pandas as pd
import pprint
from sklearn.preprocessing import StandardScaler


def load_model(path_to_model_pkl):
    return joblib.load(path_to_model_pkl)


if __name__ == "__main__":
    DF = pd.read_csv(p("result/featuresEXTERNAL.csv"))
    decision_tree_model = load_model(p("result/models/2_MegaClassifier/KNN/KNN.pkl"))
    #print(type(decision_tree_model))
    DF_first_3 = DF.head(7)
    # DF_first_3["feat_asymmetry"] = pd.to_numeric(DF_first_3["feat_asymmetry"], errors="coerce")

    feature_columns = [col for col in DF_first_3.columns if col.startswith("feat_")]
    print(feature_columns)
    X = DF_first_3[feature_columns]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #print(X)
    Y = DF_first_3["BIOPSED"]

    Y_predicted = decision_tree_model.predict_proba(X)

    pprint.pprint(Y_predicted)
    pprint.pprint(Y)
