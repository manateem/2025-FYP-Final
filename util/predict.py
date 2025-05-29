import joblib
from constants import p
import pandas as pd
import pprint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    accuracy_score,
    precision_score
)


def load_model(path_to_model_pkl):
    return joblib.load(path_to_model_pkl)


def get_model_performance_on_extern_data(
        path_to_model_pkl: str,
        model_name: str,
        feature_columns: list[str],
        scale: bool = True):
    """
    Given a model, print the model's performance statistics
    on an external dataset.

    :param path_to_model_pkl: Path to model pickle file
    :param model_name: Name of the model
    :param feature_columns: List of features to make predictions
    based on. MUST be the same as the features used to train the
    model.
    :param scale: Whether to scale the data. Should only be set to
    `False` for decisiont tree models.
    """
    
    DF = pd.read_csv(p("result/featuresEXTERNAL.csv"))
    model = load_model(p(path_to_model_pkl))


    X = DF[feature_columns]

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    #print(X)
    Y = DF["BIOPSED"]

    Y_predicted = model.predict(X)

    print(f"----\t{model_name}\t----")
    print("Confusion matrix:")
    pprint.pprint(confusion_matrix(Y, Y_predicted))
    
    print(f"Accuracy: {accuracy_score(Y, Y_predicted)}")
    print(f"Recall: {recall_score(Y, Y_predicted)}")
    print(f"Precision: {precision_score(Y, Y_predicted)}")


if __name__ == "__main__":
    get_model_performance_on_extern_data(
        path_to_model_pkl="result/models/1_ABC_Classifiers/KNN/KNN.pkl",
        model_name="KNN Classifier - ABC Features",
        feature_columns=[
            "feat_asymmetry",
            "feat_border_irregularity",
            "feat_multiColorRate"
        ]
    )
