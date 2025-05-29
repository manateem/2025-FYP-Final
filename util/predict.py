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
        feature_columns: list[str] | None,
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

    if feature_columns is None:
        feature_columns = [feature for feature in DF if feature.startswith("feat_")]

    X = DF[feature_columns]

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    #print(X)
    Y = DF["BIOPSED"]

    Y_predicted = model.predict(X)

    accuracy = accuracy_score(Y, Y_predicted)
    recall = recall_score(Y, Y_predicted)
    precision = precision_score(Y, Y_predicted)
    conf_matrix = confusion_matrix(Y, Y_predicted)

    print(f"----\t{model_name}\t----")
    print("Confusion matrix:")
    pprint.pprint(conf_matrix)
    
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")

    return {
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }


if __name__ == "__main__":
    models = [
        ("KNN Classifier - ABC Features", "result/models/1_ABC_Classifiers/KNN/KNN.pkl"),
        ("Decision Tree Classifier - ABC Features", "result/models/1_ABC_Classifiers/decision_tree/DecisionTree.pkl"),
        ("Logistic Regression Classifier - ABC Features", "result/models/1_ABC_Classifiers/logistic_regression/LogisticRegression.pkl"),

        ("KNN Classifier - Every image feature", "result/models/2_MegaClassifier/KNN/KNN.pkl"),
        ("Decision Tree Classifier - Every image feature", "result/models/2_MegaClassifier/decision_tree/DecisionTree.pkl"),
        ("Logistic Regression Classifier - Every image feature", "result/models/2_MegaClassifier/logistic_regression/LogisticRegression.pkl"),
    ]



    for model_name, model_file_path in models:
        feature_columns: list[str] | None
        if "ABC" in model_name:
            feature_columns = [
                "feat_asymmetry",
                "feat_border_irregularity",
                "feat_multiColorRate"
            ]
        else:
            # defaults to all features
            feature_columns = None
        
        get_model_performance_on_extern_data(
            path_to_model_pkl=model_file_path,
            model_name=model_name,
            feature_columns=feature_columns
        )
