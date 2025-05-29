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
from matplotlib import pyplot as plt
from matplotlib import colormaps as plt_cm
import numpy as np
from sklearn import set_config as sklearn_set_config


def load_model(path_to_model_pkl):
    return joblib.load(path_to_model_pkl)


def get_model_performance_on_extern_data(
        path_to_model_pkl: str,
        model_name: str,
        feature_columns: list[str] | None,
        scale: bool = True):
    """
    Given a model, print the model's performance statistics
    on the external dataset.

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
        feature_columns = [feature for feature in DF if feature.startswith("feat_")]  # type: ignore


    # if scale:
    #     X = DF[feature_columns]
    #     scaler = StandardScaler()
    #     X = scaler.fit_transform(X)
    # else:
    #     X = DF[feature_columns]
    X = DF[feature_columns]
    if scale:
        print(f"{model_name}: scaling...")
        X = DF[feature_columns]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        if isinstance(X, pd.DataFrame):
            X = X.values
        print(f"Will use features: {feature_columns}")
    else:
        print(f"{model_name}: not scaling")
        X = X.values
        print(f"Will use features: {feature_columns}")

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
        "name": model_name,
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }


if __name__ == "__main__":
    sklearn_set_config(transform_output="pandas")

    models = [
        ("KNN Classifier - ABC Features", "result/models/1_ABC_Classifiers/KNN/KNN.pkl"),
        ("Decision Tree Classifier - ABC Features", "result/models/1_ABC_Classifiers/decision_tree/DecisionTree.pkl"),
        ("Logistic Regression Classifier - ABC Features", "result/models/1_ABC_Classifiers/logistic_regression/LogisticRegression.pkl"),

        ("KNN Classifier - Every image feature", "result/models/2_MegaClassifier/KNN/KNN.pkl"),
        ("Decision Tree Classifier - Every image feature", "result/models/2_MegaClassifier/decision_tree/DecisionTree.pkl"),
        ("Logistic Regression Classifier - Every image feature", "result/models/2_MegaClassifier/logistic_regression/LogisticRegression.pkl"),
    ]

    model_perfomance_analyses = []

    for model_name, model_file_path in models:
        scale = "Decision Tree" not in model_name

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
        
        model_perfomance_analyses.append(get_model_performance_on_extern_data(
            path_to_model_pkl=model_file_path,
            model_name=model_name,
            feature_columns=feature_columns,
            scale=scale
        ))
    
    fig, axs = plt.subplots(nrows=2, ncols=3)
    for i, model in enumerate(model_perfomance_analyses):
        row_idx = i // 3
        col_idx = i % 3

        subplot = axs[row_idx, col_idx]

        conf_matrix = np.array(model["confusion_matrix"])
        model_name = model["name"]

        subplot.set_title(model_name)

        # subplot.figure(figsize=(4, 4))
        im = subplot.imshow(conf_matrix, cmap=plt_cm["summer"])
        # pprint.pprint(conf_matrix)

        # print(f"---- {model_name} ({model_feat_count} features) -----")
        # print("Confusion matrix:")
        # pprint.pprint(conf_matrix)

        xlabels = ["False", "True"]
        ylabels = ["False", "True"]

        subplot.set_xlabel("Predicted")
        subplot.set_ylabel("Actual")

        subplot.set_xticks([0, 1], labels=xlabels)
        subplot.set_yticks([0, 1], labels=ylabels,
                            rotation=90, ha="right", rotation_mode="anchor")
        
        subplot.text(0, 0, conf_matrix[0, 0],
                    ha="center", va="center", color='r')
        subplot.text(0, 1, conf_matrix[1, 0],
                    ha="center", va="center", color='r')
        subplot.text(1, 0, conf_matrix[0, 1],
                    ha="center", va="center", color='r')
        subplot.text(1, 1, conf_matrix[1, 1],
                    ha="center", va="center", color='r')
    
    fig.suptitle("Confusion matrices of models on external dataset")

    plt.show()
