from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_score
from sklearn.model_selection import GroupKFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize, StandardScaler
from utility import saveModel
from constants import p
import pandas as pd
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray
import json
import os

KNN_MODEL_DIR = p("models/KNN")
DTREE_MODEL_DIR = p("models/decision_tree")
LR_MODEL_DIR = p("models/logistic_regression")


@dataclass
class ModelData:
    name: str
    model: Any
    confusion_matrix: NDArray
    false_positive_rate: NDArray
    true_positive_rate: NDArray
    area_under_curve: float
    features: list[str]
    feature_importances: NDArray | None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None

    @property
    def num_features(self):
        return len(self.features)

    def to_json(self):
        return json.dumps({
            "name": self.name,
            "numFeatures": self.num_features,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "confusionMatrix": self.confusion_matrix.tolist(),
            "falsePositiveRate": self.false_positive_rate.tolist(),
            "truePositiveRate": self.true_positive_rate.tolist(),
            "areaUnderCurve": self.area_under_curve,
            "features": self.features,
            "featureImportances": self.feature_importances.tolist() if self.feature_importances is not None else None,
        }, indent=4)


def train_knn_model(x_train, x_test, y_train, y_test, features, n_neighbors = 5) -> ModelData:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(x_train, y_train)

    y_pred = knn_model.predict(x_test)
    y_proba = knn_model.predict_proba(x_test)[:, 1]

    knn_confusion_matrix = confusion_matrix(y_test, y_pred)
    knn_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    return ModelData(
        name="K-Nearest-Neighbors classifier",
        model=knn_model,
        accuracy=knn_accuracy,
        confusion_matrix=knn_confusion_matrix,
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate,
        area_under_curve=roc_auc,
        features=features,
        feature_importances=None,
        precision=precision,
        recall=recall
    )


def train_decision_tree(x_train, x_test, y_train, y_test, features):
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model = decision_tree_model.fit(x_train, y_train)

    y_pred = decision_tree_model.predict(x_test)
    y_proba = decision_tree_model.predict_proba(x_test)[:, 1]

    tree_confusion_matrix = confusion_matrix(y_test, y_pred)
    tree_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    feature_importances = decision_tree_model.feature_importances_

    return ModelData(
        name="Decision Tree classifier",
        model=decision_tree_model,
        accuracy=tree_accuracy,
        confusion_matrix=tree_confusion_matrix,
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate,
        area_under_curve=roc_auc,
        features=features,
        feature_importances=feature_importances,
        precision=precision,
        recall=recall
    )


def train_logistic_regression(x_train, x_test, y_train, y_test, features):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    logistic_model = LogisticRegression(solver="liblinear")
    logistic_model.fit(x_train, y_train)

    y_pred = logistic_model.predict(x_test)
    y_proba = logistic_model.predict_proba(x_test)[:, 1]

    logistic_confusion_matrix = confusion_matrix(y_test, y_pred)
    logistic_accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    return ModelData(
        name="Logistic Regression model",
        model=logistic_model,
        accuracy=logistic_accuracy,
        confusion_matrix=logistic_confusion_matrix,
        false_positive_rate=false_positive_rate,
        true_positive_rate=true_positive_rate,
        area_under_curve=roc_auc,
        features=features,
        feature_importances=None,
        precision=precision,
        recall=recall
    )


def train_models(
        data_frame: pd.DataFrame,
        features: list[str],
        test_size: float = 0.2,
        knn_n_neighbors: int = 5,
        save_to_directory: str = "result/models/",
        save_knn_model_to: str = "KNN",
        save_tree_model_to: str = "DecisionTree",
        save_logistic_model_to: str = "LogisticRegression"):
    save_to_directory = p(save_to_directory)

    if not os.path.exists(save_to_directory):
        os.makedirs(save_to_directory)

    print(
        "Amount of True vs False biopsed:",
        len(DF[DF["biopsed"] == True]),
        "false:",
        len(DF[DF["biopsed"] == False])
    )

    knn_model_dir = os.path.join(save_to_directory, "KNN/")
    tree_model_dir = os.path.join(save_to_directory, "decision_tree/")
    lr_model_dir = os.path.join(save_to_directory, "logistic_regression/")

    for folder in [knn_model_dir, tree_model_dir, lr_model_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    x_all = data_frame[features]
    y_all = data_frame["biopsed"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all,
        test_size=test_size, random_state=42
    )

    knn_model_data = train_knn_model(
        x_train, x_test,
        y_train, y_test,
        features
    )

    saveModel(knn_model_data.model, save_knn_model_to, knn_model_dir)
    with open(os.path.join(knn_model_dir, "metrics.json"), "w", encoding="utf-8") as model_metrics_file:
        model_metrics_file.write(knn_model_data.to_json())

    decision_tree_model_data = train_decision_tree(
        x_train, x_test,
        y_train, y_test,
        features
    )

    saveModel(decision_tree_model_data.model, save_tree_model_to, tree_model_dir)
    with open(os.path.join(tree_model_dir, "metrics.json"), "w", encoding="utf-8") as model_metrics_file:
        model_metrics_file.write(decision_tree_model_data.to_json())

    logistic_model_data = train_logistic_regression(
        x_train, x_test, 
        y_train, y_test,
        features
    )
    saveModel(logistic_model_data.model, save_logistic_model_to, lr_model_dir)
    with open(os.path.join(lr_model_dir, "metrics.json"), "w", encoding="utf-8") as model_metrics_file:
        model_metrics_file.write(logistic_model_data.to_json())

    return (
        knn_model_data,
        decision_tree_model_data,
        logistic_model_data
    )



if __name__ == "__main__":
    DF = pd.read_csv(p("result/dataset3.csv"))
    feature_columns = [col for col in DF.columns if col.startswith("feat_")]
    print(feature_columns)
    DF = DF.dropna(subset=feature_columns)

    _ = train_models(
        DF, features=["feat_compactness", "feat_multicolor"],
        save_to_directory="result/models/2features"
    )

    _ = train_models(
        DF, features=["feat_asymmetry", "feat_compactness", "feat_multicolor"],
        save_to_directory="result/models/3features"
    )

    _ = train_models(
        DF, features=["feat_hair", "feat_asymmetry", "feat_compactness", "feat_multicolor"],
        save_to_directory="result/models/4features"
    )

    # print(knn_confusion_matrix)
    # print(decision_tree_confusion_matrix)
    # print(log_conf_matrix)