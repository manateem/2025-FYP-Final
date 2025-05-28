from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.model_selection import GroupKFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from utility import saveModel
from constants import p
import pandas as pd

KNN_MODEL_DIR = p("models/KNN")
DTREE_MODEL_DIR = p("models/decision_tree")


def train_knn_model(x_train, x_test, y_train, y_test, n_neighbors = 5):
    x_train, x_test = normalize(x_train), normalize(x_test)

    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(x_train, y_train)

    y_pred = knn_model.predict(x_test)

    knn_confusion_matrix = confusion_matrix(y_test, y_pred)

    return knn_model, knn_confusion_matrix


def train_decision_tree(x_train, x_test, y_train, y_test):
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model = decision_tree_model.fit(x_train, y_train)

    y_pred = decision_tree_model.predict(x_test)

    tree_confusion_matrix = confusion_matrix(y_test, y_pred)

    return decision_tree_model, tree_confusion_matrix


def train_models(
        data_frame: pd.DataFrame,
        features: list[str],
        test_size: float = 0.3,
        knn_n_neighbors: int = 5,
        save_knn_model_to: str = "KNN",
        save_tree_model_to: str = "DecisionTree"):
    x_all = data_frame[features]
    y_all = data_frame["biopsed"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all,
        test_size=test_size, random_state=42
    )

    knn_model, knn_confusion_matrix = train_knn_model(
        x_train,
        x_test,
        y_train,
        y_test
    )

    saveModel(knn_model, save_knn_model_to, KNN_MODEL_DIR)

    decision_tree_model, decision_tree_confusion_matrix = train_decision_tree(
        x_train, x_test,
        y_train, y_test
    )

    saveModel(decision_tree_model, save_tree_model_to, DTREE_MODEL_DIR)

    return (knn_model, knn_confusion_matrix), (decision_tree_model, decision_tree_confusion_matrix)



if __name__ == "__main__":
    DF = pd.read_csv(p("result/dataset3.csv"))
    DF.dropna(inplace=True)

    (knn_model, knn_confusion_matrix), (decision_tree_model, decision_tree_confusion_matrix) = train_models(
        DF, features=["feat_hair", "feat_asymmetry", "feat_compactness", "feat_multicolor"]
    )

    print(knn_confusion_matrix)
    print(decision_tree_confusion_matrix)
