import matplotlib.pyplot as plt
from matplotlib import colormaps as plt_cm
from constants import p
import os
import json
from typing import Any
import pprint
import math
import numpy as np

MODELS_DIR = p("result/models/")
PLOTS_DIR = p("result/plots/")

NUM_PLOT_COLUMNS = 3

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)


# load models' statistics
def model_stats_files() -> list[str]:
    json_file_paths: list[str] = []
    for root, _, files in os.walk(MODELS_DIR):
        for file in files:
            if file.endswith("metrics.json"):
                json_file_paths.append(
                    os.path.join(root, file)
                )
    
    return json_file_paths


def get_model_data(json_file_paths: list[str]) -> list[dict[str, Any]]:
    model_data: list[dict[str, Any]] = []

    for json_file_path in json_file_paths:
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            model_data.append({
                "directory": os.path.dirname(json_file_path),
                **json_data
            })
    
    return model_data


def generate_plots(model_data: list[dict[str, Any]]):
    NUM_PLOT_ROWS = max(math.ceil(len(model_data) / 3), 2)
    # create confusion matrices
    #figure(figsize=(30, 30))
    fig, axs = plt.subplots(NUM_PLOT_ROWS, NUM_PLOT_COLUMNS, figsize=(15, 5 * NUM_PLOT_ROWS))
    for i, model in enumerate(model_data):
        row_idx = i // NUM_PLOT_COLUMNS
        col_idx = i % NUM_PLOT_COLUMNS
        subplot = axs[row_idx, col_idx]

        conf_matrix = np.array(model["confusionMatrix"])
        model_name = model["name"]
        model_feat_count = model["numFeatures"]

        subplot.set_title(f"{model_name} ({model_feat_count} features)")

        # subplot.figure(figsize=(4, 4))
        im = subplot.imshow(conf_matrix, cmap=plt_cm["summer"])
        pprint.pprint(conf_matrix)

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
    
    fig.suptitle("Confusion matrices")

    # plt.show()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"))


    # plot the ROC curves
    fig, axs = plt.subplots(NUM_PLOT_ROWS, NUM_PLOT_COLUMNS, figsize=(18, 5 * NUM_PLOT_ROWS))
    for i, model in enumerate(model_data):
        row_idx = i // NUM_PLOT_COLUMNS
        col_idx = i % NUM_PLOT_COLUMNS
        subplot = axs[row_idx, col_idx]

        fpr = model["falsePositiveRate"]
        tpr = model["truePositiveRate"]
        roc_auc = model["areaUnderCurve"]
        model_name = model["name"]
        num_features = model["numFeatures"]

        subplot.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        subplot.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        subplot.set_xlabel('False Positive Rate')
        subplot.set_ylabel('True Positive Rate')
        subplot.set_title(f"{model_name} - ROC Curve ({num_features} features)")
        subplot.legend(loc="lower right")
        subplot.grid(True)

    fig.suptitle("ROC curves")

    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"))


    # plot feature importances
    decision_tree_models = [model for model in model_data if model["featureImportances"] is not None]
    # pprint.pprint(decision_tree_models)
    ft_importance_plot_rows = max(2, math.ceil(len(decision_tree_models) / 2))
    fig, axs = plt.subplots(ft_importance_plot_rows, 2, figsize=(15, 5 * ft_importance_plot_rows))
    for i, model in enumerate(decision_tree_models):
        row_idx = i // 2
        col_idx = i % 2
        subplot = axs[row_idx, col_idx]

        features = model["features"]
        feature_importances = model["featureImportances"]
        model_name = model["name"]
        num_features = model["numFeatures"]

        subplot.bar(x=features, height=feature_importances)
        subplot.set_title(f"{model_name} ({num_features} features)")
        subplot.set_xlabel("Feature")
        subplot.set_ylabel("Gini Importance")
        subplot.grid(True)

    fig.suptitle("Feature importances for decision trees")

    plt.savefig(os.path.join(PLOTS_DIR, "feature_importances.png"))


if __name__ == "__main__":
    generate_plots(get_model_data(model_stats_files()))
