import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

# Load dataset
df = pd.read_csv(r"C:\Users\valan\OneDrive\Desktop\Projects in D Science\MANDATORY\metadata2.csv")

# Define selected numerical features
selected_numeric_cols = ['feat_asymmetry', 'feat_border_irregularity', 'feat_color']

# Fill missing values using median
df[selected_numeric_cols] = df[selected_numeric_cols].apply(lambda x: x.fillna(x.median()))

# Convert `biopsed` to numeric (1 = True, 0 = False)
df['biopsed'] = df['biopsed'].astype(int)

# Map patient IDs to group indices
unique_patients = {pid: i for i, pid in enumerate(df["patient_id"].unique())}
df["group"] = df["patient_id"].map(unique_patients)

# Define features, target, and group
X = df[selected_numeric_cols].copy()
y = df['biopsed']
groups = df["group"].values

# Split data
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42, stratify=y
)

# Define models using pipelines
models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear'))
    ]),
    "Decision Tree": Pipeline([
        ('scaler', StandardScaler()),  # optional for tree
        ('clf', DecisionTreeClassifier(random_state=42))
    ]),
    "K-Nearest Neighbors": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ])
}

# Visualization functions
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_name} - ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_feature_importance(model, feature_names):
    importances = model.named_steps['clf'].feature_importances_
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances, y=feature_names)
    plt.title("Decision Tree - Feature Importances")
    plt.xlabel("Gini Importance")
    plt.ylabel("Feature")
    plt.grid(True)
    plt.show()

# Train and evaluate models
cv_results = {}
group_kf = GroupKFold(n_splits=5)

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    auc_score = roc_auc_score(y_test, y_proba)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"AUC: {auc_score:.2f}")

    # Plots
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_proba, model_name)

    if model_name == "Decision Tree":
        plot_feature_importance(model, X.columns)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=group_kf, groups=groups_train, scoring='accuracy')
    cv_results[model_name] = cv_scores
    print(f"{model_name} - GroupKFold Mean Accuracy: {cv_scores.mean():.2f}")

# Cross-validation boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=pd.DataFrame(cv_results))
plt.title("Cross-Validation Accuracy (GroupKFold)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
