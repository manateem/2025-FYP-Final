import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupKFold, GridSearchCV, cross_val_score
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

selected_numeric_cols = ['feat_hair', 'feat_asymmetry', 'feat_border_irregularity', 'feat_color','feat_convexity']
df[selected_numeric_cols] = df[selected_numeric_cols].apply(lambda x: x.fillna(x.median()))
df['biopsed'] = df['biopsed'].astype(int)

unique_patients = {pid: i for i, pid in enumerate(df["patient_id"].unique())}
df["group"] = df["patient_id"].map(unique_patients)

X = df[selected_numeric_cols].copy()
y = df['biopsed']
groups = df["group"].values

X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42, stratify=y
)

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

group_kf = GroupKFold(n_splits=5)

# Logistic Regression
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
param_grid_lr = {
    'clf__C': [0.1, 1, 10],
    'clf__penalty': ['l1', 'l2'],
    'clf__solver': ['liblinear']
}
grid_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=group_kf, scoring='accuracy', n_jobs=-1)
grid_lr.fit(X_train, y_train, groups=groups_train)

# Decision Tree
pipeline_dt = Pipeline([
    ('scaler', StandardScaler()),  # Not needed, but kept for uniformity
    ('clf', DecisionTreeClassifier(random_state=42))
])
param_grid_dt = {
    'clf__max_depth': [3, 5],
    'clf__min_samples_split': [2, 5],
    'clf__criterion': ['gini', 'entropy']
}
grid_dt = GridSearchCV(pipeline_dt, param_grid_dt, cv=group_kf, scoring='accuracy', n_jobs=-1)
grid_dt.fit(X_train, y_train, groups=groups_train)

# KNN
pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KNeighborsClassifier())
])
param_grid_knn = {
    'clf__n_neighbors': [3, 5],
    'clf__weights': ['uniform', 'distance']
}
grid_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=group_kf, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train, y_train, groups=groups_train)

best_lr = grid_lr.best_estimator_
best_dt = grid_dt.best_estimator_
best_knn = grid_knn.best_estimator_

models = {
    "Logistic Regression": best_lr,
    "Decision Tree": best_dt,
    "K-Nearest Neighbors": best_knn
}

cv_results = {}

# Evaluation loop
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=1):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=1):.2f}")
    print(f"AUC: {roc_auc_score(y_test, y_proba):.2f}")

    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_proba, model_name)

    if model_name == "Decision Tree":
        plot_feature_importance(model, X.columns)

    cv_scores = cross_val_score(model, X_train, y_train, cv=group_kf, groups=groups_train, scoring='accuracy')
    cv_results[model_name] = cv_scores

# Cross-validation boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=pd.DataFrame(cv_results))
plt.title("Cross-Validation Accuracy (GroupKFold)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
