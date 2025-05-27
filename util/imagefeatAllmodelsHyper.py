import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GroupKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load dataset
df = pd.read_csv(r"C:\Users\valan\OneDrive\Desktop\Projects in D Science\MANDATORY\metadata1.csv")

# Define selected numerical features
selected_numeric_cols = ['feat_hair', 'feat_asymmetry', 'feat_border_irregularity', 'feat_color']

# Fill missing values using MEDIAN
df[selected_numeric_cols] = df[selected_numeric_cols].apply(lambda x: x.fillna(x.median()))

# Convert `biopsed` to numeric (1 = True, 0 = False)
df['biopsed'] = df['biopsed'].astype(int)

# Extract patient groups for GroupKFold
groups = df['patient_id'].values  

# Select only the specified features
X = df[selected_numeric_cols].copy()
y = df['biopsed']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# ✅ Standardizing ONLY on the training set to prevent data leakage
scaler = StandardScaler()
X_train[selected_numeric_cols] = scaler.fit_transform(X_train[selected_numeric_cols])
X_test[selected_numeric_cols] = scaler.transform(X_test[selected_numeric_cols])  # Apply same scaling without refitting

### **Hyperparameter Tuning using GridSearchCV** ###
param_grid_lr = {
    'C': [0.1, 0.5, 1, 5, 10], 
    'penalty': ['l1', 'l2'], 
    'solver': ['liblinear']
}
grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_lr.fit(X_train, y_train)

param_grid_dt = {
    'max_depth': [3, 5, 10], 
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=None), param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)
grid_dt.fit(X_train, y_train)

# Best hyperparameters found
best_lr = grid_lr.best_estimator_
best_dt = grid_dt.best_estimator_

print("\nBest Logistic Regression Hyperparameters:", grid_lr.best_params_)
print("Best Decision Tree Hyperparameters:", grid_dt.best_params_)

### **Training Optimized Models** ###
models = {
    "Logistic Regression": best_lr,
    "Decision Tree": best_dt
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate Accuracy, Precision, and Recall
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)

    print(f"\n{model_name} Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    # Feature Importance (only for Decision Tree)
    if model_name == "Decision Tree":
        feature_importance = model.feature_importances_
        print("\nFeature Importance (Gini Importance):")
        for feature, importance in zip(X.columns, feature_importance):
            print(f"{feature}: {importance:.2f}")

### **Cross-Validation** ###
kf = KFold(n_splits=5, shuffle=True, random_state=None)
group_kf = GroupKFold(n_splits=5)

for model_name, model in models.items():
    # Standard KFold
    cv_scores_kf = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    
    # GroupKFold using patient_id groups
    cv_scores_group_kf = cross_val_score(model, X_train, y_train, cv=group_kf, groups=groups[:len(y_train)], scoring='accuracy')

    print(f"\n{model_name} - KFold Accuracy Scores: {cv_scores_kf}")
    print(f"Mean Accuracy: {cv_scores_kf.mean():.2f}")

    print(f"\n{model_name} - GroupKFold Accuracy Scores: {cv_scores_group_kf}")
    print(f"Mean Accuracy: {cv_scores_group_kf.mean():.2f}")
