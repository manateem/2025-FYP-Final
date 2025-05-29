import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, accuracy_score
import statsmodels.api as sm
import json

# === Load dataset ===
df = pd.read_csv(r"C:\Users\valan\OneDrive\Desktop\Github\FINAL PROJECT DATA SCIENCE\2025-FYP-Final\result\features.csv")

# === Selected features ===
selected_numeric_cols = [
    "feat_asymmetry", "feat_border_irregularity", "feat_homogeneity", "feat_colorUniformity", 
    "feat_hair", "feat_multiColorRate", "feat_convexity", "feat_maxBrightness", 
    "feat_minBrightness", "feat_convexVariance", "feat_convexMax","feat_convexAverage",
    "feat_contrast", "feat_energy", "feat_averageColor", "feat_averageRedness"
    ]

# === Handle missing values ===
df[selected_numeric_cols] = df[selected_numeric_cols].apply(lambda x: x.fillna(x.median()))

# === Correlation check ===
corr_matrix = df[selected_numeric_cols].corr()
correlated_pairs = [
    (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
    for i in range(len(corr_matrix.columns))
    for j in range(i)
    if abs(corr_matrix.iloc[i, j]) > 0.85
]
print("\n--- Highly Correlated Feature Pairs ---")
for feat1, feat2, corr in correlated_pairs:
    print(f"{feat1} ↔ {feat2}: Correlation = {corr:.4f}")

# === Target & Grouping ===
df['biopsed'] = df['biopsed'].astype(int)
df["group"] = df["patient_id"].factorize()[0]

# === Define X, y, groups ===
X = df[selected_numeric_cols].copy()
y = df['biopsed']
groups = df["group"].values

# === Train/test split (test untouched) ===
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42, stratify=y
)

# === Models with Pipelines ===
models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear'))
    ]),
    "Decision Tree": Pipeline([
        ('clf', DecisionTreeClassifier(random_state=42))
    ]),
    "K-Nearest Neighbors": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ])
}

# === GroupKFold cross-validation ===
group_kf = GroupKFold(n_splits=5)
folds = list(group_kf.split(X_train, y_train, groups_train))

# === Logistic regression significance & tree importance ===
logit_coefs, logit_pvals, dt_importances = [], [], []

for train_idx, val_idx in folds:
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]

    # Logistic regression (manual scaling + statsmodels)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_train_sm = sm.add_constant(X_train_scaled)
    logit_model = sm.Logit(y_train_fold, X_train_sm)
    result = logit_model.fit(disp=0)

    logit_coefs.append(result.params.values)
    logit_pvals.append(result.pvalues.values)

    # Decision tree importance
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_fold, y_train_fold)
    dt_importances.append(dt_model.feature_importances_)

# === Aggregate significance results ===
logit_coefs = np.array(logit_coefs)
logit_pvals = np.array(logit_pvals)
dt_importances = np.array(dt_importances)

avg_logit_coefs = np.mean(logit_coefs[:, 1:], axis=0)  # Skip intercept
avg_logit_pvals = np.mean(logit_pvals[:, 1:], axis=0)
avg_dt_importances = np.mean(dt_importances, axis=0)

print("\n--- Logistic Regression Coefficients & P-values (Averaged) ---")
for feat, coef, pval in zip(selected_numeric_cols, avg_logit_coefs, avg_logit_pvals):
    print(f"{feat}: Coefficient = {coef:.4f}, P-value = {pval:.4f}")

print("\n--- Decision Tree Feature Importances (Averaged) ---")
for feat, imp in zip(selected_numeric_cols, avg_dt_importances):
    print(f"{feat}: Importance = {imp:.4f}")

# === Cross-validated model evaluation ===
cv_results = {}

for model_name, model in models.items():
    precision_scores, recall_scores, accuracy_scores = [], [], []

    for train_idx, val_idx in folds:
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)

        precision_scores.append(precision_score(y_val_fold, y_pred, zero_division=1))
        recall_scores.append(recall_score(y_val_fold, y_pred, zero_division=1))
        accuracy_scores.append(accuracy_score(y_val_fold, y_pred))

    cv_results[model_name] = {
        "Mean Accuracy": np.mean(accuracy_scores),
        "Mean Precision": np.mean(precision_scores),
        "Mean Recall": np.mean(recall_scores)
    }

print("\n--- GroupKFold Validation Results ---")
for model_name, metrics in cv_results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

# === Unique group count ===
print(f"\nTotal number of unique patient groups: {df['group'].nunique()}")

# === Optional: Save results to disk ===
# Save metrics
# with open("cv_metrics.json", "w") as f:
#     json.dump(cv_results, f, indent=4)

# Save significance
# np.savetxt("logit_coefs.csv", avg_logit_coefs, delimiter=",", header=",".join(selected_numeric_cols))
# np.savetxt("logit_pvals.csv", avg_logit_pvals, delimiter=",", header=",".join(selected_numeric_cols))
# np.savetxt("dt_importances.csv", avg_dt_importances, delimiter=",", header=",".join(selected_numeric_cols))
