import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

# Load dataset
df = pd.read_csv(r"C:\Users\valan\OneDrive\Desktop\Projects in D Science\MANDATORY\metadata2.csv")

# Define selected numerical features
selected_numeric_cols = ['feat_asymmetry', 'feat_border_irregularity', 'feat_color']

# Handle missing values
df[selected_numeric_cols] = df[selected_numeric_cols].apply(lambda x: x.fillna(x.median()))

# Convert target variable
df['biopsed'] = df['biopsed'].astype(int)

# Assign unique patient groups for GroupKFold
df["group"] = df["patient_id"].factorize()[0]  # Faster mapping

# Define features, target, and groups
X = df[selected_numeric_cols].copy()
y = df['biopsed']
groups = df["group"].values

# Train-test split (test set is untouched after this)
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42, stratify=y
)

# Models using Pipelines (StandardScaler inside each fold)
models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear'))
    ]),
    "Decision Tree": Pipeline([
        ('clf', DecisionTreeClassifier(random_state=42))
    ]),  # Trees don’t need scaling
    "K-Nearest Neighbors": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=5))
    ])
}

# Cross-validation using GroupKFold
group_kf = GroupKFold(n_splits=5)
cv_results = {}

# Store logistic regression coefficients and p-values
logit_coefs = []
logit_pvals = []

# Store Decision Tree feature importance values
dt_importances = []

for train_idx, val_idx in group_kf.split(X_train, y_train, groups_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Standardize inside fold manually for logistic regression significance analysis
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)

    # Logistic Regression Feature Significance
    X_train_sm = sm.add_constant(X_train_scaled)  # Add intercept
    logit_model = sm.Logit(y_train_fold, X_train_sm)
    result = logit_model.fit(disp=0)

    logit_coefs.append(result.params.values)  # Includes intercept as first param
    logit_pvals.append(result.pvalues.values)

    # Decision Tree Feature Importance
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_fold, y_train_fold)
    dt_importances.append(dt_model.feature_importances_)

# Convert lists to numpy arrays and average
logit_coefs = np.array(logit_coefs)
logit_pvals = np.array(logit_pvals)
dt_importances = np.array(dt_importances)

avg_logit_coefs = np.mean(logit_coefs[:, 1:], axis=0)  # Excluding intercept
avg_logit_pvals = np.mean(logit_pvals[:, 1:], axis=0)  # Excluding intercept
avg_dt_importances = np.mean(dt_importances, axis=0)

# Display averaged significance scores
print("\n--- Logistic Regression Coefficients & P-values (Averaged) ---")
for feat, coef, pval in zip(selected_numeric_cols, avg_logit_coefs, avg_logit_pvals):
    print(f"{feat}: Coefficient = {coef:.4f}, P-value = {pval:.4f}")

print("\n--- Decision Tree Feature Importances (Averaged) ---")
for feat, imp in zip(selected_numeric_cols, avg_dt_importances):
    print(f"{feat}: Importance = {imp:.4f}")

# Final GroupKFold validation results for models
for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=group_kf, groups=groups_train, scoring='accuracy')
    
    cv_results[model_name] = cv_scores
    print(f"\n{model_name} - GroupKFold Accuracy Scores: {cv_scores}")
    print(f"Mean Accuracy: {cv_scores.mean():.2f}")
