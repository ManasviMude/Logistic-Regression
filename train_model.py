# ================================================
# 1. DATA EXPLORATION (EDA)
# ================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("diabetes.csv")

print("Shape:", df.shape)
print(df.info())
print(df.describe())
print("Target distribution:\n", df['Outcome'].value_counts())

# Check for missing or zero-like values
cols_zero_missing = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
print("Zeros in key columns:\n", (df[cols_zero_missing] == 0).sum())

# Histograms
df.hist(figsize=(12,10))
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()


# ================================================
# 2. DATA PREPROCESSING
# ================================================
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df2 = df.copy()
df2[cols_zero_missing] = df2[cols_zero_missing].replace(0, np.nan)

X = df2.drop(columns=['Outcome'])
y = df2['Outcome']

numeric_features = X.columns.tolist()  # all features are numeric

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# No categorical features, so only numeric
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features)
])


# ================================================
# 3. MODEL BUILDING
# ================================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(solver='liblinear', random_state=42, max_iter=1000))
])

param_grid = {
    'clf__C': [0.01, 0.1, 1, 10],
    'clf__penalty': ['l1','l2']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
search.fit(X_train, y_train)

best_model = search.best_estimator_
print("Best params:", search.best_params_)


# ================================================
# 4. MODEL EVALUATION
# ================================================
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_proba):.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# ================================================
# 5. INTERPRETATION
# ================================================
# Extract coefficients
feat_names = numeric_features
coefs = best_model.named_steps['clf'].coef_[0]

coef_df = pd.DataFrame({
    'feature': feat_names,
    'coef': coefs,
    'odds_ratio': np.exp(coefs)
}).sort_values(by='odds_ratio', ascending=False)

print("\nFeature importance (logistic regression coefficients):\n")
print(coef_df)

# Statistical significance with statsmodels
import statsmodels.api as sm

X_train_proc = best_model.named_steps['preprocessor'].transform(X_train)
X_sm = sm.add_constant(X_train_proc)
sm_logit = sm.Logit(y_train.reset_index(drop=True), X_sm).fit(disp=False)
print(sm_logit.summary())
