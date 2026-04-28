
# =========================================
# Employee Attrition Prediction (Final Code)
# Dataset: WA_Fn-UseC_-HR-Employee-Attrition.csv
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ------------------------------
# 2. Basic Info
# ------------------------------
print("Shape:", df.shape)
print(df['Attrition'].value_counts())

# Convert target
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# ------------------------------
# 3. Cleaning
# ------------------------------
df.drop_duplicates(inplace=True)

# Drop unnecessary columns (if exist)
drop_cols = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

# ------------------------------
# 4. Feature Engineering
# ------------------------------
df['OverTime_Flag'] = df['OverTime'].map({'Yes': 1, 'No': 0})

# ------------------------------
# 5. Encoding
# ------------------------------
df = pd.get_dummies(df, drop_first=True)

# ------------------------------
# 6. Split
# ------------------------------
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 7. Scaling (for LR)
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------
# 8. Models
# ------------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)

# ------------------------------
# 9. Evaluation
# ------------------------------
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    print(f"\n===== {name} =====")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

evaluate(lr, X_test_scaled, y_test, "Logistic Regression")
evaluate(rf, X_test, y_test, "Random Forest")

# ------------------------------
# 10. Feature Importance
# ------------------------------
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop Features:")
print(feat_imp.head(10))

# ------------------------------
# 11. Visualization
# ------------------------------
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp.head(10))
plt.title("Top 10 Important Features")
plt.tight_layout()
plt.show()

# ------------------------------
# 12. ROC Curve
# ------------------------------
fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Random Forest)")
plt.show()
