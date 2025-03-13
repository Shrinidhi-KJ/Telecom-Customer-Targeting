import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess data (as before)
print("Loading data...")
df = pd.read_csv('wallacecommunications.csv')
X = df.drop('new_contract_this_campaign', axis=1)
y = df['new_contract_this_campaign'].map({'yes': 1, 'no': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define feature groups and preprocessing steps (as before)
numeric_features = ['age', 'current_balance', 'conn_tr', 'last_contact_this_campaign_day', 
                    'this_campaign', 'days_since_last_contact_previous_campaign', 
                    'contacted_during_previous_campaign']
categorical_features = ['country', 'job', 'married', 'education', 'arrears', 'housing', 
                        'has_tv_package', 'last_contact', 'last_contact_this_campaign_month', 
                        'outcome_previous_campaign']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Function to evaluate and print results
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_proba):.4f}")


# 1. Baseline Logistic Regression
print("\nTraining Baseline Logistic Regression...")
baseline_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])
baseline_model.fit(X_train, y_train)
evaluate_model(baseline_model, X_test, y_test)

# 2. Logistic Regression with SMOTE
print("\nTraining Logistic Regression with SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(preprocessor.fit_transform(X_train), y_train)
lr_smote = LogisticRegression(random_state=42, max_iter=1000)
lr_smote.fit(X_train_smote, y_train_smote)
evaluate_model(Pipeline([('preprocessor', preprocessor), ('classifier', lr_smote)]), X_test, y_test)

# 3. Tuned Logistic Regression
print("\nTuning Logistic Regression...")
param_dist = {'classifier__C': np.logspace(-4, 4, 20),
              'classifier__class_weight': [None, 'balanced']}
lr_tuned = RandomizedSearchCV(
    Pipeline([('preprocessor', preprocessor),
              ('classifier', LogisticRegression(random_state=42, max_iter=1000))]),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)
lr_tuned.fit(X_train, y_train)
print(f"Best parameters: {lr_tuned.best_params_}")
evaluate_model(lr_tuned, X_test, y_test)

# 4. Random Forest (for comparison)
print("\nTraining Random Forest...")
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_model.fit(X_train, y_train)
evaluate_model(rf_model, X_test, y_test)


# Feature Importance Analysis
feature_names = (numeric_features + 
                 categorical_transformer.fit(X_train[categorical_features]).named_steps['onehot'].get_feature_names_out(categorical_features).tolist())

# For Logistic Regression
lr_coefficients = pd.DataFrame({
    'feature': feature_names,
    'coefficient': lr_tuned.best_estimator_.named_steps['classifier'].coef_[0]
})
lr_coefficients = lr_coefficients.sort_values('coefficient', key=abs, ascending=False)

print("\nTop 10 Important Features (Logistic Regression):")
print(lr_coefficients.head(10))

# For Random Forest
rf_importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.named_steps['classifier'].feature_importances_
})
rf_importances = rf_importances.sort_values('importance', ascending=False)

print("\nTop 10 Important Features (Random Forest):")
print(rf_importances.head(10))

# Plotting ROC curves
plt.figure(figsize=(10, 8))
models = [baseline_model, Pipeline([('preprocessor', preprocessor), ('classifier', lr_smote)]), lr_tuned, rf_model]
model_names = ['Baseline LR', 'LR with SMOTE', 'Tuned LR', 'Random Forest']

for model, name in zip(models, model_names):
    if name == 'LR with SMOTE':
        # For SMOTE model, we need to preprocess X_test separately
        X_test_preprocessed = preprocessor.transform(X_test)
        y_proba = model.named_steps['classifier'].predict_proba(X_test_preprocessed)[:, 1]
    else:
        y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend()
plt.show()
