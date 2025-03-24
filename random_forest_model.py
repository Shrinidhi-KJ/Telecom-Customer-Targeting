import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load and preprocess data (same as before)
print("Loading data...")
df = pd.read_csv('wallacecommunications.csv')

# Create interaction features
df['age_balance_interaction'] = df['age'] * df['current_balance']
df['age_campaign_interaction'] = df['age'] * df['this_campaign']

X = df.drop('new_contract_this_campaign', axis=1)
y = df['new_contract_this_campaign'].map({'yes': 1, 'no': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature lists (same as before)
numeric_features = ['age', 'current_balance', 'conn_tr', 'last_contact_this_campaign_day', 
                    'this_campaign', 'days_since_last_contact_previous_campaign', 
                    'contacted_during_previous_campaign', 'age_balance_interaction', 'age_campaign_interaction']
categorical_features = ['country', 'job', 'married', 'education', 'arrears', 'housing', 
                        'has_tv_package', 'last_contact', 'last_contact_this_campaign_month', 
                        'outcome_previous_campaign']

# Preprocessing (same as before)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ])

# Random Forest (Bagging)
rf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

param_dist_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}

# Gradient Boosting
gb_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(random_state=42))
])

param_dist_gb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 4, 5],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Function to train and evaluate model
def train_and_evaluate(model, param_dist, model_name):
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=3, scoring='roc_auc', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    
    print(f"\nBest {model_name} parameters:", random_search.best_params_)
    print(f"Best {model_name} AUC-ROC score:", random_search.best_score_)
    
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{model_name} AUC-ROC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    return best_model, y_proba

# Train and evaluate Random Forest
best_rf, y_proba_rf = train_and_evaluate(rf_model, param_dist_rf, "Random Forest")

# Train and evaluate Gradient Boosting
best_gb, y_proba_gb = train_and_evaluate(gb_model, param_dist_gb, "Gradient Boosting")

# Plot ROC curves
plt.figure(figsize=(10, 8))
for model_name, y_proba in [("Random Forest", y_proba_rf), ("Gradient Boosting", y_proba_gb)]:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_proba):.3f})')

plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Comparison')
plt.legend()
plt.show()

# Feature importance for Random Forest
feature_names = (numeric_features + 
                 best_rf.named_steps['preprocessor'].named_transformers_['cat']
                 .named_steps['onehot'].get_feature_names_out(categorical_features).tolist())
importances_rf = best_rf.named_steps['classifier'].feature_importances_
feature_importance_rf = pd.DataFrame({'feature': feature_names, 'importance': importances_rf})
feature_importance_rf = feature_importance_rf.sort_values('importance', ascending=False)

print("\nTop 10 Important Features (Random Forest):")
print(feature_importance_rf.head(10))

# Plot feature importance for Random Forest
plt.figure(figsize=(10, 6))
feature_importance_rf.head(10).plot(x='feature', y='importance', kind='bar')
plt.title('Top 10 Feature Importances - Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()
