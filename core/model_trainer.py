from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier


def train_enhanced_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Configure XGBoost for binary classification
    model = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='binary:logistic',  # Explicitly set for binary classification
        eval_metric='logloss'        # Appropriate metric for binary classification
    )

    # Train the model with basic parameters
    model.fit(X_train, y_train)

    return model, X_test, y_test