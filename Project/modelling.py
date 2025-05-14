import pandas as pd
import numpy as np 
import mlflow
import dagshub
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split 

from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, f1_score, 
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import os
import sys

file_path = sys.argv[4] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sleep-health-and-lifestyle-dataset_preprocessing.csv")
df = pd.read_csv(file_path)

X = df.drop(columns=['Sleep Disorder'])
y = df['Sleep Disorder']

# Deteksi kolom kategorik
categorical_cols = X.select_dtypes(include='object').columns.tolist()
low_cardinality_cols = [col for col in categorical_cols if X[col].nunique() <= 4]
high_cardinality_cols = [col for col in categorical_cols if X[col].nunique() > 4]

# Siapkan preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('low_card_cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_cardinality_cols),
    ('high_card_cat', CountFrequencyEncoder(variables=high_cardinality_cols), high_cardinality_cols)
])

n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 144   
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2226055605640655

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate))
])

# Split untuk evaluasi akhir dan logging
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

with mlflow.start_run():

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.sklearn.log_model(clf, "xgb_pipeline")
    signature = mlflow.models.signature.infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(clf, "xgb_pipeline", signature=signature, input_example=X_test.iloc[:5])

    mlflow.log_params(clf.get_params())



