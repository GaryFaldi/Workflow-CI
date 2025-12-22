import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dagshub_url = "https://dagshub.com/GaryFaldi/Membangun_model.mlflow"
mlflow.set_tracking_uri(dagshub_url)
mlflow.set_experiment("Airline Passenger Satisfaction - CI")

df = pd.read_csv("Airline Passenger Satisfaction_Cleaned/train_cleaned.csv") 

X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

    mlflow.sklearn.log_model(model, "model", registered_model_name="Airline-Satisfaction-Model")