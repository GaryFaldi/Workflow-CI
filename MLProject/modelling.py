import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

dagshub_url = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/GaryFaldi/Membangun_model.mlflow")
mlflow.set_tracking_uri(dagshub_url)
mlflow.set_experiment("Airline Passenger Satisfaction - CI")

dataset_path = "Airline Passenger Satisfaction_Cleaned/train_cleaned.csv"
if not os.path.exists(dataset_path):
    dataset_path = "MLProject/Airline Passenger Satisfaction_Cleaned/train_cleaned.csv"

df = pd.read_csv(dataset_path) 

X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.autolog()

with mlflow.start_run(nested=True) as run:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc}")

    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="model", 
        registered_model_name="Airline-Satisfaction-Model"
    )
    
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/run_id.txt", "w") as f:
        f.write(run.info.run_id)

print("Training selesai dan model telah diregistrasi.")