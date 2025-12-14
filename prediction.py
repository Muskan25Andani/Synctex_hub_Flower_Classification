import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")

print("🌸 Iris Flower Prediction (CLI)")

sl = float(input("Enter Sepal Length: "))
sw = float(input("Enter Sepal Width: "))
pl = float(input("Enter Petal Length: "))
pw = float(input("Enter Petal Width: "))

# ⚠️ IMPORTANT: Use SAME column names as training
sample = pd.DataFrame(
    [[sl, sw, pl, pw]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)

# Transform and predict
sample_scaled = scaler.transform(sample)

pred_class = model.predict(sample_scaled)[0]
confidence = model.predict_proba(sample_scaled).max() * 100

# Class label mapping
species_map = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

print("\nPredicted Species:", species_map[pred_class])
print(f"Confidence: {confidence:.2f}%")
