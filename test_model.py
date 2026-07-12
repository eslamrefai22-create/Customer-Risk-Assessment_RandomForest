"""
Quick sanity check: loads the model + preprocessing artifacts and runs one
prediction on a sample customer, without needing Flask running.

Run:  python test_model.py
"""

from app import build_feature_vector, model, CHURN_CLASS_INDEX, GEOGRAPHY_OPTIONS, GENDER_OPTIONS

sample_customer = {
    "CreditScore": 650,
    "Geography": GEOGRAPHY_OPTIONS[0],
    "Gender": GENDER_OPTIONS[0],
    "Age": 40,
    "Tenure": 3,
    "Balance": 60000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000,
}

features = build_feature_vector(sample_customer)
probability = float(model.predict_proba(features)[0][CHURN_CLASS_INDEX])

print("Feature vector:", features)
print(f"Churn probability: {probability * 100:.2f}%")
print("Prediction:", "Will churn" if probability >= 0.5 else "Will stay")
