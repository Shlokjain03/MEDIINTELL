import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load all CSV files
df = pd.read_csv('dataset.csv')
severity_df = pd.read_csv('Symptom_severity.csv')
description_df = pd.read_csv('Symptom_description.csv')
precaution_df = pd.read_csv('Symptom_precaution.csv')

# Create Symptom → Weight dict
severity_dict = dict(zip(severity_df['Symptom'].str.lower(), severity_df['weight']))
 
# Get unique list of symptoms from dataset.csv
symptom_cols = [col for col in df.columns if col.startswith("Symptom_")]
all_symptoms = set()
for col in symptom_cols:
    all_symptoms.update(df[col].dropna().str.lower())
all_symptoms = sorted(all_symptoms)
print(f"Total unique symptoms: {len(all_symptoms)}")

# Create binary feature matrix with severity weight
def encode_symptoms(row):
    symptom_vector = []
    for symptom in all_symptoms:
        if symptom in row.values:
            weight = severity_dict.get(symptom, 1)
            symptom_vector.append(weight)
        else:
            symptom_vector.append(0)
    return pd.Series(symptom_vector)
X = df[symptom_cols].apply(lambda row: row.str.lower() if row.notna().any() else row, axis=1)
X_encoded = X.apply(encode_symptoms, axis=1)
X_encoded.columns = all_symptoms

# Encode target labels (diseases)
le = LabelEncoder()
y = le.fit_transform(df['Disease'])

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and encodings
joblib.dump(model, 'model.pkl')
joblib.dump(all_symptoms, 'symptom_list.pkl')
joblib.dump(le, 'label_encoder.pkl')  # Save for decoding predictions

# Prepare precautions and descriptions
precaution_dict = {}
for _, row in precaution_df.iterrows():
    disease = row['Disease']
    precaution_dict[disease] = [row[f'Precaution_{i}'] for i in range(1, 5) if pd.notna(row[f'Precaution_{i}'])]
description_dict = dict(zip(description_df['Disease'], description_df['Description']))
joblib.dump(precaution_dict, 'precautions.pkl')
joblib.dump(description_dict, 'descriptions.pkl')