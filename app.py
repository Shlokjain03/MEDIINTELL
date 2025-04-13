from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
import joblib
from ocr_local import detect_text_local
from fuzzywuzzy import fuzz

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model and supporting files
model = joblib.load('model.pkl')
symptom_list = joblib.load('symptom_list.pkl')          
label_encoder = joblib.load('label_encoder.pkl')        
precautions_dict = joblib.load('precautions.pkl')       
descriptions_dict = joblib.load('descriptions.pkl')     

df1 = pd.read_csv('medcine_dataset.csv')
df2 = pd.read_csv('indian_medcine_data.csv')

def match_medicine(ocr_text, med_list):
    best_match = ""
    best_score = 0
    for med in med_list:
        score = fuzz.ratio(ocr_text.lower(), med.lower())
        if score > best_score:
            best_match = med
            best_score = score
    return best_match if best_score >= 70 else None

# Home page
@app.route('/')
def home():
    return render_template('index.html')
# ----------------------------------SYMTOM CHECKER--------------------------------------#
# Symptom checker form
@app.route('/checker')
def checker():
    # Convert symptom list to title format for UI
    symptoms_ui = [s.replace('_', ' ').title() for s in symptom_list]
    return render_template('checker.html', symptoms=symptom_list)

# Handle result
@app.route('/result', methods=['POST'])
def result():
    selected_symptoms = request.form.getlist('symptoms')
    selected_symptoms = [s.lower() for s in selected_symptoms]
    # Encode input symptoms as a weighted vector
    input_vector = []
    for symptom in symptom_list:
        if symptom in selected_symptoms:
            input_vector.append(5)  # Give a fixed high weight to selected symptoms
        else:
            input_vector.append(0)
    # Get prediction probabilities for all diseases
    probabilities = model.predict_proba([input_vector])[0]
    # Get indices of top 3 diseases
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_diseases = label_encoder.inverse_transform(top_indices)
    top_probs = probabilities[top_indices]
    match_percentages = [int(prob * 100) for prob in top_probs]

    # Prepare results for display
    results = zip(top_diseases, match_percentages)
    # Use the top disease for additional info
    top_disease = top_diseases[0]
    description = descriptions_dict.get(top_disease, "No description available.")
    precautions = precautions_dict.get(top_disease, ["No precautions available."])
    return render_template('result.html', results=results, description=description, precautions=precautions)

#---------------------------------------------------IMAGE ANALYSIS-----------------------------#
@app.route('/upload')
def upload():
    return render_template('upload_image.html')  # Upload image page

@app.route('/process_upload', methods=['POST'])
def process_upload():
    if 'image' not in request.files:
        return "No image file"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Local OCR
    detected_text = detect_text_local(filepath)
    medicine_names = df1['name'].tolist()
    matched_name = match_medicine(detected_text, medicine_names)

    basic_info = df1[df1['name'] == matched_name].to_dict(orient='records')[0] if matched_name in df1['name'].values else {}
    extra_info = df2[df2['name'] == matched_name].to_dict(orient='records')[0] if matched_name in df2['name'].values else {}

    return render_template("medcine_result.html", name=matched_name or detected_text,
                           basic_info=basic_info, extra_info=extra_info)
if __name__ == '__main__':
    app.run(debug=True)
