# 🧠 AI Symptom & Medicine Identifier Web App
--------------------------------------------------
A dual-function AI-powered Flask web app:

✅ Symptom Checker — Predicts diseases based on user-input symptoms  
✅ Medicine Identifier — Detects medicine name from an uploaded image using OCR
_______________________________________________________________________________

## ⚙️ Features
-------------------
🔍 Disease prediction using a trained Random Forest model  
💊 OCR-based medicine detection using Tesseract + OpenCV  
📋 Retrieves details from local medical CSV datasets  
🌐 Clean Flask-based user interface with multiple routes
________________________________________________________________________________

## 🚀 Getting Started
---------------------------
1️⃣ Clone the repository:  
   git clone https://github.com/your-username/ai-health-assistant.git  
   cd ai-health-assistant

2️⃣ Install Python dependencies:  
   pip install -r requirements.txt

3️⃣ Install Tesseract OCR:  
   📥 https://github.com/UB-Mannheim/tesseract/wiki  
   🛠 Set path in `ocr_local.py`:  
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

4️⃣ Run the app:  
   python app.py  
   🌐 Visit: http://127.0.0.1:5000/
________________________________________________________________________________

## 🧰 Tech Stack & Tools
-------------------------------
🧠 Python, Flask, Scikit-learn, Pandas  
🧪 RandomForestClassifier (ML)  
📷 OpenCV & Pytesseract (OCR)  
🔍 fuzzywuzzy (Text matching)  
🗂 joblib (Model persistence)
________________________________________________________________________________

## 📁 Data Files
--------------------
- dataset.csv  
- Symptom_severity.csv  
- Symptom_description.csv  
- Symptom_precaution.csv  
- medcine_dataset.csv  
- indian_medcine_data.csv
________________________________________________________________________________
🎯 Conclusion
------------------------------------------------------------------------------------------------------------------------------------------
The AI Symptom & Medicine Identifier Web App offers an innovative way to assist users in identifying potential diseases based on
symptoms and medicine information from images. By leveraging machine learning, OCR technology, and a user-friendly web interface,
this app provides a valuable tool for healthcare support. Whether you're a student, researcher, or developer, this project offers
insights into AI, Flask, and image processing integration. Feel free to contribute, enhance, or customise the app for your specific needs!
