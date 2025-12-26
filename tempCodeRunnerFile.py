# main.py — TF-IDF + Logistic Regression + Gemini API Chatbot

import json, os, re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv

# -------------------- Load Gemini API Key --------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------------- Load datasets --------------------
SYMPTOM_DISEASE_PATH = r"F:\ChatBot\symptoms_diseases.json"
DOCTOR_MAPPING_PATH = r"F:\ChatBot\DocsInfoNew.json"

with open(SYMPTOM_DISEASE_PATH, "r", encoding="utf-8") as f:
    symptom_disease_dict = json.load(f)

with open(DOCTOR_MAPPING_PATH, "r", encoding="utf-8") as f:
    doctor_map = json.load(f)

# -------------------- Prepare ML model --------------------
diseases = []
symptom_texts = []

for entry in symptom_disease_dict:
    diseases.append(entry["disease"])
    symptom_texts.append(" ".join(entry["symptoms"]))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(symptom_texts)
clf = LogisticRegression(max_iter=500)
clf.fit(X, diseases)

# -------------------- Helper Functions --------------------
def extract_symptoms(text):
    text = text.lower()
    return re.findall(r"[a-z]{3,}", text)

def predict_disease(user_symptom_text):
    X_test = vectorizer.transform([user_symptom_text])
    prediction = clf.predict(X_test)[0]
    proba = clf.predict_proba(X_test)[0]
    confidence = float(max(proba))
    return prediction, confidence

def get_doctor_for_disease(disease):
    return doctor_map.get(disease, "General Physician")

# -------------------- Gemini API Chat --------------------
def call_gemini_api(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 500
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"❌ Error contacting Gemini API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"❌ Exception while calling Gemini API: {str(e)}"

def generate_response(disease, confidence, symptoms, doctor, user_message):
    prompt = f"""
You are MediGuide AI, a safe medical guidance assistant.
Do NOT provide a diagnosis.
Explain possible conditions and suggest a specialist.
Include a safety warning.

User message: {user_message}
Symptoms extracted: {symptoms}
Most likely condition (not a diagnosis): {disease}
Confidence score (ML model only): {confidence:.2f}
Recommended specialist: {doctor}

Write a friendly explanation covering:
- why these symptoms match the condition
- what the condition generally means
- why this doctor is appropriate
- a safety note advising to see a real doctor
"""
    return call_gemini_api(prompt)

# -------------------- Chat Loop --------------------
def chat():
    print("\n🩺 MediGuide AI — Healthcare Chatbot")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Patient: ")
        if user_input.lower() == "exit":
            break

        symptoms = extract_symptoms(user_input)
        symptom_text = " ".join(symptoms)

        predicted_disease, confidence = predict_disease(symptom_text)
        recommended_doctor = get_doctor_for_disease(predicted_disease)

        ai_reply = generate_response(
            predicted_disease,
            confidence,
            symptoms,
            recommended_doctor,
            user_input
        )

        print("\nChatBot:", ai_reply, "\n")

# -------------------- Run --------------------
if __name__ == "__main__":
    chat()
