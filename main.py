# minn_gpu.py
# Used MiniLM embeddings and cosine similarity for diseases prediction
# Used distilgpt2 for asking question
# Used Dictionary + string match for Doctor matching
# 🩺 Constrained LLM-based Medical Symptom Chatbot (GPU-friendly, small model)

import os
import json
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------------------- Paths --------------------
BASE_DIR = os.getcwd()
SYMPTOM_DISEASE_PATH = os.path.join(BASE_DIR, "symptoms_diseases.json")
DOCTOR_MAPPING_PATH = os.path.join(BASE_DIR, "DocsInfoNew.json")

# -------------------- Load Data --------------------
with open(SYMPTOM_DISEASE_PATH, "r", encoding="utf-8") as f:
    disease_data = json.load(f)

with open(DOCTOR_MAPPING_PATH, "r", encoding="utf-8") as f:
    doctor_map = json.load(f)

# -------------------- Disease → Specialist Mapping --------------------
DISEASE_TO_SPECIALIST = {
    "asthma": "pulmonologist",
    "alzheimer’s disease": "neurologist",
    "arthritis": "rheumatologist",
    "diabetes": "endocrinologist",
    "hypertension": "cardiologist",
    "cancer": "cancer specialist",
    "flu": "general physician"
}

# -------------------- Load Embedding Model --------------------
print("Loading SentenceTransformer...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

disease_texts = [" ".join(d["symptoms"]) for d in disease_data]
disease_labels = [d["disease"] for d in disease_data]
disease_embeddings = embed_model.encode(disease_texts, convert_to_tensor=True)

print("Embedding model loaded.")

# -------------------- Load SMALL GPU-Compatible Model --------------------
print("Loading small language model...")

MODEL_NAME = "distilgpt2"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None
)

model.config.pad_token_id = tokenizer.eos_token_id

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=120,
    do_sample=True,
    top_p=0.9,
    temperature=0.7
)

print("Language model loaded.")

# -------------------- Utility Functions --------------------
def normalize(text):
    return text.strip().lower()

def clean_disease_name(disease_str):
    """
    Remove numeric prefix like '1. ' and lowercase.
    Example: '1. Asthma' -> 'asthma'
    """
    return disease_str.split(".")[-1].strip().lower()

# -------------------- Core Functions --------------------
def predict_top_disease(symptom_text, top_k=1):
    user_emb = embed_model.encode(symptom_text, convert_to_tensor=True)
    scores = util.cos_sim(user_emb, disease_embeddings)[0]
    top_score, idx = torch.topk(scores, k=top_k)
    return disease_data[idx[0]], float(top_score[0])


def generate_questions_constrained(disease_obj):
    symptoms = disease_obj["symptoms"]

    prompt = f"""
You are a medical assistant.

RULES:
- You MUST use ONLY the symptoms listed below.
- DO NOT add new symptoms.
- DO NOT explain anything.
- Ask ONLY yes/no questions in Bengali.
- Ask exactly 3 questions.

Symptoms:
{chr(10).join(symptoms)}
"""

    output = llm(prompt)[0]["generated_text"]

    valid_questions = []
    for line in output.split("\n"):
        line = line.strip()
        for s in symptoms:
            if s.lower() in line.lower():
                valid_questions.append(line)
                break

    # Fallback if model fails
    if len(valid_questions) < 3:
        for s in symptoms[:3]:
            valid_questions.append(f"আপনার কি {s} আছে?")

    return valid_questions[:3]


def refine_confidence(base_conf, answers):
    yes_count = sum(1 for a in answers.values() if a == "yes")
    total = len(answers)

    adjustment = (yes_count / total) * 0.2
    final = base_conf + adjustment - 0.1

    return max(min(final, 0.95), 0.1)


def get_doctors_for_disease(disease_name, max_results=3):
    disease_key = clean_disease_name(disease_name)
    specialist_needed = DISEASE_TO_SPECIALIST.get(disease_key)

    if not specialist_needed:
        return []

    results = []

    for doc in doctor_map.values():
        if specialist_needed in normalize(doc["Specialist"]):
            results.append({
                "specialist": doc["Specialist"],
                "name": doc["Name"],
                "details": doc["Speciality"],
                "chamber": doc["Chamber & Location"]
            })

        if len(results) >= max_results:
            break

    return results

# -------------------- Chat Loop --------------------
def chat():
    print("\n⚕⚕ SAFE Constrained Medical Chatbot")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("Patient symptom: ").strip()
        if user_input.lower() == "exit":
            break

        disease_obj, confidence = predict_top_disease(user_input)
        disease_name = disease_obj["disease"]

        print(f"\nŸ‹ Possible Disease: {disease_name}")
        print("Answer the following questions:\n")

        questions = generate_questions_constrained(disease_obj)
        answers = {}

        for q in questions:
            ans = input(q + " (yes/no): ").lower()
            answers[q] = ans

        final_conf = refine_confidence(confidence, answers)
        doctors = get_doctors_for_disease(disease_name)

        print("\n✅ FINAL RESULT")
        print(f"Disease: {disease_name}")
        print(f"Confidence: {final_conf:.2f}")

        if doctors:
            for i, d in enumerate(doctors, 1):
                print(f"\nŸ‘ Doctor {i}")
                print(f"Specialist: {d['specialist']}")
                print(f"Name: {d['name']}")
                print(f"Details:\n{d['details']}")
                print(f"Ÿ’ Chamber & Location:\n{d['chamber']}")
        else:
            print("Ÿ‘ Recommended Specialist: General Physician")

        print("⚠ This is not a medical diagnosis. Consult a doctor.\n")

# -------------------- Run --------------------
if __name__ == "__main__":
    chat()
