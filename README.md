# 🩺 Constrained LLM-Based Medical Symptom Chatbot (GPU-Friendly)

A **lightweight, constrained medical symptom chatbot** that predicts possible diseases, asks **controlled follow-up questions**, and recommends appropriate doctors — all while remaining **GPU-friendly and resource-efficient**.

This system intentionally avoids large medical LLMs and instead combines:

* **Sentence embeddings + cosine similarity** for disease prediction
* **Small causal language model (DistilGPT-2)** for constrained question generation
* **Rule-based specialist & doctor matching**

⚠️ **Disclaimer:** This project is for **educational and research purposes only** and is **not a medical diagnostic system**.

---

## 🚀 Key Features

* ✅ **Symptom → Disease prediction** using MiniLM embeddings
* ✅ **Constrained LLM question generation**

  * Uses **only known symptoms**
  * Asks **exactly 3 yes/no questions**
  * No hallucinated symptoms
* ✅ **GPU-friendly**

  * Uses `distilgpt2`
  * Works on low-VRAM GPUs or CPU
* ✅ **Doctor recommendation**

  * Disease → Specialist mapping
  * Dictionary + string matching for doctors
* ✅ **Confidence refinement**

  * Adjusts prediction confidence based on user answers

---

## 🧠 System Architecture

```
User Symptoms
      ↓
MiniLM Embedding Model
      ↓
Cosine Similarity
      ↓
Top Disease Prediction
      ↓
DistilGPT-2 (Constrained Questions)
      ↓
User Yes/No Answers
      ↓
Confidence Refinement
      ↓
Specialist + Doctor Recommendation
```

---

## 🧪 Models Used

| Task                | Model              |
| ------------------- | ------------------ |
| Symptom Embedding   | `all-MiniLM-L6-v2` |
| Question Generation | `distilgpt2`       |
| Similarity Metric   | Cosine Similarity  |

---

## 📂 Project Structure

```
.
├── minn_gpu.py                 # Main chatbot logic
├── symptoms_diseases.json      # Disease ↔ symptom dataset
├── DocsInfoNew.json            # Doctor & specialist information
├── README.md
```

---

## ⚙️ Installation

### 1️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2️⃣ Install dependencies

```bash
pip install torch transformers sentence-transformers
```

> 💡 For GPU usage, install CUDA-compatible PyTorch.

---

## ▶️ How to Run

```bash
python main.py
```

Then enter patient symptoms in free text:

```
Patient symptom: chest pain and shortness of breath
```

Type `exit` to quit.

---

## 🧩 How It Works (Core Logic)

### 🔹 Disease Prediction

* Converts user symptoms into embeddings
* Matches against known disease symptom embeddings
* Selects top disease using cosine similarity

### 🔹 Question Generation (Constrained)

* LLM is **strictly constrained**
* Can only use **existing symptoms**
* Generates **3 yes/no questions in Bengali**
* Fallback logic ensures safety if LLM fails

### 🔹 Confidence Refinement

* Base confidence from similarity score
* Adjusted using yes/no answers
* Output bounded between `0.1 – 0.95`

### 🔹 Doctor Recommendation

* Maps disease → required specialist
* Matches doctors using normalized string comparison
* Returns top 3 results

---

## 🔒 Safety & Constraints

✔ No new symptoms are hallucinated
✔ No medical explanations are generated
✔ Clear disclaimer is always shown
✔ Designed to assist — **not diagnose**

---

## 📌 Example Output

```
Possible Disease: Asthma
Confidence: 0.74

Doctor 1
Specialist: Pulmonologist
Name: Dr. XYZ
Chamber & Location: ABC Hospital

⚠ This is not a medical diagnosis. Consult a doctor.
```

---

## 🔮 Future Improvements

* Multi-disease ranking instead of top-1
* Multilingual symptom input
* Dynamic follow-up question count
* Web / Flutter frontend integration
* Better specialist ontology mapping

---

## 📜 Disclaimer

This software **does not provide medical advice**.
Always consult a qualified healthcare professional for diagnosis and treatment.

---

## 👨‍💻 Author

**Qm Asif Tanjim**
CSE Student | 2211402042 | North South University
