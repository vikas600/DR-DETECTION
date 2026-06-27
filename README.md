# 🩺 Diabetic Retinopathy Detection using EfficientNetB0

## 📌 Project Overview

This project is an AI-based web application that detects the stage of Diabetic Retinopathy from retinal fundus images using Deep Learning. The system uses EfficientNetB0 with Transfer Learning to classify retinal images into five stages of diabetic retinopathy.

The application also provides:
- Disease information
- Precautions
- Recommended diet
- AI medical assistant
- Prediction confidence
- Probability for all classes
- Downloadable medical PDF report

---

## 🎯 Objectives

- Detect Diabetic Retinopathy automatically.
- Reduce screening time.
- Assist doctors in early diagnosis.
- Provide health recommendations based on prediction.

---

## 🧠 Model Used

- EfficientNetB0
- Transfer Learning (ImageNet Weights)
- TensorFlow / Keras

---

## 📂 Dataset

APTOS 2019 Blindness Detection Dataset

Classes:

- No Diabetic Retinopathy
- Mild
- Moderate
- Severe
- Proliferative

Approximately:

- 3662 Images
- 5 Classes

---

## ⚙️ Technologies Used

### Backend

- Python
- Flask

### Deep Learning

- TensorFlow
- Keras
- EfficientNetB0

### Frontend

- HTML
- CSS
- JavaScript
- Bootstrap
- jQuery

### Other Libraries

- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- ReportLab

---

## ✨ Features

- Upload retinal image
- Automatic disease prediction
- Confidence score
- Prediction reliability
- Risk level
- Probability bars
- AI medical assistant
- Medical PDF report generation
- Low confidence warning
- Responsive UI

---

## 📊 Model Performance

Validation Accuracy

75%

Evaluation Accuracy

66%

Model:

EfficientNetB0

---

## 📁 Project Structure

```
drd/
│
├── app.py
├── ai_assistant.py
├── train_efficientnet.py
├── evaluate_model.py
├── requirements.txt
├── README.md
│
├── models/
│     └── dr_multiclass_model.h5
│
├── templates/
├── static/
├── uploads/
├── results/
└── dataset/
```

---

## 🚀 Installation

Clone Repository

```bash
git clone <repository-link>
```

Install Requirements

```bash
pip install -r requirements.txt
```

Run Application

```bash
python app.py
```

Open Browser

```
http://127.0.0.1:5000
```

---

## 📸 Output

The application provides:

- Disease Stage
- Confidence Score
- Prediction Reliability
- Risk Level
- AI Medical Assistant
- Medical Report PDF

---

## ⚠ Disclaimer

This AI model is developed for educational and screening purposes only.

It should not be considered as a substitute for professional medical diagnosis. Patients should always consult a qualified ophthalmologist for confirmation.

---

## 👨‍💻 Developed By

Final Year Computer Engineering Project