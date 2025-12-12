# PotatoCare – Potato Leaf Disease Detection Using Machine Learning

PotatoCare is a machine learning–based system that detects potato leaf diseases using image classification.  
The model identifies **Early Blight**, **Late Blight**, and **Healthy** leaves using a Convolutional Neural Network (CNN).

The system includes:

- A **FastAPI backend** for handling predictions  
- A **TensorFlow/Keras model** (`potato_model.h5`)  
- A simple **HTML/JavaScript frontend** for uploading images and viewing results  

This project was developed as part of IA-2 mini project requirements.

---

## ⭐ Features

- Detects three potato leaf classes:
  - Early Blight  
  - Late Blight  
  - Healthy  
- Fast prediction with a lightweight CNN model  
- Simple and clean web interface  
- Works locally without internet connection  
- Easy to install and run  

---

## ⭐ Tech Stack

**Backend:** FastAPI, Uvicorn  
**Model:** TensorFlow / Keras  
**Frontend:** HTML, CSS, JavaScript  
**Languages:** Python, JavaScript  
**Packages Used:** NumPy, Pillow, python-multipart  

---

## ⭐ Project StructurePotatoCare/
│── backend/
│ ├── main.py
│ ├── potato_model.h5
│ ├── requirements.txt
│── frontend/
│ ├── index.html
│ ├── script.js
│ ├── styles.css
│── docs/
│ ├── PRS.pdf
│ ├── Data_Plan.pdf
│ ├── Design_Document.pdf
│ ├── TestPlan.pdf
│ ├── Install_User_Guide.pdf
│── README.md
│── .gitignore


---

## ⭐ Installation Guide

### 1. Create Virtual Environment
python -m venv venv
venv\Scripts\activate # Windows


### 2. Install Required Libraries
pip install -r backend/requirements.txt


### 3. Run FastAPI Backend
Navigate into backend folder:

Backend runs at:
http://127.0.0.1:8000


### 4. Run Frontend
Open:
frontend/index.html


---

## ⭐ API Endpoint

### POST /predict
**Input:** Image file  
**Output:** JSON containing predicted class & confidence  
```json
{
  "class": "Early Blight",
  "confidence": "92.44%"
}


feat: added FastAPI prediction endpoint
fix: corrected model loading bug
docs: updated README and guides
refactor: improved frontend UI layout
test: added API test cases

