# Boarding SBERT Semantic Matching Platform

An AI/ML-powered matching pipeline designed to perform highly accurate semantic matching and text-overlap comparison between resumes and job descriptions using **Sentence-BERT (SBERT)** and FastAPI.

This repository is structured as a modular **monorepo**, keeping the AI matching service, data assets, models, and frontend client cleanly separated for frictionless collaboration.

---

## 📂 Directory Layout

```text
Boarding_SBERT_approche/
├── backend/                  # Python FastAPI Backend (AI Pipeline Service)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI app code (contains matching & endpoint logic)
│   │   └── services/         # Custom business logic (expandable matching algorithms)
│   ├── requirements.txt      # Python package dependencies
│   └── run.py                # Simple startup script to run FastAPI dev server
│
├── data/                     # Raw and processed datasets (Ignored in Git to avoid bloat)
│   ├── raw/
│   │   └── dataset.csv       # Original dataset
│   └── processed/
│       ├── boarding_dataset_index.parquet
│       ├── jd_embeddings.npy
│       └── resume_embeddings.npy
│
├── models/                   # Local ML model weights and configurations
│   ├── boarding_sbert_model/ # Local SentenceTransformers model directory
│   └── matching_config.joblib
│
├── research/                 # EDA (Exploratory Data Analysis) & evaluations
│   ├── notebooks/
│   │   └── boarding_matching_pipeline.ipynb
│   └── plots/                # Data exploration and evaluation charts (ROC, PR curves)
│
├── web/                      # RESERVED FOR THE WEB TEAM (Frontend Application)
│   └── ... (their React, Next.js, or HTML dashboard)
│
└── .gitignore                # Repository-wide Git ignore rules
```

---

## ⚡ Quick Start: Backend API Service

### 1. Prerequisites
Ensure you have **Python 3.10+** installed.

### 2. Setup Virtual Environment
Navigate to the `backend/` directory, create a virtual environment, and activate it:

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install all backend libraries including PyTorch, SentenceTransformers, and FastAPI:
```bash
pip install -r requirements.txt
```

### 4. Run the Dev Server
To run the server with auto-reloading enabled, execute:
```bash
python run.py
```
The server will start running locally at: **`http://localhost:8000`**

---

## 🌐 Consuming the API (For the Web Team)

The SBERT backend automatically generates comprehensive, interactive API documentation. 
* Open your browser and navigate to **`http://localhost:8000/docs`** to view the **Swagger UI** containing schemas, models, and interactive request testing.

### Core Endpoints

#### 1. Health Check
* **Route**: `GET /health`
* **Response**:
  ```json
  {
    "status": "ok",
    "model": "boarding_sbert_model"
  }
  ```

#### 2. Match Resume against Job Descriptions (Including Personality Questionnaire)
Use this endpoint to take a candidate resume's text and find the best-matched job descriptions. You can optionally pass the student's personality test options and enable preference-based scoring.
* **Route**: `POST /match/resume`
* **Request Body Schema (Basic SBERT + Overlap)**:
  ```json
  {
    "text": "Full stack software engineer with 2 years of experience in Python, React, and FastAPI.",
    "top_k": 3,
    "semantic_weight": 0.8,
    "overlap_weight": 0.2
  }
  ```
* **Request Body Schema (Hybrid Personality Match)**:
  ```json
  {
    "text": "Student developer specialized in Python, React, and FastAPI. Love working in agile startup teams.",
    "top_k": 3,
    "semantic_weight": 0.50,
    "overlap_weight": 0.15,
    "preference_weight": 0.35,
    "student_preferences": {
      "languages": ["Français", "Anglais"],
      "travel_destination": "🇿🇦 Afrique du Sud (ambiance anglophone & internationale)",
      "english_level": "Avancé",
      "duration": "2-3 mois",
      "desired_domain": "Dév / IT",
      "company_style": "Start-up cool",
      "mission_type": "Technique",
      "ideal_atmosphere": "Libre et flexible",
      "work_style": "En équipe"
    }
  }
  ```
* **Response Schema**:
  ```json
  {
    "matches": [
      {
        "rank": 1,
        "match_score": 0.8924,
        "role": "Software Engineer",
        "text": "Looking for a Python backend developer with FastAPI skills. We offer a high-performance agile startup environment in Cape Town...",
        "label": 1,
        "reason": "Outstanding skills alignment and perfect startup culture & location compatibility."
      }
    ]
  }
  ```

#### 3. Match Job Description against Resumes
Use this endpoint to take a job description's text and find the best-matched student resumes.
* **Route**: `POST /match/jd`
* **Request Body Schema**: (Same as `POST /match/resume`, note that `student_preferences` only applies when matching resumes to JDs).

---

## 🛠️ Customizing Weights
The matching model uses a highly customizable hybrid retrieval score combining:
$$\text{Score} = w_{\text{semantic}} \cdot \text{CosSim}_{\text{SBERT}} + w_{\text{overlap}} \cdot \text{Overlap}_{\text{Jaccard}} + w_{\text{preference}} \cdot S_{\text{Preference}}$$

You can fine-tune these weights in your API calls dynamically:
* To run **standard semantic skills matching only**, set `semantic_weight = 1.0` (all others 0.0).
* To run **hybrid matching including the personality questionnaire**, set your custom weights (e.g., `semantic_weight = 0.50`, `overlap_weight = 0.15`, `preference_weight = 0.35`). The system automatically normalizes weights to sum to `1.0` before computing the scoring.

