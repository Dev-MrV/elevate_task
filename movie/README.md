# 🎬 CinematicVerse: Multi-Lingual Hybrid Movie Recommendation System

![Python Version](https://img.shields.io/badge/Python-v3.9+-blue?style=flat-square&logo=python)
![Framework](https://img.shields.io/badge/Framework-Streamlit_v1.30+-FF4B4B?style=flat-square&logo=streamlit)
![ML Engine](https://img.shields.io/badge/Engine-Scikit--Learn_&_Pandas-F7931E?style=flat-square&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-success?style=flat-square)
![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen?style=flat-square)

CinematicVerse is a high-performance, content-based recommendation engine designed to intelligently match and suggest films across five independent regional movie industries. By utilizing advanced NLP vectorization on real-world Kaggle datasets, the system allows users to seamlessly discover related movies tailored to their exact cultural and genre preferences.

---

## 🏗️ Core Architecture & Features Breakdown

*   **Cross-Industry Discovery:** Native, optimized feature support for cross-referencing films from Malayalam, Tamil, Telugu, Hindi, and Hollywood film assets.
*   **Hybrid NLP Logic:** Textual vectorization powered by Scikit-Learn's `TfidfVectorizer` and `cosine_similarity` mapping out complex plot overviews, genres, and cast metadata into mathematical similarity matrices.
*   **Dynamic Visual Presentation:** Automated poster URL engine that gracefully falls back to live TMDB image CDN integrations if local structural data is missing or incomplete.

### ⚙️ How It Works
The underlying data lifecycle pipeline operates in five discrete stages:
1.  **Data Ingestion:** Reads local CSV datasets into Pandas DataFrames and sanitizes the feature columns.
2.  **Metadata Soup Concatenation:** Merges parsed genres, cast names, directors, and plot overviews into a single, comprehensive text string.
3.  **TF-IDF Vectorization:** Analyzes the soup to assign statistical weight to critical keywords and themes.
4.  **Cosine Similarity Evaluation:** Calculates the spatial mathematical proximity between the target movie and all other movies in the vector space.
5.  **Regional Filter Sorting & Streamlit Injection:** Filters the closest matches by the user's selected industry and injects the live results into the interactive Streamlit UI grid.

---

## 🚀 Quick-Start & Local Setup Instructions

Follow these steps to deploy the recommendation engine in your local development environment.

### Prerequisites
*   Python 3.9 or higher
*   Git installed on your local machine

### 1. Cloning the Repository
```bash
git clone https://github.com/yourusername/cinematicverse.git
cd cinematicverse
```

### 2. Environment Configuration
> [!WARNING]
> It is highly recommended to use a virtual environment to prevent dependency conflicts with your system's global Python packages.

```bash
# Create a virtual environment
python -m venv venv

# Activate on Windows:
.\venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 3. Dependency Installation
```bash
pip install -r requirements.txt
```

### 4. Dataset Initialization
For this system to function, you must manually provide the Kaggle dataset. 
1. Download the TMDB/IMDb movie datasets from Kaggle.
2. Place the resulting CSV files directly into the project directory at the following path:
   `data/movies_data.csv`

### 5. Execution
Spin up the local FastAPI and HTML web application dashboard:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## 📂 Interactive Configuration & Directory Tree

```text
mmovie/
├── imdb_top_1000.csv          # Your downloaded Kaggle dataset
├── app.py                     # FastAPI Backend Application
├── engine.py                  # Core ML recommendation logic
├── index.html                 # Frontend HTML User Interface
├── main.js                    # Frontend JavaScript Logic
├── styles.css                 # Frontend Tailwind/Custom CSS
└── README.md                  # Project documentation
```

### 🔑 TMDB API Key Integration (Optional)
If your provided local CSV dataset is missing direct image links for movie posters, CinematicVerse features a built-in dynamic fallback strategy. 

To activate this, launch the Streamlit dashboard and locate the **TMDB API Key** input field in the left sidebar. Inputting a valid (and free) API key from The Movie Database will allow the application to securely ping the TMDB servers at runtime to retrieve and render gorgeous, high-resolution official graphic covers for all recommendations.
