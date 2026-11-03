# Pokelytics: a State-of-the-Art AI-powered Pokedex! <br /> 🔥💧🌿⚡   
**⋆˚꩜｡ By Olan, Pabico, Pesino, and Regala | CMSC 170 - W2L ˚꩜｡** <br /><br />
**૮₍´｡ᵔ ꈊ ᵔ｡`₎ა**
<br /><br />
**Pokelytics** is an app that aims to improve upon the Pokedex we know and love by creating an AI-powered Pokemon Classification tool using Naive-Bayes, K-Nearest Neighbors, and K-Means Clustering, basing on the [Ultimate Pokemon Dataset (2026)](https://www.kaggle.com/datasets/darkmatternet/ultimate-pokmon-dataset-2025) from Kaggle. Through this tool, Pokemon trainers will have no trouble organizing their Pokedex and grouping their collected Pokemon because our state-of-the-art algorithms will be able to predict its type based on your beloved Pokemon's predetermined features! Neat, right? 

## How to Run
1. API
    - Navigate to api folder `cd api`
    - Create a Virtual Environment `python -m venv ./venv`
    - Activate Virtual Environment `venv\Scripts\activate`
    - Install Requirements `pip install -r requirements.txt`
    - Run API `uvicorn api:app`

## File Structure
```
api/
├── api.py               # FastAPI, CORS Middleware
├── classifiers/         # Naive-Bayes, KNN, K-Means
├── schemas/             # Pydantic Models
├── routes/              # API Endpoints
└── data/                
    └── pokemon.csv      # Dataset
```