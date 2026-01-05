import re
import random
import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------- DATA LOAD ----------------
BASE_DIR = os.path.dirname(__file__)
training = pd.read_csv(os.path.join(BASE_DIR, 'Data', 'Training.csv'))
testing = pd.read_csv(os.path.join(BASE_DIR, 'Data', 'Testing.csv'))

training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]
testing = testing.loc[:, ~testing.columns.duplicated()]

cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# ---------------- MODELS ----------------
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=200, random_state=42)


rf_model.fit(x_train, y_train)
gb_model.fit(x_train, y_train)

# ---------------- DICTIONARIES ----------------
severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x)}

def load_descriptions():
    path = os.path.join(BASE_DIR, 'MasterData', 'symptom_Description.csv')
    desc_dict = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                desc_dict[row[0].strip()] = row[1].strip()
    return desc_dict

def load_severity():
    path = os.path.join(BASE_DIR, 'MasterData', 'symptom_severity.csv')
    sev_dict = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            try:
                sev_dict[row[0].strip()] = int(row[1])
            except:
                pass
    return sev_dict

def load_precautions():
    path = os.path.join(BASE_DIR, 'MasterData', 'symptom_precaution.csv')
    prec_dict = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 5:
                prec_dict[row[0].strip()] = [row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()]
    return prec_dict

description_list = load_descriptions()
severityDictionary = load_severity()
precautionDictionary = load_precautions()

# ---------------- SYMPTOM SYNONYMS ----------------
symptom_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "tummy pain": "stomach_pain",
    "loose motion": "diarrhea",
    "motions": "diarrhea",
    "high temperature": "fever",
    "temperature": "fever",
    "feaver": "fever",
    "coughing": "cough",
    "throat pain": "sore_throat",
    "cold": "chills",
    "breathing issue": "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache": "muscle_pain",
}

# ---------------- SYMPTOM EXTRACTION ----------------
def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)
    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.8)
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)
    return list(set(extracted))

# ---------------- PREDICTION ----------------
def predict_disease(symptoms_list, top_n=3):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    # Ensemble predictions
    rf_pred_proba = rf_model.predict_proba([input_vector])[0]
    gb_pred_proba = gb_model.predict_proba([input_vector])[0]

    avg_proba = (rf_pred_proba + gb_pred_proba) / 2
    top_idx = np.argsort(avg_proba)[::-1][:top_n]
    diseases = le.inverse_transform(top_idx)
    confidences = [round(avg_proba[i]*100, 2) for i in top_idx]

    return diseases, confidences, avg_proba
