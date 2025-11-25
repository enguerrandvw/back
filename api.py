# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from xgboost import XGBRegressor
import json

app = FastAPI()

# 1. Charger le modèle au démarrage
model = XGBRegressor()
model.load_model("restaurant_model.json")

# 2. Définir la structure des données reçues du site (les inputs)
class RestaurantInput(BaseModel):
    day_of_week: str
    month: int
    is_weekend: int
    is_holiday: int
    local_event: int
    temp_c: float
    precip_mm: float
    staff_on_duty: int
    promotion: int

# 3. L'endpoint de prédiction
@app.post("/predict")
def predict_sales(data: RestaurantInput):
    # Créer le DataFrame comme dans ton script original
    nouvelle_personne = pd.DataFrame([{
        'day_of_week': data.day_of_week,
        'month': data.month,
        'is_weekend': data.is_weekend,
        'is_holiday': data.is_holiday,
        'local_event': data.local_event,
        'temp_c': data.temp_c,
        'precip_mm': data.precip_mm,
        'staff_on_duty': data.staff_on_duty,
        'promotion': data.promotion
    }])

    # One Hot Encoding manuel pour s'assurer que ça correspond à l'entraînement
    # On recrée les colonnes exactes utilisées lors de l'entraînement
    # Note: Idéalement, sauvegarde la liste des colonnes d'entraînement dans un fichier json à part
    # Pour l'exemple, je liste les jours possibles
    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    
    # Créer les colonnes dummy
    for day in days:
        col_name = f"day_of_week_{day}"
        nouvelle_personne[col_name] = 1 if data.day_of_week == day else 0
        
    # Supprimer la colonne originale texte
    nouvelle_personne = nouvelle_personne.drop('day_of_week', axis=1)

    # Réorganiser les colonnes pour matcher l'ordre du modèle (très important avec XGBoost)
    # L'ordre doit être EXACTEMENT celui de X_encoded.columns lors de l'entraînement
    # Tu devras peut-être ajuster cette liste selon ton X_encoded.columns précis
    expected_columns = [
        'month', 'is_weekend', 'is_holiday', 'local_event', 'temp_c', 'precip_mm', 
        'staff_on_duty', 'promotion', 
        'day_of_week_Dimanche', 'day_of_week_Jeudi', 'day_of_week_Lundi', 
        'day_of_week_Mardi', 'day_of_week_Mercredi', 'day_of_week_Samedi', 'day_of_week_Vendredi'
    ]
    
    # On s'assure d'avoir toutes les colonnes, remplies à 0 si manquantes
    for col in expected_columns:
        if col not in nouvelle_personne.columns:
            nouvelle_personne[col] = 0
            
    # On reordonne
    input_vector = nouvelle_personne[expected_columns]

    # Prédiction
    prediction = model.predict(input_vector.values)
    
    # Résultat (ajout des 10 clients comme dans ton script)
    result = prediction[0].astype(int) + 10 # array de 5 valeurs

    return {
        "total_customers": int(result[0]),
        "steak_sold": int(result[1]),
        "chicken_sold": int(result[2]),
        "salad_sold": int(result[3]),
        "fries_sold": int(result[4])
    }