# api.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import pandas as pd
from xgboost import XGBRegressor
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import date 
import psycopg2 

app = FastAPI()

# --- 1. CONFIGURATION DE LA BASE DE DONNÉES SUPABASE (PostgreSQL) ---
# ⚠️ ACTION REQUISE : REMPLACEZ VOTRE_MOT_DE_PASSE_DB_A_REMPLACER
DB_CONFIG = {
    "host": "vqwsatrwtrvarumewxbw.supabase.co", 
    "database": "postgres",
    "user": "postgres",
    "password": "VOTRE_MOT_DE_PASSE_DB_A_REMPLACER" 
}

# --- 2. CONFIGURATION CORS ---
origins = [
    "https://h59xlh-5173.csb.app", 
    "*", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Charger le modèle au démarrage
model = XGBRegressor()
model.load_model("restaurant_model.json")

# 4. Définir la structure des données pour la PRÉDICTION
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

# 5. Définir la structure pour le LOG de DONNÉES RÉELLES
class LogDataInput(RestaurantInput):
    total_customers_real: int
    steak_sold_real: int
    chicken_sold_real: int
    salad_sold_real: int
    fries_sold_real: int


# --- ENDPOINT DE PRÉDICTION (/predict) ---
@app.post("/predict")
def predict_sales(data: RestaurantInput, user_id: str = Header(None, alias="X-User-ID")):
    
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

    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    
    for day in days:
        col_name = f"day_of_week_{day}"
        nouvelle_personne[col_name] = 1 if data.day_of_week == day else 0
        
    nouvelle_personne = nouvelle_personne.drop('day_of_week', axis=1)

    expected_columns = [
        'month', 'is_weekend', 'is_holiday', 'local_event', 'temp_c', 'precip_mm', 
        'staff_on_duty', 'promotion', 
        'day_of_week_Dimanche', 'day_of_week_Jeudi', 'day_of_week_Lundi', 
        'day_of_week_Mardi', 'day_of_week_Mercredi', 'day_of_week_Samedi', 'day_of_week_Vendredi'
    ]
    
    for col in expected_columns:
        if col not in nouvelle_personne.columns:
            nouvelle_personne[col] = 0
            
    input_vector = nouvelle_personne[expected_columns]

    prediction = model.predict(input_vector.values)
    
    result = prediction[0].astype(int) + 10

    return {
        "total_customers": int(result[0]),
        "steak_sold": int(result[1]),
        "chicken_sold": int(result[2]),
        "salad_sold": int(result[3]),
        "fries_sold": int(result[4])
    }


# --- ENDPOINT DE JOURNALISATION DES DONNÉES RÉELLES (/log_data) ---
@app.post("/log_data")
def log_daily_data(data: LogDataInput, user_id: str = Header(..., alias="X-User-ID")):
    
    if not user_id:
        raise HTTPException(status_code=401, detail="L'utilisateur n'est pas authentifié (Header 'X-User-ID' manquant).")
        
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Préparation des données d'entrée au format JSON
        input_data_json = data.model_dump_json(
            exclude={'total_customers_real', 'steak_sold_real', 'chicken_sold_real', 'salad_sold_real', 'fries_sold_real'}
        )
        
        insert_query = """
            INSERT INTO predictions (
                user_id, date, 
                total_customers_real, steak_sold_real, chicken_sold_real, salad_sold_real, fries_sold_real, 
                input_data
            ) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """
        
        # Exécution de la requête avec paramétrisation sécurisée
        cur.execute(insert_query, (
            user_id,
            date.today(),
            data.total_customers_real,
            data.steak_sold_real,
            data.chicken_sold_real,
            data.salad_sold_real,
            data.fries_sold_real,
            input_data_json
        ))
        
        conn.commit()
        cur.close()
        conn.close()

        return {"status": "success", "message": "Données journalières sauvegardées pour le ré-entraînement futur."}
        
    except psycopg2.Error as e:
        # Erreur spécifique à la DB (ex: table 'predictions' inexistante)
        raise HTTPException(status_code=500, detail=f"Erreur de Base de Données. Vérifiez le mot de passe DB et la table 'predictions': {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Interne du Serveur: {e}")
