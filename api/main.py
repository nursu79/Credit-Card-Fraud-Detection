from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import uuid
import os
from sqlalchemy.orm import Session
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import engine, Base, get_db
from database.models import Transaction
from rag.generator import FraudExplanationGenerator

# Create DB tables if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Fraud Detection API")

# Load trained Model
try:
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "logistic_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Warning: logistic_model.pkl not found. Please train the model first.")

try:
    rag_generator = FraudExplanationGenerator()
except Exception as e:
    rag_generator = None
    print("Warning: Vector DB not initialized. Run the build_vector_db script first.", e)

class TransactionInput(BaseModel):
    Time: float = 0.0
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = 0.0

@app.post("/predict")
def predict_fraud(transaction: TransactionInput, db: Session = Depends(get_db)):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    data_dict = transaction.model_dump()
    df = pd.DataFrame([data_dict])
    
    # Predict
    prob = float(model.predict_proba(df)[0, 1])
    pred = int(model.predict(df)[0])
    
    risk_level = "HIGH" if prob > 0.70 else ("MEDIUM" if prob > 0.30 else "LOW")
    
    return {
        "fraud_probability": round(prob, 4),
        "fraud_prediction": pred,
        "risk_level": risk_level
    }

@app.post("/investigate")
def investigate_fraud(transaction: TransactionInput, db: Session = Depends(get_db)):
    if not model or not rag_generator:
        raise HTTPException(status_code=500, detail="System not fully initialized (missing model or vector DB).")
        
    data_dict = transaction.model_dump()
    
    # 1. Run prediction
    prediction_result = predict_fraud(transaction, db)
    prob = prediction_result["fraud_probability"]
    pred = prediction_result["fraud_prediction"]
    risk = prediction_result["risk_level"]
    
    # 2. Get RAG explanation
    explanation_report = rag_generator.generate_explanation(data_dict, prob, risk)
    
    # 3. Store in Database
    tx_id = f"txn_{uuid.uuid4().hex[:10]}"
    db_transaction = Transaction(
        transaction_id=tx_id,
        **data_dict,
        fraud_probability=prob,
        fraud_prediction=pred,
        risk_level=risk,
        investigation_explanation=explanation_report["ai_explanation"],
        timestamp=datetime.utcnow()
    )
    db.add(db_transaction)
    db.commit()
    db.refresh(db_transaction)
    
    return {
        "transaction_id": tx_id,
        **explanation_report
    }

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fraud Detection API running."}
