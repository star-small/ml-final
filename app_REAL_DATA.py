"""
Student Performance Classifier - FastAPI Application
Real Data Project - Predicts student performance using UCI ML Repository data
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
import os

# Initialize FastAPI app
app = FastAPI(
    title="Student Performance Classifier (Real Data)",
    description="ML model API for predicting student performance levels",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and preprocessing objects
MODEL_DIR = 'ml_models'

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'XGBoost_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    print("✓ All models loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print("Make sure ml_models/ directory contains all required files")

# ==================== PYDANTIC MODELS ====================

class StudentDataInput(BaseModel):
    """Input model for prediction request - Key features from UCI dataset"""
    age: int = Field(..., ge=15, le=22, description="Student age (15-22)")
    studytime: int = Field(..., ge=1, le=4, description="Weekly study time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)")
    absences: int = Field(..., ge=0, le=100, description="Number of absences (0-100)")
    failures: int = Field(default=0, ge=0, le=4, description="Past class failures (0-4)")
    g1: int = Field(..., ge=0, le=20, description="First period grade (0-20)")
    g2: int = Field(..., ge=0, le=20, description="Second period grade (0-20)")
    health: int = Field(default=3, ge=1, le=5, description="Current health status (1=very bad, 5=very good)")
    freetime: int = Field(default=3, ge=1, le=5, description="Free time after school (1=very low, 5=very high)")
    goout: int = Field(default=3, ge=1, le=5, description="Goes out with friends (1=very low, 5=very high)")
    dalc: int = Field(default=1, ge=1, le=5, description="Workday alcohol consumption (1=very low, 5=very high)")
    walc: int = Field(default=2, ge=1, le=5, description="Weekend alcohol consumption (1=very low, 5=very high)")

    class Config:
        example = {
            "age": 18,
            "studytime": 3,
            "absences": 5,
            "failures": 0,
            "g1": 14,
            "g2": 15,
            "health": 4,
            "freetime": 3,
            "goout": 3,
            "dalc": 1,
            "walc": 2
        }

class PredictionOutput(BaseModel):
    """Output model for prediction response"""
    performance_level: str = Field(..., description="Predicted performance level")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: dict = Field(..., description="Probability for each performance level")
    recommendations: List[str] = Field(..., description="Personalized recommendations")

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    dataset: str
    version: str

class ModelInfoResponse(BaseModel):
    """Model information"""
    model_name: str
    task_type: str
    classes: List[str]
    dataset_source: str
    n_features: int
    version: str

# ==================== HELPER FUNCTIONS ====================

def get_recommendations(performance_level: str) -> List[str]:
    """Generate personalized recommendations based on performance level"""
    recommendations = {
        'High': [
            "✓ Excellent performance! Maintain your study habits",
            "✓ Your grade trend is positive",
            "✓ Consider helping other students",
            "✓ Explore advanced topics in the subject",
            "✓ Keep up the good attendance",
            "✓ Balance study with health and wellbeing"
        ],
        'Medium': [
            "⚠ Good progress - room for improvement",
            "⚠ Increase study time to 5-10 hours per week",
            "⚠ Improve attendance (reduce absences)",
            "⚠ Review previous exam results (G1, G2)",
            "⚠ Consider forming study groups",
            "⚠ Manage workload and stress levels",
            "⚠ Get more sleep and maintain health"
        ],
        'Low': [
            "🔴 Performance needs significant improvement",
            "🔴 Please consult with your instructor",
            "🔴 Dedicate at least 10+ hours per week to studying",
            "🔴 Attend all classes (high absences hurting grades)",
            "🔴 Get tutoring or academic support",
            "🔴 Consider dropping other commitments temporarily",
            "🔴 Improve lifestyle: sleep, exercise, reduce stress",
            "🔴 Set clear, achievable study goals"
        ]
    }
    return recommendations.get(performance_level, [])

# ==================== API ENDPOINTS ====================

@app.get("/", tags=["Information"])
def root():
    """Root endpoint - API information"""
    return {
        "message": "Student Performance Classifier API (Real Data)",
        "version": "2.0.0",
        "description": "ML-powered API for predicting student performance using real UCI dataset",
        "dataset": "Student Performance - Portuguese Language Course",
        "endpoints": {
            "documentation": "/docs",
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "sample": "/sample-prediction"
        }
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        model_loaded=True,
        dataset="Student Performance (UCI ML Repository)",
        version="2.0.0"
    )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Information"])
def get_model_info():
    """Get model information"""
    return ModelInfoResponse(
        model_name="XGBoost",
        task_type="Multiclass Classification (Real Data)",
        classes=list(encoder.classes_),
        dataset_source="UCI Machine Learning Repository - Student Performance",
        n_features=len(feature_names),
        version="2.0.0"
    )

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(input_data: StudentDataInput):
    """
    Predict student performance level
    
    **Request Body:**
    - age: Student age (15-22)
    - studytime: Weekly study time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)
    - absences: Number of school absences (0-100)
    - failures: Past class failures (0-4)
    - g1: First period grade (0-20)
    - g2: Second period grade (0-20)
    - health: Current health status (1=very bad, 5=very good)
    - freetime: Free time after school (1=very low, 5=very high)
    - goout: Goes out with friends (1=very low, 5=very high)
    - dalc: Workday alcohol (1=very low, 5=very high)
    - walc: Weekend alcohol (1=very low, 5=very high)
    
    **Response:**
    - performance_level: Predicted level (Low/Medium/High)
    - confidence: Model confidence (0-1)
    - probabilities: Probability for each class
    - recommendations: Personalized advice
    """
    try:
        # Create DataFrame from input
        data_dict = input_data.dict()
        df_input = pd.DataFrame([data_dict])
        
        # Ensure all required features exist
        for col in feature_names:
            if col not in df_input.columns:
                df_input[col] = 0
        
        # Select only required features in correct order
        df_input = df_input[feature_names]
        
        # Scale features
        scaled_input = scaler.transform(df_input)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]
        
        # Get performance level and confidence
        performance_level = encoder.classes_[prediction]
        confidence = float(probabilities.max())
        
        # Create probability dictionary
        prob_dict = {
            encoder.classes_[i]: float(probabilities[i])
            for i in range(len(encoder.classes_))
        }
        
        # Get recommendations
        recommendations = get_recommendations(performance_level)
        
        return PredictionOutput(
            performance_level=performance_level,
            confidence=confidence,
            probabilities=prob_dict,
            recommendations=recommendations
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict", tags=["Prediction"])
def batch_predict(inputs: List[StudentDataInput]):
    """Make predictions for multiple students"""
    try:
        if not inputs:
            raise ValueError("Empty input list")
        
        if len(inputs) > 1000:
            raise ValueError("Maximum batch size is 1000")
        
        results = []
        for input_data in inputs:
            try:
                result = predict(input_data)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        return {
            "predictions": results,
            "count": len(results),
            "source": "Real Student Data"
        }
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/sample-prediction", tags=["Examples"])
def sample_prediction():
    """Get a sample prediction result"""
    sample_input = StudentDataInput(
        age=18,
        studytime=3,
        absences=5,
        failures=0,
        g1=14,
        g2=15,
        health=4,
        freetime=3,
        goout=3,
        dalc=1,
        walc=2
    )
    return predict(sample_input)

@app.get("/dataset-info", tags=["Information"])
def dataset_info():
    """Get information about the dataset"""
    return {
        "name": "Student Performance (Portuguese Language Course)",
        "source": "UCI Machine Learning Repository",
        "samples": 649,
        "features": 30,
        "target": "G3 (final grade)",
        "classes": 3,
        "classes_names": ["Low (0-9)", "Medium (10-14)", "High (15-20)"],
        "url": "https://archive.ics.uci.edu/ml/datasets/student+performance",
        "citation": "Paulo Cortez, Alice Silva. Using Data Mining to Predict Secondary School Student Performance. EUROCON, 2008."
    }

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🎓 Student Performance Classifier API (Real Data)")
    print("="*60)
    print("Dataset: UCI ML Repository - Student Performance")
    print("Model: XGBoost trained on real student data")
    print("="*60)
    print("\nStarting API server...\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
