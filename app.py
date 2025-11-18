"""
Student Mental Health Classifier - FastAPI Application
A REST API for predicting student stress levels using XGBoost model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
import os

# Initialize FastAPI app
app = FastAPI(
    title="Student Mental Health Classifier",
    description="ML model API for predicting student stress levels",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
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

class StudentHealthInput(BaseModel):
    """Input model for prediction request"""
    sleep_hours: float = Field(..., ge=0, le=24, description="Daily sleep hours (0-24)")
    study_hours_per_day: float = Field(..., ge=0, le=24, description="Daily study hours")
    social_interaction_score: int = Field(..., ge=1, le=10, description="Social engagement (1-10)")
    exercise_hours_per_week: float = Field(..., ge=0, le=100, description="Weekly exercise hours")
    academic_performance: float = Field(..., ge=0, le=100, description="Academic score (0-100)")
    exam_anxiety_level: int = Field(..., ge=1, le=10, description="Exam anxiety level (1-10)")
    family_income_level: str = Field(..., description="Income level: 'low', 'medium', or 'high'")
    caffeine_intake: float = Field(..., ge=0, le=10, description="Daily caffeine intake")
    assignment_overload: int = Field(..., ge=1, le=10, description="Workload pressure (1-10)")
    extracurricular_activities: int = Field(..., ge=0, le=10, description="Number of activities")

    class Config:
        example = {
            "sleep_hours": 7,
            "study_hours_per_day": 4,
            "social_interaction_score": 8,
            "exercise_hours_per_week": 5,
            "academic_performance": 85,
            "exam_anxiety_level": 5,
            "family_income_level": "medium",
            "caffeine_intake": 2,
            "assignment_overload": 6,
            "extracurricular_activities": 2
        }

class PredictionOutput(BaseModel):
    """Output model for prediction response"""
    stress_level: str = Field(..., description="Predicted stress level")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    probabilities: dict = Field(..., description="Probability for each class")
    recommendations: List[str] = Field(..., description="Health recommendations")

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    model_loaded: bool
    version: str

class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    task_type: str
    classes: List[str]
    n_features: int
    version: str

# ==================== HELPER FUNCTIONS ====================

def get_recommendations(stress_level: str) -> List[str]:
    """Generate personalized recommendations based on stress level"""
    recommendations = {
        'Low': [
            "✓ Maintain your current stress management strategies",
            "✓ Continue regular exercise and social activities",
            "✓ Keep up your good sleep schedule",
            "✓ Consider mentoring other students"
        ],
        'Medium': [
            "⚠ Increase exercise to at least 3 hours per week",
            "⚠ Aim for 7-8 hours of sleep per night",
            "⚠ Consider time management strategies",
            "⚠ Talk to friends or mentors about stress",
            "⚠ Reduce caffeine intake"
        ],
        'High': [
            "🔴 Please reach out to student counseling services",
            "🔴 Consider reducing course load if possible",
            "🔴 Prioritize sleep (aim for 8+ hours)",
            "🔴 Engage in stress-relief activities (yoga, meditation)",
            "🔴 Talk to academic advisor about deadline extensions",
            "🔴 Contact mental health professionals: hotline/counselor"
        ]
    }
    return recommendations.get(stress_level, [])

# ==================== API ENDPOINTS ====================

@app.get("/", tags=["Information"])
def root():
    """Root endpoint - API information"""
    return {
        "message": "Student Mental Health Classifier API",
        "version": "1.0.0",
        "description": "ML-powered API for predicting student stress levels",
        "endpoints": {
            "documentation": "/docs",
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict"
        }
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        model_loaded=True,
        version="1.0.0"
    )

@app.get("/model-info", response_model=ModelInfoResponse, tags=["Information"])
def get_model_info():
    """Get model information"""
    return ModelInfoResponse(
        model_name="XGBoost",
        task_type="Multiclass Classification",
        classes=list(encoder.classes_),
        n_features=len(feature_names),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(input_data: StudentHealthInput):
    """
    Make prediction for student stress level
    
    **Request Body:**
    - sleep_hours: Daily sleep duration (0-24 hours)
    - study_hours_per_day: Daily study time (0-24 hours)
    - social_interaction_score: Social engagement (1-10)
    - exercise_hours_per_week: Weekly exercise (0-100 hours)
    - academic_performance: Academic score (0-100)
    - exam_anxiety_level: Anxiety level during exams (1-10)
    - family_income_level: Socioeconomic status ('low', 'medium', 'high')
    - caffeine_intake: Daily caffeine consumption
    - assignment_overload: Workload pressure (1-10)
    - extracurricular_activities: Number of activities (0-10)
    
    **Response:**
    - stress_level: Predicted stress category (Low/Medium/High)
    - confidence: Model confidence (0-1)
    - probabilities: Probability distribution across classes
    - recommendations: Personalized health recommendations
    """
    try:
        # Validate input
        if input_data.family_income_level not in ['low', 'medium', 'high']:
            raise ValueError("family_income_level must be 'low', 'medium', or 'high'")
        
        # Create dictionary from input
        data_dict = input_data.dict()
        income_level = data_dict.pop('family_income_level')
        
        # One-hot encode family_income_level
        data_dict['family_income_level_high'] = 1 if income_level == 'high' else 0
        data_dict['family_income_level_medium'] = 1 if income_level == 'medium' else 0
        
        # Create DataFrame with same structure as training data
        df_input = pd.DataFrame([data_dict])
        
        # Ensure column order matches training features
        df_input = df_input[feature_names]
        
        # Scale features using fitted scaler
        scaled_input = scaler.transform(df_input)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]
        
        # Get stress level and confidence
        stress_level = encoder.classes_[prediction]
        confidence = float(probabilities.max())
        
        # Create probability dictionary
        prob_dict = {
            encoder.classes_[i]: float(probabilities[i])
            for i in range(len(encoder.classes_))
        }
        
        # Get recommendations
        recommendations = get_recommendations(stress_level)
        
        return PredictionOutput(
            stress_level=stress_level,
            confidence=confidence,
            probabilities=prob_dict,
            recommendations=recommendations
        )
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict", tags=["Prediction"])
def batch_predict(inputs: List[StudentHealthInput]):
    """
    Make predictions for multiple students
    
    Returns list of predictions with stress levels and confidence scores
    """
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
                results.append({
                    "error": str(e),
                    "input": input_data.dict()
                })
        
        return {"predictions": results, "count": len(results)}
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/sample-prediction", tags=["Examples"])
def sample_prediction():
    """Get a sample prediction result"""
    sample_input = StudentHealthInput(
        sleep_hours=7.5,
        study_hours_per_day=4.0,
        social_interaction_score=8,
        exercise_hours_per_week=5.0,
        academic_performance=85.0,
        exam_anxiety_level=4,
        family_income_level="medium",
        caffeine_intake=2.0,
        assignment_overload=5,
        extracurricular_activities=2
    )
    return predict(sample_input)

# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Student Mental Health Classifier API...")
    print("📍 API running on http://localhost:8000")
    print("📚 Interactive docs: http://localhost:8000/docs")
    print("🔄 Alternative docs: http://localhost:8000/redoc")
    print("\nPress Ctrl+C to stop")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
