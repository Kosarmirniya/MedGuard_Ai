import logging
from fastapi import  FastAPI , Depends ,Request , HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware . cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
import pandas as pd
from app.schemas.risk import (RiskRequest,PatientCreate , PatientResponse , RiskResponse , RiskAssessmentResponse,
                              RiskAssessmentCountResponse,PatientUpdateResponse, PatientDeleteResponse)
from app.db.database import engine , Base ,SessionLocal , get_db
from app.models.patient import Patient
from app.models.risk_assessment import RiskAsessment
from app.services.risk_service import predict_risk
from app.core.config import FEATURE_IMPORTANCE_PATH
from app.core.config import  DEBUG , ALLOWED_ORIGINS




logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")



app = FastAPI( title="MedGuard AI" , description = "Intelligent Healthcare Risk Assessment Platform" , version="0.1.0" , debug=DEBUG)
@app.exception_handler(Exception)
async def global_exception_handler(request:Request, exc:Exception):
    logger.exception("Unhandled exception occurred")
    return JSONResponse(status_code=500, content= {"success": False,"error":"Internal Server Error" , "detail":"An unexpected error occurred."})

app.add_middleware( CORSMiddleware ,allow_origins=ALLOWED_ORIGINS , allow_credentials=True , allow_methods=["*"], allow_headers=["*"] , )

Base.metadata.create_all(bind = engine)

@app.get("/")
def home(): 
    return {
        "message" : "MedGuard AI is running !" , "status" : "success" ,}


@app.get("/health",
    tags=["Health"] , summary="Check API and database health")

def health_check() :
    db = SessionLocal()

    try:
        db.execute(text("SELECT 1"))
        return {
            "status" : "healthy",
            "database" : "connected"
        }

    except Exception:
        return JSONResponse(status_code=503 ,
        content={"status": "unhealthy" , "database":"disconnected"})

    finally:
        db.close()


@app.post("/risk-assessment" , response_model=RiskResponse , tags= ["Risk Assesment"] , summary="Perform a healthcare risk assessment")
def risk_assessment(data:RiskRequest , db:Session = Depends(get_db)):
    result = predict_risk(data)
    assessment =RiskAsessment(prediction= result["prediction"] ,
                               risk_level = result["risk_level"] , 
                               confidence = result["confidence"])

    db.add(assessment)

    try:
        db.commit()
        db.refresh(assessment)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500 , detail="Failed to save risk assessment")
    return {"message": "Risk assessment completed successfully" , "result":result }

@app.get("/risk-assessments/count" , response_model= RiskAssessmentCountResponse ,tags=["Risk Assessment"] ,
          summary="Get total number of risk assessments" )
def get_risk_assessments_count(db:Session = Depends(get_db)):

    count = db.query(RiskAsessment).count()

    return{"total_assessments": count
    }



@app.get("/risk-assessments" , response_model=list[RiskAssessmentResponse] , tags=["Risk Assessment"] , summary="Get all risk assessments")
def get_risk_assessments(db:Session = Depends(get_db)):
    assessments = db.query(RiskAsessment).all()
    return assessments
                    
                
@app.post("/patients" , status_code=201 , response_model=PatientResponse , tags=["Patients"] , summary="Create  a new patient")

def create_patient(patient : PatientCreate, db : Session = Depends(get_db)):
    new_patient = Patient(**patient.model_dump())
    db.add(new_patient)

    try:
        db.commit()
        db.refresh(new_patient)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500 , detail="Failed to create patient")
    return new_patient
@app.get("/patients" , response_model=list[PatientResponse] , tags=["Patients"] , summary="Get all patients")
def get_patients(db : Session = Depends(get_db)):
    patients = db.query(Patient).all()
    return patients
@app.get("/patients/{patient_id}" , response_model=PatientResponse , tags=["Patients"] , summary="Get a patient by ID")
def get_patient(patient_id : int , db : Session = Depends(get_db)):

    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient :
        raise HTTPException(status_code=404 , detail="Patient not found")
    return patient

@app.put("/patients/{patient_id}" , response_model=PatientUpdateResponse , tags=["Patients"] , summary="Update a patient")
def update_patient(patient_id : int , patient_data : PatientCreate, db : Session= Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient :
        raise HTTPException(status_code=404 , detail="Patient not found")

    patient.age =patient_data.age
    patient.systolic_bp=patient_data.systolic_bp
    patient.diastolic_bp=patient_data.diastolic_bp
    patient.blood_sugar =patient_data.blood_sugar
    patient.heart_rate =patient_data.heart_rate
    try:
        db.commit()
        db.refresh(patient)
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500 , detail="Failed to update patient")
    return{"message": "Patient updated successfully" , "patient_id" : patient.id}


@app.delete("/patients/{patient_id}" , response_model=PatientDeleteResponse , tags=["Patients"] , summary= "Delete a patient")
def delete_patient(patient_id : int, db : Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code= 404 , detail="Patient not found")
        db.delete(patient)
    try:
        db.commit()
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500 , detail="Failed to delete patient")
    return{"message" : "Patient deleted successfully" , "patient_id" :patient_id}

@app.get("/feature-importance" , tags=["Machine Learning"] , summary="Get model feature importance")
def get_feature_importance():
    feature_path = FEATURE_IMPORTANCE_PATH

    if not feature_path.exists():
        raise HTTPException(status_code=404 , detail="Feature importance file not found")
    df = pd.read_csv(feature_path)
    return{"feature": df.to_dict(orient="records")}