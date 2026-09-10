
from fastapi import  FastAPI , Depends
from fastapi.middleware . cors import CORSMiddleware
from sqlalchemy.orm import Session
from pathlib import Path
import pandas as pd
from app.schemas.risk import RiskRequest,PatientCreate
from app.db.database import engine , Base ,SessionLocal , get_db
from app.models.patient import Patient
from app.models.risk_assessment import RiskAsessment
from app.services.risk_service import predict_risk



app = FastAPI( title="MedGuard AI" , description = "Intelligent Healthcare Risk Assessment Platform" , version="0.1.0" ,)

app.add_middleware( CORSMiddleware ,allow_origins=["*"] , allow_credentials=True , allow_methods=["*"], allow_headers=["*"] , )

Base.metadata.create_all(bind = engine)

@app.get("/")
def home(): 
    return {
        "message" : "MedGuard AI is running !" , "status" : "success" ,}


@app.get("/health")
def health_check() :
    return{
        "status" : "healthy"
    }

@app.get("/feature_importance")
def get_feature_importance():
    BASE_DIR = Path(__file__).resolve().parent
    FEATURE_IMPORTANCE_PATH = (BASE_DIR / "ml" / "feature_importance.csv")

    if not FEATURE_IMPORTANCE_PATH.exists():
        return {"message":"Feature importance file not found" , "status": "error"}

    df = pd.read_csv(FEATURE_IMPORTANCE_PATH)

    return {"status": "success" , "features": df.to_dict(orient="records")}

@app.post("/risk-assessment")
def risk_assessment(data:RiskRequest , db:Session = Depends(get_db)):
    result = predict_risk(data)
    assessment =RiskAsessment(prediction= result["prediction"] ,
                               risk_level = result["risk_level"] , 
                               confidence = result["confidence"])

    db.add(assessment)
    db.commit()
    db.refresh(assessment)

    return {"message": "Risk assessment completed successfully" , "result":result }

@app.get("/risk-assessments/count")
def get_risk_assessments_count(db:Session = Depends(get_db)):

    count = db.query(RiskAsessment).count()

    return{"total_assessments": count
    }



@app.get("/risk-assessments")
def get_risk_assessments(db:Session = Depends(get_db)):
    assessments = db.query(RiskAsessment).all()

    return {
        "assessments":[{"id":assessment.id,"prediction" :assessment.prediction,"risk_level":assessment.risk_level ,
                         "confidence": assessment.confidence}

        for assessment in assessments
        ]
    }
                    


                
@app.post("/patients")
def create_patient(patient : PatientCreate, db : Session = Depends(get_db)):
    new_patient = Patient(**patient.model_dump())
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)

    return {"message" : "Patient created successfuly" , "patient_id" : new_patient.id}
@app.get("/patients")
def get_patients(db : Session = Depends(get_db)):
    Patients = db.query(Patient).all()
    return { "patients": Patients}
@app.get("/patients/{patient_id}")
def get_patient(patient_id : int , db : Session = Depends(get_db)):

    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient :
        return{"message" : "Patient not found"}
    return{"patient": patient}

@app.put("/patient/{patient_id}")
def update_patient(patient_id : int , patient_data : PatientCreate, db : Session= Depends(get_db)):

    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient :
        return{"message" : "Patient not found"}
    patient.age = patient_data.age
    patient.systolic_bp=patient_data.systolic_bp
    patient.diastolic_bp=patient_data.diastolic_bp
    patient.blood_sugar = patient_data.blood_sugar 
    patient.heart_rate = patient_data.heart_rate

    db.commit()
    db.refresh(patient)

    return{"message": "Patient updated successfully" , "patient_id" : patient.id}


@app.delete("/patients/{patient_id}")
def deleted_patient(patient_id : int, db : Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        return{"message": "Patient not found"}
    db.delete(patient)
    db.commit()

    return{"message" : "Patient deleted successfuly" , "patient_id" :patient_id}

@app.get("/feature-importance")
def get_feature_importance():
    base_dir = Path(__file__).resolve().parents[1]
    feature_path =(base_dir / "app" / "ml" / "feature_importance.csv")

    if not feature_path.exists():
        return{"message" : "Feature importance file not found"}
    df = pd.read_csv(feature_path)
    return{"feature": df.to_dict(orient="records")}