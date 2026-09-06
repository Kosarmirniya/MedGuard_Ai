from fastapi import FastAPI
from fastapi import Depends
from sqlalchemy.orm import Session
from app.schemas.risk import RiskRequest,PatientCreate
from app.db.database import engine,Base,SessionLocal,get_db
from app.models.patient import Patient
from app.services.risk_service import predict_risk


app = FastAPI( title="MedGuard AI" , description = "Intelligent Healthcare Risk Assessment Platform" , version="0.1.0" ,)

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

@app.post("/risk-assessment")
def risk_assessment(data:RiskRequest):
    result = predict_risk(data)
    return{"massage" : "Risk assessment completed successfully" , "result" : result}


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