from pydantic import BaseModel 

class RiskRequest(BaseModel):
    HighBP: int
    HighChol: int
    CholCheck: int
    BMI : int
    Smoker: int
    Stroke:int
    HeartDiseaseorAttack: int
    PhysActivity: int
    Fruits: int
    Veggies: int
    HvyAlcoholConsump: int
    AnyHealthcare: int
    NoDocbcCost: int
    GenHlth: int
    MentHlth: int
    PhysHlth: int
    DiffWalk: int
    Sex: int
    Age: int
    Education: int
    Income: int

class PatientCreate(BaseModel):
    age: int
    systolic_bp: int
    diastolic_bp: int
    blood_sugar: float
    heart_rate: int