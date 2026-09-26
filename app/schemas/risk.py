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


class PatientResponse(BaseModel):
    id : int
    age: int
    systolic_bp: int
    diastolic_bp: int
    blood_sugar: float
    heart_rate : int

    class Config:
        from_attributes = True

class RiskResult(BaseModel):
    prediction: int
    risk_level: str
    confidence:float
    probabilities: dict[str , float]


class RiskResponse(BaseModel):
    message:str
    result:RiskResult

class RiskAssessmentResponse(BaseModel):
    id:int
    prediction:int
    risk_level:str
    confidence:float
class RiskAssessmentCountResponse(BaseModel):
    total_assessments:int

class PatientUpdateResponse(BaseModel):
    message:str
    patient_id:int


class PatientDeleteResponse(BaseModel):
    message: str
    patient_id: int