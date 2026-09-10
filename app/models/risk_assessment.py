from sqlalchemy import Column , Integer , Float , String 
from app.db.database import Base

class RiskAsessment(Base):
    __tablename__ = "risk_assessments"

    id = Column(Integer , primary_key=True , index=True)
    patient_id = Column(Integer , nullable=True)
    prediction = Column(Integer , nullable=False)
    risk_level = Column(String , nullable=False)
    confidence = Column(Float, nullable=False)