from sqlalchemy import Column , Integer , Float

from app.db.database import Base

class Patient(Base):
    __tablename__="patients"

    id = Column(Integer , primary_key = True , index = True)
    age = Column(Integer , nullable = False)
    systolic_bp = Column(Integer , nullable = False)
    diastolic_bp = Column(Integer , nullable = False)
    blood_sugar = Column (Float , nullable = False)
    heart_rate = Column (Integer , nullable = False)