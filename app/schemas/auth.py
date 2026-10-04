from pydantic import BaseModel,Field

class UserCreate(BaseModel):
    username: str = Field (min_length=3 , max_length=50)
    email: str = Field(min_length= 5 , max_length=255)
    password:str = Field(min_length=8 , max_length=128)


class UserLogin(BaseModel):
    username:str
    password:str


class TokenResponse(BaseModel):
    access_token:str
    token_type: str ="bearer"


class UserResponse(BaseModel):
    id: int
    username:str
    email:str
    is_active:bool

    class Config:
        from_attributes = True