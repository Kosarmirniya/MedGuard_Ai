from pwdlib import PasswordHash
password_hash = PasswordHash.recommended()
from datetime import datetime , timedelta , timezone
from jose import jwt
from app.core.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES

def hash_password(password: str) -> str:
    return password_hash.hash(password)

def verify_password(plain_password: str , hashed_password:str ) -> bool:
    return password_hash.verify(plain_password , hashed_password)

def create_access_token(data: dict) -> str :
    to_encode = data.copy()

    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})

    encoded_jwt=jwt.encode(to_encode , SECRET_KEY ,algorithm=ALGORITHM)

    return encoded_jwt