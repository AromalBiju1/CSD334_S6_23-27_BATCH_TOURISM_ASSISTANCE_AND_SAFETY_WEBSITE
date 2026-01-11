from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database.database import get_db
from database import models, schemas
from auth.utility import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/signup", response_model=schemas.Token)
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):

    # check if user exists
    existing_user = db.query(models.User).filter(
        models.User.email == user.email
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    # hash password
    hashed_password = hash_password(user.password)

    # create user
    new_user = models.User(
        email=user.email,
        name=user.name,
        password=hashed_password,
        google_id=user.google_id
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # create JWT
    token = create_access_token({"user_id": new_user.id})

    return {"access_token": token, "token_type": "bearer"}





@router.post("/login", response_model=schemas.Token)
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):

    db_user = db.query(models.User).filter(
        models.User.email == user.email
    ).first()

    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    if not verify_password(user.password, db_user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    token = create_access_token({"user_id": db_user.id})

    return {
    "access_token": token,
    "token_type": "bearer"
}

