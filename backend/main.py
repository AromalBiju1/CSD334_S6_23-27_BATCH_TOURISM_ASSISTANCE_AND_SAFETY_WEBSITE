from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.deps import get_current_user
from database import schemas

from database.database import engine, Base   
from database import models   

from auth.routes import router as auth_router

app = FastAPI()

Base.metadata.create_all(bind=engine)
app.include_router(auth_router)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/me", response_model=schemas.UserResponse)
def read_me(current_user = Depends(get_current_user)):
    return current_user