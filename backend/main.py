from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.auth.deps import get_current_user
from database import schemas,models
from database.database import engine, Base   
from routers.auth.routes import router as auth_router
from routers.cities.routes import router as cities_router
from routers.routes.routes import router as routes_router
from routers.emergency.routes import router as emergency_router

app = FastAPI()

Base.metadata.create_all(bind=engine)
app.include_router(auth_router)
app.include_router(cities_router)
app.include_router(routes_router)
app.include_router(emergency_router)

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
