from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from routers.auth.deps import get_current_user
from database import schemas,models
from database.database import engine, Base   
from routers.auth.routes import router as auth_router
from routers.cities.routes import router as cities_router
from routers.routes.routes import router as routes_router
from routers.emergency.routes import router as emergency_router
from routers.attractions.routes import router as attractions_router
from routers.profile import routes as profile_routes
from routers.recommendations.routes import router as recommendations_router
 
app = FastAPI()

Base.metadata.create_all(bind=engine)

# Include routers
app.include_router(auth_router)
app.include_router(cities_router)
app.include_router(routes_router)
app.include_router(emergency_router)
app.include_router(attractions_router)
app.include_router(profile_routes.router)
app.include_router(recommendations_router)

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# Add production Vercel URL from env
vercel_url = os.getenv("FRONTEND_URL")
if vercel_url:
    origins.append(vercel_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "GuardMyTrip API is running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/me", response_model=schemas.UserResponse)
def read_me(current_user = Depends(get_current_user)):
    return current_user
