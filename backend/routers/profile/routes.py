# routers/profile/routes.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from database import models, schemas
from database.database import get_db
# Assuming your authentication/deps.py has a get_current_user function
from routers.auth.deps import get_current_user 

router = APIRouter(prefix="/api/profile", tags=["User Profile"])

@router.get("/me", response_model=schemas.UserProfileResponse)
def get_profile(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    # 1. Calculate Stats
    routes_count = db.query(models.SavedRoute).filter(models.SavedRoute.user_id == current_user.id).count()
    
    total_km = db.query(func.sum(models.SavedRoute.distance_km)).filter(
        models.SavedRoute.user_id == current_user.id
    ).scalar() or 0.0
    
    cities_count = db.query(func.count(func.distinct(models.SavedRoute.destination))).filter(
        models.SavedRoute.user_id == current_user.id
    ).scalar() or 0

    # 2. Return data - KEYS MUST MATCH SCHEMAS EXACTLY
    return {
        "name": current_user.name,
        "email": current_user.email,
        "profile_pic": current_user.profile_pic,
        "stats": {
            "routes_planned": routes_count,
            "cities_explored": cities_count,
            "total_km": round(total_km, 2)
        },
        "preferences": {
            "language": current_user.language,
            "theme": current_user.theme,
            "notifications_enabled": current_user.notifications_enabled  # Fixed from 'notifications'
        },
        "privacy": {
            "is_public": current_user.is_public  # Fixed from 'public_profile'
        }
    }

@router.post("/save-route")
def save_route(route_data: schemas.RouteSaveRequest, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    new_route = models.SavedRoute(
        user_id=current_user.id,
        origin=route_data.origin,
        destination=route_data.destination,
        distance_km=route_data.distance_km,
        safety_score=route_data.safety_score
    )
    db.add(new_route)
    db.commit()
    return {"message": "Route saved successfully"}


@router.delete("/clear-history")
def clear_history(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    db.query(models.SavedRoute).filter(models.SavedRoute.user_id == current_user.id).delete()
    db.commit()
    return {"message": "All saved routes deleted"}


@router.post("/change-password")
def change_password(
    data: schemas.PasswordChangeRequest, 
    db: Session = Depends(get_db), 
    current_user: models.User = Depends(get_current_user)
):
    # 1. Verify the old password
    if current_user.password != data.old_password:
        raise HTTPException(status_code=400, detail="The old password you entered is incorrect.")

    # 2. Update to the new password
    current_user.password = data.new_password
    db.commit()
    
    return {"message": "Password changed successfully!"}