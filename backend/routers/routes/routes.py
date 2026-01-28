"""
Routes Router - A* Pathfinding with Safety Weights
Implements safe routing that avoids high-risk areas
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Tuple
import heapq
import math

from database.database import get_db
from database import models

router = APIRouter(prefix="/api/routes", tags=["routes"])


# ----- Pydantic Models -----
class RouteRequest(BaseModel):
    origin: str
    destination: str


class RoutePoint(BaseModel):
    name: str
    state: str
    latitude: float
    longitude: float
    safety_zone: str


class RouteResponse(BaseModel):
    origin: RoutePoint
    destination: RoutePoint
    path: List[List[float]]  # [[lat, lng], ...]
    waypoints: List[RoutePoint]
    distance: str
    duration: str
    safety_score: float
    route_type: str
    warnings: List[str] = []
    distance_km: float = 0.0  # Raw distance for comparison


class MultiRouteResponse(BaseModel):
    """Response containing multiple route alternatives."""
    routes: List[RouteResponse]
    recommended_index: int = 0  # Index of recommended (safest) route


# Weight profiles for different route types
ROUTE_PROFILES = {
    "safest": {
        "green": 1.0,
        "orange": 2.0,
        "red": 10.0,  # Heavy penalty - strongly avoid
        "avoid_red": True,
        "label": "Safest Route"
    },
    "balanced": {
        "green": 1.0,
        "orange": 1.3,
        "red": 2.0,  # Moderate penalty
        "avoid_red": False,
        "label": "Balanced Route"
    },
    "fastest": {
        "green": 1.0,
        "orange": 1.1,
        "red": 1.2,  # Minimal penalty - prioritize distance
        "avoid_red": False,
        "label": "Fastest Route"
    }
}


# ----- A* Algorithm Implementation -----

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in km."""
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def get_safety_weight(zone: str, weights: dict = None) -> float:
    """
    Get safety weight multiplier for A* algorithm.
    Higher weight = less likely to choose this path.
    """
    if weights is None:
        weights = {
            "green": 1.0,    # Safe - no penalty
            "orange": 1.5,   # Moderate - 50% penalty
            "red": 3.0,      # High risk - 200% penalty
        }
    return weights.get(zone, 1.5)


def build_graph(
    cities: List[models.City], 
    max_connection_distance: float = 300,
    avoid_red: bool = True,
    required_ids: set = None,
    weight_profile: dict = None
) -> dict:
    """
    Build adjacency graph from cities.
    Only connect cities within max_connection_distance km.
    
    If avoid_red=True, red zones are EXCLUDED from the graph entirely,
    UNLESS they are in required_ids (origin/destination).
    
    weight_profile: Custom weights for safety zones (from ROUTE_PROFILES)
    """
    graph = {}
    required_ids = required_ids or set()
    weights = weight_profile or {"green": 1.0, "orange": 1.5, "red": 3.0}
    
    # Filter cities - exclude red zones unless required
    filtered_cities = []
    for city in cities:
        if avoid_red and city.safety_zone == "red" and city.id not in required_ids:
            # Skip red zones that aren't required (origin/destination)
            continue
        filtered_cities.append(city)
        graph[city.id] = {
            "city": city,
            "neighbors": []
        }
    
    # Connect cities that are within distance threshold
    for i, city1 in enumerate(filtered_cities):
        for city2 in filtered_cities[i + 1:]:
            dist = haversine_distance(
                city1.latitude, city1.longitude,
                city2.latitude, city2.longitude
            )
            
            if dist <= max_connection_distance:
                # Calculate weighted cost (distance * safety penalty)
                weight1 = get_safety_weight(city1.safety_zone, weights)
                weight2 = get_safety_weight(city2.safety_zone, weights)
                avg_weight = (weight1 + weight2) / 2
                cost = dist * avg_weight
                
                graph[city1.id]["neighbors"].append((city2.id, cost, dist))
                graph[city2.id]["neighbors"].append((city1.id, cost, dist))
    
    return graph, len(filtered_cities)


def astar_search(
    graph: dict,
    start_id: int,
    goal_id: int,
    cities_lookup: dict
) -> Tuple[List[int], float, float]:
    """
    A* search algorithm to find safest path.
    
    Returns:
        - path: List of city IDs
        - total_cost: Weighted cost (includes safety penalties)
        - total_distance: Actual distance in km
    """
    if start_id not in graph or goal_id not in graph:
        return None, float('inf'), float('inf')
    
    start_city = cities_lookup[start_id]
    goal_city = cities_lookup[goal_id]
    
    # Heuristic: straight-line distance to goal
    def heuristic(city_id: int) -> float:
        city = cities_lookup[city_id]
        return haversine_distance(
            city.latitude, city.longitude,
            goal_city.latitude, goal_city.longitude
        )
    
    # Priority queue: (f_score, city_id)
    open_set = [(heuristic(start_id), start_id)]
    heapq.heapify(open_set)
    
    came_from = {}
    g_score = {start_id: 0}
    distance_to = {start_id: 0}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal_id:
            # Reconstruct path
            path = [current]
            total_dist = distance_to[current]
            
            while current in came_from:
                current = came_from[current]
                path.append(current)
            
            return path[::-1], g_score[goal_id], total_dist
        
        for neighbor_id, cost, dist in graph[current]["neighbors"]:
            tentative_g = g_score[current] + cost
            tentative_dist = distance_to[current] + dist
            
            if neighbor_id not in g_score or tentative_g < g_score[neighbor_id]:
                came_from[neighbor_id] = current
                g_score[neighbor_id] = tentative_g
                distance_to[neighbor_id] = tentative_dist
                f_score = tentative_g + heuristic(neighbor_id)
                heapq.heappush(open_set, (f_score, neighbor_id))
    
    return None, float('inf'), float('inf')


def calculate_safety_score(path_cities: List[models.City]) -> float:
    """Calculate overall safety score for a path (0-100)."""
    if not path_cities:
        return 0.0
    
    zone_scores = {"green": 100, "orange": 60, "red": 20}
    total_score = sum(zone_scores.get(c.safety_zone, 50) for c in path_cities)
    return round(total_score / len(path_cities), 1)


def format_distance(km: float) -> str:
    """Format distance for display."""
    if km < 1:
        return f"{int(km * 1000)} m"
    return f"{km:.1f} km"


def estimate_duration(km: float, avg_speed_kmh: float = 60) -> str:
    """Estimate travel duration."""
    hours = km / avg_speed_kmh
    if hours < 1:
        return f"{int(hours * 60)} min"
    else:
        h = int(hours)
        m = int((hours - h) * 60)
        return f"{h}h {m}m"


# ----- API Endpoints -----

@router.post("/safe", response_model=RouteResponse)
def get_safe_route(request: RouteRequest, db: Session = Depends(get_db)):
    """
    Find the safest route between two cities using A* algorithm.
    The algorithm weighs paths through high-risk areas more heavily.
    """
    # Find origin and destination cities
    origin_city = db.query(models.City).filter(
        models.City.name.ilike(f"%{request.origin}%")
    ).first()
    
    dest_city = db.query(models.City).filter(
        models.City.name.ilike(f"%{request.destination}%")
    ).first()
    
    if not origin_city:
        raise HTTPException(status_code=404, detail=f"Origin city '{request.origin}' not found")
    if not dest_city:
        raise HTTPException(status_code=404, detail=f"Destination city '{request.destination}' not found")
    
    # Get all cities for pathfinding
    all_cities = db.query(models.City).all()
    
    if len(all_cities) < 2:
        raise HTTPException(status_code=500, detail="Not enough cities in database for routing")
    
    # Build graph and lookup - AVOID red zones except for origin/destination
    cities_lookup = {c.id: c for c in all_cities}
    required_ids = {origin_city.id, dest_city.id}
    graph, graph_size = build_graph(
        all_cities, 
        avoid_red=True, 
        required_ids=required_ids
    )
    
    # Run A* search
    path_ids, total_cost, total_distance = astar_search(
        graph, origin_city.id, dest_city.id, cities_lookup
    )
    
    if path_ids is None:
        # No path found - return direct route with warning
        direct_distance = haversine_distance(
            origin_city.latitude, origin_city.longitude,
            dest_city.latitude, dest_city.longitude
        )
        
        return RouteResponse(
            origin=RoutePoint(
                name=origin_city.name,
                state=origin_city.state,
                latitude=origin_city.latitude,
                longitude=origin_city.longitude,
                safety_zone=origin_city.safety_zone
            ),
            destination=RoutePoint(
                name=dest_city.name,
                state=dest_city.state,
                latitude=dest_city.latitude,
                longitude=dest_city.longitude,
                safety_zone=dest_city.safety_zone
            ),
            path=[
                [origin_city.latitude, origin_city.longitude],
                [dest_city.latitude, dest_city.longitude]
            ],
            waypoints=[],
            distance=format_distance(direct_distance),
            duration=estimate_duration(direct_distance),
            safety_score=calculate_safety_score([origin_city, dest_city]),
            route_type="direct",
            warnings=["No intermediate waypoints found. Showing direct route."]
        )
    
    # Get cities in path
    path_cities = [cities_lookup[cid] for cid in path_ids]
    
    # Build response
    path_coords = [[c.latitude, c.longitude] for c in path_cities]
    
    waypoints = [
        RoutePoint(
            name=c.name,
            state=c.state,
            latitude=c.latitude,
            longitude=c.longitude,
            safety_zone=c.safety_zone
        )
        for c in path_cities[1:-1]  # Exclude origin and destination
    ]
    
    # Generate warnings for high-risk segments
    warnings = []
    for city in path_cities:
        if city.safety_zone == "red":
            warnings.append(f"High-risk area: {city.name}, {city.state}")
    
    return RouteResponse(
        origin=RoutePoint(
            name=origin_city.name,
            state=origin_city.state,
            latitude=origin_city.latitude,
            longitude=origin_city.longitude,
            safety_zone=origin_city.safety_zone
        ),
        destination=RoutePoint(
            name=dest_city.name,
            state=dest_city.state,
            latitude=dest_city.latitude,
            longitude=dest_city.longitude,
            safety_zone=dest_city.safety_zone
        ),
        path=path_coords,
        waypoints=waypoints,
        distance=format_distance(total_distance),
        duration=estimate_duration(total_distance),
        safety_score=calculate_safety_score(path_cities),
        route_type="safe_optimized",
        warnings=warnings
    )


@router.get("/alternatives", response_model=MultiRouteResponse)
def get_route_alternatives(
    origin: str,
    destination: str,
    db: Session = Depends(get_db)
):
    """
    Get multiple route alternatives with different safety/distance tradeoffs.
    Returns up to 3 routes: safest, balanced, and fastest.
    Uses A* algorithm with different weight profiles.
    """
    # Find origin and destination cities
    origin_city = db.query(models.City).filter(
        models.City.name.ilike(f"%{origin}%")
    ).first()
    
    dest_city = db.query(models.City).filter(
        models.City.name.ilike(f"%{destination}%")
    ).first()
    
    if not origin_city:
        raise HTTPException(status_code=404, detail=f"Origin city '{origin}' not found")
    if not dest_city:
        raise HTTPException(status_code=404, detail=f"Destination city '{destination}' not found")
    
    # Get all cities for pathfinding
    all_cities = db.query(models.City).all()
    cities_lookup = {c.id: c for c in all_cities}
    required_ids = {origin_city.id, dest_city.id}
    
    routes = []
    seen_paths = set()  # Track unique paths to avoid duplicates
    
    # Run A* with each weight profile
    for profile_name, profile in ROUTE_PROFILES.items():
        # Build graph with this profile's weights
        graph, graph_size = build_graph(
            all_cities,
            max_connection_distance=500,
            avoid_red=profile.get("avoid_red", False),
            required_ids=required_ids,
            weight_profile=profile
        )
        
        # Run A* search
        path_ids, total_cost, total_distance = astar_search(
            graph, origin_city.id, dest_city.id, cities_lookup
        )
        
        if path_ids is None:
            continue
        
        # Create path signature to check for duplicates
        path_sig = tuple(path_ids)
        if path_sig in seen_paths:
            continue
        seen_paths.add(path_sig)
        
        # Get cities in path
        path_cities = [cities_lookup[cid] for cid in path_ids]
        path_coords = [[c.latitude, c.longitude] for c in path_cities]
        
        waypoints = [
            RoutePoint(
                name=c.name,
                state=c.state,
                latitude=c.latitude,
                longitude=c.longitude,
                safety_zone=c.safety_zone
            )
            for c in path_cities[1:-1]
        ]
        
        # Generate warnings for high-risk segments
        warnings = []
        for city in path_cities:
            if city.safety_zone == "red":
                warnings.append(f"High-risk area: {city.name}, {city.state}")
        
        route = RouteResponse(
            origin=RoutePoint(
                name=origin_city.name,
                state=origin_city.state,
                latitude=origin_city.latitude,
                longitude=origin_city.longitude,
                safety_zone=origin_city.safety_zone
            ),
            destination=RoutePoint(
                name=dest_city.name,
                state=dest_city.state,
                latitude=dest_city.latitude,
                longitude=dest_city.longitude,
                safety_zone=dest_city.safety_zone
            ),
            path=path_coords,
            waypoints=waypoints,
            distance=format_distance(total_distance),
            duration=estimate_duration(total_distance),
            distance_km=total_distance,
            safety_score=calculate_safety_score(path_cities),
            route_type=profile_name,
            warnings=warnings
        )
        routes.append(route)
    
    # If no routes found, try direct route
    if not routes:
        direct_distance = haversine_distance(
            origin_city.latitude, origin_city.longitude,
            dest_city.latitude, dest_city.longitude
        )
        routes.append(RouteResponse(
            origin=RoutePoint(
                name=origin_city.name,
                state=origin_city.state,
                latitude=origin_city.latitude,
                longitude=origin_city.longitude,
                safety_zone=origin_city.safety_zone
            ),
            destination=RoutePoint(
                name=dest_city.name,
                state=dest_city.state,
                latitude=dest_city.latitude,
                longitude=dest_city.longitude,
                safety_zone=dest_city.safety_zone
            ),
            path=[
                [origin_city.latitude, origin_city.longitude],
                [dest_city.latitude, dest_city.longitude]
            ],
            waypoints=[],
            distance=format_distance(direct_distance),
            duration=estimate_duration(direct_distance),
            distance_km=direct_distance,
            safety_score=calculate_safety_score([origin_city, dest_city]),
            route_type="direct",
            warnings=["No intermediate waypoints found. Showing direct route."]
        ))
    
    # Sort by safety score (highest first) - safest is index 0
    routes.sort(key=lambda r: r.safety_score, reverse=True)
    
    return MultiRouteResponse(
        routes=routes,
        recommended_index=0  # First route is safest
    )
