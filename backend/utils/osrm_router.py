"""
OSRM Router — Real Road Routing via OpenStreetMap
Uses the public OSRM API to get actual road geometries instead of straight lines.
Supports multiple route alternatives.
"""

import httpx
import polyline
import math
from typing import List, Dict, Any, Optional, Tuple


# Public OSRM demo server (free, no API key required)
OSRM_BASE_URL = "https://router.project-osrm.org"

# Timeout settings
REQUEST_TIMEOUT = 15.0  # seconds


def decode_osrm_geometry(encoded: str) -> List[List[float]]:
    """
    Decode OSRM's encoded polyline geometry into [[lat, lng], ...].
    OSRM uses Google's polyline encoding (precision 5).
    """
    decoded = polyline.decode(encoded)  # returns [(lat, lng), ...]
    return [[lat, lng] for lat, lng in decoded]


async def get_osrm_routes(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float,
    alternatives: int = 3,
    profile: str = "driving"
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch real road routes from OSRM.

    Args:
        origin_lat, origin_lng: Start coordinates
        dest_lat, dest_lng: End coordinates
        alternatives: Number of alternative routes to request (max ~3)
        profile: OSRM profile — 'driving', 'walking', or 'cycling'

    Returns:
        List of route dicts, each containing:
        - geometry: [[lat, lng], ...] points following actual roads
        - distance_km: total distance in km
        - duration_min: estimated duration in minutes
        - steps: turn-by-turn instructions
        Or None if OSRM request fails
    """
    # OSRM expects coordinates as lng,lat (reversed from our lat,lng format)
    coords = f"{origin_lng},{origin_lat};{dest_lng},{dest_lat}"

    url = f"{OSRM_BASE_URL}/route/v1/{profile}/{coords}"

    params = {
        "alternatives": "true",          # Request alternatives (OSRM treats 'true' as boolean or number)
        "geometries": "polyline",        # encoded polyline format
        "overview": "full",              # full route geometry
        "steps": "true",                 # turn-by-turn steps
        "annotations": "distance,duration"  # segment-level data
    }
    
    print(f"🌍 Requesting OSRM: {url}")

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(url, params=params)
            
            if response.status_code != 200:
                print(f"❌ OSRM HTTP Error: {response.status_code} - {response.text}")
                return None
                
            data = response.json()

        if data.get("code") != "Ok":
            print(f"⚠️ OSRM API Error: {data.get('code')} — {data.get('message', 'Unknown error')}")
            return None

        found_routes = data.get("routes", [])
        print(f"✅ OSRM Success: Found {len(found_routes)} route(s)")
        
        routes = []
        for i, route in enumerate(found_routes):
            # Decode the polyline geometry to lat/lng points
            geometry = decode_osrm_geometry(route["geometry"])

            # Extract distance and duration
            distance_m = route.get("distance", 0)  # meters
            duration_s = route.get("duration", 0)   # seconds

            # Extract turn-by-turn steps
            steps = []
            for leg in route.get("legs", []):
                for step in leg.get("steps", []):
                    # Filter out 'depart' and 'arrive' if desired, but keeping them is fine
                    maneuver = step.get("maneuver", {})
                    steps.append({
                        "instruction": step.get("name") or maneuver.get("type", "maneuver"),
                        "distance_m": step.get("distance", 0),
                        "duration_s": step.get("duration", 0),
                        "type": maneuver.get("type", "")
                    })

            routes.append({
                "geometry": geometry,
                "distance_km": round(distance_m / 1000, 1),
                "duration_min": round(duration_s / 60, 1),
                "steps": steps,
                "osrm_index": i
            })

        return routes if routes else None

    except httpx.TimeoutException:
        print("❌ OSRM Request Timed Out")
        return None
    except Exception as e:
        print(f"❌ OSRM Exception: {e}")
        return None


def get_osrm_routes_sync(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float,
    alternatives: int = 3,
    profile: str = "driving"
) -> Optional[List[Dict[str, Any]]]:
    """
    Synchronous version of get_osrm_routes for non-async contexts.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an existing event loop — create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    get_osrm_routes(origin_lat, origin_lng, dest_lat, dest_lng, alternatives, profile)
                )
                return future.result(timeout=20)
        else:
            return loop.run_until_complete(
                get_osrm_routes(origin_lat, origin_lng, dest_lat, dest_lng, alternatives, profile)
            )
    except Exception:
        return asyncio.run(
            get_osrm_routes(origin_lat, origin_lng, dest_lat, dest_lng, alternatives, profile)
        )


def format_duration(minutes: float) -> str:
    """Format duration in minutes to human-readable string."""
    if minutes < 60:
        return f"{int(minutes)} min"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}m"


def format_distance(km: float) -> str:
    """Format distance in km to human-readable string."""
    if km < 1:
        return f"{int(km * 1000)} m"
    return f"{km:.1f} km"
