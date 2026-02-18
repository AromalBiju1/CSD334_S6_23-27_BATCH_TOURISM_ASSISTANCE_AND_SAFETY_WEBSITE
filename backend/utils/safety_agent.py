"""
Safety Agent — Route Safety Scoring & Decision Making
Overlays OSRM road routes against district safety zones from the database
to score, rank, and explain route recommendations.
"""

import math
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session
from database import models


# ────────────────────────────────────────────
# Zone scoring constants
# ────────────────────────────────────────────
ZONE_SCORES = {"green": 100, "orange": 55, "red": 10}

# Composite score weights
WEIGHT_SAFETY = 0.60
WEIGHT_DISTANCE = 0.20
WEIGHT_DURATION = 0.20


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def find_nearest_city(lat: float, lng: float, cities: List[models.City], max_distance_km: float = 100) -> Optional[models.City]:
    """Find the nearest city/district to a given point."""
    nearest = None
    min_dist = float("inf")
    for city in cities:
        dist = haversine_distance(lat, lng, city.latitude, city.longitude)
        if dist < min_dist and dist <= max_distance_km:
            min_dist = dist
            nearest = city
    return nearest


def sample_route_points(geometry: List[List[float]], num_samples: int = 20) -> List[List[float]]:
    """
    Sample evenly-spaced points along a route geometry.
    We don't check every single point (could be thousands), just a representative sample.
    """
    if len(geometry) <= num_samples:
        return geometry

    step = len(geometry) / num_samples
    samples = []
    for i in range(num_samples):
        idx = int(i * step)
        samples.append(geometry[idx])

    # Always include the last point
    if geometry[-1] not in samples:
        samples.append(geometry[-1])

    return samples


def score_route_safety(
    geometry: List[List[float]],
    cities: List[models.City],
    num_samples: int = 25
) -> Dict[str, Any]:
    """
    Score a route's safety by sampling points and checking which safety zones they pass through.

    Returns:
        {
            "safety_score": float (0-100),
            "zone_breakdown": {"green": %, "orange": %, "red": %, "unknown": %},
            "risk_zones": [{"name": ..., "zone": ..., "lat": ..., "lng": ...}],
            "warnings": ["..."]
        }
    """
    sample_points = sample_route_points(geometry, num_samples)

    zone_counts = {"green": 0, "orange": 0, "red": 0, "unknown": 0}
    risk_zones_seen = set()
    risk_zones = []

    for point in sample_points:
        lat, lng = point[0], point[1]
        nearest = find_nearest_city(lat, lng, cities, max_distance_km=80)

        if nearest:
            zone = nearest.safety_zone or "orange"
            zone_counts[zone] = zone_counts.get(zone, 0) + 1

            # Track high-risk areas
            if zone == "red" and nearest.id not in risk_zones_seen:
                risk_zones_seen.add(nearest.id)
                risk_zones.append({
                    "name": nearest.name,
                    "state": nearest.state,
                    "zone": zone,
                    "lat": nearest.latitude,
                    "lng": nearest.longitude,
                    "crime_index": nearest.crime_index
                })
        else:
            zone_counts["unknown"] += 1

    total = sum(zone_counts.values()) or 1

    # Calculate safety score (weighted average of zone scores)
    safety_score = 0
    for zone, count in zone_counts.items():
        if zone != "unknown":
            safety_score += ZONE_SCORES.get(zone, 50) * count
        else:
            safety_score += 60 * count  # Assume moderate for unknown areas
    safety_score = round(safety_score / total, 1)

    # Zone percentages
    zone_breakdown = {
        zone: round(count / total * 100, 1) for zone, count in zone_counts.items()
    }

    # Generate warnings
    warnings = []
    for rz in risk_zones:
        warnings.append(f"Route passes near high-risk area: {rz['name']}, {rz['state']}")

    if zone_breakdown.get("red", 0) > 30:
        warnings.insert(0, "⚠️ Over 30% of this route passes through high-risk zones")

    return {
        "safety_score": safety_score,
        "zone_breakdown": zone_breakdown,
        "risk_zones": risk_zones,
        "warnings": warnings
    }


def compute_composite_score(
    safety_score: float,
    distance_km: float,
    duration_min: float,
    all_distances: List[float],
    all_durations: List[float]
) -> float:
    """
    Compute a composite score that balances safety, distance, and duration.
    All scores are normalized to 0-100.
    """
    # Normalize distance (shorter = better)
    max_dist = max(all_distances) if all_distances else distance_km
    min_dist = min(all_distances) if all_distances else distance_km
    dist_range = max_dist - min_dist if max_dist != min_dist else 1
    distance_score = 100 - ((distance_km - min_dist) / dist_range * 100)

    # Normalize duration (shorter = better)
    max_dur = max(all_durations) if all_durations else duration_min
    min_dur = min(all_durations) if all_durations else duration_min
    dur_range = max_dur - min_dur if max_dur != min_dur else 1
    duration_score = 100 - ((duration_min - min_dur) / dur_range * 100)

    composite = (
        WEIGHT_SAFETY * safety_score +
        WEIGHT_DISTANCE * distance_score +
        WEIGHT_DURATION * duration_score
    )

    return round(composite, 1)


def classify_route_type(
    index: int,
    safety_score: float,
    distance_km: float,
    all_safety: List[float],
    all_distances: List[float]
) -> str:
    """Determine the route type label based on its characteristics."""
    max_safety = max(all_safety) if all_safety else safety_score
    min_dist = min(all_distances) if all_distances else distance_km

    if safety_score == max_safety:
        return "safest"
    elif distance_km == min_dist:
        return "fastest"
    else:
        return "balanced"


def generate_agent_reasoning(
    route_type: str,
    safety_score: float,
    zone_breakdown: Dict[str, float],
    risk_zones: List[Dict],
    distance_km: float,
    duration_min: float,
    is_recommended: bool
) -> str:
    """
    Generate human-readable agent reasoning for why a route was ranked/recommended.
    This is the 'agent decision making' part.
    """
    reasons = []

    if is_recommended:
        reasons.append("✅ **Recommended by Safety Agent**")

    # Safety assessment
    if safety_score >= 80:
        reasons.append(f"🟢 High safety score ({safety_score}/100) — majority of route passes through safe zones")
    elif safety_score >= 55:
        reasons.append(f"🟡 Moderate safety score ({safety_score}/100) — some segments pass through moderate-risk areas")
    else:
        reasons.append(f"🔴 Low safety score ({safety_score}/100) — significant portions pass through high-risk zones")

    # Zone breakdown
    green_pct = zone_breakdown.get("green", 0)
    red_pct = zone_breakdown.get("red", 0)

    if green_pct > 60:
        reasons.append(f"🛡️ {green_pct:.0f}% of route is in safe zones")
    if red_pct > 0:
        reasons.append(f"⚠️ {red_pct:.0f}% of route passes through high-risk zones")

    # Risk zone details
    if risk_zones:
        names = [f"{rz['name']}" for rz in risk_zones[:3]]
        reasons.append(f"📍 High-risk areas near route: {', '.join(names)}")
    else:
        reasons.append("🛡️ Route avoids all known high-risk areas")

    # Route type assessment
    if route_type == "safest":
        reasons.append("🏆 This route prioritizes your safety with the least exposure to high-risk districts")
    elif route_type == "fastest":
        reasons.append(f"⚡ Shortest route at {distance_km:.1f} km, but may pass through less safe areas")
    elif route_type == "balanced":
        reasons.append("⚖️ This route balances safety and distance for an optimal journey")

    return "\n".join(reasons)


def evaluate_routes(
    osrm_routes: List[Dict[str, Any]],
    cities: List[models.City],
    origin_name: str = "",
    dest_name: str = ""
) -> Dict[str, Any]:
    """
    Main agent function: evaluate all OSRM routes and return scored, ranked results.

    Args:
        osrm_routes: List of route dicts from OSRM (with geometry, distance_km, duration_min)
        cities: All cities from DB
        origin_name: Display name for origin
        dest_name: Display name for destination

    Returns:
        {
            "routes": [...scored and ranked routes...],
            "recommended_index": int,
            "agent_summary": str
        }
    """
    scored_routes = []

    # First pass: score all routes
    all_distances = [r["distance_km"] for r in osrm_routes]
    all_durations = [r["duration_min"] for r in osrm_routes]

    for route in osrm_routes:
        safety_result = score_route_safety(route["geometry"], cities)
        scored_routes.append({
            **route,
            **safety_result,
            "composite_score": 0  # will be computed after
        })

    # Second pass: compute composite scores and classify
    all_safety = [r["safety_score"] for r in scored_routes]

    for i, route in enumerate(scored_routes):
        route["composite_score"] = compute_composite_score(
            route["safety_score"],
            route["distance_km"],
            route["duration_min"],
            all_distances,
            all_durations
        )
        route["route_type"] = classify_route_type(
            i, route["safety_score"], route["distance_km"],
            all_safety, all_distances
        )

    # Sort by composite score (highest first)
    scored_routes.sort(key=lambda r: r["composite_score"], reverse=True)

    # Determine recommended route
    recommended_idx = 0  # Best composite score

    # Generate agent reasoning for each route
    for i, route in enumerate(scored_routes):
        route["agent_reasoning"] = generate_agent_reasoning(
            route["route_type"],
            route["safety_score"],
            route["zone_breakdown"],
            route["risk_zones"],
            route["distance_km"],
            route["duration_min"],
            is_recommended=(i == recommended_idx)
        )

    # Generate agent summary
    best = scored_routes[recommended_idx]
    agent_summary = (
        f"Analyzed {len(scored_routes)} route{'s' if len(scored_routes) > 1 else ''} "
        f"from {origin_name or 'origin'} to {dest_name or 'destination'}. "
        f"Recommended the {best['route_type']} route with safety score "
        f"{best['safety_score']}/100 ({best['distance_km']:.1f} km, "
        f"{int(best['duration_min'])} min)."
    )

    if best.get("risk_zones"):
        avoided = len(best["risk_zones"])
        agent_summary += f" Route passes near {avoided} high-risk area{'s' if avoided > 1 else ''}."
    else:
        agent_summary += " Route successfully avoids all high-risk areas."

    return {
        "routes": scored_routes,
        "recommended_index": recommended_idx,
        "agent_summary": agent_summary
    }
