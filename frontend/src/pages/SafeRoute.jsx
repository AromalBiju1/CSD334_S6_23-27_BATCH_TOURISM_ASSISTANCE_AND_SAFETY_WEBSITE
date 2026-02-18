import React, { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { Search, Navigation, ChevronDown, MapPin, Loader, AlertCircle, Shield, Zap, Scale, Bot, ChevronRight, Car, Locate } from "lucide-react";
import SafetyMap from "../components/SafetyMap";
import { getCities, getSmartRoutes, getRouteAlternatives, checkPositionSafety, rerouteFromPosition } from "../api/services";
import toast from "react-hot-toast";

// Route type icons and labels
const ROUTE_INFO = {
    safest: { icon: Shield, label: "Safest", color: "emerald", description: "Avoids all high-risk areas" },
    balanced: { icon: Scale, label: "Balanced", color: "amber", description: "Good safety, reasonable distance" },
    fastest: { icon: Zap, label: "Fastest", color: "rose", description: "Shortest distance" },
    direct: { icon: MapPin, label: "Direct", color: "slate", description: "Straight line" },
    safe_optimized: { icon: Shield, label: "Safe", color: "emerald", description: "Optimized for safety" }
};

// Route colors for map display
const ROUTE_COLORS = {
    safest: "#22c55e",      // green
    balanced: "#f59e0b",    // amber
    fastest: "#f43f5e",     // rose
    direct: "#64748b",      // slate
    safe_optimized: "#22c55e"
};

// Custom debounce hook for search optimization
function useDebounce(value, delay) {
    const [debouncedValue, setDebouncedValue] = useState(value);
    useEffect(() => {
        const handler = setTimeout(() => setDebouncedValue(value), delay);
        return () => clearTimeout(handler);
    }, [value, delay]);
    return debouncedValue;
}

export default function SafeRoute() {
    const [cities, setCities] = useState([]);
    const [startPoint, setStartPoint] = useState("");
    const [destination, setDestination] = useState("");
    const [startCity, setStartCity] = useState(null);
    const [destCity, setDestCity] = useState(null);

    // Multi-route state
    const [routes, setRoutes] = useState([]);
    const [selectedRouteIndex, setSelectedRouteIndex] = useState(0);
    const [recommendedIndex, setRecommendedIndex] = useState(0);
    const [loading, setLoading] = useState(false);
    const [citiesLoading, setCitiesLoading] = useState(true);
    const [animating, setAnimating] = useState(false);
    const [agentSummary, setAgentSummary] = useState("");
    const [showAgentReasoning, setShowAgentReasoning] = useState(false);

    // Navigation state
    const [isNavigating, setIsNavigating] = useState(false);
    const [currentLocation, setCurrentLocation] = useState(null);
    const [navigationStep, setNavigationStep] = useState(0);
    const [safetyAlert, setSafetyAlert] = useState(null);
    const navigationIntervalRef = useRef(null);

    // Travel mode state
    const [travelMode, setTravelMode] = useState("driving"); // driving, walking, cycling

    // Debounce search inputs (300ms delay)
    const debouncedStart = useDebounce(startPoint, 150);
    const debouncedDest = useDebounce(destination, 150);

    // Fetch cities for autocomplete
    useEffect(() => {
        const fetchCities = async () => {
            try {
                const data = await getCities();
                setCities(data);
            } catch (err) {
                console.error("Failed to load cities:", err);
            } finally {
                setCitiesLoading(false);
            }
        };
        fetchCities();
    }, []);

    // Memoized city filter function using Trie-like prefix matching for speed
    const filterCities = useCallback((query) => {
        if (!query || query.length < 2) return [];
        const q = query.toLowerCase();
        const results = [];
        for (let i = 0; i < cities.length && results.length < 10; i++) {
            const c = cities[i];
            if (c.name?.toLowerCase().includes(q) || c.state?.toLowerCase().includes(q)) {
                results.push(c);
            }
        }
        return results;
    }, [cities]);

    // Memoized suggestions using debounced values
    const startSuggestions = useMemo(() =>
        debouncedStart.length > 1 && !startCity ? filterCities(debouncedStart) : [],
        [debouncedStart, startCity, filterCities]
    );

    const destSuggestions = useMemo(() =>
        debouncedDest.length > 1 && !destCity ? filterCities(debouncedDest) : [],
        [debouncedDest, destCity, filterCities]
    );

    const handleFindRoute = async () => {
        if (!startCity || !destCity) {
            toast.error("Please select both starting point and destination");
            return;
        }

        setLoading(true);
        setRoutes([]);
        setSelectedRouteIndex(0);
        setAgentSummary("");

        try {
            // Try smart routing first (OSRM + Safety Agent)
            const data = await getSmartRoutes(startCity, destCity, travelMode);

            if (data.routes && data.routes.length > 0) {
                setRoutes(data.routes);
                setRecommendedIndex(data.recommended_index || 0);
                setSelectedRouteIndex(data.recommended_index || 0);
                setAgentSummary(data.agent_summary || "");

                // Trigger animation for recommended route
                setAnimating(true);
                setTimeout(() => setAnimating(false), 3000);

                toast.success(`🛡️ Safety Agent found ${data.routes.length} route${data.routes.length > 1 ? 's' : ''}!`);
            } else {
                toast.error("No routes found");
            }
        } catch (error) {
            console.error("Smart routing failed, trying fallback:", error);
            // Fallback to A* routing
            try {
                const fallback = await getRouteAlternatives(startCity.name, destCity.name);
                if (fallback.routes && fallback.routes.length > 0) {
                    setRoutes(fallback.routes);
                    setRecommendedIndex(fallback.recommended_index || 0);
                    setSelectedRouteIndex(fallback.recommended_index || 0);
                    setAgentSummary("Using fallback routing (road routing service unavailable).");
                    toast.success(`Found ${fallback.routes.length} route(s) via fallback`);
                } else {
                    toast.error("No routes found");
                }
            } catch (fallbackError) {
                toast.error("Failed to find route. Try again.");
                console.error(fallbackError);
            }
        } finally {
            setLoading(false);
        }
    };

    // Zone filter state - can be 'all', 'red', 'orange', 'green'
    const [zoneFilter, setZoneFilter] = useState('all');

    const selectedRoute = routes[selectedRouteIndex];

    const getZoneColor = useCallback((zone) => {
        switch (zone) {
            case "green": return "text-green-400";
            case "orange": return "text-orange-400";
            case "red": return "text-red-400";
            case "user": return "text-blue-400";
            default: return "text-slate-400";
        }
    }, []);

    // Navigation Simulation Logic
    const startNavigation = useCallback(() => {
        if (!selectedRoute?.path || selectedRoute.path.length === 0) return;

        setIsNavigating(true);
        setNavigationStep(0);
        setCurrentLocation({
            lat: selectedRoute.path[0][0],
            lng: selectedRoute.path[0][1]
        });
        setSafetyAlert(null);
        toast.success("Starting live navigation simulation...", { icon: "🚗" });
    }, [selectedRoute]);

    const stopNavigation = useCallback(() => {
        setIsNavigating(false);
        setCurrentLocation(null);
        setNavigationStep(0);
        setSafetyAlert(null);
        if (navigationIntervalRef.current) clearInterval(navigationIntervalRef.current);
    }, []);

    // Actual Navigation Loop
    useEffect(() => {
        if (!isNavigating || !selectedRoute) return;

        navigationIntervalRef.current = setInterval(async () => {
            setNavigationStep(prev => {
                const nextStep = prev + 1; // Move faster for demo: +5 points
                if (nextStep >= selectedRoute.path.length) {
                    stopNavigation();
                    toast.success("You have arrived at your destination!");
                    return prev;
                }

                const currentPos = selectedRoute.path[nextStep];
                const lat = currentPos[0];
                const lng = currentPos[1];
                setCurrentLocation({ lat, lng });

                // Check safety every 10 steps (~2 seconds in sim)
                if (nextStep % 10 === 0) {
                    checkPositionSafety(lat, lng).then(safetyData => {
                        if (safetyData.trigger_reroute) {
                            // Pause navigation and reroute
                            clearInterval(navigationIntervalRef.current);
                            setSafetyAlert({
                                type: "danger",
                                message: safetyData.message,
                                district: safetyData.nearest_district
                            });

                            toast.error("⚠️ Entering High-Risk Zone! Rerouting...", { duration: 4000 });

                            // Trigger auto-reroute
                            setTimeout(() => handleReroute(lat, lng), 2000);
                        } else if (safetyData.zone === 'orange') {
                            toast("⚠️ Caution: Moderate risk area nearby", { icon: "⚠️" });
                        }
                    });
                }

                return nextStep;
            });
        }, 200); // 200ms tick for smooth simulation

        return () => {
            if (navigationIntervalRef.current) clearInterval(navigationIntervalRef.current);
        };
    }, [isNavigating, selectedRoute]);

    const handleReroute = async (currentLat, currentLng) => {
        setLoading(true);
        try {
            const data = await rerouteFromPosition(
                currentLat,
                currentLng,
                destCity.latitude,
                destCity.longitude,
                destCity.name
            );

            if (data.routes && data.routes.length > 0) {
                setRoutes(data.routes);
                setSelectedRouteIndex(data.recommended_index);
                setNavigationStep(0); // Reset to start of new route
                setSafetyAlert(null);

                // Resume navigation on new route
                navigationIntervalRef.current = setInterval(() => {
                    // ... logic resumes in next effect cycle ...
                    // Actually we need to set isNavigating to true to ensure effect picks up new route
                    setIsNavigating(true);
                }, 200);

                toast.success("✅ Rerouted successfully to safer path!");
                setAgentSummary(data.agent_summary);
            } else {
                toast.error("Could not find a safer route. Proceed with extreme caution.");
            }
        } catch (error) {
            console.error("Reroute failed:", error);
            toast.error("Rerouting failed. Please check connection.");
        } finally {
            setLoading(false);
        }
    };

    // Memoize displayed cities to prevent re-filtering on every keystroke
    const displayedCities = useMemo(() =>
        cities
            .filter(c => zoneFilter === 'all' || c.safety_zone === zoneFilter)
            .map(c => ({
                id: c.id,
                name: c.name,
                state: c.state,
                lat: c.latitude,
                lng: c.longitude,
                zone: c.safety_zone,
                safety_zone: c.safety_zone,
                crime_index: c.crime_index
            })),
        [cities, zoneFilter]
    );

    // Memoize origin/destination/user markers
    const mapCities = useMemo(() => [
        startCity && { ...startCity, lat: startCity.latitude, lng: startCity.longitude, zone: "green", isStart: true },
        destCity && { ...destCity, lat: destCity.latitude, lng: destCity.longitude, zone: "red", isDest: true },
        currentLocation && {
            id: "user-loc",
            name: "Current Location",
            lat: currentLocation.lat,
            lng: currentLocation.lng,
            zone: "user", // Custom zone for blue marker
            safety_zone: "user",
            isUser: true
        }
    ].filter(Boolean), [startCity, destCity, currentLocation]);

    // Handlers for Map Interactions
    const handleSelectStart = useCallback((city) => {
        setStartPoint(city.name);
        setStartCity(city);
        toast.success(`Starting point set to ${city.name}`, { icon: '📍' });
    }, []);

    const handleSelectDest = useCallback((city) => {
        setDestination(city.name);
        setDestCity(city);
        toast.success(`Destination set to ${city.name}`, { icon: '🏁' });
    }, []);

    const handleUseMyLocation = useCallback(() => {
        if (!navigator.geolocation) {
            toast.error("Geolocation is not supported by your browser");
            return;
        }

        toast.loading("Locating...", { id: 'locating' });

        navigator.geolocation.getCurrentPosition(
            (position) => {
                const { latitude, longitude } = position.coords;
                const userLoc = {
                    id: 'gps-user',
                    name: 'Current Location',
                    state: 'GPS',
                    lat: latitude,
                    latitude: latitude,
                    lng: longitude,
                    longitude: longitude,
                    zone: 'user',
                    safety_zone: 'user',
                    isUserLocation: true
                };

                setStartCity(userLoc);
                setStartPoint("📍 Current Location");
                toast.success("Location set!", { id: 'locating' });
            },
            (error) => {
                console.error("Geolocation error:", error);

                let msg = "Could not get location.";
                if (error.code === 1) msg = "Location permission denied.";
                else if (error.code === 2) msg = "Position unavailable.";
                else if (error.code === 3) msg = "Location request timed out.";

                toast.error(msg, { id: 'locating' });
            },
            { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
        );
    }, []);

    // Memoize map routes array for the map component
    const mapRoutes = useMemo(() => {
        return routes.map((route, idx) => ({
            path: route.path,
            color: ROUTE_COLORS[route.route_type] || ROUTE_COLORS.direct,
            selected: idx === selectedRouteIndex,
            safe: route.safety_score >= 60,
            info: {
                distance: route.distance,
                duration: route.duration,
                type: route.route_type
            },
            animate: idx === recommendedIndex && animating
        }));
    }, [routes, selectedRouteIndex, recommendedIndex, animating]);

    return (
        <main className="pt-16 min-h-screen w-full">
            <div className="w-full max-w-6xl mx-auto px-6 md:px-16 py-8">
                {/* Header */}
                <div className="mb-8 flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                        <h1 className="text-2xl sm:text-3xl font-bold mb-2">
                            Plan Your <span className="text-emerald-400">Safe Route</span>
                        </h1>
                        <p className="text-slate-400">
                            Compare routes optimized for safety, balance, or speed
                        </p>
                    </div>

                    {/* Zone Filter Buttons */}
                    <div className="flex items-center gap-2 bg-slate-900/50 p-2 rounded-xl border border-slate-800 flex-wrap">
                        <span className="text-sm text-slate-400 pl-2">Filter:</span>
                        <button
                            onClick={() => setZoneFilter('all')}
                            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${zoneFilter === 'all'
                                ? 'bg-slate-500/20 text-white border border-slate-500/50'
                                : 'text-slate-400 hover:text-white'
                                }`}
                        >
                            All ({cities.length})
                        </button>
                        <button
                            onClick={() => setZoneFilter('green')}
                            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${zoneFilter === 'green'
                                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50'
                                : 'text-slate-400 hover:text-emerald-400'
                                }`}
                        >
                            🟢 Safe
                        </button>
                        <button
                            onClick={() => setZoneFilter('orange')}
                            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${zoneFilter === 'orange'
                                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/50'
                                : 'text-slate-400 hover:text-amber-400'
                                }`}
                        >
                            🟠 Moderate
                        </button>
                        <button
                            onClick={() => setZoneFilter('red')}
                            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${zoneFilter === 'red'
                                ? 'bg-rose-500/20 text-rose-400 border border-rose-500/50'
                                : 'text-slate-400 hover:text-rose-400'
                                }`}
                        >
                            🔴 High Risk
                        </button>
                    </div>
                </div>

                {/* Main Content Grid */}
                <div className="grid lg:grid-cols-3 gap-6">
                    {/* Left Panel - Inputs */}
                    <div className="lg:col-span-1 space-y-4">
                        {/* Starting Point Card */}
                        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-5">
                            <div className="flex items-center gap-2 text-sm text-slate-400 mb-3">
                                <div className="w-3 h-3 rounded-full bg-green-500 flex-shrink-0"></div>
                                <span>Starting Point</span>
                            </div>
                            <div className="relative">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
                                <input
                                    type="text"
                                    placeholder={citiesLoading ? "Loading cities..." : "Search city or use location..."}
                                    value={startPoint}
                                    onChange={(e) => {
                                        setStartPoint(e.target.value);
                                        setStartCity(null);
                                    }}
                                    disabled={citiesLoading}
                                    className="w-full bg-slate-800/50 border border-slate-700 rounded-xl h-11 pl-10 pr-12 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500/50 text-sm transition-colors disabled:opacity-50"
                                />

                                {/* Use My Location Button */}
                                <button
                                    onClick={handleUseMyLocation}
                                    className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-lg text-slate-400 hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors"
                                    title="Use My Location"
                                    type="button"
                                >
                                    <Locate size={18} />
                                </button>

                                {startSuggestions.length > 0 && (
                                    <div className="absolute z-10 w-full mt-1 bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow-lg">
                                        {startSuggestions.map((city) => (
                                            <button
                                                key={city.id}
                                                className="w-full px-4 py-3 text-left text-sm text-white hover:bg-slate-700 transition-colors flex justify-between items-center"
                                                onClick={() => {
                                                    setStartPoint(city.name);
                                                    setStartCity(city);
                                                }}
                                            >
                                                <span>{city.name}, {city.state}</span>
                                                <span className={`text-xs ${getZoneColor(city.safety_zone)}`}>
                                                    {city.safety_zone?.toUpperCase()}
                                                </span>
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Arrow Connector */}
                        <div className="flex justify-center py-1">
                            <div className="w-8 h-8 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center">
                                <ChevronDown className="text-slate-400" size={18} />
                            </div>
                        </div>

                        {/* Destination Card */}
                        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-5">
                            <div className="flex items-center gap-2 text-sm text-slate-400 mb-3">
                                <div className="w-3 h-3 rounded-full bg-red-500 flex-shrink-0"></div>
                                <span>Destination</span>
                            </div>
                            <div className="relative">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={16} />
                                <input
                                    type="text"
                                    placeholder={citiesLoading ? "Loading cities..." : "Search city..."}
                                    value={destination}
                                    onChange={(e) => {
                                        setDestination(e.target.value);
                                        setDestCity(null);
                                    }}
                                    disabled={citiesLoading}
                                    className="w-full bg-slate-800/50 border border-slate-700 rounded-xl h-11 pl-10 pr-4 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500/50 text-sm transition-colors disabled:opacity-50"
                                />
                                {destSuggestions.length > 0 && (
                                    <div className="absolute z-10 w-full mt-1 bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow-lg">
                                        {destSuggestions.map((city) => (
                                            <button
                                                key={city.id}
                                                className="w-full px-4 py-3 text-left text-sm text-white hover:bg-slate-700 transition-colors flex justify-between items-center"
                                                onClick={() => {
                                                    setDestination(city.name);
                                                    setDestCity(city);
                                                }}
                                            >
                                                <span>{city.name}, {city.state}</span>
                                                <span className={`text-xs ${getZoneColor(city.safety_zone)}`}>
                                                    {city.safety_zone?.toUpperCase()}
                                                </span>
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Travel Mode - Driving Only */}
                        <div className="mb-4">
                            <div className="flex flex-col items-center justify-center py-3 rounded-xl border bg-emerald-500/10 border-emerald-500/50 text-emerald-400 ring-1 ring-emerald-500/30">
                                <Car size={22} className="mb-1.5" />
                                <span className="text-[10px] uppercase font-bold tracking-wider">Drive</span>
                            </div>
                        </div>

                        {/* Find Route Button */}
                        <button
                            onClick={handleFindRoute}
                            disabled={!startCity || !destCity || loading}
                            className="w-full flex items-center justify-center gap-2 bg-emerald-500 hover:bg-emerald-600 disabled:bg-slate-700 disabled:cursor-not-allowed text-white h-12 rounded-xl font-medium transition-all"
                        >
                            {loading ? (
                                <Loader className="animate-spin" size={18} />
                            ) : (
                                <Navigation size={18} />
                            )}
                            <span>{loading ? "Finding Routes..." : "Find Routes"}</span>
                        </button>

                        {/* Route Alternatives */}
                        {routes.length > 0 && (
                            <div className="space-y-3">
                                <h3 className="text-sm font-medium text-slate-400 flex items-center gap-2">
                                    <span>Route Options</span>
                                    {routes.length > 1 && (
                                        <span className="px-2 py-0.5 bg-slate-700 rounded-full text-xs">
                                            {routes.length} available
                                        </span>
                                    )}
                                </h3>

                                {routes.map((route, idx) => {
                                    const info = ROUTE_INFO[route.route_type] || ROUTE_INFO.direct;
                                    const Icon = info.icon;
                                    const isSelected = idx === selectedRouteIndex;
                                    const isRecommended = idx === recommendedIndex;

                                    return (
                                        <button
                                            key={idx}
                                            onClick={() => setSelectedRouteIndex(idx)}
                                            className={`w-full text-left p-4 rounded-xl border transition-all ${isSelected
                                                ? `bg-${info.color}-500/10 border-${info.color}-500/50 ring-2 ring-${info.color}-500/30`
                                                : 'bg-slate-900/50 border-slate-800 hover:border-slate-700'
                                                } ${isRecommended && animating ? 'animate-pulse' : ''}`}
                                            style={{
                                                backgroundColor: isSelected ? `${ROUTE_COLORS[route.route_type]}15` : undefined,
                                                borderColor: isSelected ? `${ROUTE_COLORS[route.route_type]}50` : undefined
                                            }}
                                        >
                                            <div className="flex items-start gap-3">
                                                <div
                                                    className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                                                    style={{ backgroundColor: `${ROUTE_COLORS[route.route_type]}20` }}
                                                >
                                                    <Icon size={20} style={{ color: ROUTE_COLORS[route.route_type] }} />
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex items-center gap-2 mb-1">
                                                        <span className="font-medium text-white">{info.label}</span>
                                                        {isRecommended && (
                                                            <span className="px-1.5 py-0.5 bg-emerald-500/20 text-emerald-400 text-xs rounded">
                                                                Recommended
                                                            </span>
                                                        )}
                                                    </div>
                                                    <div className="flex items-center gap-3 text-sm text-slate-400">
                                                        <span>{route.distance}</span>
                                                        <span>•</span>
                                                        <span>{route.duration}</span>
                                                    </div>
                                                    <div className="flex items-center gap-2 mt-2">
                                                        <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                                            <div
                                                                className="h-full rounded-full transition-all"
                                                                style={{
                                                                    width: `${route.safety_score}%`,
                                                                    backgroundColor: route.safety_score >= 80 ? '#22c55e' :
                                                                        route.safety_score >= 60 ? '#f59e0b' : '#f43f5e'
                                                                }}
                                                            />
                                                        </div>
                                                        <span className={`text-xs font-medium ${route.safety_score >= 80 ? 'text-emerald-400' :
                                                            route.safety_score >= 60 ? 'text-amber-400' : 'text-rose-400'
                                                            }`}>
                                                            {route.safety_score}%
                                                        </span>
                                                    </div>
                                                    {route.warnings?.length > 0 && (
                                                        <p className="text-xs text-orange-400 mt-2 flex items-center gap-1">
                                                            <AlertCircle size={12} />
                                                            {route.warnings.length} warning{route.warnings.length > 1 ? 's' : ''}
                                                        </p>
                                                    )}
                                                </div>
                                            </div>
                                        </button>
                                    );
                                })}
                            </div>
                        )}

                        {/* Selected Route Details - or Navigation Dashboard */}
                        {isNavigating ? (
                            <div className="bg-slate-800 border border-slate-700 rounded-2xl p-5 shadow-xl ring-1 ring-cyan-500/50 relative overflow-hidden">
                                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent animate-pulse"></div>

                                <div className="flex justify-between items-center mb-4">
                                    <h4 className="flex items-center gap-2 text-cyan-400 font-bold text-lg animate-pulse">
                                        <Navigation className="animate-bounce" size={20} />
                                        Live Navigation
                                    </h4>
                                    <button
                                        onClick={stopNavigation}
                                        className="px-3 py-1 bg-red-500/20 text-red-400 rounded-lg text-xs hover:bg-red-500/30 transition-colors"
                                    >
                                        Stop
                                    </button>
                                </div>

                                {safetyAlert ? (
                                    <div className="mb-4 bg-red-500/20 border border-red-500/50 p-4 rounded-xl animate-pulse">
                                        <div className="flex items-center gap-2 text-red-400 font-bold mb-1">
                                            <AlertCircle size={20} />
                                            REROUTING ALERT
                                        </div>
                                        <p className="text-white text-sm">{safetyAlert.message}</p>
                                    </div>
                                ) : (
                                    <div className="grid grid-cols-2 gap-3 mb-4">
                                        <div className="bg-slate-900/50 p-3 rounded-xl border border-slate-700">
                                            <p className="text-slate-400 text-xs">Speed</p>
                                            <p className="text-white font-mono text-lg">45 km/h</p>
                                        </div>
                                        <div className="bg-slate-900/50 p-3 rounded-xl border border-slate-700">
                                            <p className="text-slate-400 text-xs">Safety</p>
                                            <p className="text-emerald-400 font-bold text-lg">Stable</p>
                                        </div>
                                    </div>
                                )}

                                <div className="space-y-2">
                                    <div className="flex items-center gap-3">
                                        <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-white font-bold">
                                            <ChevronDown size={18} />
                                        </div>
                                        <div>
                                            <p className="text-slate-300 text-sm">Follow current route</p>
                                            <p className="text-slate-500 text-xs">{(selectedRoute.distance_km - (navigationStep * 0.1)).toFixed(1)} km remaining</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ) : (
                            selectedRoute && (
                                <div className={`border rounded-2xl p-5 ${selectedRoute.safety_score >= 60
                                    ? 'bg-emerald-500/10 border-emerald-500/30'
                                    : 'bg-orange-500/10 border-orange-500/30'
                                    }`}>
                                    <div className="flex justify-between items-start mb-3">
                                        <h4 className={`font-semibold ${selectedRoute.safety_score >= 60 ? 'text-emerald-400' : 'text-orange-400'
                                            }`}>
                                            📍 Selected Route Details
                                        </h4>
                                        <button
                                            onClick={startNavigation}
                                            className="px-4 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-sm font-medium transition-colors shadow-lg shadow-cyan-900/20 flex items-center gap-2"
                                        >
                                            <Navigation size={14} /> Start
                                        </button>
                                    </div>

                                    <div className="space-y-2 text-sm">
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Type:</span>
                                            <span className="text-white capitalize">{selectedRoute.route_type.replace('_', ' ')}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Total Distance:</span>
                                            <span className="text-white">{selectedRoute.distance}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Est. Duration:</span>
                                            <span className="text-white">{selectedRoute.duration}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Safety Score:</span>
                                            <span className={`font-bold ${selectedRoute.safety_score >= 60 ? 'text-emerald-400' : 'text-orange-400'
                                                }`}>
                                                {selectedRoute.safety_score}%
                                            </span>
                                        </div>
                                    </div>

                                    {/* Zone Breakdown Bar */}
                                    {selectedRoute.zone_breakdown && Object.keys(selectedRoute.zone_breakdown).length > 0 && (
                                        <div className="mt-3 pt-3 border-t border-slate-700">
                                            <p className="text-xs text-slate-400 mb-2">Zone Breakdown</p>
                                            <div className="flex rounded-full overflow-hidden h-2.5">
                                                {selectedRoute.zone_breakdown.green > 0 && (
                                                    <div
                                                        className="bg-emerald-500 h-full"
                                                        style={{ width: `${selectedRoute.zone_breakdown.green}%` }}
                                                        title={`Safe: ${selectedRoute.zone_breakdown.green}%`}
                                                    />
                                                )}
                                                {selectedRoute.zone_breakdown.orange > 0 && (
                                                    <div
                                                        className="bg-amber-500 h-full"
                                                        style={{ width: `${selectedRoute.zone_breakdown.orange}%` }}
                                                        title={`Moderate: ${selectedRoute.zone_breakdown.orange}%`}
                                                    />
                                                )}
                                                {selectedRoute.zone_breakdown.red > 0 && (
                                                    <div
                                                        className="bg-rose-500 h-full"
                                                        style={{ width: `${selectedRoute.zone_breakdown.red}%` }}
                                                        title={`High Risk: ${selectedRoute.zone_breakdown.red}%`}
                                                    />
                                                )}
                                                {selectedRoute.zone_breakdown.unknown > 0 && (
                                                    <div
                                                        className="bg-slate-600 h-full"
                                                        style={{ width: `${selectedRoute.zone_breakdown.unknown}%` }}
                                                        title={`Unknown: ${selectedRoute.zone_breakdown.unknown}%`}
                                                    />
                                                )}
                                            </div>
                                            <div className="flex justify-between text-xs text-slate-500 mt-1">
                                                {selectedRoute.zone_breakdown.green > 0 && (
                                                    <span className="text-emerald-400">🟢 {selectedRoute.zone_breakdown.green}%</span>
                                                )}
                                                {selectedRoute.zone_breakdown.orange > 0 && (
                                                    <span className="text-amber-400">🟡 {selectedRoute.zone_breakdown.orange}%</span>
                                                )}
                                                {selectedRoute.zone_breakdown.red > 0 && (
                                                    <span className="text-rose-400">🔴 {selectedRoute.zone_breakdown.red}%</span>
                                                )}
                                            </div>
                                        </div>
                                    )}

                                    {/* Agent Reasoning Toggle */}
                                    {selectedRoute.agent_reasoning && (
                                        <div className="mt-3 pt-3 border-t border-slate-700">
                                            <button
                                                onClick={() => setShowAgentReasoning(!showAgentReasoning)}
                                                className="w-full flex items-center justify-between text-xs text-cyan-400 hover:text-cyan-300 transition-colors"
                                            >
                                                <span className="flex items-center gap-1">
                                                    <Bot size={14} /> Why this route?
                                                </span>
                                                <ChevronRight
                                                    size={14}
                                                    className={`transition-transform ${showAgentReasoning ? 'rotate-90' : ''}`}
                                                />
                                            </button>
                                            {showAgentReasoning && (
                                                <div className="mt-2 p-3 bg-slate-800/80 rounded-lg text-xs text-slate-300 whitespace-pre-line leading-relaxed">
                                                    {selectedRoute.agent_reasoning}
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    {/* Warnings */}
                                    {selectedRoute.warnings?.length > 0 && (
                                        <div className="mt-3 pt-3 border-t border-slate-700">
                                            <p className="text-xs text-orange-400 mb-1 flex items-center gap-1">
                                                <AlertCircle size={12} /> Cautions:
                                            </p>
                                            {selectedRoute.warnings.slice(0, 3).map((warning, i) => (
                                                <p key={i} className="text-xs text-slate-400">{warning}</p>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )
                        )}

                        {/* Agent Summary */}
                        {agentSummary && (
                            <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-2xl p-4">
                                <div className="flex items-center gap-2 mb-2">
                                    <Bot size={16} className="text-cyan-400" />
                                    <span className="text-sm font-medium text-cyan-400">Safety Agent</span>
                                </div>
                                <p className="text-xs text-slate-300 leading-relaxed">{agentSummary}</p>
                            </div>
                        )}
                    </div>

                    {/* Right Panel - Map */}
                    <div className="lg:col-span-2">
                        <SafetyMap
                            cities={[
                                ...displayedCities,
                                ...mapCities,
                                ...(selectedRoute?.waypoints?.map((wp, i) => ({
                                    id: `waypoint-${i}`,
                                    name: wp.name,
                                    state: wp.state,
                                    lat: wp.latitude,
                                    lng: wp.longitude,
                                    zone: wp.safety_zone,
                                    safety_zone: wp.safety_zone,
                                    isWaypoint: true
                                })) || [])
                            ]}
                            routes={mapRoutes}
                            center={startCity ? [startCity.latitude, startCity.longitude] : [20.5937, 78.9629]}
                            zoom={startCity && destCity ? 6 : 5}
                            height="600px"
                            showLegend={true}
                            showZoneCircles={false}
                            zoneFilter={zoneFilter}
                            className="border border-slate-800"
                            fitBounds={selectedRoute?.path?.length >= 2 ? selectedRoute.path : null}
                            onSelectStart={handleSelectStart}
                            onSelectDest={handleSelectDest}
                        />
                    </div>
                </div>
            </div>
        </main>
    );
}
