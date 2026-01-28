import React, { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { Search, Navigation, ChevronDown, MapPin, Loader, AlertCircle, Shield, Zap, Scale } from "lucide-react";
import SafetyMap from "../components/SafetyMap";
import { getCities, getRouteAlternatives } from "../api/services";
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

    // Debounce search inputs (300ms delay)
    const debouncedStart = useDebounce(startPoint, 300);
    const debouncedDest = useDebounce(destination, 300);

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

        try {
            const data = await getRouteAlternatives(startCity.name, destCity.name);

            if (data.routes && data.routes.length > 0) {
                setRoutes(data.routes);
                setRecommendedIndex(data.recommended_index || 0);
                setSelectedRouteIndex(data.recommended_index || 0);

                // Trigger animation for safest route
                setAnimating(true);
                setTimeout(() => setAnimating(false), 3000);

                toast.success(`Found ${data.routes.length} route${data.routes.length > 1 ? 's' : ''}!`);
            } else {
                toast.error("No routes found");
            }
        } catch (error) {
            toast.error("Failed to find route. Try again.");
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    // Zone filter state - can be 'all', 'red', 'orange', 'green'
    const [zoneFilter, setZoneFilter] = useState('all');

    const selectedRoute = routes[selectedRouteIndex];

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

    // Memoize origin/destination markers
    const mapCities = useMemo(() => [
        startCity && { ...startCity, lat: startCity.latitude, lng: startCity.longitude, zone: "green", isStart: true },
        destCity && { ...destCity, lat: destCity.latitude, lng: destCity.longitude, zone: "red", isDest: true },
    ].filter(Boolean), [startCity, destCity]);

    const getZoneColor = useCallback((zone) => {
        switch (zone) {
            case "green": return "text-green-400";
            case "orange": return "text-orange-400";
            case "red": return "text-red-400";
            default: return "text-slate-400";
        }
    }, []);

    // Memoize map routes array with colors and animation
    const mapRoutes = useMemo(() => routes.map((route, idx) => ({
        path: route.path,
        safe: route.safety_score >= 60,
        color: ROUTE_COLORS[route.route_type] || "#22c55e",
        selected: idx === selectedRouteIndex,
        animate: idx === recommendedIndex && animating,
        opacity: idx === selectedRouteIndex ? 0.9 : 0.4,
        weight: idx === selectedRouteIndex ? 6 : 3,
        info: {
            distance: route.distance,
            duration: route.duration,
            type: route.route_type
        }
    })), [routes, selectedRouteIndex, recommendedIndex, animating]);

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
                                    placeholder={citiesLoading ? "Loading cities..." : "Search city..."}
                                    value={startPoint}
                                    onChange={(e) => {
                                        setStartPoint(e.target.value);
                                        setStartCity(null);
                                    }}
                                    disabled={citiesLoading}
                                    className="w-full bg-slate-800/50 border border-slate-700 rounded-xl h-11 pl-10 pr-4 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500/50 text-sm transition-colors disabled:opacity-50"
                                />
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

                        {/* Selected Route Details */}
                        {selectedRoute && (
                            <div className={`border rounded-2xl p-5 ${selectedRoute.safety_score >= 60
                                ? 'bg-emerald-500/10 border-emerald-500/30'
                                : 'bg-orange-500/10 border-orange-500/30'
                                }`}>
                                <h4 className={`font-semibold mb-3 ${selectedRoute.safety_score >= 60 ? 'text-emerald-400' : 'text-orange-400'
                                    }`}>
                                    📍 Selected Route Details
                                </h4>
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
                                    {selectedRoute.waypoints?.length > 0 && (
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Waypoints:</span>
                                            <span className="text-white">{selectedRoute.waypoints.length}</span>
                                        </div>
                                    )}
                                </div>

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
                            showZoneCircles={zoneFilter === 'all'}
                            zoneFilter={zoneFilter}
                            className="border border-slate-800"
                            fitBounds={selectedRoute?.path?.length >= 2 ? selectedRoute.path : null}
                        />
                    </div>
                </div>
            </div>
        </main>
    );
}
