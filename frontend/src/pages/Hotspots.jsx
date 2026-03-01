import React, { useState, useEffect, useRef } from "react";
import { Search, MapPin, Star, ChevronDown, Loader, Navigation, X, Route } from "lucide-react";
import SafetyMap from "../components/SafetyMap";
import { getAllAttractions, getCities } from "../api/services";
import toast from "react-hot-toast";

const CATEGORIES = [
    { value: "all", label: "All Categories" },
    { value: "Monument", label: "Monuments" },
    { value: "Temple", label: "Temples" },
    { value: "Palace", label: "Palaces" },
    { value: "Fort", label: "Forts" },
    { value: "Beach", label: "Beaches" },
    { value: "Museum", label: "Museums" },
    { value: "Park", label: "Parks" },
];

const TOUR_COLORS = [
    "#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6",
    "#ec4899", "#06b6d4", "#f97316", "#14b8a6", "#a855f7"
];

export default function Hotspots() {
    // State
    const [attractions, setAttractions] = useState([]);
    const [filteredAttractions, setFilteredAttractions] = useState([]);
    const [cities, setCities] = useState([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [cityQuery, setCityQuery] = useState("");
    const [selectedCity, setSelectedCity] = useState(null);
    const [selectedCategory, setSelectedCategory] = useState("all");
    const [loading, setLoading] = useState(true);
    const [citiesLoading, setCitiesLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedAttraction, setSelectedAttraction] = useState(null);

    // Tour mode state — accumulates stops sequentially
    const [tourStops, setTourStops] = useState([]);
    // tourRouteSegments[i] = real OSRM path between tourStops[i] and tourStops[i+1]
    const [tourRouteSegments, setTourRouteSegments] = useState([]);
    const [fetchingRoute, setFetchingRoute] = useState(false);
    const isFetchingRef = useRef(false);

    // Fetch cities for autocomplete
    useEffect(() => {
        const fetchCities = async () => {
            setCitiesLoading(true);
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

    // Fetch attractions
    useEffect(() => {
        const fetchAttractions = async () => {
            setLoading(true);
            setError(null);
            try {
                const params = {};
                if (selectedCity) {
                    params.city = selectedCity.name;
                }
                if (selectedCategory !== "all") {
                    params.category = selectedCategory;
                }
                const data = await getAllAttractions(params);
                setAttractions(data);
                setFilteredAttractions(data);
            } catch (err) {
                setError("Failed to load attractions. Make sure the backend is running.");
                setAttractions([]);
                toast.error("Failed to load attractions");
            } finally {
                setLoading(false);
            }
        };
        fetchAttractions();
    }, [selectedCity, selectedCategory]);

    // City suggestions
    const citySuggestions = cityQuery.length > 1 && !selectedCity
        ? cities.filter(c => c.name?.toLowerCase().includes(cityQuery.toLowerCase())).slice(0, 5)
        : [];

    // Filter attractions by search
    useEffect(() => {
        let filtered = attractions;

        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter(
                (a) =>
                    a.name?.toLowerCase().includes(query) ||
                    a.city_name?.toLowerCase().includes(query) ||
                    a.category?.toLowerCase().includes(query)
            );
        }

        setFilteredAttractions(filtered);
    }, [searchQuery, attractions]);

    const handleClearCity = () => {
        setSelectedCity(null);
        setCityQuery("");
    };

    // Fetch real road route via OSRM public API for a segment between two attractions
    const fetchOSRMSegment = async (from, to) => {
        // OSRM uses lng,lat order
        const url = `https://router.project-osrm.org/route/v1/driving/${from.longitude},${from.latitude};${to.longitude},${to.latitude}?overview=full&geometries=geojson&alternatives=false`;
        const resp = await fetch(url, { signal: AbortSignal.timeout(10000) });
        if (!resp.ok) throw new Error(`OSRM ${resp.status}`);
        const data = await resp.json();
        if (data.code !== "Ok" || !data.routes?.length) throw new Error("OSRM no route");
        // GeoJSON geometry: coordinates are [lng, lat] — flip to [lat, lng] for Leaflet
        return data.routes[0].geometry.coordinates.map(([lng, lat]) => [lat, lng]);
    };

    // Fetch OSRM road route for latest segment when tourStops change
    useEffect(() => {
        if (tourStops.length < 2) return;
        const from = tourStops[tourStops.length - 2];
        const to = tourStops[tourStops.length - 1];
        const segIdx = tourStops.length - 2;

        // Don't re-fetch if segment already exists
        if (tourRouteSegments[segIdx] !== undefined) return;

        const fetchSegment = async () => {
            setFetchingRoute(true);
            isFetchingRef.current = true;
            try {
                const path = await fetchOSRMSegment(from, to);
                setTourRouteSegments(prev => {
                    const updated = [...prev];
                    updated[segIdx] = path;
                    return updated;
                });
            } catch (e) {
                console.warn("OSRM route fetch failed, using straight line:", e.message);
                setTourRouteSegments(prev => {
                    const updated = [...prev];
                    updated[segIdx] = [[from.latitude, from.longitude], [to.latitude, to.longitude]];
                    return updated;
                });
            } finally {
                setFetchingRoute(false);
                isFetchingRef.current = false;
            }
        };
        fetchSegment();
    }, [tourStops]);


    // Add attraction to the tour (sequential routing)
    const handleAttractionClick = (attraction) => {
        setSelectedAttraction(attraction);

        if (!attraction.latitude || !attraction.longitude) {
            toast.error("This attraction has no location data for routing.");
            return;
        }

        setTourStops(prev => {
            // Prevent duplicate consecutive stops
            if (prev.length > 0 && prev[prev.length - 1].id === attraction.id) return prev;
            const updated = [...prev, attraction];
            if (updated.length >= 2) {
                const last = updated[updated.length - 2];
                toast.success(
                    `Route: ${last.name} → ${attraction.name}`,
                    { icon: "🗺️", duration: 2500 }
                );
            } else {
                toast.success(`Tour started at ${attraction.name}`, { icon: "📍", duration: 2000 });
            }
            return updated;
        });
    };

    const handleRemoveTourStop = (index) => {
        setTourStops(prev => prev.filter((_, i) => i !== index));
    };

    const handleClearTour = () => {
        setTourStops([]);
        setTourRouteSegments([]);
        toast("Tour cleared", { icon: "🗑️" });
    };

    const handleRemoveTourStopWithRoutes = (index) => {
        setTourStops(prev => {
            const updated = prev.filter((_, i) => i !== index);
            // Rebuild segment list - segments that touch removed stop are invalidated
            setTourRouteSegments([]);  // Reset all; will re-fetch on next click
            return updated;
        });
    };

    const getRatingStars = (rating) => {
        return Math.min(5, Math.max(0, Math.round(rating || 0)));
    };

    // Prepare attractions for map (with tour stop highlighting)
    const tourStopIds = new Set(tourStops.map(s => s.id));
    const mapAttractions = filteredAttractions
        .filter(a => a.latitude && a.longitude)
        .map(a => {
            const stopIdx = tourStops.findIndex(s => s.id === a.id);
            const isTourStop = stopIdx >= 0;
            return {
                ...a,
                id: a.id.toString(),
                lat: a.latitude,
                lng: a.longitude,
                zone: isTourStop ? 'orange' : 'green',
                ...(isTourStop ? {
                    stopNumber: stopIdx + 1,
                    stopColor: TOUR_COLORS[stopIdx % TOUR_COLORS.length],
                } : {}),
            };
        });

    // Build sequential tour route segments for SafetyMap using real OSRM paths
    const tourRoutes = tourStops.length >= 2
        ? tourStops.slice(0, -1).map((stop, i) => ({
            // Use real road path if available, otherwise fall back to straight line
            path: tourRouteSegments[i] || [
                [stop.latitude, stop.longitude],
                [tourStops[i + 1].latitude, tourStops[i + 1].longitude],
            ],
            color: TOUR_COLORS[i % TOUR_COLORS.length],
            weight: 5,
            opacity: tourRouteSegments[i] ? 0.9 : 0.5,
            selected: true,
            animate: true,
            info: {
                type: `Stop ${i + 1} → Stop ${i + 2}`,
                distance: `${stop.name} → ${tourStops[i + 1].name}`,
                duration: ""
            }
        }))
        : [];

    // Fit map to tour stops if present
    const tourBounds = tourStops.length >= 2
        ? tourStops.map(s => [s.latitude, s.longitude])
        : null;

    // Calculate map center
    const getMapCenter = () => {
        if (tourStops.length > 0) {
            const last = tourStops[tourStops.length - 1];
            return [last.latitude, last.longitude];
        }
        if (selectedAttraction && selectedAttraction.latitude && selectedAttraction.longitude) {
            return [selectedAttraction.latitude, selectedAttraction.longitude];
        }
        if (selectedCity) {
            return [selectedCity.latitude, selectedCity.longitude];
        }
        if (mapAttractions.length > 0) {
            const avgLat = mapAttractions.reduce((sum, a) => sum + a.latitude, 0) / mapAttractions.length;
            const avgLng = mapAttractions.reduce((sum, a) => sum + a.longitude, 0) / mapAttractions.length;
            return [avgLat, avgLng];
        }
        return [20.5937, 78.9629]; // India center
    };

    const getMapZoom = () => {
        if (tourStops.length >= 2) return null; // fitBounds will handle it
        if (selectedAttraction) return 12;
        if (selectedCity) return 10;
        if (mapAttractions.length <= 5) return 6;
        return 5;
    };

    return (
        <main className="pt-16 min-h-screen w-full">
            <div className="w-full max-w-7xl mx-auto px-6 md:px-16 py-8">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-2xl sm:text-3xl font-bold mb-2">
                        Tourist <span className="text-emerald-400">Hotspots</span>
                    </h1>
                    <p className="text-slate-400">
                        Discover attractions across India — click any two to build your tour route
                    </p>
                </div>

                {/* Controls */}
                <div className="flex flex-col gap-4 mb-6">
                    {/* Search and Filters Row */}
                    <div className="flex flex-col sm:flex-row gap-4">
                        {/* Search */}
                        <div className="flex-1">
                            <div className="relative">
                                <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                <input
                                    type="text"
                                    placeholder="Search attractions..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="w-full bg-slate-900/50 border border-slate-800 rounded-xl h-12 pl-12 pr-4 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500/50 transition-colors"
                                />
                            </div>
                        </div>
                    </div>

                    {/* City and Category Filters */}
                    <div className="flex flex-col sm:flex-row gap-4">
                        {/* City Search with Autocomplete */}
                        <div className="flex-1 relative">
                            <div className="relative">
                                <MapPin className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                                <input
                                    type="text"
                                    placeholder={citiesLoading ? "Loading cities..." : "Search by city..."}
                                    value={selectedCity ? selectedCity.name : cityQuery}
                                    onChange={(e) => {
                                        setCityQuery(e.target.value);
                                        setSelectedCity(null);
                                    }}
                                    disabled={citiesLoading}
                                    className="w-full bg-slate-900/50 border border-slate-800 rounded-xl h-12 pl-12 pr-20 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500/50 transition-colors disabled:opacity-50"
                                />
                                {selectedCity && (
                                    <button
                                        onClick={handleClearCity}
                                        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white text-sm"
                                    >
                                        Clear
                                    </button>
                                )}
                            </div>

                            {/* City Suggestions Dropdown */}
                            {citySuggestions.length > 0 && (
                                <div className="absolute z-10 w-full mt-1 bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow-lg">
                                    {citySuggestions.map((city) => (
                                        <button
                                            key={city.id}
                                            className="w-full px-4 py-3 text-left text-sm text-white hover:bg-slate-700 transition-colors"
                                            onClick={() => {
                                                setSelectedCity(city);
                                                setCityQuery(city.name);
                                            }}
                                        >
                                            <span className="font-medium">{city.name}</span>
                                            <span className="text-slate-400">, {city.state}</span>
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Category Filter */}
                        <div className="relative w-full sm:w-auto">
                            <select
                                value={selectedCategory}
                                onChange={(e) => setSelectedCategory(e.target.value)}
                                className="appearance-none w-full bg-slate-900/50 border border-slate-800 rounded-xl h-12 pl-4 pr-10 text-white focus:outline-none focus:border-emerald-500/50 cursor-pointer transition-colors"
                            >
                                {CATEGORIES.map((cat) => (
                                    <option key={cat.value} value={cat.value}>
                                        {cat.label}
                                    </option>
                                ))}
                            </select>
                            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" size={18} />
                        </div>
                    </div>
                </div>

                {/* Active Filters Display */}
                {(selectedCity || selectedCategory !== "all") && (
                    <div className="flex items-center gap-2 mb-6 flex-wrap">
                        <span className="text-slate-400 text-sm">Active filters:</span>
                        {selectedCity && (
                            <span className="px-3 py-1 bg-emerald-500/20 text-emerald-400 rounded-full text-sm flex items-center gap-2">
                                {selectedCity.name}
                                <button onClick={handleClearCity} className="hover:text-emerald-300">×</button>
                            </span>
                        )}
                        {selectedCategory !== "all" && (
                            <span className="px-3 py-1 bg-emerald-500/20 text-emerald-400 rounded-full text-sm flex items-center gap-2">
                                {CATEGORIES.find(c => c.value === selectedCategory)?.label}
                                <button onClick={() => setSelectedCategory("all")} className="hover:text-emerald-300">×</button>
                            </span>
                        )}
                    </div>
                )}

                {/* Error Message */}
                {error && (
                    <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-6 text-red-400">
                        {error}
                    </div>
                )}

                {/* Results Count */}
                {!loading && !error && (
                    <p className="text-slate-400 text-sm mb-4">
                        Showing {filteredAttractions.length} attraction{filteredAttractions.length !== 1 ? 's' : ''}
                        {mapAttractions.length < filteredAttractions.length &&
                            ` (${mapAttractions.length} with location data)`
                        }
                    </p>
                )}

                {/* Loading */}
                {loading && (
                    <div className="flex items-center justify-center py-20">
                        <Loader className="animate-spin text-emerald-400" size={40} />
                    </div>
                )}

                {/* Map + Tour Panel Layout */}
                {!loading && (
                    <div className="flex gap-4 mb-6">
                        {/* Map */}
                        <div className="flex-1 min-w-0">
                            <SafetyMap
                                cities={mapAttractions}
                                routes={tourRoutes}
                                onCityClick={handleAttractionClick}
                                center={getMapCenter()}
                                zoom={getMapZoom() || 5}
                                fitBounds={tourBounds}
                                showZoneCircles={false}
                                showLegend={false}
                                height="600px"
                                className="border border-slate-800"
                            />
                        </div>

                        {/* Tour Sidebar */}
                        {tourStops.length > 0 && (
                            <div className="w-72 flex-shrink-0">
                                <div className="bg-slate-900/70 border border-slate-700 rounded-2xl overflow-hidden sticky top-20">
                                    {/* Tour Header */}
                                    <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700 bg-slate-800/60">
                                        <div className="flex items-center gap-2">
                                            <Route size={16} className="text-emerald-400" />
                                            <span className="text-white font-semibold text-sm">Your Tour</span>
                                            <span className="bg-emerald-500/20 text-emerald-400 text-xs px-2 py-0.5 rounded-full">
                                                {tourStops.length} stops
                                            </span>
                                            {fetchingRoute && (
                                                <Loader size={12} className="animate-spin text-slate-400" />
                                            )}
                                        </div>
                                        <button
                                            onClick={handleClearTour}
                                            className="text-slate-400 hover:text-red-400 transition-colors"
                                            title="Clear tour"
                                        >
                                            <X size={16} />
                                        </button>
                                    </div>

                                    {/* Tour Stops List */}
                                    <div className="max-h-[520px] overflow-y-auto divide-y divide-slate-800">
                                        {tourStops.map((stop, index) => (
                                            <div key={`${stop.id}-${index}`} className="flex items-start gap-3 px-4 py-3 hover:bg-slate-800/40 transition-colors group">
                                                {/* Stop Number with color dot */}
                                                <div
                                                    className="mt-0.5 w-6 h-6 rounded-full flex-shrink-0 flex items-center justify-center text-white text-xs font-bold shadow"
                                                    style={{ backgroundColor: TOUR_COLORS[index % TOUR_COLORS.length] }}
                                                >
                                                    {index + 1}
                                                </div>

                                                {/* Stop Info */}
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-white text-sm font-medium truncate">{stop.name}</p>
                                                    <div className="flex items-center gap-1 text-slate-400 text-xs mt-0.5">
                                                        <MapPin size={10} />
                                                        <span className="truncate">{stop.city_name}</span>
                                                    </div>
                                                    {stop.category && (
                                                        <span className="inline-block mt-1 text-xs px-1.5 py-0.5 bg-slate-800 text-slate-400 rounded">
                                                            {stop.category}
                                                        </span>
                                                    )}
                                                </div>

                                                {/* Remove button */}
                                                <button
                                                    onClick={() => handleRemoveTourStopWithRoutes(index)}
                                                    className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-red-400 transition-all flex-shrink-0 mt-0.5"
                                                >
                                                    <X size={14} />
                                                </button>
                                            </div>
                                        ))}
                                    </div>

                                    {/* Tour Footer */}
                                    {tourStops.length >= 2 && (
                                        <div className="px-4 py-3 border-t border-slate-700 bg-slate-800/40">
                                            <p className="text-slate-400 text-xs flex items-center gap-2">
                                                <Navigation size={12} className="text-emerald-400" />
                                                {tourStops.length - 1} route segment{tourStops.length > 2 ? 's' : ''} on map
                                            </p>
                                        </div>
                                    )}

                                    {tourStops.length === 1 && (
                                        <div className="px-4 py-3 border-t border-slate-700 bg-slate-800/40">
                                            <p className="text-slate-400 text-xs flex items-center gap-2">
                                                <MapPin size={12} className="text-emerald-400" />
                                                Click another attraction to route
                                            </p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Selected Attraction Details */}
                {selectedAttraction && (
                    <div className="mb-6 p-6 bg-slate-900/50 border border-slate-800 rounded-2xl">
                        <div className="flex items-start justify-between">
                            <div className="flex-1">
                                <span className="text-xs text-emerald-400 bg-emerald-500/20 px-2 py-1 rounded-full">
                                    {selectedAttraction.category}
                                </span>
                                <h3 className="text-2xl font-bold text-white mt-3 mb-1">
                                    {selectedAttraction.name}
                                </h3>
                                <p className="text-slate-400 mb-3 flex items-center gap-2">
                                    <MapPin size={16} />
                                    {selectedAttraction.city_name}
                                </p>
                                {selectedAttraction.description && (
                                    <p className="text-slate-300 mb-4">{selectedAttraction.description}</p>
                                )}
                                <div className="flex items-center gap-1">
                                    {[...Array(5)].map((_, i) => (
                                        <Star
                                            key={i}
                                            className={
                                                i < getRatingStars(selectedAttraction.rating)
                                                    ? "text-yellow-500 fill-yellow-500"
                                                    : "text-slate-600"
                                            }
                                            size={18}
                                        />
                                    ))}
                                    <span className="text-white font-medium ml-2">
                                        {selectedAttraction.rating?.toFixed(1) || "N/A"}
                                    </span>
                                </div>
                            </div>
                            <button
                                onClick={() => setSelectedAttraction(null)}
                                className="text-slate-400 hover:text-white"
                            >
                                ×
                            </button>
                        </div>
                    </div>
                )}

                {/* Attractions List Below Map */}
                {!loading && filteredAttractions.length > 0 && (
                    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                        {filteredAttractions.map((attraction) => {
                            const stopIndex = tourStops.findIndex(s => s.id === attraction.id);
                            const isInTour = stopIndex >= 0;
                            return (
                                <div
                                    key={attraction.id}
                                    onClick={() => handleAttractionClick(attraction)}
                                    className={`bg-slate-900/50 border rounded-2xl overflow-hidden hover:border-slate-700 transition-all group cursor-pointer ${isInTour
                                        ? 'border-emerald-500 ring-1 ring-emerald-500/40'
                                        : selectedAttraction?.id === attraction.id
                                            ? 'border-emerald-500'
                                            : 'border-slate-800'
                                        }`}
                                >
                                    {/* Image or Placeholder */}
                                    <div className="h-40 bg-slate-800/50 relative overflow-hidden flex items-center justify-center border-b border-slate-800">
                                        <MapPin className="text-slate-600 absolute" size={40} />
                                        {attraction.image_url && (
                                            <img
                                                src={attraction.image_url}
                                                alt={attraction.name}
                                                className="absolute inset-0 w-full h-full object-cover group-hover:scale-105 transition-transform duration-300 z-10"
                                                onError={(e) => {
                                                    e.currentTarget.style.display = 'none';
                                                }}
                                            />
                                        )}
                                        {/* Tour stop badge */}
                                        {isInTour && (
                                            <div
                                                className="absolute top-2 right-2 w-7 h-7 z-20 rounded-full flex items-center justify-center text-white text-xs font-bold shadow-lg"
                                                style={{ backgroundColor: TOUR_COLORS[stopIndex % TOUR_COLORS.length] }}
                                            >
                                                {stopIndex + 1}
                                            </div>
                                        )}
                                    </div>

                                    {/* Content */}
                                    <div className="p-5">
                                        {/* Category */}
                                        <span className="text-xs text-slate-400 bg-slate-800 px-2 py-1 rounded-full">
                                            {attraction.category || "Attraction"}
                                        </span>

                                        {/* Name */}
                                        <h3 className="text-lg font-semibold text-white mt-3 mb-1 group-hover:text-emerald-400 transition-colors">
                                            {attraction.name}
                                        </h3>

                                        {/* Location */}
                                        <p className="text-slate-400 text-sm mb-3">
                                            {attraction.city_name}
                                        </p>

                                        {/* Rating */}
                                        <div className="flex items-center gap-1">
                                            {[...Array(5)].map((_, i) => (
                                                <Star
                                                    key={i}
                                                    className={
                                                        i < getRatingStars(attraction.rating)
                                                            ? "text-yellow-500 fill-yellow-500"
                                                            : "text-slate-600"
                                                    }
                                                    size={14}
                                                />
                                            ))}
                                            <span className="text-slate-400 text-sm ml-1">
                                                {attraction.rating?.toFixed(1) || "N/A"}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                )}

                {/* Empty State */}
                {!loading && filteredAttractions.length === 0 && !error && (
                    <div className="text-center py-16">
                        <MapPin className="text-slate-600 mx-auto mb-4" size={48} />
                        <h3 className="text-xl font-semibold text-white mb-2">No attractions found</h3>
                        <p className="text-slate-400">Try adjusting your search or filters</p>
                    </div>
                )}
            </div>
        </main>
    );
}