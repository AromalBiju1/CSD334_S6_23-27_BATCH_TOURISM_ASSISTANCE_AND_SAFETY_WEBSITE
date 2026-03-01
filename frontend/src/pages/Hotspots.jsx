import React, { useState, useEffect } from "react";
import { Search, MapPin, Star, ChevronDown, Loader } from "lucide-react";
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

    const handleAttractionClick = (attraction) => {
        setSelectedAttraction(attraction);
    };

    const getRatingStars = (rating) => {
        return Math.min(5, Math.max(0, Math.round(rating || 0)));
    };

    // Prepare attractions for map
    const mapAttractions = filteredAttractions
        .filter(a => a.latitude && a.longitude)
        .map(a => ({
            ...a,
            id: a.id.toString(),
            lat: a.latitude,
            lng: a.longitude,
            zone: 'green', // Default zone for attractions
        }));

    // Calculate map center
    const getMapCenter = () => {
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
                        Discover popular attractions across India with interactive map
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

                {/* Loading */}
                {loading && (
                    <div className="flex items-center justify-center py-20">
                        <Loader className="animate-spin text-emerald-400" size={40} />
                    </div>
                )}

                {/* Map View */}
                {!loading && (
                    <div className="mb-6">
                        <SafetyMap
                            cities={mapAttractions}
                            onCityClick={handleAttractionClick}
                            center={getMapCenter()}
                            zoom={getMapZoom()}
                            showZoneCircles={false}
                            showLegend={false}
                            height="600px"
                            className="border border-slate-800"
                        />
                    </div>
                )}

                {/* Attractions List Below Map */}
                {!loading && filteredAttractions.length > 0 && (
                    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                        {filteredAttractions.map((attraction) => (
                            <div
                                key={attraction.id}
                                onClick={() => handleAttractionClick(attraction)}
                                className={`bg-slate-900/50 border rounded-2xl overflow-hidden hover:border-slate-700 transition-all group cursor-pointer ${selectedAttraction?.id === attraction.id
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
                        ))}
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