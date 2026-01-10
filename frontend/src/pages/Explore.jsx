import React, { useState, useEffect } from "react";
import { Search, Map, List, ChevronDown, Loader, AlertCircle } from "lucide-react";
import SafetyMap from "../components/SafetyMap";
import { getCities, getCitiesByZone, searchCities } from "../api/services";

export default function Explore() {
    const [viewMode, setViewMode] = useState("map");
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedZone, setSelectedZone] = useState("all");
    const [cities, setCities] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [selectedCity, setSelectedCity] = useState(null);

    // Fetch cities from backend
    useEffect(() => {
        const fetchCities = async () => {
            setLoading(true);
            setError(null);
            try {
                const params = {};
                if (selectedZone !== "all") {
                    params.zone = selectedZone;
                }
                const data = await getCities(params);
                setCities(data);
            } catch (err) {
                setError("Failed to load cities. Make sure the backend is running.");
                setCities([]);
            } finally {
                setLoading(false);
            }
        };
        fetchCities();
    }, [selectedZone]);

    // Filter cities based on search
    const filteredCities = cities.filter((city) => {
        if (!searchQuery) return true;
        const query = searchQuery.toLowerCase();
        return (
            city.name?.toLowerCase().includes(query) ||
            city.state?.toLowerCase().includes(query)
        );
    });

    const handleCityClick = (city) => {
        setSelectedCity(city);
    };

    const getZoneColor = (zone) => {
        switch (zone) {
            case "green": return "bg-green-500/20 text-green-400";
            case "orange": return "bg-orange-500/20 text-orange-400";
            case "red": return "bg-red-500/20 text-red-400";
            default: return "bg-slate-500/20 text-slate-400";
        }
    };

    return (
        <main className="pt-16 min-h-screen w-full">
            <div className="w-full max-w-6xl mx-auto px-6 md:px-16 py-8">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-2xl sm:text-3xl font-bold mb-2">
                        Explore <span className="text-emerald-400">Safety Zones</span>
                    </h1>
                    <p className="text-slate-400">
                        View cities across India classified by safety levels based on crime data
                    </p>
                </div>

                {/* Controls Row */}
                <div className="flex flex-col sm:flex-row gap-4 mb-6">
                    {/* Search Input */}
                    <div className="flex-1">
                        <div className="relative">
                            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
                            <input
                                type="text"
                                placeholder="Search cities or states..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                                className="w-full bg-slate-900/50 border border-slate-800 rounded-xl h-12 pl-12 pr-4 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500/50 transition-colors"
                            />
                        </div>
                    </div>

                    {/* Right Controls */}
                    <div className="flex gap-3">
                        {/* Zone Filter */}
                        <div className="relative">
                            <select
                                value={selectedZone}
                                onChange={(e) => setSelectedZone(e.target.value)}
                                className="appearance-none bg-slate-900/50 border border-slate-800 rounded-xl h-12 pl-4 pr-10 text-white focus:outline-none focus:border-emerald-500/50 cursor-pointer transition-colors"
                            >
                                <option value="all">All Zones</option>
                                <option value="green">Green Zone</option>
                                <option value="orange">Orange Zone</option>
                                <option value="red">Red Zone</option>
                            </select>
                            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" size={18} />
                        </div>

                        {/* View Toggle */}
                        <div className="flex bg-slate-900/50 border border-slate-800 rounded-xl p-1">
                            <button
                                onClick={() => setViewMode("map")}
                                className={`flex items-center gap-2 px-4 h-10 rounded-lg text-sm font-medium transition-all ${viewMode === "map"
                                        ? "bg-emerald-500/20 text-emerald-400"
                                        : "text-slate-400 hover:text-white"
                                    }`}
                            >
                                <Map size={16} />
                                <span>Map</span>
                            </button>
                            <button
                                onClick={() => setViewMode("list")}
                                className={`flex items-center gap-2 px-4 h-10 rounded-lg text-sm font-medium transition-all ${viewMode === "list"
                                        ? "bg-emerald-500/20 text-emerald-400"
                                        : "text-slate-400 hover:text-white"
                                    }`}
                            >
                                <List size={16} />
                                <span>List</span>
                            </button>
                        </div>
                    </div>
                </div>

                {/* Map or List View */}
                {viewMode === "map" ? (
                    <SafetyMap
                        cities={filteredCities.map(c => ({
                            ...c,
                            lat: c.latitude,
                            lng: c.longitude,
                            zone: c.safety_zone
                        }))}
                        onCityClick={handleCityClick}
                        showZoneCircles={true}
                        showLegend={true}
                        height="500px"
                        loading={loading}
                        error={error}
                        className="border border-slate-800"
                    />
                ) : (
                    <div className="bg-slate-900/30 border border-slate-800 rounded-2xl overflow-hidden">
                        {loading ? (
                            <div className="flex items-center justify-center py-20">
                                <Loader className="animate-spin text-emerald-400" size={40} />
                            </div>
                        ) : error ? (
                            <div className="flex items-center justify-center py-20 text-red-400">
                                <AlertCircle className="mr-2" size={20} />
                                {error}
                            </div>
                        ) : (
                            <div className="grid gap-2 p-4">
                                {filteredCities.map((city) => (
                                    <div
                                        key={city.id}
                                        className="flex items-center justify-between p-4 bg-slate-800/30 rounded-xl hover:bg-slate-800/50 transition-colors cursor-pointer"
                                        onClick={() => handleCityClick(city)}
                                    >
                                        <div>
                                            <h3 className="font-semibold text-white">{city.name}</h3>
                                            <p className="text-sm text-slate-400">{city.state}</p>
                                        </div>
                                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${getZoneColor(city.safety_zone)}`}>
                                            {city.safety_zone?.toUpperCase()}
                                        </span>
                                    </div>
                                ))}
                                {filteredCities.length === 0 && (
                                    <p className="text-center text-slate-400 py-8">No cities found</p>
                                )}
                            </div>
                        )}
                    </div>
                )}

                {/* Selected City Info */}
                {selectedCity && (
                    <div className="mt-6 p-6 bg-slate-900/50 border border-slate-800 rounded-2xl">
                        <div className="flex items-start justify-between">
                            <div>
                                <h3 className="text-xl font-bold text-white mb-1">{selectedCity.name}</h3>
                                <p className="text-slate-400 mb-3">{selectedCity.state}</p>
                                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getZoneColor(selectedCity.safety_zone)}`}>
                                    {selectedCity.safety_zone?.toUpperCase()} ZONE
                                </span>
                            </div>
                            {selectedCity.crime_index !== undefined && (
                                <div className="text-right">
                                    <p className="text-slate-400 text-sm">Crime Index</p>
                                    <p className="text-2xl font-bold text-white">{selectedCity.crime_index?.toFixed(1)}</p>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </main>
    );
}
