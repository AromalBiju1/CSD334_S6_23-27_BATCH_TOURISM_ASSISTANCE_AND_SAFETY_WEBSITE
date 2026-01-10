import React, { useState, useEffect } from "react";
import { Search, Navigation, ChevronDown, MapPin, Loader, AlertCircle } from "lucide-react";
import SafetyMap from "../components/SafetyMap";
import { getCities, getSafeRoute } from "../api/services";
import toast from "react-hot-toast";

export default function SafeRoute() {
    const [cities, setCities] = useState([]);
    const [startPoint, setStartPoint] = useState("");
    const [destination, setDestination] = useState("");
    const [startCity, setStartCity] = useState(null);
    const [destCity, setDestCity] = useState(null);
    const [route, setRoute] = useState(null);
    const [loading, setLoading] = useState(false);
    const [citiesLoading, setCitiesLoading] = useState(true);

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

    // Filter cities for suggestions
    const startSuggestions = startPoint.length > 1 && !startCity
        ? cities.filter(c => c.name?.toLowerCase().includes(startPoint.toLowerCase())).slice(0, 5)
        : [];
    const destSuggestions = destination.length > 1 && !destCity
        ? cities.filter(c => c.name?.toLowerCase().includes(destination.toLowerCase())).slice(0, 5)
        : [];

    const handleFindRoute = async () => {
        if (!startCity || !destCity) {
            toast.error("Please select both starting point and destination");
            return;
        }

        setLoading(true);
        try {
            const routeData = await getSafeRoute(startCity.name, destCity.name);
            setRoute(routeData);
            toast.success("Route found!");
        } catch (error) {
            toast.error("Failed to find route. Try again.");
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const mapCities = [
        startCity && { ...startCity, lat: startCity.latitude, lng: startCity.longitude, zone: "green" },
        destCity && { ...destCity, lat: destCity.latitude, lng: destCity.longitude, zone: "red" },
    ].filter(Boolean);

    const getZoneColor = (zone) => {
        switch (zone) {
            case "green": return "text-green-400";
            case "orange": return "text-orange-400";
            case "red": return "text-red-400";
            default: return "text-slate-400";
        }
    };

    return (
        <main className="pt-16 min-h-screen w-full">
            <div className="w-full max-w-6xl mx-auto px-6 md:px-16 py-8">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-2xl sm:text-3xl font-bold mb-2">
                        Plan Your <span className="text-emerald-400">Safe Route</span>
                    </h1>
                    <p className="text-slate-400">
                        Get routes optimized for safety, with detailed zone breakdowns
                    </p>
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
                            <span>{loading ? "Finding Route..." : "Find Safe Route"}</span>
                        </button>

                        {/* Route Info */}
                        {route && (
                            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-2xl p-5">
                                <h4 className="font-semibold text-emerald-400 mb-3">Route Found!</h4>
                                <div className="space-y-2 text-sm">
                                    {route.distance && (
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Distance:</span>
                                            <span className="text-white">{route.distance}</span>
                                        </div>
                                    )}
                                    {route.duration && (
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Duration:</span>
                                            <span className="text-white">{route.duration}</span>
                                        </div>
                                    )}
                                    {route.safety_score !== undefined && (
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Safety Score:</span>
                                            <span className="text-emerald-400 font-bold">{route.safety_score}%</span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Right Panel - Map */}
                    <div className="lg:col-span-2">
                        {startCity || destCity ? (
                            <SafetyMap
                                cities={mapCities}
                                center={startCity ? [startCity.latitude, startCity.longitude] : [20.5937, 78.9629]}
                                zoom={startCity && destCity ? 6 : 8}
                                height="500px"
                                showLegend={false}
                                className="border border-slate-800"
                            />
                        ) : (
                            <div className="bg-slate-900/30 border border-slate-800 rounded-2xl min-h-[500px] flex flex-col items-center justify-center p-6">
                                <MapPin className="text-slate-600 mb-4" size={48} strokeWidth={1.5} />
                                <h3 className="text-xl font-semibold text-white mb-2 text-center">Plan Your Journey</h3>
                                <p className="text-slate-400 text-center text-sm max-w-sm leading-relaxed">
                                    Select a <span className="text-emerald-400">starting point</span> and{" "}
                                    <span className="text-emerald-400">destination</span> to see safe route options
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </main>
    );
}
