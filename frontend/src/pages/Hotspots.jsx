import React, { useState, useEffect } from "react";
import { Search, MapPin, Star, ChevronDown, Loader, ExternalLink } from "lucide-react";
import { getAllAttractions } from "../api/services";

// Indian cities for filtering
const CITIES = [
    { value: "all", label: "All Cities" },
    { value: "Delhi", label: "Delhi" },
    { value: "Mumbai", label: "Mumbai" },
    { value: "Jaipur", label: "Jaipur" },
    { value: "Agra", label: "Agra" },
    { value: "Goa", label: "Goa" },
    { value: "Varanasi", label: "Varanasi" },
    { value: "Chennai", label: "Chennai" },
    { value: "Kolkata", label: "Kolkata" },
];

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
    const [attractions, setAttractions] = useState([]);
    const [filteredAttractions, setFilteredAttractions] = useState([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedCity, setSelectedCity] = useState("all");
    const [selectedCategory, setSelectedCategory] = useState("all");
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);


    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            try {
                const data = await getAllAttractions();
                setAttractions(data);
                setFilteredAttractions(data);
            } catch (err) {
                setError("Failed to load attractions. Make sure the backend is running.");
                setAttractions([]);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    // Filter attractions when filters change
    useEffect(() => {
        let filtered = attractions;

        // Search filter
        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            filtered = filtered.filter(
                (a) =>
                    a.name?.toLowerCase().includes(query) ||
                    a.city_name?.toLowerCase().includes(query)
            );
        }

        // City filter
        if (selectedCity !== "all") {
            filtered = filtered.filter((a) => a.city_name === selectedCity);
        }

        // Category filter
        if (selectedCategory !== "all") {
            filtered = filtered.filter((a) => a.category === selectedCategory);
        }

        setFilteredAttractions(filtered);
    }, [searchQuery, selectedCity, selectedCategory, attractions]);

    const getRatingStars = (rating) => {
        return Math.min(5, Math.max(0, Math.round(rating || 0)));
    };

    return (
        <main className="pt-16 min-h-screen w-full">
            <div className="w-full max-w-6xl mx-auto px-6 md:px-16 py-8">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-2xl sm:text-3xl font-bold mb-2">
                        Tourist <span className="text-emerald-400">Hotspots</span>
                    </h1>
                    <p className="text-slate-400">
                        Discover popular attractions across India with safety ratings
                    </p>
                </div>

                {/* Controls */}
                <div className="flex flex-col md:flex-row gap-4 mb-8">
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

                    {/* City Filter */}
                    <div className="relative">
                        <select
                            value={selectedCity}
                            onChange={(e) => setSelectedCity(e.target.value)}
                            className="appearance-none bg-slate-900/50 border border-slate-800 rounded-xl h-12 pl-4 pr-10 text-white focus:outline-none focus:border-emerald-500/50 cursor-pointer transition-colors"
                        >
                            {CITIES.map((city) => (
                                <option key={city.value} value={city.value}>
                                    {city.label}
                                </option>
                            ))}
                        </select>
                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" size={18} />
                    </div>

                    {/* Category Filter */}
                    <div className="relative">
                        <select
                            value={selectedCategory}
                            onChange={(e) => setSelectedCategory(e.target.value)}
                            className="appearance-none bg-slate-900/50 border border-slate-800 rounded-xl h-12 pl-4 pr-10 text-white focus:outline-none focus:border-emerald-500/50 cursor-pointer transition-colors"
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

                {/* Error Message */}
                {error && (
                    <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-6 text-red-400">
                        {error}
                    </div>
                )}

                {/* Loading */}
                {loading && (
                    <div className="flex items-center justify-center py-20">
                        <Loader className="animate-spin text-emerald-400" size={40} />
                    </div>
                )}

                {/* Results Count */}
                {!loading && !error && (
                    <p className="text-slate-400 text-sm mb-4">
                        Showing {filteredAttractions.length} attractions
                    </p>
                )}

                {/* Attractions Grid */}
                {!loading && (
                    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                        {filteredAttractions.map((attraction) => (
                            <div
                                key={attraction.id}
                                className="bg-slate-900/50 border border-slate-800 rounded-2xl overflow-hidden hover:border-slate-700 transition-all group"
                            >
                                {/* Image Placeholder */}
                                <div className="h-40 bg-slate-800/50 flex items-center justify-center">
                                    <MapPin className="text-slate-600" size={40} />
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
