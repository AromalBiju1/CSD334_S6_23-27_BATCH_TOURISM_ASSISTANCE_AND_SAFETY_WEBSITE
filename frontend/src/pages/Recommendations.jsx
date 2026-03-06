import React, { useState, useEffect, useCallback } from "react";
import {
    Heart, Settings, Star, MapPin, Search, X, Plus, Trash2,
    Loader, RefreshCw, Shield, Sparkles, ChevronDown
} from "lucide-react";
import { useTheme } from "../context/ThemeContext";
import {
    getUserPreferences, updateUserPreferences, getVisitedPlaces,
    addVisitedPlace, removeVisitedPlace, getRecommendations,
    getAvailableCategories, searchAttractionsForMarking
} from "../api/services";
import toast from "react-hot-toast";

const BUDGET_LEVELS = ["low", "medium", "high"];
const TRAVEL_STYLES = ["adventure", "relaxed", "balanced"];
const SAFETY_OPTIONS = ["all", "green", "orange"];

export default function Recommendations() {
    const { theme } = useTheme();
    const isDark = theme === "dark";

    // State
    const [preferences, setPreferences] = useState(null);
    const [visited, setVisited] = useState([]);
    const [recommendations, setRecommendations] = useState([]);
    const [categories, setCategories] = useState([]);
    const [loading, setLoading] = useState(true);
    const [recsLoading, setRecsLoading] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");
    const [searchResults, setSearchResults] = useState([]);
    const [showSearch, setShowSearch] = useState(false);
    const [selectedCats, setSelectedCats] = useState([]);
    const [budgetLevel, setBudgetLevel] = useState("medium");
    const [travelStyle, setTravelStyle] = useState("balanced");
    const [preferredSafety, setPreferredSafety] = useState("all");
    const [prefsOpen, setPrefsOpen] = useState(false);

    // Load all data on mount
    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        setLoading(true);
        try {
            const [prefsData, visitedData, catsData, recsData] = await Promise.all([
                getUserPreferences().catch(() => null),
                getVisitedPlaces().catch(() => []),
                getAvailableCategories().catch(() => []),
                getRecommendations().catch(() => []),
            ]);

            if (prefsData) {
                setPreferences(prefsData);
                setSelectedCats(prefsData.preferred_categories ? prefsData.preferred_categories.split(",").filter(Boolean) : []);
                setBudgetLevel(prefsData.budget_level || "medium");
                setTravelStyle(prefsData.travel_style || "balanced");
                setPreferredSafety(prefsData.preferred_safety || "all");
            }
            setVisited(visitedData);
            setCategories(catsData);
            setRecommendations(recsData);
        } catch (err) {
            console.error("Failed to load data:", err);
        } finally {
            setLoading(false);
        }
    };

    const handleSavePreferences = async () => {
        try {
            await updateUserPreferences({
                preferred_categories: selectedCats.join(","),
                budget_level: budgetLevel,
                travel_style: travelStyle,
                preferred_safety: preferredSafety,
            });
            toast.success("Preferences saved!");
            // Refresh recommendations
            setRecsLoading(true);
            const recs = await getRecommendations();
            setRecommendations(recs);
            setRecsLoading(false);
        } catch (err) {
            toast.error("Failed to save preferences");
        }
    };

    const handleToggleCategory = (cat) => {
        setSelectedCats((prev) =>
            prev.includes(cat) ? prev.filter((c) => c !== cat) : [...prev, cat]
        );
    };

    const handleSearchAttractions = useCallback(async (query) => {
        if (!query || query.length < 2) {
            setSearchResults([]);
            return;
        }
        try {
            const results = await searchAttractionsForMarking(query);
            setSearchResults(results);
        } catch {
            setSearchResults([]);
        }
    }, []);

    useEffect(() => {
        const timer = setTimeout(() => handleSearchAttractions(searchQuery), 300);
        return () => clearTimeout(timer);
    }, [searchQuery, handleSearchAttractions]);

    const handleAddVisited = async (attraction) => {
        try {
            await addVisitedPlace({ attraction_id: attraction.id, rating: null });
            toast.success(`Added "${attraction.name}" to visited places`);
            setSearchQuery("");
            setSearchResults([]);
            setShowSearch(false);
            // Refresh
            const [v, r] = await Promise.all([getVisitedPlaces(), getRecommendations()]);
            setVisited(v);
            setRecommendations(r);
        } catch (err) {
            toast.error(err.response?.data?.detail || "Failed to add");
        }
    };

    const handleRemoveVisited = async (id) => {
        try {
            await removeVisitedPlace(id);
            toast.success("Removed from visited places");
            const [v, r] = await Promise.all([getVisitedPlaces(), getRecommendations()]);
            setVisited(v);
            setRecommendations(r);
        } catch {
            toast.error("Failed to remove");
        }
    };

    const handleRefreshRecs = async () => {
        setRecsLoading(true);
        try {
            const recs = await getRecommendations();
            setRecommendations(recs);
        } catch {
            toast.error("Failed to refresh");
        } finally {
            setRecsLoading(false);
        }
    };

    const getSafetyBadge = (zone) => {
        const config = {
            green: { label: "Safe", cls: "bg-green-500/20 text-green-400" },
            orange: { label: "Moderate", cls: "bg-orange-500/20 text-orange-400" },
            red: { label: "High Risk", cls: "bg-red-500/20 text-red-400" },
        };
        const c = config[zone] || config.orange;
        return <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${c.cls}`}>{c.label}</span>;
    };

    if (loading) {
        return (
            <main className="pt-16 min-h-screen w-full flex items-center justify-center">
                <div className="text-center">
                    <Loader className="animate-spin text-emerald-400 mx-auto mb-3" size={40} />
                    <p className={isDark ? "text-slate-400" : "text-slate-500"}>Loading recommendations...</p>
                </div>
            </main>
        );
    }

    return (
        <main className="pt-16 min-h-screen w-full">
            <div className="w-full max-w-6xl mx-auto px-6 md:px-16 py-8">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-2xl sm:text-3xl font-bold mb-2">
                        <Sparkles className="inline-block text-amber-400 mr-2" size={28} />
                        For <span className="text-emerald-400">You</span>
                    </h1>
                    <p className={isDark ? "text-slate-400" : "text-slate-500"}>
                        Personalized attraction recommendations based on your preferences and travel history
                    </p>
                </div>

                <div className="grid lg:grid-cols-3 gap-6">
                    {/* Left Column: Preferences + Visited */}
                    <div className="lg:col-span-1 space-y-6">

                        {/* ── Preferences Section ── */}
                        <div className={`rounded-2xl border p-6 ${isDark ? "bg-slate-900/50 border-slate-800" : "bg-white border-slate-200 shadow-sm"}`}>
                            <button
                                onClick={() => setPrefsOpen(!prefsOpen)}
                                className="w-full flex items-center justify-between mb-4"
                            >
                                <h2 className={`text-lg font-semibold flex items-center gap-2 ${isDark ? "text-white" : "text-slate-900"}`}>
                                    <Settings size={18} className="text-emerald-400" />
                                    Your Preferences
                                </h2>
                                <ChevronDown size={18} className={`transition-transform ${prefsOpen ? "rotate-180" : ""} ${isDark ? "text-slate-400" : "text-slate-500"}`} />
                            </button>

                            {prefsOpen && (
                                <div className="space-y-5">
                                    {/* Categories */}
                                    <div>
                                        <label className={`block text-sm font-medium mb-2 ${isDark ? "text-slate-300" : "text-slate-600"}`}>
                                            Interests
                                        </label>
                                        <div className="flex flex-wrap gap-2">
                                            {categories.map((cat) => (
                                                <button
                                                    key={cat}
                                                    onClick={() => handleToggleCategory(cat)}
                                                    className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${selectedCats.includes(cat)
                                                        ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/50"
                                                        : isDark
                                                            ? "bg-slate-800 text-slate-400 border border-slate-700 hover:border-slate-600"
                                                            : "bg-slate-100 text-slate-500 border border-slate-300 hover:border-slate-400"
                                                        }`}
                                                >
                                                    {cat}
                                                </button>
                                            ))}
                                            {categories.length === 0 && (
                                                <p className={`text-sm ${isDark ? "text-slate-500" : "text-slate-400"}`}>
                                                    No categories available
                                                </p>
                                            )}
                                        </div>
                                    </div>

                                    {/* Budget */}
                                    <div>
                                        <label className={`block text-sm font-medium mb-2 ${isDark ? "text-slate-300" : "text-slate-600"}`}>
                                            Budget Level
                                        </label>
                                        <div className="flex gap-2">
                                            {BUDGET_LEVELS.map((b) => (
                                                <button
                                                    key={b}
                                                    onClick={() => setBudgetLevel(b)}
                                                    className={`flex-1 py-2 rounded-xl text-xs font-medium capitalize transition-all ${budgetLevel === b
                                                        ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/50"
                                                        : isDark
                                                            ? "bg-slate-800 text-slate-400 border border-slate-700"
                                                            : "bg-slate-100 text-slate-500 border border-slate-300"
                                                        }`}
                                                >
                                                    {b}
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Travel Style */}
                                    <div>
                                        <label className={`block text-sm font-medium mb-2 ${isDark ? "text-slate-300" : "text-slate-600"}`}>
                                            Travel Style
                                        </label>
                                        <div className="flex gap-2">
                                            {TRAVEL_STYLES.map((s) => (
                                                <button
                                                    key={s}
                                                    onClick={() => setTravelStyle(s)}
                                                    className={`flex-1 py-2 rounded-xl text-xs font-medium capitalize transition-all ${travelStyle === s
                                                        ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/50"
                                                        : isDark
                                                            ? "bg-slate-800 text-slate-400 border border-slate-700"
                                                            : "bg-slate-100 text-slate-500 border border-slate-300"
                                                        }`}
                                                >
                                                    {s}
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Safety Preference */}
                                    <div>
                                        <label className={`block text-sm font-medium mb-2 ${isDark ? "text-slate-300" : "text-slate-600"}`}>
                                            Safety Preference
                                        </label>
                                        <div className="flex gap-2">
                                            {SAFETY_OPTIONS.map((s) => (
                                                <button
                                                    key={s}
                                                    onClick={() => setPreferredSafety(s)}
                                                    className={`flex-1 py-2 rounded-xl text-xs font-medium capitalize transition-all ${preferredSafety === s
                                                        ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/50"
                                                        : isDark
                                                            ? "bg-slate-800 text-slate-400 border border-slate-700"
                                                            : "bg-slate-100 text-slate-500 border border-slate-300"
                                                        }`}
                                                >
                                                    {s === "all" ? "All Zones" : s === "green" ? "🟢 Safe" : "🟠 Moderate"}
                                                </button>
                                            ))}
                                        </div>
                                    </div>

                                    <button
                                        onClick={handleSavePreferences}
                                        className="w-full bg-emerald-500 hover:bg-emerald-600 text-white py-2.5 rounded-xl text-sm font-medium transition-all"
                                    >
                                        Save Preferences
                                    </button>
                                </div>
                            )}
                        </div>

                        {/* ── Visited Places Section ── */}
                        <div className={`rounded-2xl border p-6 ${isDark ? "bg-slate-900/50 border-slate-800" : "bg-white border-slate-200 shadow-sm"}`}>
                            <div className="flex items-center justify-between mb-4">
                                <h2 className={`text-lg font-semibold flex items-center gap-2 ${isDark ? "text-white" : "text-slate-900"}`}>
                                    <MapPin size={18} className="text-blue-400" />
                                    Visited Places
                                    <span className={`text-xs ml-1 px-2 py-0.5 rounded-full ${isDark ? "bg-slate-800 text-slate-400" : "bg-slate-100 text-slate-500"}`}>
                                        {visited.length}
                                    </span>
                                </h2>
                                <button
                                    onClick={() => setShowSearch(!showSearch)}
                                    className="text-emerald-400 hover:text-emerald-300 transition-colors"
                                >
                                    <Plus size={20} />
                                </button>
                            </div>

                            {/* Search to add */}
                            {showSearch && (
                                <div className="mb-4">
                                    <div className="relative">
                                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                                        <input
                                            type="text"
                                            value={searchQuery}
                                            onChange={(e) => setSearchQuery(e.target.value)}
                                            placeholder="Search attractions..."
                                            className={`w-full rounded-xl h-10 pl-10 pr-8 text-sm focus:outline-none focus:border-emerald-500/50 transition-colors ${isDark
                                                ? "bg-slate-800 border border-slate-700 text-white placeholder-slate-500"
                                                : "bg-slate-50 border border-slate-300 text-slate-900 placeholder-slate-400"
                                                }`}
                                        />
                                        <button onClick={() => { setShowSearch(false); setSearchQuery(""); setSearchResults([]); }} className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400">
                                            <X size={16} />
                                        </button>
                                    </div>
                                    {searchResults.length > 0 && (
                                        <div className={`mt-2 rounded-xl border overflow-hidden max-h-60 overflow-y-auto ${isDark ? "bg-slate-800 border-slate-700" : "bg-white border-slate-200"}`}>
                                            {searchResults.map((a) => (
                                                <button
                                                    key={a.id}
                                                    onClick={() => handleAddVisited(a)}
                                                    className={`w-full text-left p-3 transition-colors flex items-center justify-between ${isDark ? "hover:bg-slate-700" : "hover:bg-slate-50"}`}
                                                >
                                                    <div>
                                                        <p className={`text-sm font-medium ${isDark ? "text-white" : "text-slate-900"}`}>{a.name}</p>
                                                        <p className={`text-xs ${isDark ? "text-slate-400" : "text-slate-500"}`}>{a.city_name} · {a.category}</p>
                                                    </div>
                                                    <Plus size={16} className="text-emerald-400" />
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Visited list */}
                            <div className="space-y-2 max-h-80 overflow-y-auto">
                                {visited.map((v) => (
                                    <div
                                        key={v.id}
                                        className={`flex items-center justify-between p-3 rounded-xl ${isDark ? "bg-slate-800/50" : "bg-slate-50"}`}
                                    >
                                        <div>
                                            <p className={`text-sm font-medium ${isDark ? "text-white" : "text-slate-900"}`}>
                                                {v.attraction_name}
                                            </p>
                                            <p className={`text-xs ${isDark ? "text-slate-400" : "text-slate-500"}`}>
                                                {v.city_name} · {v.attraction_category}
                                            </p>
                                        </div>
                                        <button
                                            onClick={() => handleRemoveVisited(v.id)}
                                            className="text-red-400 hover:text-red-300 transition-colors p-1"
                                        >
                                            <Trash2 size={14} />
                                        </button>
                                    </div>
                                ))}
                                {visited.length === 0 && (
                                    <p className={`text-center text-sm py-4 ${isDark ? "text-slate-500" : "text-slate-400"}`}>
                                        No visited places yet. Add some to improve recommendations!
                                    </p>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Right Column: Recommendations */}
                    <div className="lg:col-span-2">
                        <div className="flex items-center justify-between mb-4">
                            <h2 className={`text-xl font-bold flex items-center gap-2 ${isDark ? "text-white" : "text-slate-900"}`}>
                                <Heart size={20} className="text-pink-400" />
                                You May Like
                            </h2>
                            <button
                                onClick={handleRefreshRecs}
                                disabled={recsLoading}
                                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${isDark
                                    ? "text-slate-400 hover:text-white hover:bg-slate-800"
                                    : "text-slate-500 hover:text-slate-900 hover:bg-slate-100"
                                    }`}
                            >
                                <RefreshCw size={14} className={recsLoading ? "animate-spin" : ""} />
                                Refresh
                            </button>
                        </div>

                        {recsLoading ? (
                            <div className="flex items-center justify-center py-20">
                                <Loader className="animate-spin text-emerald-400" size={40} />
                            </div>
                        ) : recommendations.length === 0 ? (
                            <div className={`rounded-2xl border p-12 text-center ${isDark ? "bg-slate-900/30 border-slate-800" : "bg-white border-slate-200"}`}>
                                <Sparkles className="text-amber-400 mx-auto mb-4" size={48} />
                                <h3 className={`text-lg font-semibold mb-2 ${isDark ? "text-white" : "text-slate-900"}`}>
                                    No recommendations yet
                                </h3>
                                <p className={`text-sm ${isDark ? "text-slate-400" : "text-slate-500"}`}>
                                    Set your preferences and mark some visited places to get personalized recommendations!
                                </p>
                            </div>
                        ) : (
                            <div className="grid sm:grid-cols-2 gap-4">
                                {recommendations.map((rec) => (
                                    <div
                                        key={rec.id}
                                        className={`rounded-2xl border p-5 transition-all hover:scale-[1.02] ${isDark
                                            ? "bg-slate-900/50 border-slate-800 hover:border-slate-700"
                                            : "bg-white border-slate-200 hover:border-slate-300 shadow-sm hover:shadow-md"
                                            }`}
                                    >
                                        <div className="flex items-start justify-between mb-3">
                                            <div className="flex-1">
                                                <h3 className={`font-semibold mb-1 ${isDark ? "text-white" : "text-slate-900"}`}>
                                                    {rec.name}
                                                </h3>
                                                <p className={`text-sm flex items-center gap-1 ${isDark ? "text-slate-400" : "text-slate-500"}`}>
                                                    <MapPin size={12} />
                                                    {rec.city_name}
                                                </p>
                                            </div>
                                            <div className="text-right">
                                                <div className="text-emerald-400 font-bold text-lg">
                                                    {Math.round(rec.match_score * 100)}%
                                                </div>
                                                <p className={`text-xs ${isDark ? "text-slate-500" : "text-slate-400"}`}>match</p>
                                            </div>
                                        </div>

                                        <div className="flex items-center gap-2 flex-wrap">
                                            {rec.category && (
                                                <span className={`px-2 py-0.5 rounded-full text-xs ${isDark ? "bg-slate-800 text-slate-300" : "bg-slate-100 text-slate-600"}`}>
                                                    {rec.category}
                                                </span>
                                            )}
                                            {rec.rating && (
                                                <span className="flex items-center gap-1 text-xs text-amber-400">
                                                    <Star size={12} fill="currentColor" />
                                                    {rec.rating.toFixed(1)}
                                                </span>
                                            )}
                                            {getSafetyBadge(rec.safety_zone)}
                                        </div>

                                        {rec.description && (
                                            <p className={`text-xs mt-3 line-clamp-2 ${isDark ? "text-slate-400" : "text-slate-500"}`}>
                                                {rec.description}
                                            </p>
                                        )}

                                        <button
                                            onClick={() => handleAddVisited({ id: rec.id, name: rec.name })}
                                            className={`mt-3 w-full text-center py-2 rounded-xl text-xs font-medium transition-all ${isDark
                                                ? "bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
                                                : "bg-slate-50 hover:bg-slate-100 text-slate-600 border border-slate-200"
                                                }`}
                                        >
                                            Mark as Visited
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </main>
    );
}
