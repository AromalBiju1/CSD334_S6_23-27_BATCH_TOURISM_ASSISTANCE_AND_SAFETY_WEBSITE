import React, { useState } from "react";
import { Search, Map, List, ChevronDown, Loader } from "lucide-react";

export default function Explore() {
    const [viewMode, setViewMode] = useState("map");
    const [searchQuery, setSearchQuery] = useState("");
    const [selectedZone, setSelectedZone] = useState("all");

    return (
        <main className="pt-16 min-h-screen w-full">
            <div className="w-full max-w-6xl mx-auto px-16 py-8">
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

                {/* Map Container */}
                <div className="bg-slate-900/30 border border-slate-800 rounded-2xl min-h-[500px] flex flex-col items-center justify-center">
                    <Loader className="text-emerald-400 animate-spin mb-4" size={40} />
                    <p className="text-slate-400">Loading map...</p>
                </div>
            </div>
        </main>
    );
}
