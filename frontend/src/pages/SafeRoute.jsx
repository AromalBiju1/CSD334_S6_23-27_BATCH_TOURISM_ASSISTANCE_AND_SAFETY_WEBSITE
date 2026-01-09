import React, { useState } from "react";
import { Search, Navigation, ChevronDown } from "lucide-react";

export default function SafeRoute() {
    const [startPoint, setStartPoint] = useState("");
    const [destination, setDestination] = useState("");

    return (
        <main className="pt-16 min-h-screen w-full">
            <div className="w-full max-w-6xl mx-auto px-16 py-8">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-2xl sm:text-3xl font-bold mb-2">
                        Plan Your <span className="text-emerald-400">Safe Route</span>
                    </h1>
                    <p className="text-slate-400">
                        Get routes optimized for safety, with detailed zone breakdowns and recommendations
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
                                    placeholder="Search city..."
                                    value={startPoint}
                                    onChange={(e) => setStartPoint(e.target.value)}
                                    className="w-full bg-slate-800/50 border border-slate-700 rounded-xl h-11 pl-10 pr-4 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500/50 text-sm transition-colors"
                                />
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
                                    placeholder="Search city..."
                                    value={destination}
                                    onChange={(e) => setDestination(e.target.value)}
                                    className="w-full bg-slate-800/50 border border-slate-700 rounded-xl h-11 pl-10 pr-4 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500/50 text-sm transition-colors"
                                />
                            </div>
                        </div>

                        {/* Find Route Button */}
                        <button className="w-full flex items-center justify-center gap-2 bg-emerald-500 hover:bg-emerald-600 text-white h-12 rounded-xl font-medium transition-all">
                            <Navigation size={18} />
                            <span>Find Safe Route</span>
                        </button>
                    </div>

                    {/* Right Panel - Map Placeholder */}
                    <div className="lg:col-span-2 bg-slate-900/30 border border-slate-800 rounded-2xl min-h-[400px] lg:min-h-[500px] flex flex-col items-center justify-center p-6">
                        <Navigation className="text-slate-600 mb-4" size={48} strokeWidth={1.5} />
                        <h3 className="text-xl font-semibold text-white mb-2 text-center">Plan Your Journey</h3>
                        <p className="text-slate-400 text-center text-sm max-w-sm leading-relaxed">
                            Select a <span className="text-emerald-400">starting point</span> and{" "}
                            <span className="text-emerald-400">destination</span> to see safe route options
                        </p>
                    </div>
                </div>
            </div>
        </main>
    );
}
