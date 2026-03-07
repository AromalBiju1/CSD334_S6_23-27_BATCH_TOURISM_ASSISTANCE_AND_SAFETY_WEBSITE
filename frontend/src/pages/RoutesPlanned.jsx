import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { Navigation, ArrowLeft, Loader, Shield, MapPin, Calendar } from 'lucide-react';
import toast from 'react-hot-toast';
import { getSavedRoutes } from '../api/services';

export default function RoutesPlanned() {
    const { user } = useAuth();
    const { theme } = useTheme();
    const isDark = theme === 'dark';
    const navigate = useNavigate();

    const [routes, setRoutes] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchRoutes = async () => {
            try {
                const data = await getSavedRoutes();
                setRoutes(data);
            } catch (error) {
                console.error("Failed to load saved routes", error);
                toast.error("Could not load your planned routes.");
            } finally {
                setLoading(false);
            }
        };
        if (user) fetchRoutes();
    }, [user]);

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <Loader className="animate-spin text-emerald-400" size={40} />
            </div>
        );
    }

    return (
        <main className="pt-24 pb-12 min-h-screen w-full">
            <div className="w-full max-w-3xl mx-auto px-6">

                {/* Header */}
                <div className="flex items-center gap-4 mb-8">
                    <button
                        onClick={() => navigate('/profile')}
                        className={`p-2 rounded-xl transition-colors ${isDark ? 'hover:bg-slate-800 text-slate-400 hover:text-white' : 'hover:bg-slate-100 text-slate-500 hover:text-slate-900'}`}
                    >
                        <ArrowLeft size={24} />
                    </button>
                    <div>
                        <h1 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-slate-900'}`}>Routes Planned</h1>
                        <p className={`text-sm ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>Review your past navigation and saved routes</p>
                    </div>
                </div>

                <div className="space-y-4">
                    {routes.length === 0 ? (
                        <div className={`text-center py-20 rounded-3xl border ${isDark ? 'bg-slate-900/50 border-slate-800' : 'bg-white border-slate-200'}`}>
                            <Navigation className="mx-auto mb-4 text-slate-500" size={48} />
                            <h3 className={`text-lg font-bold mb-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>No Routes Planned</h3>
                            <p className="text-slate-500 mb-6">You haven't planned any routes yet.</p>
                            <button
                                onClick={() => navigate('/safe-route')}
                                className="px-6 py-2 bg-emerald-500 hover:bg-emerald-600 text-white rounded-xl transition-colors font-medium"
                            >
                                Plan a Route
                            </button>
                        </div>
                    ) : (
                        routes.map((route, idx) => (
                            <div key={idx} className={`p-5 rounded-2xl border transition-all ${isDark ? 'bg-slate-900/50 border-slate-800 hover:border-slate-700' : 'bg-white border-slate-200 hover:border-slate-300'}`}>
                                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                                    <div className="flex-1">
                                        <div className="flex items-center gap-3 mb-2">
                                            <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 flex-shrink-0" />
                                            <h3 className={`font-semibold ${isDark ? 'text-white' : 'text-slate-900'}`}>{route.origin}</h3>
                                        </div>
                                        <div className="pl-1 border-l-2 border-slate-700/50 ml-1 py-1">
                                            {/* connection line visual */}
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <div className="w-2.5 h-2.5 rounded-full bg-red-500 flex-shrink-0" />
                                            <h3 className={`font-semibold ${isDark ? 'text-white' : 'text-slate-900'}`}>{route.destination}</h3>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-2 md:grid-cols-3 gap-y-4 gap-x-8 text-sm bg-slate-800/20 p-4 rounded-xl">
                                        <div>
                                            <p className="text-slate-500 mb-1 flex items-center gap-1"><Navigation size={14} /> Distance</p>
                                            <p className={`font-medium ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>{route.distance_km} km</p>
                                        </div>
                                        <div>
                                            <p className="text-slate-500 mb-1 flex items-center gap-1"><Shield size={14} /> Safety</p>
                                            <p className={`font-medium ${route.safety_score > 60 ? 'text-emerald-400' : 'text-orange-400'}`}>{route.safety_score}%</p>
                                        </div>
                                        <div className="col-span-2 md:col-span-1 border-t border-slate-700/50 pt-3 md:border-t-0 md:pt-0">
                                            <p className="text-slate-500 mb-1 flex items-center gap-1"><Calendar size={14} /> Date</p>
                                            <p className={`font-medium ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>{new Date(route.created_at).toLocaleDateString()}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                </div>

            </div>
        </main>
    );
}
