import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { Shield, Bell, ArrowLeft, Save, Loader } from 'lucide-react';
import toast from 'react-hot-toast';
import { getProfileInfo, updatePrivacySettings } from '../api/services';

export default function PrivacySettings() {
    const { user } = useAuth();
    const { theme } = useTheme();
    const isDark = theme === 'dark';
    const navigate = useNavigate();

    const [isPublic, setIsPublic] = useState(false);
    const [notificationsEnabled, setNotificationsEnabled] = useState(false);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        const fetchPrivacy = async () => {
            try {
                const info = await getProfileInfo();
                setIsPublic(info.privacy?.is_public ?? false);
                setNotificationsEnabled(info.preferences?.notifications_enabled ?? false);
            } catch (error) {
                console.error("Failed to load privacy settings", error);
                toast.error("Could not load current settings.");
            } finally {
                setLoading(false);
            }
        };
        if (user) fetchPrivacy();
    }, [user]);

    const handleSave = async () => {
        setSaving(true);
        try {
            await updatePrivacySettings({
                is_public: isPublic,
                notifications_enabled: notificationsEnabled
            });
            toast.success("Privacy settings updated successfully!");
            navigate('/profile');
        } catch (error) {
            console.error("Failed to update privacy settings", error);
            toast.error("Failed to update settings. Please try again.");
        } finally {
            setSaving(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-screen">
                <Loader className="animate-spin text-emerald-400" size={40} />
            </div>
        );
    }

    return (
        <main className="pt-24 pb-12 min-h-screen w-full">
            <div className="w-full max-w-2xl mx-auto px-6">

                {/* Header */}
                <div className="flex items-center gap-4 mb-8">
                    <button
                        onClick={() => navigate('/profile')}
                        className={`p-2 rounded-xl transition-colors ${isDark ? 'hover:bg-slate-800 text-slate-400 hover:text-white' : 'hover:bg-slate-100 text-slate-500 hover:text-slate-900'}`}
                    >
                        <ArrowLeft size={24} />
                    </button>
                    <div>
                        <h1 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-slate-900'}`}>Privacy &amp; Settings</h1>
                        <p className={`text-sm ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>Manage your public profile and notifications</p>
                    </div>
                </div>

                <div className={`p-6 rounded-3xl border shadow-sm ${isDark ? 'bg-slate-900/50 border-slate-800' : 'bg-white border-slate-200'}`}>

                    {/* Public Profile Toggle */}
                    <div className="flex items-start justify-between py-5 border-b border-slate-800/50">
                        <div className="flex gap-4 pr-4">
                            <div className={`mt-1 p-2 rounded-xl ${isDark ? 'bg-slate-800 text-emerald-400' : 'bg-emerald-50 text-emerald-600'}`}>
                                <Shield size={20} />
                            </div>
                            <div>
                                <h3 className={`font-semibold mb-1 ${isDark ? 'text-white' : 'text-slate-900'}`}>Public Profile</h3>
                                <p className={`text-sm leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                                    Allow other users to view your public travel statistics and general achievements.
                                    Your exact routes and active locations are never shared.
                                </p>
                            </div>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer flex-shrink-0 mt-2">
                            <input
                                type="checkbox"
                                className="sr-only peer"
                                checked={isPublic}
                                onChange={(e) => setIsPublic(e.target.checked)}
                            />
                            <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-emerald-500"></div>
                        </label>
                    </div>

                    {/* Notifications Toggle */}
                    <div className="flex items-start justify-between py-5 mb-4">
                        <div className="flex gap-4 pr-4">
                            <div className={`mt-1 p-2 rounded-xl ${isDark ? 'bg-slate-800 text-blue-400' : 'bg-blue-50 text-blue-600'}`}>
                                <Bell size={20} />
                            </div>
                            <div>
                                <h3 className={`font-semibold mb-1 ${isDark ? 'text-white' : 'text-slate-900'}`}>Push Notifications</h3>
                                <p className={`text-sm leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                                    Receive alerts when you are entering a high risk danger zone while navigating a planned route.
                                </p>
                            </div>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer flex-shrink-0 mt-2">
                            <input
                                type="checkbox"
                                className="sr-only peer"
                                checked={notificationsEnabled}
                                onChange={(e) => setNotificationsEnabled(e.target.checked)}
                            />
                            <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-500"></div>
                        </label>
                    </div>

                    <button
                        onClick={handleSave}
                        disabled={saving}
                        className="w-full mt-6 py-4 rounded-xl flex justify-center items-center gap-2 bg-emerald-500 hover:bg-emerald-600 text-white font-medium transition-all disabled:opacity-50"
                    >
                        {saving ? <Loader className="animate-spin" size={20} /> : <Save size={20} />}
                        {saving ? "Saving Changes..." : "Save Settings"}
                    </button>

                </div>
            </div>
        </main>
    );
}
