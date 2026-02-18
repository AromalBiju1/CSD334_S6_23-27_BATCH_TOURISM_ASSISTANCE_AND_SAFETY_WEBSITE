import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
    User, Mail, LogOut, Shield, Calendar, MapPin, Navigation,
    Star, Bell, Settings, ChevronRight, Map, Phone, Heart
} from 'lucide-react';
import toast from 'react-hot-toast';

export default function Profile() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();

    const handleLogout = () => {
        logout();
        toast.success('Logged out successfully');
        navigate('/');
    };

    if (!user) {
        return null;
    }

    const quickActions = [
        { icon: <Map size={20} />, label: 'Explore Map', path: '/explore', color: 'bg-emerald-500/20 text-emerald-400' },
        { icon: <Navigation size={20} />, label: 'Plan Route', path: '/safe-route', color: 'bg-blue-500/20 text-blue-400' },
        { icon: <MapPin size={20} />, label: 'Hotspots', path: '/hotspots', color: 'bg-orange-500/20 text-orange-400' },
        { icon: <Phone size={20} />, label: 'Emergency', path: '/emergency', color: 'bg-red-500/20 text-red-400' },
    ];

    const stats = [
        { label: 'Trips Viewed', value: '12', icon: <Map size={16} /> },
        { label: 'Routes Planned', value: '5', icon: <Navigation size={16} /> },
        { label: 'Cities Explored', value: '8', icon: <MapPin size={16} /> },
    ];

    const recentActivity = [
        { title: 'Explored Delhi Safety Map', time: '2 days ago', icon: <Map size={16} className="text-emerald-400" /> },
        { title: 'Planned route to Jaipur', time: '5 days ago', icon: <Navigation size={16} className="text-blue-400" /> },
        { title: 'Checked Mumbai hotspots', time: '1 week ago', icon: <MapPin size={16} className="text-orange-400" /> },
    ];

    return (
        <main className="pt-16 min-h-screen w-full">
            <div className="w-full max-w-4xl mx-auto px-6 md:px-16 py-8">

                {/* Profile Header */}
                <div className="bg-gradient-to-br from-emerald-500/10 to-blue-500/10 border border-slate-800 rounded-3xl p-8 mb-6">
                    <div className="flex flex-col md:flex-row items-center gap-6">
                        {/* Avatar */}
                        <div className="w-24 h-24 rounded-full bg-emerald-500/20 border-4 border-emerald-500/30 flex items-center justify-center overflow-hidden">
                            {user.profile_pic ? (
                                <img src={user.profile_pic} alt={user.name} className="w-full h-full object-cover" />
                            ) : (
                                <User className="text-emerald-400" size={40} />
                            )}
                        </div>

                        {/* User Info */}
                        <div className="text-center md:text-left flex-1">
                            <h1 className="text-2xl font-bold text-white mb-1">{user.name}</h1>
                            <p className="text-slate-400 flex items-center justify-center md:justify-start gap-2">
                                <Mail size={14} />
                                {user.email}
                            </p>
                            <div className="flex items-center justify-center md:justify-start gap-2 mt-2">
                                <span className="inline-flex items-center gap-1 bg-emerald-500/20 text-emerald-400 px-3 py-1 rounded-full text-xs font-medium">
                                    <Shield size={12} />
                                    Verified Traveler
                                </span>
                            </div>
                        </div>

                        {/* Edit Profile Button */}
                        <button
                            onClick={() => navigate('/profile/edit')}
                            className="bg-slate-800 hover:bg-slate-700 border border-slate-700 text-white px-4 py-2 rounded-xl text-sm font-medium transition-all flex items-center gap-2"
                        >
                            <Settings size={16} />
                            Edit Profile
                        </button>
                    </div>
                </div>

                {/* Stats Row */}
                <div className="grid grid-cols-3 gap-4 mb-6">
                    {stats.map((stat, index) => (
                        <div key={index} className="bg-slate-900/50 border border-slate-800 rounded-2xl p-4 text-center">
                            <div className="flex items-center justify-center gap-2 text-slate-400 mb-1">
                                {stat.icon}
                                <span className="text-xs">{stat.label}</span>
                            </div>
                            <p className="text-2xl font-bold text-white">{stat.value}</p>
                        </div>
                    ))}
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                    {/* Quick Actions */}
                    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <Star size={18} className="text-yellow-400" />
                            Quick Actions
                        </h2>
                        <div className="grid grid-cols-2 gap-3">
                            {quickActions.map((action, index) => (
                                <button
                                    key={index}
                                    onClick={() => navigate(action.path)}
                                    className="flex items-center gap-3 p-3 rounded-xl bg-slate-800/50 hover:bg-slate-800 border border-slate-700 transition-all group"
                                >
                                    <div className={`w-10 h-10 rounded-lg ${action.color} flex items-center justify-center`}>
                                        {action.icon}
                                    </div>
                                    <span className="text-sm text-slate-300 group-hover:text-white font-medium">{action.label}</span>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Recent Activity */}
                    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <Bell size={18} className="text-blue-400" />
                            Recent Activity
                        </h2>
                        <div className="space-y-3">
                            {recentActivity.map((activity, index) => (
                                <div key={index} className="flex items-center gap-3 p-3 rounded-xl bg-slate-800/30">
                                    <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center">
                                        {activity.icon}
                                    </div>
                                    <div className="flex-1">
                                        <p className="text-sm text-white">{activity.title}</p>
                                        <p className="text-xs text-slate-500">{activity.time}</p>
                                    </div>
                                    <ChevronRight size={16} className="text-slate-600" />
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Preferences */}
                    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <Heart size={18} className="text-pink-400" />
                            Preferences
                        </h2>
                        <div className="space-y-3">
                            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-800/30">
                                <span className="text-sm text-slate-300">Preferred Zone</span>
                                <span className="text-sm text-emerald-400 font-medium">Green Zones Only</span>
                            </div>
                            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-800/30">
                                <span className="text-sm text-slate-300">Notifications</span>
                                <span className="text-sm text-blue-400 font-medium">Enabled</span>
                            </div>
                            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-800/30">
                                <span className="text-sm text-slate-300">Language</span>
                                <span className="text-sm text-white font-medium">English</span>
                            </div>
                        </div>
                    </div>

                    {/* Account Actions */}
                    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                        <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                            <Settings size={18} className="text-slate-400" />
                            Account
                        </h2>
                        <div className="space-y-3">
                            <button className="w-full flex items-center justify-between p-3 rounded-xl bg-slate-800/30 hover:bg-slate-800/50 transition-colors text-left">
                                <span className="text-sm text-slate-300">Change Password</span>
                                <ChevronRight size={16} className="text-slate-600" />
                            </button>
                            <button className="w-full flex items-center justify-between p-3 rounded-xl bg-slate-800/30 hover:bg-slate-800/50 transition-colors text-left">
                                <span className="text-sm text-slate-300">Privacy Settings</span>
                                <ChevronRight size={16} className="text-slate-600" />
                            </button>
                            <button
                                onClick={handleLogout}
                                className="w-full flex items-center justify-center gap-2 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 text-red-400 py-3 rounded-xl font-medium transition-all"
                            >
                                <LogOut size={18} />
                                <span>Logout</span>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </main>
    );
}
