import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, Map, Navigation, Phone, AlertTriangle, LogIn, Menu, X, MapPin, User, UserPlus } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import Logo from '../assets/logo.jpeg';

export default function Navbar() {
    const location = useLocation();
    const { isAuthenticated, user } = useAuth();
    const isActive = (path) => location.pathname === path;
    const [mobileOpen, setMobileOpen] = useState(false);

    // Nav items only show when logged in
    const navItems = isAuthenticated ? [
        { name: 'Home', path: '/', icon: <Home size={16} /> },
        { name: 'Explore', path: '/explore', icon: <Map size={16} /> },
        { name: 'Hotspots', path: '/hotspots', icon: <MapPin size={16} /> },
        { name: 'Safe Route', path: '/safe-route', icon: <Navigation size={16} /> },
        { name: 'Emergency', path: '/emergency', icon: <Phone size={16} /> },
    ] : [];

    return (
        <nav className="fixed top-0 left-0 right-0 z-[9999] bg-[#0a1628] border-b border-slate-800/50">
            <div className="w-full max-w-7xl mx-auto px-6">
                <div className="h-16 flex items-center justify-between">

                    {/* Logo */}
                    <Link to="/" className="flex items-center gap-2">
                        <img src={Logo} alt="GuardMyTrip" className="w-8 h-8 md:w-10 md:h-10 rounded-md object-cover" />
                        <div className="flex flex-col">
                            <span className="font-bold text-sm leading-tight text-white">GuardMyTrip</span>
                            <span className="text-[9px] leading-tight text-slate-400">Your Safety Companion</span>
                        </div>
                    </Link>

                    {/* Nav Links - Desktop */}
                    <div className="hidden md:flex items-center gap-6">
                        {navItems.map((item) => (
                            <Link
                                key={item.path}
                                to={item.path}
                                className={`flex items-center gap-2 text-sm font-medium transition-colors ${isActive(item.path)
                                    ? 'text-emerald-400'
                                    : 'text-slate-400 hover:text-white'
                                    }`}
                            >
                                {item.icon}
                                <span>{item.name}</span>
                            </Link>
                        ))}
                    </div>

                    {/* Mobile Menu Button */}
                    <button
                        onClick={() => setMobileOpen(!mobileOpen)}
                        className="md:hidden text-slate-200 hover:text-white"
                    >
                        {mobileOpen ? <X size={24} /> : <Menu size={24} />}
                    </button>

                    {/* Right Buttons - Desktop */}
                    <div className="hidden md:flex items-center gap-3">
                        {isAuthenticated ? (
                            <>
                                {/* Profile Icon */}
                                <Link
                                    to="/profile"
                                    className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
                                >
                                    <div className="w-8 h-8 rounded-full bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center overflow-hidden">
                                        {user?.profile_pic ? (
                                            <img src={user.profile_pic} alt={user.name} className="w-full h-full object-cover" />
                                        ) : (
                                            <User size={16} className="text-emerald-400" />
                                        )}
                                    </div>
                                </Link>

                                {/* SOS Button */}
                                <Link
                                    to="/emergency"
                                    className="flex items-center gap-1.5 bg-red-600 hover:bg-red-700 text-white px-3 py-1.5 rounded-full text-xs font-semibold transition-all"
                                >
                                    <AlertTriangle size={14} />
                                    SOS
                                </Link>
                            </>
                        ) : (
                            <>
                                {/* Sign In */}
                                <Link
                                    to="/login"
                                    className="flex items-center gap-1.5 bg-emerald-500 hover:bg-emerald-600 text-white px-3 py-1.5 rounded-full text-xs font-medium transition-all"
                                >
                                    <LogIn size={14} />
                                    Sign In
                                </Link>

                                {/* Sign Up */}
                                <Link
                                    to="/signup"
                                    className="flex items-center gap-1.5 bg-slate-700 hover:bg-slate-600 border border-slate-600 text-white px-3 py-1.5 rounded-full text-xs font-medium transition-all"
                                >
                                    <UserPlus size={14} />
                                    Sign Up
                                </Link>
                            </>
                        )}
                    </div>
                </div>
            </div>

            {/* Mobile Menu */}
            {mobileOpen && (
                <div className="md:hidden bg-[#0a1628] border-t border-slate-800/50">
                    <div className="flex flex-col px-6 py-4 gap-4">
                        {navItems.map((item) => (
                            <Link
                                key={item.path}
                                to={item.path}
                                onClick={() => setMobileOpen(false)}
                                className={`flex items-center gap-3 text-sm font-medium ${isActive(item.path) ? 'text-emerald-400' : 'text-slate-300'
                                    }`}
                            >
                                {item.icon}
                                {item.name}
                            </Link>
                        ))}

                        <div className="pt-4 border-t border-slate-800 flex flex-col gap-3">
                            {isAuthenticated ? (
                                <>
                                    <Link
                                        to="/profile"
                                        onClick={() => setMobileOpen(false)}
                                        className="flex items-center gap-2 text-white py-2"
                                    >
                                        <User size={16} />
                                        Profile
                                    </Link>
                                    <Link
                                        to="/emergency"
                                        onClick={() => setMobileOpen(false)}
                                        className="bg-red-600 text-white py-2 rounded-full text-center font-bold"
                                    >
                                        SOS
                                    </Link>
                                </>
                            ) : (
                                <>
                                    <Link
                                        to="/login"
                                        onClick={() => setMobileOpen(false)}
                                        className="bg-emerald-500 text-white py-2 rounded-full text-center font-medium"
                                    >
                                        Sign In
                                    </Link>
                                    <Link
                                        to="/signup"
                                        onClick={() => setMobileOpen(false)}
                                        className="bg-slate-700 text-white py-2 rounded-full text-center font-medium"
                                    >
                                        Sign Up
                                    </Link>
                                </>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </nav>
    );
}
