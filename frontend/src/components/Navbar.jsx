import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, Map, Navigation, Phone, AlertTriangle, LogIn } from 'lucide-react';
import Logo from '../assets/logo2.jpeg';

export default function Navbar() {
    const location = useLocation();
    const isActive = (path) => location.pathname === path;

    const navItems = [
        { name: 'Home', path: '/', icon: <Home size={16} /> },
        { name: 'Explore', path: '/explore', icon: <Map size={16} /> },
        { name: 'Safe Route', path: '/safe-route', icon: <Navigation size={16} /> },
        { name: 'Emergency', path: '/emergency', icon: <Phone size={16} /> },
    ];

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 bg-[#0a1628] border-b border-slate-800/50">
            <div className="w-full max-w-7xl mx-auto px-6">
                <div className="h-16 flex items-center justify-between">

                    {/* Logo */}
                    <Link to="/" className="flex items-center gap-3">
                        <img src={Logo} alt="GuardMyTrip" className="w-10 h-10 rounded-lg object-cover" />
                        <div className="flex flex-col">
                            <span className="font-bold text-base text-white">GuardMyTrip</span>
                            <span className="text-[10px] text-slate-400">Your Safety Companion</span>
                        </div>
                    </Link>

                    {/* Nav Links */}
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

                    {/* Right Buttons */}
                    <div className="flex items-center gap-3">
                        <button className="flex items-center gap-2 bg-emerald-500 hover:bg-emerald-600 text-white px-5 py-2 rounded-full text-sm font-medium transition-all">
                            <LogIn size={16} />
                            Sign In
                        </button>
                        <button className="flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white px-5 py-2 rounded-full text-sm font-bold transition-all">
                            <AlertTriangle size={16} />
                            SOS
                        </button>
                    </div>
                </div>
            </div>
        </nav>
    );
}
