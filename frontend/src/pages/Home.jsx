import React from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  Map,
  Navigation,
  MapPin,
  Phone,
  Shield,
  ChevronRight,
  AlertTriangle,
} from "lucide-react";
import { useAuth } from "../context/AuthContext";
import toast from "react-hot-toast";

export default function Home() {
  const { isAuthenticated } = useAuth();
  const navigate = useNavigate();

  const handleProtectedClick = (path, label) => {
    if (isAuthenticated) {
      navigate(path);
    } else {
      toast.error(`Please login to access ${label}`);
      navigate("/login");
    }
  };

  const stats = [
    { value: "50+", label: "Cities Covered" },
    { value: "3", label: "Safety Zones" },
    { value: "100+", label: "Attractions" },
    { value: "24/7", label: "Emergency Help" },
  ];

  const features = [
    {
      icon: <Map size={24} />,
      title: "Safety Zones Map",
      description:
        "Visualize cities across India color-coded by safety levels based on crime data.",
      color: "bg-gradient-to-r from-purple-500 via-purple-400 to-pink-400",
      path: "/explore",
    },
    {
      icon: <Navigation size={24} />,
      title: "Safe Route Planning",
      description:
        "Get routes optimized for safety, avoiding high risk areas when possible.",
      color: "bg-gradient-to-r from-blue-500 via-cyan-400 to-teal-400",
      path: "/safe-route",
    },
    {
      icon: <MapPin size={24} />,
      title: "Tourist Hotspots",
      description:
        "Discover popular attractions with safety ratings for each location.",
      color: "bg-gradient-to-r from-red-500 via-orange-500 to-yellow-400",
      path: "/hotspots",
    },
    {
      icon: <Phone size={24} />,
      title: "Emergency Support",
      description:
        "Quick access to emergency contacts and step by step guidance for crisis situations.",
      color: "bg-gradient-to-r from-emerald-500 via-green-400 to-lime-400",
      path: "/emergency",
    },
  ];

  const zones = [
    {
      title: "Green Zone - Safe",
      description: "Low crime rate, highly recommended for tourists",
      borderColor: "border-l-green-500",
      bgColor: "bg-green-500/10",
      textColor: "text-green-400",
      icon: <Shield className="text-green-500" size={20} />,
    },
    {
      title: "Orange Zone - Moderate",
      description: "Exercise normal precautions, stay aware",
      borderColor: "border-l-orange-500",
      bgColor: "bg-orange-500/10",
      textColor: "text-orange-400",
      icon: <AlertTriangle className="text-orange-500" size={20} />,
    },
    {
      title: "Red Zone - High Risk",
      description: "Increased caution advised, avoid isolated areas",
      borderColor: "border-l-red-500",
      bgColor: "bg-red-500/10",
      textColor: "text-red-400",
      icon: <AlertTriangle className="text-red-500" size={20} />,
    },
  ];

  return (
    <main className="pt-16 w-full">
      {/* Hero Section */}
      <section className="min-h-[70vh] flex items-center justify-center text-center py-14 md:py-20">
        <div className="w-full max-w-5xl mx-auto px-6 md:px-16">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 px-4 py-2 rounded-full text-sm mb-8">
            <Shield size={16} />
            <span>Your Safety is Our Priority</span>
          </div>

          {/* Headline */}
          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold mb-5 leading-tight">
            Travel <span className="text-emerald-400">Safely</span> Across
            <br />
            <span className="text-emerald-400">India</span>
          </h1>

          {/* Description */}
          <p className="text-slate-400 text-sm sm:text-base md:text-lg max-w-xs sm:max-w-xl mx-auto mb-8 leading-relaxed">
            Navigate India confidently with real-time safety data, smart route planning, and instant emergency assistance.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center">
            <button
              onClick={() => handleProtectedClick("/explore", "Safety Map")}
              className="inline-flex items-center gap-2 bg-emerald-500 hover:bg-emerald-600 text-white px-5 py-3 text-sm rounded-full font-semibold transition-all active:scale-[0.98]"
            >
              <Map size={16} strokeWidth={1.8} />
              <span>Explore Safety Map</span>
              <ChevronRight size={16} strokeWidth={1.8} />
            </button>
            <button
              onClick={() => handleProtectedClick("/safe-route", "Route Planning")}
              className="inline-flex items-center gap-2 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-white px-5 py-3 text-sm rounded-full font-semibold transition-all"
            >
              <Navigation size={16} strokeWidth={1.8} />
              <span>Plan Safe Route</span>
            </button>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-12 md:py-16 border-y border-slate-800/50">
        <div className="w-full max-w-5xl mx-auto px-6 md:px-16">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-3xl md:text-4xl font-bold text-emerald-400 mb-1">
                  {stat.value}
                </div>
                <div className="text-slate-400 text-sm">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 md:py-24">
        <div className="w-full max-w-5xl mx-auto px-6 md:px-16">
          <div className="text-center mb-12">
            <h2 className="text-2xl sm:text-3xl font-bold mb-3">
              Smart Features for <span className="text-emerald-400">Safe Travel</span>
            </h2>
            <p className="text-slate-400 max-w-lg mx-auto">
              Everything you need to travel safely across India in one platform
            </p>
          </div>

          <div className="grid sm:grid-cols-2 gap-4">
            {features.map((feature, index) => (
              <button
                key={index}
                onClick={() => handleProtectedClick(feature.path, feature.title)}
                className="text-left bg-slate-900/30 border border-slate-800 rounded-2xl p-6 hover:border-slate-700 transition-all group"
              >
                <div
                  className={`w-12 h-12 ${feature.color} rounded-xl flex items-center justify-center text-white mb-4 group-hover:scale-105 transition-transform`}
                >
                  {feature.icon}
                </div>
                <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-emerald-400 transition-colors">
                  {feature.title}
                </h3>
                <p className="text-slate-400 text-sm leading-relaxed">
                  {feature.description}
                </p>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Safety Zones Section */}
      <section className="py-16 md:py-24 bg-slate-900/20">
        <div className="w-full max-w-5xl mx-auto px-6 md:px-16">
          <div className="text-center mb-12">
            <h2 className="text-2xl sm:text-3xl font-bold mb-3">
              Understanding <span className="text-emerald-400">Safety Zones</span>
            </h2>
            <p className="text-slate-400 max-w-lg mx-auto">
              Cities are classified based on crime data and safety metrics
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            {zones.map((zone, index) => (
              <div
                key={index}
                className={`border-l-4 ${zone.borderColor} ${zone.bgColor} rounded-lg p-5`}
              >
                <div className="flex items-center gap-2 mb-2">
                  {zone.icon}
                  <h3 className={`font-semibold ${zone.textColor}`}>
                    {zone.title}
                  </h3>
                </div>
                <p className="text-slate-400 text-sm">{zone.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 md:py-24">
        <div className="w-full max-w-3xl mx-auto px-6 md:px-16 text-center">
          <h2 className="text-2xl sm:text-3xl font-bold mb-4">
            Ready to Travel <span className="text-emerald-400">Safely</span>?
          </h2>
          <p className="text-slate-400 mb-8 max-w-md mx-auto">
            Join thousands of travelers who trust GuardMyTrip for their safety
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Link
              to="/signup"
              className="inline-flex items-center justify-center gap-2 bg-emerald-500 hover:bg-emerald-600 text-white px-6 py-3 rounded-full font-semibold transition-all"
            >
              Get Started Free
              <ChevronRight size={16} />
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}
