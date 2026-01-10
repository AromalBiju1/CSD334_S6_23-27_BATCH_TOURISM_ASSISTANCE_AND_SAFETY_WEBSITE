import React from "react";
import { Link } from "react-router-dom";
import {
  Map,
  Navigation,
  MapPin,
  Phone,
  Shield,
  ChevronRight,
  AlertTriangle,
} from "lucide-react";

export default function Home() {
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
      color: "bg-emerald-500",
    },
    {
      icon: <Navigation size={24} />,
      title: "Safe Route Planning",
      description:
        "Get routes optimized for safety, avoiding high risk areas when possible.",
      color: "bg-blue-500",
    },
    {
      icon: <MapPin size={24} />,
      title: "Tourist Hotspots",
      description:
        "Discover popular attractions with safety ratings for each location.",
      color: "bg-orange-500",
    },
    {
      icon: <Phone size={24} />,
      title: "Emergency Support",
      description:
        "Quick access to emergency contacts and step by step guidance for crisis situations.",
      color: "bg-green-500",
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
          <h1
            className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl 
font-bold mb-5 leading-tight"
          >
            Travel <span className="text-emerald-400">Safely</span> Across
            <br />
            <span className="text-emerald-400">India</span>
          </h1>

          {/* Description */}
          <p
            className="text-slate-400 text-sm sm:text-base md:text-lg 
max-w-xs sm:max-w-xl mx-auto mb-8 leading-relaxed"
          ></p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center">
            <Link
              to="/explore"
              className="inline-flex items-center gap-2
  bg-emerald-500 hover:bg-emerald-600
  text-white
  px-5 py-3
  text-sm
  rounded-full
  font-semibold
  transition-all
  active:scale-[0.98]
  active:opacity-90

"
            >
              <Map size={16} strokeWidth={1.8}/>
              <span>Explore Safety Map</span>
              <ChevronRight size={16} strokeWidth={1.8} />
            </Link>
            <Link
              to="/safe-route"
              className="
  inline-flex items-center gap-2
  bg-slate-800 hover:bg-slate-700
  border border-slate-700
  text-white
  px-5 py-3
  text-sm
  rounded-full
  font-medium
  transition-all
  active:scale-[0.98]
  active:opacity-90

">
              <Navigation size={16} strokeWidth={1.8} />
              <span>Plan Safe Route</span>
            </Link>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 border-t border-b border-slate-800/50">
        <div className="w-full max-w-6xl mx-auto px-6 md:px-16">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            {stats.map((stat, index) => (
              <div key={index}>
                <div className="text-3xl md:text-5xl font-bold text-emerald-400 mb-1">
                  {stat.value}
                </div>
                <div className="text-slate-400">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="w-full max-w-6xl mx-auto px-6 md:px-16">
          {/* Section Header */}
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Everything You Need for{" "}
              <span className="text-emerald-400">Safe Travel</span>
            </h2>
            <p className="text-slate-400 max-w-2xl mx-auto">
              Our comprehensive platform combines safety data, navigation, and
              emergency support to ensure you have a worry-free travel
              experience.
            </p>
          </div>

          {/* Feature Cards */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <div
                key={index}
                className="bg-slate-900/50 border border-slate-800 rounded-2xl 
p-5 md:p-6 hover:border-slate-700 transition-all"
              >
                <div
                  className={`${feature.color} w-12 h-12 rounded-xl flex items-center justify-center text-white mb-4`}
                >
                  {feature.icon}
                </div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  {feature.title}
                </h3>
                <p className="text-slate-400 text-sm leading-relaxed">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Safety Zones Section */}
      <section className="py-14 md:py-20">
        <div className="w-full max-w-6xl mx-auto px-6 md:px-16">
          <div className="grid lg:grid-cols-2 gap-12 items-start">
            {/* Left - Zone Info */}
            <div>
              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                Understanding Our{" "}
                <span className="text-emerald-400">Safety Zones</span>
              </h2>
              <p className="text-slate-400 mb-8 leading-relaxed">
                We analyze crime data from official sources to classify cities
                into three safety zones, helping you make informed travel
                decisions.
              </p>

              <div className="space-y-4">
                {zones.map((zone, index) => (
                  <div
                    key={index}
                    className={`${zone.bgColor} border-l-4 ${zone.borderColor} rounded-r-xl p-4 flex items-start gap-4`}
                  >
                    <div className="flex-shrink-0 mt-0.5">{zone.icon}</div>
                    <div>
                      <h4 className={`font-semibold ${zone.textColor} mb-1`}>
                        {zone.title}
                      </h4>
                      <p className="text-slate-400 text-sm">
                        {zone.description}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right - Map Preview */}
            <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-8 flex flex-col items-center justify-center min-h-[280px] md:min-h-[400px]">
              <div className="text-slate-600 mb-4">
                <Map size={56} strokeWidth={1.5} />
              </div>
              <p className="text-slate-500 mb-4 text-center">
                Interactive Safety Map
              </p>
              <Link
                to="/explore"
                className="text-emerald-400 hover:text-emerald-300 inline-flex items-center gap-1 text-sm font-medium"
              >
                View Full Map <ChevronRight size={16} />
              </Link>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
