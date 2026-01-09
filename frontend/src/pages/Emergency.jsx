import React from "react";
import { AlertTriangle, Phone, Shield, Copy, Flame, Plane } from "lucide-react";

export default function Emergency() {
    const emergencyContacts = [
        {
            number: "112",
            title: "Universal Emergency",
            description: "Works across all services",
            icon: <Phone size={22} />,
            color: "border-red-500/40 bg-red-500/5",
            iconBg: "bg-red-500/20 text-red-400",
        },
        {
            number: "100",
            title: "Police",
            description: "Law enforcement emergency",
            icon: <Shield size={22} />,
            color: "border-slate-700 bg-slate-800/30",
            iconBg: "bg-slate-700 text-slate-300",
        },
        {
            number: "102",
            title: "Ambulance",
            description: "Medical emergencies",
            icon: <Phone size={22} />,
            color: "border-green-500/40 bg-green-500/5",
            iconBg: "bg-green-500/20 text-green-400",
        },
        {
            number: "101",
            title: "Fire",
            description: "Fire emergencies",
            icon: <Flame size={22} />,
            color: "border-slate-700 bg-slate-800/30",
            iconBg: "bg-slate-700 text-slate-300",
        },
        {
            number: "1091",
            title: "Women Helpline",
            description: "Women safety support",
            icon: <Phone size={22} />,
            color: "border-slate-700 bg-slate-800/30",
            iconBg: "bg-slate-700 text-slate-300",
        },
        {
            number: "1363",
            title: "Tourist Helpline",
            description: "Tourist assistance",
            icon: <Plane size={22} />,
            color: "border-slate-700 bg-slate-800/30",
            iconBg: "bg-slate-700 text-slate-300",
        },
    ];

    const quickCall = [
        { number: "112", color: "bg-red-500 hover:bg-red-600", icon: <Phone size={14} /> },
        { number: "100", color: "bg-blue-500 hover:bg-blue-600", icon: <Shield size={14} /> },
        { number: "102", color: "bg-green-500 hover:bg-green-600", icon: <Phone size={14} /> },
    ];

    return (
        <main className="pt-16 min-h-screen w-full">
            <div className="w-full max-w-6xl mx-auto px-16 py-8">
                {/* Header */}
                <div className="text-center mb-10">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-red-500/10 border border-red-500/30 mb-6">
                        <AlertTriangle className="text-red-500" size={30} />
                    </div>
                    <h1 className="text-2xl sm:text-3xl font-bold mb-3">
                        <span className="text-red-400">Emergency</span> Support
                    </h1>
                    <p className="text-slate-400 max-w-lg mx-auto leading-relaxed">
                        Quick access to emergency services and important contacts. Stay calm and call for help when needed.
                    </p>
                </div>

                {/* Emergency Cards Grid */}
                <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
                    {emergencyContacts.map((contact, index) => (
                        <div
                            key={index}
                            className={`relative border rounded-2xl p-5 ${contact.color}`}
                        >
                            {/* Copy Button */}
                            <button className="absolute top-4 right-4 text-slate-500 hover:text-white transition-colors p-1">
                                <Copy size={16} />
                            </button>

                            {/* Icon */}
                            <div className={`w-11 h-11 rounded-xl flex items-center justify-center mb-4 ${contact.iconBg}`}>
                                {contact.icon}
                            </div>

                            {/* Number */}
                            <div className="text-3xl font-bold text-white mb-1">{contact.number}</div>

                            {/* Title & Description */}
                            <h3 className="font-semibold text-white mb-1">{contact.title}</h3>
                            <p className="text-slate-400 text-sm mb-4">{contact.description}</p>

                            {/* Call Button */}
                            <a
                                href={`tel:${contact.number}`}
                                className="inline-flex items-center gap-2 bg-slate-700/50 hover:bg-slate-700 border border-slate-600 text-white px-4 py-2 rounded-lg text-sm transition-all"
                            >
                                <Phone size={14} />
                                <span>Call Now</span>
                            </a>
                        </div>
                    ))}
                </div>

                {/* Quick Call Bar */}
                <div className="flex flex-wrap items-center justify-center gap-3 py-4 bg-slate-900/30 border border-slate-800 rounded-2xl">
                    <span className="text-slate-400 text-sm px-2">Quick Call:</span>
                    <div className="flex gap-2">
                        {quickCall.map((item, index) => (
                            <a
                                key={index}
                                href={`tel:${item.number}`}
                                className={`${item.color} text-white px-4 py-2 rounded-full text-sm font-medium inline-flex items-center gap-2 transition-all`}
                            >
                                {item.icon}
                                <span>{item.number}</span>
                            </a>
                        ))}
                    </div>
                </div>
            </div>
        </main>
    );
}
