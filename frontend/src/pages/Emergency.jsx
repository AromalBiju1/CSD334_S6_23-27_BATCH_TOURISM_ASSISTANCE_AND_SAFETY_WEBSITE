import React, { useState, useEffect } from "react";
import { AlertTriangle, Phone, Shield, Copy, Flame, Plane, ChevronDown, User, Baby, Car } from "lucide-react";
import { getCities, getEmergencyContacts, getAllEmergencyServices } from "../api/services";
import toast from "react-hot-toast";

// National helplines (always available)
const NATIONAL_CONTACTS = [
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
        iconBg: "bg-blue-500/20 text-blue-400",
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
        color: "border-orange-500/40 bg-orange-500/5",
        iconBg: "bg-orange-500/20 text-orange-400",
    },
    {
        number: "181",
        title: "Women Helpline",
        description: "Women safety support",
        icon: <User size={22} />,
        color: "border-slate-700 bg-slate-800/30",
        iconBg: "bg-pink-500/20 text-pink-400",
    },
    {
        number: "1098",
        title: "Child Helpline",
        description: "Child safety & protection",
        icon: <Baby size={22} />,
        color: "border-slate-700 bg-slate-800/30",
        iconBg: "bg-purple-500/20 text-purple-400",
    },
    {
        number: "1363",
        title: "Tourist Helpline",
        description: "Tourist assistance",
        icon: <Plane size={22} />,
        color: "border-slate-700 bg-slate-800/30",
        iconBg: "bg-cyan-500/20 text-cyan-400",
    },
    {
        number: "1073",
        title: "Road Accident",
        description: "Road accident emergency",
        icon: <Car size={22} />,
        color: "border-slate-700 bg-slate-800/30",
        iconBg: "bg-slate-700 text-slate-300",
    },
];

export default function Emergency() {
    const [cities, setCities] = useState([]);
    const [selectedCity, setSelectedCity] = useState(null);
    const [cityContacts, setCityContacts] = useState([]);
    const [loading, setLoading] = useState(false);

    // Fetch cities for dropdown
    useEffect(() => {
        getCities()
            .then(setCities)
            .catch(() => setCities([]));
    }, []);

    // Fetch city-specific contacts when city selected
    useEffect(() => {
        if (selectedCity) {
            setLoading(true);
            getEmergencyContacts(selectedCity.id)
                .then(setCityContacts)
                .catch(() => setCityContacts([]))
                .finally(() => setLoading(false));
        } else {
            setCityContacts([]);
        }
    }, [selectedCity]);

    const copyNumber = (number) => {
        navigator.clipboard.writeText(number);
        toast.success(`Copied ${number}`);
    };

    const quickCall = [
        { number: "112", color: "bg-red-500 hover:bg-red-600", icon: <Phone size={14} /> },
        { number: "100", color: "bg-blue-500 hover:bg-blue-600", icon: <Shield size={14} /> },
        { number: "102", color: "bg-green-500 hover:bg-green-600", icon: <Phone size={14} /> },
    ];

    return (
        <main className="pt-16 min-h-screen w-full">
            <div className="w-full max-w-6xl mx-auto px-6 md:px-16 py-8">
                {/* Header */}
                <div className="text-center mb-10">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-red-500/10 border border-red-500/30 mb-6">
                        <AlertTriangle className="text-red-500" size={30} />
                    </div>
                    <h1 className="text-2xl sm:text-3xl font-bold mb-3">
                        <span className="text-red-400">Emergency</span> Support
                    </h1>
                    <p className="text-slate-400 max-w-lg mx-auto leading-relaxed">
                        Quick access to emergency services. These numbers work across India.
                    </p>
                </div>

                {/* City Selector */}
                <div className="mb-8">
                    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-5">
                        <p className="text-slate-400 text-sm mb-3">Select a city for local emergency contacts:</p>
                        <div className="relative">
                            <select
                                value={selectedCity?.id || ""}
                                onChange={(e) => {
                                    const city = cities.find(c => c.id === parseInt(e.target.value));
                                    setSelectedCity(city || null);
                                }}
                                className="w-full appearance-none bg-slate-800/50 border border-slate-700 rounded-xl h-12 pl-4 pr-10 text-white focus:outline-none focus:border-emerald-500/50 cursor-pointer transition-colors"
                            >
                                <option value="">National helplines only</option>
                                {cities.map((city) => (
                                    <option key={city.id} value={city.id}>
                                        {city.name}, {city.state}
                                    </option>
                                ))}
                            </select>
                            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" size={18} />
                        </div>
                    </div>
                </div>

                {/* City-specific Contacts */}
                {selectedCity && cityContacts.length > 0 && (
                    <div className="mb-8">
                        <h2 className="text-lg font-semibold text-white mb-4">
                            {selectedCity.name} Emergency Contacts
                        </h2>
                        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
                            {cityContacts.map((contact, index) => (
                                <div
                                    key={index}
                                    className="border border-emerald-500/30 bg-emerald-500/5 rounded-2xl p-5"
                                >
                                    <button
                                        onClick={() => copyNumber(contact.number)}
                                        className="absolute top-4 right-4 text-slate-500 hover:text-white transition-colors p-1"
                                    >
                                        <Copy size={16} />
                                    </button>
                                    <div className="text-2xl font-bold text-emerald-400 mb-1">{contact.number}</div>
                                    <h3 className="font-semibold text-white mb-1">{contact.name}</h3>
                                    <p className="text-slate-400 text-sm mb-4">{contact.service_type}</p>
                                    <a
                                        href={`tel:${contact.number.replace(/[^0-9]/g, '')}`}
                                        className="inline-flex items-center gap-2 bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/30 text-emerald-400 px-4 py-2 rounded-lg text-sm transition-all"
                                    >
                                        <Phone size={14} />
                                        <span>Call Now</span>
                                    </a>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* National Emergency Cards */}
                <h2 className="text-lg font-semibold text-white mb-4">
                    National Emergency Numbers
                </h2>
                <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                    {NATIONAL_CONTACTS.map((contact, index) => (
                        <div
                            key={index}
                            className={`relative border rounded-2xl p-5 ${contact.color}`}
                        >
                            <button
                                onClick={() => copyNumber(contact.number)}
                                className="absolute top-4 right-4 text-slate-500 hover:text-white transition-colors p-1"
                            >
                                <Copy size={16} />
                            </button>

                            <div className={`w-11 h-11 rounded-xl flex items-center justify-center mb-4 ${contact.iconBg}`}>
                                {contact.icon}
                            </div>

                            <div className="text-3xl font-bold text-white mb-1">{contact.number}</div>
                            <h3 className="font-semibold text-white mb-1">{contact.title}</h3>
                            <p className="text-slate-400 text-sm mb-4">{contact.description}</p>

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
