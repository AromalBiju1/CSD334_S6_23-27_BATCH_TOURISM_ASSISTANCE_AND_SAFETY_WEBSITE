import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { User, Mail, Phone, ArrowLeft, Save, Camera, Shield } from 'lucide-react';
import { updateProfile } from '../api/services';
import toast from 'react-hot-toast';

export default function EditProfile() {
    const { user, login } = useAuth();
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        phone: '',
    });
    const [errors, setErrors] = useState({});

    useEffect(() => {
        if (user) {
            setFormData({
                name: user.name || '',
                email: user.email || '',
                phone: user.phone || '',
            });
        }
    }, [user]);

    const validateForm = () => {
        const newErrors = {};

        if (!formData.name.trim()) {
            newErrors.name = 'Name is required';
        }

        if (!formData.email.trim()) {
            newErrors.email = 'Email is required';
        } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
            newErrors.email = 'Please enter a valid email';
        }

        if (formData.phone && !/^[0-9]{10}$/.test(formData.phone.replace(/\D/g, ''))) {
            newErrors.phone = 'Please enter a valid 10-digit phone number';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
        // Clear error when user starts typing
        if (errors[name]) {
            setErrors(prev => ({ ...prev, [name]: '' }));
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!validateForm()) {
            return;
        }

        setLoading(true);
        try {
            const response = await updateProfile(formData);

            // Update local auth state with new user data
            const updatedUser = { ...user, ...formData };
            const token = localStorage.getItem('token');
            login(updatedUser, token);

            toast.success('Profile updated successfully!');
            navigate('/profile');
        } catch (error) {
            toast.error(error.response?.data?.detail || 'Failed to update profile');
        } finally {
            setLoading(false);
        }
    };

    if (!user) {
        return null;
    }

    return (
        <main className="pt-16 min-h-screen w-full">
            <div className="w-full max-w-2xl mx-auto px-6 md:px-16 py-8">
                {/* Header */}
                <div className="flex items-center gap-4 mb-8">
                    <button
                        onClick={() => navigate('/profile')}
                        className="w-10 h-10 rounded-xl bg-slate-800 hover:bg-slate-700 flex items-center justify-center transition-colors"
                    >
                        <ArrowLeft size={20} className="text-slate-400" />
                    </button>
                    <div>
                        <h1 className="text-2xl font-bold text-white">Edit Profile</h1>
                        <p className="text-slate-400 text-sm">Update your personal information</p>
                    </div>
                </div>

                {/* Avatar Section */}
                <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 mb-6">
                    <div className="flex items-center gap-6">
                        <div className="relative">
                            <div className="w-20 h-20 rounded-full bg-emerald-500/20 border-2 border-emerald-500/30 flex items-center justify-center overflow-hidden">
                                {user.profile_pic ? (
                                    <img src={user.profile_pic} alt={user.name} className="w-full h-full object-cover" />
                                ) : (
                                    <User className="text-emerald-400" size={32} />
                                )}
                            </div>
                            <button className="absolute -bottom-1 -right-1 w-8 h-8 rounded-full bg-emerald-500 flex items-center justify-center text-white hover:bg-emerald-600 transition-colors">
                                <Camera size={14} />
                            </button>
                        </div>
                        <div>
                            <h3 className="text-white font-medium mb-1">Profile Photo</h3>
                            <p className="text-slate-400 text-sm">JPG, PNG or GIF. Max size 2MB</p>
                        </div>
                    </div>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6">
                        <h2 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
                            <Shield size={18} className="text-emerald-400" />
                            Personal Information
                        </h2>

                        {/* Name */}
                        <div className="mb-5">
                            <label className="block text-sm text-slate-400 mb-2">
                                Full Name <span className="text-red-400">*</span>
                            </label>
                            <div className="relative">
                                <User className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                                <input
                                    type="text"
                                    name="name"
                                    value={formData.name}
                                    onChange={handleChange}
                                    placeholder="Enter your full name"
                                    className={`w-full bg-slate-800/50 border rounded-xl h-12 pl-12 pr-4 text-white placeholder-slate-500 focus:outline-none transition-colors ${errors.name ? 'border-red-500' : 'border-slate-700 focus:border-emerald-500/50'
                                        }`}
                                />
                            </div>
                            {errors.name && <p className="text-red-400 text-xs mt-1">{errors.name}</p>}
                        </div>

                        {/* Email */}
                        <div className="mb-5">
                            <label className="block text-sm text-slate-400 mb-2">
                                Email Address <span className="text-red-400">*</span>
                            </label>
                            <div className="relative">
                                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                                <input
                                    type="email"
                                    name="email"
                                    value={formData.email}
                                    onChange={handleChange}
                                    placeholder="Enter your email"
                                    className={`w-full bg-slate-800/50 border rounded-xl h-12 pl-12 pr-4 text-white placeholder-slate-500 focus:outline-none transition-colors ${errors.email ? 'border-red-500' : 'border-slate-700 focus:border-emerald-500/50'
                                        }`}
                                />
                            </div>
                            {errors.email && <p className="text-red-400 text-xs mt-1">{errors.email}</p>}
                            <p className="text-slate-500 text-xs mt-1">This will be used for login and notifications</p>
                        </div>

                        {/* Phone */}
                        <div className="mb-5">
                            <label className="block text-sm text-slate-400 mb-2">
                                Phone Number
                            </label>
                            <div className="relative">
                                <Phone className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                                <div className="absolute left-12 top-1/2 -translate-y-1/2 text-slate-400 text-sm border-r border-slate-700 pr-2">
                                    +91
                                </div>
                                <input
                                    type="tel"
                                    name="phone"
                                    value={formData.phone}
                                    onChange={handleChange}
                                    placeholder="Enter 10-digit number"
                                    maxLength={10}
                                    className={`w-full bg-slate-800/50 border rounded-xl h-12 pl-24 pr-4 text-white placeholder-slate-500 focus:outline-none transition-colors ${errors.phone ? 'border-red-500' : 'border-slate-700 focus:border-emerald-500/50'
                                        }`}
                                />
                            </div>
                            {errors.phone && <p className="text-red-400 text-xs mt-1">{errors.phone}</p>}
                            <p className="text-slate-500 text-xs mt-1">Phone number must be unique</p>
                        </div>
                    </div>

                    {/* Info Box */}
                    <div className="bg-blue-500/10 border border-blue-500/30 rounded-2xl p-4">
                        <h3 className="text-blue-400 font-medium text-sm mb-2">What changes when you update?</h3>
                        <ul className="text-slate-400 text-xs space-y-1">
                            <li>• Your profile name will be displayed across the app</li>
                            <li>• Email is used for login and receiving safety alerts</li>
                            <li>• Phone number is used for emergency SOS features</li>
                            <li>• All changes take effect immediately</li>
                        </ul>
                    </div>

                    {/* Actions */}
                    <div className="flex gap-3">
                        <button
                            type="button"
                            onClick={() => navigate('/profile')}
                            className="flex-1 bg-slate-800 hover:bg-slate-700 border border-slate-700 text-white h-12 rounded-xl font-medium transition-all"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={loading}
                            className="flex-1 flex items-center justify-center gap-2 bg-emerald-500 hover:bg-emerald-600 disabled:bg-slate-700 disabled:cursor-not-allowed text-white h-12 rounded-xl font-medium transition-all"
                        >
                            <Save size={18} />
                            <span>{loading ? 'Saving...' : 'Save Changes'}</span>
                        </button>
                    </div>
                </form>
            </div>
        </main>
    );
}
