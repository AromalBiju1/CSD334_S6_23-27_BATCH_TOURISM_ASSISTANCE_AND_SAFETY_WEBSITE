import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useTheme } from '../context/ThemeContext';
import { Lock, ArrowLeft, Save, Shield } from 'lucide-react';
import { changePassword } from '../api/services';
import toast from 'react-hot-toast';

export default function ChangePassword() {
    const { theme } = useTheme();
    const isDark = theme === 'dark';
    const navigate = useNavigate();
    const [loading, setLoading] = useState(false);
    const [formData, setFormData] = useState({
        old_password: '',
        new_password: '',
        confirm_password: '',
    });
    const [errors, setErrors] = useState({});

    const validateForm = () => {
        const newErrors = {};

        if (!formData.old_password) {
            newErrors.old_password = 'Old password is required';
        }

        if (!formData.new_password) {
            newErrors.new_password = 'New password is required';
        } else if (formData.new_password.length < 6) {
            newErrors.new_password = 'Password must be at least 6 characters';
        }

        if (formData.new_password !== formData.confirm_password) {
            newErrors.confirm_password = 'Passwords do not match';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
        if (errors[name]) {
            setErrors(prev => ({ ...prev, [name]: '' }));
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!validateForm()) return;

        setLoading(true);
        try {
            await changePassword({
                old_password: formData.old_password,
                new_password: formData.new_password
            });
            toast.success('Password changed successfully!');
            navigate('/profile');
        } catch (error) {
            toast.error(error.response?.data?.detail || 'Failed to change password');
            if (error.response?.status === 400 && error.response?.data?.detail?.toLowerCase().includes("old password")) {
                setErrors(prev => ({ ...prev, old_password: 'Incorrect old password' }));
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="pt-16 min-h-screen w-full">
            <div className="w-full max-w-xl mx-auto px-6 md:px-16 py-8">
                {/* Header */}
                <div className="flex items-center gap-4 mb-8">
                    <button
                        onClick={() => navigate('/profile')}
                        className={`w-10 h-10 rounded-xl flex items-center justify-center transition-colors ${isDark ? 'bg-slate-800 hover:bg-slate-700' : 'bg-slate-100 hover:bg-slate-200'}`}
                    >
                        <ArrowLeft size={20} className={isDark ? "text-slate-400" : "text-slate-600"} />
                    </button>
                    <div>
                        <h1 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-slate-900'}`}>Change Password</h1>
                        <p className={`text-sm ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>Secure your account with a new password</p>
                    </div>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="space-y-6">
                    <div className={`border rounded-2xl p-6 ${isDark ? 'bg-slate-900/50 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
                        <h2 className={`text-lg font-semibold mb-6 flex items-center gap-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>
                            <Shield size={18} className="text-emerald-400" />
                            Security Verification
                        </h2>

                        {/* Old Password */}
                        <div className="mb-5">
                            <label className={`block text-sm mb-2 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                Current Password <span className="text-red-400">*</span>
                            </label>
                            <div className="relative">
                                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                                <input
                                    type="password"
                                    name="old_password"
                                    value={formData.old_password}
                                    onChange={handleChange}
                                    placeholder="Enter current password"
                                    className={`w-full border rounded-xl h-12 pl-12 pr-4 focus:outline-none transition-colors ${errors.old_password
                                            ? 'border-red-500'
                                            : isDark
                                                ? 'bg-slate-800/50 border-slate-700 text-white focus:border-emerald-500/50 placeholder-slate-500'
                                                : 'bg-white border-slate-300 text-slate-900 focus:border-emerald-500/50 placeholder-slate-400'
                                        }`}
                                />
                            </div>
                            {errors.old_password && <p className="text-red-400 text-xs mt-1">{errors.old_password}</p>}
                        </div>

                        {/* New Password */}
                        <div className="mb-5">
                            <label className={`block text-sm mb-2 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                New Password <span className="text-red-400">*</span>
                            </label>
                            <div className="relative">
                                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                                <input
                                    type="password"
                                    name="new_password"
                                    value={formData.new_password}
                                    onChange={handleChange}
                                    placeholder="Enter new password"
                                    className={`w-full border rounded-xl h-12 pl-12 pr-4 focus:outline-none transition-colors ${errors.new_password
                                            ? 'border-red-500'
                                            : isDark
                                                ? 'bg-slate-800/50 border-slate-700 text-white focus:border-emerald-500/50 placeholder-slate-500'
                                                : 'bg-white border-slate-300 text-slate-900 focus:border-emerald-500/50 placeholder-slate-400'
                                        }`}
                                />
                            </div>
                            {errors.new_password && <p className="text-red-400 text-xs mt-1">{errors.new_password}</p>}
                        </div>

                        {/* Confirm Password */}
                        <div className="mb-5">
                            <label className={`block text-sm mb-2 ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                                Confirm New Password <span className="text-red-400">*</span>
                            </label>
                            <div className="relative">
                                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                                <input
                                    type="password"
                                    name="confirm_password"
                                    value={formData.confirm_password}
                                    onChange={handleChange}
                                    placeholder="Re-enter new password"
                                    className={`w-full border rounded-xl h-12 pl-12 pr-4 focus:outline-none transition-colors ${errors.confirm_password
                                            ? 'border-red-500'
                                            : isDark
                                                ? 'bg-slate-800/50 border-slate-700 text-white focus:border-emerald-500/50 placeholder-slate-500'
                                                : 'bg-white border-slate-300 text-slate-900 focus:border-emerald-500/50 placeholder-slate-400'
                                        }`}
                                />
                            </div>
                            {errors.confirm_password && <p className="text-red-400 text-xs mt-1">{errors.confirm_password}</p>}
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="flex gap-3">
                        <button
                            type="button"
                            onClick={() => navigate('/profile')}
                            className={`flex-1 h-12 rounded-xl font-medium transition-all ${isDark ? 'bg-slate-800 hover:bg-slate-700 border border-slate-700 text-white' : 'bg-white hover:bg-slate-50 border border-slate-300 text-slate-700'}`}
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={loading}
                            className="flex-1 flex items-center justify-center gap-2 bg-emerald-500 hover:bg-emerald-600 disabled:bg-slate-700 disabled:cursor-not-allowed text-white h-12 rounded-xl font-medium transition-all shadow-md hover:shadow-lg"
                        >
                            <Save size={18} />
                            <span>{loading ? 'Changing...' : 'Change Password'}</span>
                        </button>
                    </div>
                </form>
            </div>
        </main>
    );
}
