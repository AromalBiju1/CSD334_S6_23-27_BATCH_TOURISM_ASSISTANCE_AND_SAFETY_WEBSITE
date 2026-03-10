import React, { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { Mail, Lock, LogIn, Shield, Eye, EyeOff } from "lucide-react";
import { login as loginApi } from "../api/services";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";
import toast from "react-hot-toast";

export default function Login() {
    const navigate = useNavigate();
    const location = useLocation();
    const { login } = useAuth();
    const { theme } = useTheme();
    const isDark = theme === 'dark';
    const [formData, setFormData] = useState({
        email: "",
        password: "",
    });
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            const response = await loginApi(formData.email, formData.password);
            console.log('Login response:', response);

            const token = response.access_token || response.token;
            const userData = response.user || {
                email: formData.email,
                name: formData.email.split('@')[0],
            };

            if (token) {
                login(userData, token);
                toast.success("Login successful!");
                navigate('/', { replace: true });
            } else {
                toast.error("Invalid response from server");
            }
        } catch (error) {
            console.error('Login error:', error);
            toast.error(error.response?.data?.detail || "Login failed");
        } finally {
            setLoading(false);
        }
    };

    return (
        <main className="pt-16 min-h-screen w-full flex items-center justify-center px-6">
            <div className="w-full max-w-md">
                {/* Header */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-emerald-500/10 border border-emerald-500/30 mb-6">
                        <Shield className="text-emerald-400" size={30} />
                    </div>
                    <h1 className="text-2xl sm:text-3xl font-bold mb-2">
                        Welcome <span className="text-emerald-400">Back</span>
                    </h1>
                    <p className={`text-sm ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                        Sign in to continue to GuardMyTrip
                    </p>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="space-y-5">
                    {/* Email */}
                    <div>
                        <label className={`block text-sm mb-2 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>Email</label>
                        <div className="relative">
                            <Mail className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                            <input
                                type="email"
                                name="email"
                                value={formData.email}
                                onChange={handleChange}
                                placeholder="Enter your email"
                                required
                                className={`w-full rounded-xl h-12 pl-12 pr-4 focus:outline-none focus:border-emerald-500/50 transition-colors ${isDark
                                    ? 'bg-slate-900/50 border border-slate-800 text-white placeholder-slate-500'
                                    : 'bg-white border border-slate-300 text-slate-900 placeholder-slate-400'
                                    }`}
                            />
                        </div>
                    </div>

                    {/* Password */}
                    <div>
                        <label className={`block text-sm mb-2 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>Password</label>
                        <div className="relative">
                            <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                            <input
                                type={showPassword ? "text" : "password"}
                                name="password"
                                value={formData.password}
                                onChange={handleChange}
                                placeholder="Enter your password"
                                required
                                className={`w-full rounded-xl h-12 pl-12 pr-12 focus:outline-none focus:border-emerald-500/50 transition-colors ${isDark
                                    ? 'bg-slate-900/50 border border-slate-800 text-white placeholder-slate-500'
                                    : 'bg-white border border-slate-300 text-slate-900 placeholder-slate-400'
                                    }`}
                            />
                            <button
                                type="button"
                                onClick={() => setShowPassword(!showPassword)}
                                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                            >
                                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                            </button>
                        </div>
                    </div>

                    {/* Forgot Password Link */}
                    <div className="text-right">
                        <Link to="/forgot-password" className="text-sm text-emerald-400 hover:text-emerald-300">
                            Forgot password?
                        </Link>
                    </div>

                    {/* Submit Button */}
                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full flex items-center justify-center gap-2 bg-emerald-500 hover:bg-emerald-600 disabled:bg-slate-700 disabled:cursor-not-allowed text-white h-12 rounded-xl font-medium transition-all"
                    >
                        <LogIn size={18} />
                        <span>{loading ? "Signing in..." : "Sign In"}</span>
                    </button>

                    {/* Sign Up Link */}
                    <p className={`text-center text-sm mt-6 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                        Don't have an account?{" "}
                        <Link to="/signup" className="text-emerald-400 hover:text-emerald-300 font-medium">
                            Sign up
                        </Link>
                    </p>
                </form>
            </div>
        </main>
    );
}
