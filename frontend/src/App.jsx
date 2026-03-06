import React from "react";
import { Routes, Route } from "react-router-dom";
import { Toaster } from 'react-hot-toast';
import { AuthProvider } from './context/AuthContext';
import { ThemeProvider, useTheme } from './context/ThemeContext';
import ProtectedRoute from './components/ProtectedRoute';
import Navbar from "./components/Navbar";
import BackgroundBlur from './components/BackgroundBlur';
import Home from "./pages/Home";
import Explore from './pages/Explore';
import SafeRoute from './pages/SafeRoute';
import Emergency from './pages/Emergency';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Hotspots from './pages/Hotspots';
import Profile from './pages/Profile';
import EditProfile from './pages/EditProfile';
import Recommendations from './pages/Recommendations';

function AppContent() {
  const { theme } = useTheme();

  return (
    <div className={`min-h-screen theme-transition ${theme === 'dark' ? 'bg-[#020617] text-white' : 'bg-[#f8fafc] text-[#0f172a]'}`}>
      <Navbar />
      <BackgroundBlur />
      <Routes>
        {/* Public Routes */}
        <Route path='/' element={<Home />} />
        <Route path='/login' element={<Login />} />
        <Route path='/signup' element={<Signup />} />

        {/* Protected Routes */}
        <Route path='/explore' element={
          <ProtectedRoute><Explore /></ProtectedRoute>
        } />
        <Route path='/hotspots' element={
          <ProtectedRoute><Hotspots /></ProtectedRoute>
        } />
        <Route path='/safe-route' element={
          <ProtectedRoute><SafeRoute /></ProtectedRoute>
        } />
        <Route path='/emergency' element={
          <ProtectedRoute><Emergency /></ProtectedRoute>
        } />
        <Route path='/profile' element={
          <ProtectedRoute><Profile /></ProtectedRoute>
        } />
        <Route path='/profile/edit' element={
          <ProtectedRoute><EditProfile /></ProtectedRoute>
        } />
        <Route path='/recommendations' element={
          <ProtectedRoute><Recommendations /></ProtectedRoute>
        } />
      </Routes>
      <ThemedToaster />
    </div>
  );
}

function ThemedToaster() {
  const { theme } = useTheme();
  return (
    <Toaster
      position="top-center"
      toastOptions={{
        style: {
          background: theme === 'dark' ? '#1e293b' : '#ffffff',
          color: theme === 'dark' ? '#fff' : '#0f172a',
          border: `1px solid ${theme === 'dark' ? '#334155' : '#e2e8f0'}`,
        },
      }}
    />
  );
}

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;