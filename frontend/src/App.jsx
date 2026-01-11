import React from "react";
import { Routes, Route } from "react-router-dom";
import { Toaster } from 'react-hot-toast';
import { AuthProvider } from './context/AuthContext';
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

function App() {
  return (
    <AuthProvider>
      <div className="min-h-screen bg-[#020617] text-white">
        <Navbar />
        <BackgroundBlur/>
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
        </Routes>
        <Toaster
          position="top-center"
          toastOptions={{
            style: {
              background: '#1e293b',
              color: '#fff',
              border: '1px solid #334155'
            },
          }}
        />
      </div>
    </AuthProvider>
  );
}

export default App;