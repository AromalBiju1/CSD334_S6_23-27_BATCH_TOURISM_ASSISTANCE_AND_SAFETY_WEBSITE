import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Toaster } from 'react-hot-toast';
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Explore from './pages/Explore';
import SafeRoute from './pages/SafeRoute';
import Emergency from './pages/Emergency';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Hotspots from './pages/Hotspots';

function App() {
  return (
    <div className="min-h-screen bg-[#020617] text-white">
      <Navbar />
      <Routes>
        <Route path='/' element={<Home />} />
        <Route path='/explore' element={<Explore />} />
        <Route path='/safe-route' element={<SafeRoute />} />
        <Route path='/emergency' element={<Emergency />} />
        <Route path='/login' element={<Login />} />
        <Route path='/signup' element={<Signup />} />
        <Route path='/hotspots' element={<Hotspots />} />
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
  );
}

export default App;