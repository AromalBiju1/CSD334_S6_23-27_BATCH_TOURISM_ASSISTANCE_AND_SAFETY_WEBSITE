# 🧭 Tourist Assistance & Safety Website

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Visit_Site-brightgreen?style=for-the-badge)](https://csd-334-s6-23-27-batch-tourism-assi.vercel.app/)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](#)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](#)

A smart, safety-focused web platform designed to enhance the travel experience by combining seamless tourist hotspot discovery with real-time safety-aware navigation, algorithmic personalization, and emergency support tools. Featuring a premium, animated glassmorphic UI with full Dark/Light mode support.

---

## 📌 Problem Statement

Tourists visiting unfamiliar locations often face:
- Limited access to reliable local safety information.
- Risk of unknowingly traveling through high-crime or physically unsafe zones.
- Lack of personalized, safety-conscious travel recommendations.
- fragmented access to real-time emergency assistance.

Most existing platforms focus strictly on generic attraction discovery or pure navigation, **failing to integrate structural safety insights with travel planning**. This leaves travelers vulnerable in environments they do not understand.

---

## 🎯 Project Objective

To build a **highly intuitive, safety-first tourist assistance website** that empowers travelers to explore India confidently.

### Key Objectives:
- Discover relevant tourist hotspots through intelligent recommendations.
- Mitigate travel risk by visually plotting and identifying unsafe zones.
- Actively calculate and suggest visually distinct alternative routes when danger intersects a path.
- Quickly provide localized emergency assistance information.

---

## 🚀 Full Feature Breakdown

### 1️⃣ Intelligent "Safe Route" Navigation
- **Point-to-Point Routing:** Enter an Origin and Destination to instantly plot a travel path.
- **Risk Intersection Detection:** The backend actively evaluates the drawn path against classified crime city boundaries (Green=Safe, Orange=Moderate, Red=High Risk).
- **Curved Safe Detours:** If a route crosses a Red Zone, the system automatically uses OSRM polygon data to draft an alternative "Curved Safe Route" circumventing the heavy danger zone. 
- **Visual Contrast:** High-risk primary routes are highlighted red alongside calculated, glowing green alternate routes, paired with active severity warnings indicating *why* the detour was drawn.

### 2️⃣ "For You" Recommendation Engine (Machine Learning)
- Sophisticated backend suggestion engine actively comparing the user's explicitly stated profile preferences (Budget, Travel Style, Categories) via **Content-Based Filtering**.
- Enhances suggestions utilizing **Collaborative Filtering** arrays by dynamically matching users who share similar lists of "Visited Places" to elevate unknown hidden gems.
- Strictly ensures high-priority attractions are scored against real-time API ratings and the user's localized Safe-Zone tolerances.

### 3️⃣ Interactive Explore Map & Hotspots
- **Safety Heatmap:** A beautiful, responsive Leaflet.js rendering of cities across the nation color-coded to transparently reflect their historical NCRB crime-rate index.
- **OpenTripMap Deep Integration:** The database boasts over 500,000 top-rated global POIs. Filter interactive hotspot carousels seamlessly by categories (Temples, Parks, Historical, Beaches, etc.) to discover new destinations.

### 4️⃣ Authentication & Profile Ecosystem
- **Secure Persistence:** Full JWT-authorized user accounts.
- **Activity History:** The system automatically logs an activity footprint of generated routes and viewed maps.
- **Privacy Controls:** Dedicated UI to selectively delete history logs or entirely wipe history cache immediately.
- **Visited Places Tracking:** Search any attraction to log a timeline of your travel history explicitly, directly influencing algorithmic weights in the background.

### 5️⃣ Emergency Rapid Support Tracker
- Interactive localized dashboard fetching National Emergency Hotlines (Police 112, Women Helpline 1091, Ambulance 108).
- Adaptively fetches regional contact information correlated directly to the user's designated "Active Location".

### 6️⃣ Premium & Modern UI/UX
- Developed with state-of-the-art Web Design principles emphasizing rich gradient meshes, floating background blurs, animated modal transitions, and responsive Glassmorphism paneling.
- **Dynamic Theming:** Deeply mapped Context-API state enabling 1-click toggling between sleek Light Mode and immersive Dark Mode interfaces.

---

## 🗺️ Map & Location Services

- **Global Map Provider:** OpenStreetMap (OSM) via Leaflet.js
- **Routing Engine API:** OSRM (Open Source Routing Machine) Polyline decoding.
- **Attraction Seeders:** OpenTripMap (OTM) integration for highly accurate geographical coordinates and image metadata.
- Lightweight, open-source, and privacy-conscious mapping stack.

---

## 🛠️ Technology Stack

### Frontend Architecture
- **Framework:** React.js (Vite)
- **Styling:** Tailwind CSS (Vanilla + Custom Complex CSS Animations)
- **Map Rendering:** Leaflet.js & React-Leaflet
- **State Management:** Custom React Contexts / Hooks
- **Icons:** Lucide-React

### Backend Architecture
- **Framework:** FastAPI (Python)
- **Database Engineering:** PostgreSQL (Supabase Cloud Deployment)
- **ORM:** SQLAlchemy (with concurrent DB indexing & relation mapping)
- **Auth:** Passlib, bcrypt, JWT payload encoding.

### Deployed Services
- **Backend:** Render (Dockerized API)
- **Frontend:** Vercel

---

## 💻 Minimum Hardware Requirements

- **Processor:** Intel Core i3 (7th Gen) / AMD Ryzen 3 or equivalent
- **RAM:** 8 GB
- **Graphics:** Integrated GPU (Intel UHD 620+ with WebGL 2.0 rendering support)
- **Internet:** Stable broadband connection to communicate with map API tiles.

---

## 🎯 Target Audience

- Tourists traveling to new or unfamiliar locations globally.
- Solo travelers prioritizing physical safety.
- Expatriates and backpackers assessing geographical risk.
- Safety-conscious general planners.

---

## ✅ Conclusion

The **Tourist Assistance & Safety Website** breaks the mold of traditional map aggregators by strictly putting *security first*. It dynamically weaves an intricate layer of backend data analysis—from crime indexes to algorithmic profiling—into a blazing fast, gorgeous React interface. 

By leveraging **FastAPI**, **PostgreSQL**, and **React**, the platform remains highly scalable, capable of parsing hundreds of thousands of location nodes near-instantly, making the process of intelligent modern tourism genuinely stress-free.
