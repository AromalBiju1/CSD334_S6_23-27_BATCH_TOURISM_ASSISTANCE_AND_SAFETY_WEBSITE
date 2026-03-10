import React, { useEffect, useRef, useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { checkPositionSafety } from '../api/services';
import toast from 'react-hot-toast';

export default function SafetyMonitor() {
    const { user } = useAuth();
    const [isEnabled, setIsEnabled] = useState(false);
    const lastCheckRef = useRef({ lat: 0, lng: 0, time: 0 });
    const lastZoneRef = useRef(null);
    const watchIdRef = useRef(null);

    // Update local state when user preferences change
    useEffect(() => {
        if (user?.preferences?.notifications_enabled) {
            setIsEnabled(true);
        } else {
            setIsEnabled(false);
        }
    }, [user]);

    const calculateDistance = (lat1, lon1, lat2, lon2) => {
        const R = 6371; // km
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a =
            Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c;
    };

    const triggerNotification = (title, body) => {
        if (!("Notification" in window)) return;

        if (Notification.permission === "granted") {
            new Notification(title, {
                body,
                icon: '/shield-logo.png' // Fallback to a generic shield icon if available
            });
        }
    };

    const handlePositionChange = async (position) => {
        const { latitude: lat, longitude: lng } = position.coords;
        const now = Date.now();

        // Throttling: Check only if moved > 500m OR 2 minutes have passed
        const distance = calculateDistance(
            lastCheckRef.current.lat,
            lastCheckRef.current.lng,
            lat,
            lng
        );
        const timeElapsed = now - lastCheckRef.current.time;

        if (distance < 0.5 && timeElapsed < 120000) return;

        try {
            const safety = await checkPositionSafety(lat, lng);
            lastCheckRef.current = { lat, lng, time: now };

            // Only notify if entering a Red zone or if the backend specifically triggers a reroute/alert
            if (safety.zone === 'red' && lastZoneRef.current !== 'red') {
                triggerNotification(
                    "⚠️ Safety Alert: High Risk Zone",
                    `You have entered ${safety.nearest_district || 'a high-risk area'}. Please stay vigilant.`
                );
                toast.error(`⚠️ Alert: Entering High Risk Zone (${safety.nearest_district})`, { duration: 6000 });
            } else if (safety.trigger_reroute && !lastZoneRef.current) {
                // Secondary trigger for general reroute logic
                triggerNotification(
                    "🛡️ GuardMyTrip Advice",
                    "We recommend checking for safer alternative routes for your current location."
                );
            }

            lastZoneRef.current = safety.zone;
        } catch (err) {
            console.error("SafetyMonitor background check failed:", err);
        }
    };

    useEffect(() => {
        if (isEnabled && navigator.geolocation) {
            // Start watching position
            watchIdRef.current = navigator.geolocation.watchPosition(
                handlePositionChange,
                (error) => console.warn("SafetyMonitor Geolocation error:", error),
                { enableHighAccuracy: false, timeout: 30000, maximumAge: 60000 }
            );
        }

        return () => {
            if (watchIdRef.current) {
                navigator.geolocation.clearWatch(watchIdRef.current);
            }
        };
    }, [isEnabled]);

    return null; // Background component
}
