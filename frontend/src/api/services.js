import api from './axios';

// ── Request cache for map data (60s TTL) ──
const _requestCache = new Map();
const CACHE_TTL_MS = 60000;

function getCachedOrFetch(key, fetcher) {
    const cached = _requestCache.get(key);
    if (cached && Date.now() - cached.ts < CACHE_TTL_MS) {
        return Promise.resolve(cached.data);
    }
    return fetcher().then(data => {
        _requestCache.set(key, { data, ts: Date.now() });
        return data;
    });
}


export const getCities = async (params = {}) => {
    const defaultParams = { limit: 1000, ...params };
    const cacheKey = `cities:${JSON.stringify(defaultParams)}`;
    return getCachedOrFetch(cacheKey, async () => {
        const response = await api.get('/api/cities', { params: defaultParams });
        return response.data;
    });
};

export const getCityById = async (cityId) => {
    const response = await api.get(`/api/cities/${cityId}`);
    return response.data;
};

export const getCitiesByZone = async (zone) => {
    const response = await api.get('/api/cities', { params: { zone } });
    return response.data;
};

export const searchCities = async (query) => {
    const response = await api.get('/api/cities/search', { params: { q: query } });
    return response.data;
};


export const getSafeRoute = async (origin, destination) => {
    const response = await api.post('/api/routes/safe', { origin, destination });
    return response.data;
};


export const getRouteAlternatives = async (origin, destination) => {
    const response = await api.get('/api/routes/alternatives', {
        params: { origin, destination },
    });
    return response.data;
};

// Get smart routes using OSRM + Safety Agent
export const getSmartRoutes = async (origin, dest, mode = "driving") => {
    const params = { mode };

    // Resolve Origin
    if (origin && origin.lat && origin.lng) {
        params.origin_lat = origin.lat;
        params.origin_lng = origin.lng;
    } else {
        params.origin = origin.name || origin;
    }

    // Resolve Destination
    if (dest && dest.lat && dest.lng) {
        params.dest_lat = dest.lat;
        params.dest_lng = dest.lng;
    } else {
        params.destination = dest.name || dest;
    }

    const response = await api.get('/api/routes/smart', { params });
    return response.data;
};

// Live GPS: check if current position is in a danger zone
export const checkPositionSafety = async (lat, lng) => {
    const response = await api.get('/api/routes/check-safety', {
        params: { lat, lng },
    });
    return response.data;
};

// Live GPS: reroute from current position when in danger zone
export const rerouteFromPosition = async (lat, lng, destLat, destLng, destName) => {
    const response = await api.get('/api/routes/reroute', {
        params: { lat, lng, dest_lat: destLat, dest_lng: destLng, dest_name: destName },
    });
    return response.data;
};


export const getEmergencyContacts = async (cityId) => {
    const response = await api.get(`/api/emergency/${cityId}`);
    return response.data;
};


export const getAllEmergencyServices = async () => {
    const response = await api.get('/api/emergency');
    return response.data;
};

export const getAttractions = async (cityId) => {
    const response = await api.get(`/api/attractions/${cityId}`);
    return response.data;
};


export const getAllAttractions = async (params = {}) => {
    // Default limit=500 to fetch the full set and avoid silent truncation
    const defaultParams = { limit: 500, ...params };
    const response = await api.get('/api/recommendations', { params: defaultParams });
    return response.data;
};

//auth

export const login = async (email, password) => {
    const response = await api.post('/api/auth/login', { email, password });
    return response.data;
};

export const signup = async (userData) => {
    const response = await api.post('/api/auth/signup', userData);
    return response.data;
};

export const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
};

export const updateProfile = async (userData) => {
    const response = await api.put('/api/auth/profile', userData);
    return response.data;
};

export const changePassword = async (data) => {
    const response = await api.post('/api/profile/change-password', data);
    return response.data;
};

export const getCurrentUser = async () => {
    const response = await api.get('/api/auth/me');
    return response.data;
};


// ── Recommendations API ──

export const getUserPreferences = async () => {
    const response = await api.get('/api/user-recommendations/preferences');
    return response.data;
};

export const updateUserPreferences = async (data) => {
    const response = await api.put('/api/user-recommendations/preferences', data);
    return response.data;
};

export const getVisitedPlaces = async () => {
    const response = await api.get('/api/user-recommendations/visited');
    return response.data;
};

export const addVisitedPlace = async (data) => {
    const response = await api.post('/api/user-recommendations/visited', data);
    return response.data;
};

export const removeVisitedPlace = async (id) => {
    const response = await api.delete(`/api/user-recommendations/visited/${id}`);
    return response.data;
};

export const getRecommendations = async (limit = 20) => {
    const response = await api.get('/api/user-recommendations/for-you', { params: { limit } });
    return response.data;
};

export const getAvailableCategories = async () => {
    const response = await api.get('/api/user-recommendations/categories');
    return response.data;
};

export const searchAttractionsForMarking = async (search = "") => {
    const response = await api.get('/api/user-recommendations/all-attractions', { params: { search } });
    return response.data;
};

// ── Profile & Privacy API ──

export const getProfileInfo = async () => {
    const response = await api.get('/api/profile/me');
    return response.data;
};

export const getRecentActivity = async () => {
    const response = await api.get('/api/profile/activity');
    return response.data;
};

export const logActivity = async (actionType, title) => {
    try {
        const response = await api.post('/api/profile/activity', { action_type: actionType, title });
        return response.data;
    } catch (err) {
        console.error("Failed to log activity", err);
        return null; // Don't crash UI on analytics failure
    }
};

export const updatePrivacySettings = async (data) => {
    const response = await api.put('/api/profile/privacy', data);
    return response.data;
};

export const getSavedRoutes = async () => {
    const response = await api.get('/api/profile/saved-routes');
    return response.data;
};
