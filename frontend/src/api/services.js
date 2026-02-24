import api from './axios';


export const getCities = async (params = {}) => {
    // Request all cities by default (up to 1000)
    const defaultParams = { limit: 1000, ...params };
    const response = await api.get('/api/cities', { params: defaultParams });
    return response.data;
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
    const response = await api.get('/api/attractions', { params });
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

export const getCurrentUser = async () => {
    const response = await api.get('/api/auth/me');
    return response.data;
};
