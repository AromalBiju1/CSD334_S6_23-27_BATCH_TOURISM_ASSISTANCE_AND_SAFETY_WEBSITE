import React, { useState, useCallback, useMemo } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, Polyline, useMap, ZoomControl } from 'react-leaflet';
import MarkerClusterGroup from 'react-leaflet-cluster';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';
import { Loader, AlertCircle } from 'lucide-react';

// Fix Leaflet default icons
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: markerIcon2x,
    iconUrl: markerIcon,
    shadowUrl: markerShadow,
});

// Constants
const INDIA_CENTER = [20.5937, 78.9629];
const DEFAULT_ZOOM = 5;

const ZONE_COLORS = {
    green: '#22c55e',
    orange: '#f97316',
    red: '#ef4444',
    user: '#3b82f6', // Blue for user location
};

const ZONE_LABELS = {
    green: 'Safe Zone',
    orange: 'Moderate Risk',
    red: 'High Risk',
    user: 'Your Location',
};

// Tile layer config from env or default
const TILE_URL = import.meta.env.VITE_MAP_TILE_URL || 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
const TILE_ATTRIBUTION = import.meta.env.VITE_MAP_ATTRIBUTION || '&copy; OpenStreetMap &copy; CARTO';

// Memoized zone icon creator - small markers for overview
const zoneIconCache = {};
const createZoneIcon = (zone, small = true) => {
    const key = `${zone}-${small}`;
    if (zoneIconCache[key]) return zoneIconCache[key];

    const size = small ? 12 : 24;
    const border = small ? 2 : 3;

    const icon = L.divIcon({
        className: 'custom-zone-marker',
        html: `
      <div style="
        width: ${size}px;
        height: ${size}px;
        background-color: ${ZONE_COLORS[zone] || '#6b7280'};
        border: ${border}px solid white;
        border-radius: 50%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.4);
        transition: transform 0.2s;
      "></div>
    `,
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2],
        popupAnchor: [0, -size / 2],
    });

    zoneIconCache[key] = icon;
    return icon;
};

// Custom cluster icon - uses single color when filtering, gradient when showing all
const createClusterCustomIcon = (cluster, zoneFilter = 'all') => {
    const count = cluster.getChildCount();

    // Size based on count
    let size = 42;
    if (count > 50) size = 56;
    else if (count > 20) size = 48;

    // Determine background color based on filter
    let bgColor;
    if (zoneFilter === 'all') {
        // Show conic gradient for mixed zones
        bgColor = `conic-gradient(
            ${ZONE_COLORS.green} 0deg 120deg,
            ${ZONE_COLORS.orange} 120deg 240deg,
            ${ZONE_COLORS.red} 240deg 360deg
        )`;
    } else {
        // Show single solid color matching the filter
        bgColor = ZONE_COLORS[zoneFilter] || ZONE_COLORS.orange;
    }

    return L.divIcon({
        html: `
            <div style="
                width: ${size}px;
                height: ${size}px;
                background: ${bgColor};
                border: 3px solid white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            ">
                <div style="
                    width: ${size - 12}px;
                    height: ${size - 12}px;
                    background: rgba(30, 41, 59, 0.95);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: 700;
                    font-size: ${count > 99 ? '11px' : '13px'};
                ">
                    ${count}
                </div>
            </div>
        `,
        className: 'custom-cluster-icon',
        iconSize: L.point(size, size),
        iconAnchor: [size / 2, size / 2],
    });
};

// Map controller component for programmatic control
function MapController({ center, zoom, fitBounds }) {
    const map = useMap();

    React.useEffect(() => {
        if (fitBounds && fitBounds.length >= 2) {
            map.fitBounds(fitBounds, { padding: [50, 50] });
        } else if (center) {
            map.setView(center, zoom, { animate: true });
        }
    }, [center, zoom, fitBounds, map]);

    return null;
}

// City marker component with selection buttons
const CityMarker = React.memo(({ city, onClick, showCircle, onSelectStart, onSelectDest }) => {
    const handleClick = useCallback(() => {
        onClick(city);
    }, [city, onClick]);

    return (
        <>
            <Marker
                position={[city.lat || city.latitude, city.lng || city.longitude]}
                icon={createZoneIcon(city.zone || city.safety_zone)}
                eventHandlers={{ click: handleClick }}
            >
                <Popup className="custom-popup">
                    <div className="p-1 min-w-[160px]">
                        <h3 className="font-bold text-gray-900 text-base mb-1">{city.name}</h3>
                        <p className="text-gray-600 text-sm mb-2">{city.state}</p>

                        <div className="flex gap-2 mb-2">
                            <button
                                onClick={(e) => { e.stopPropagation(); onSelectStart && onSelectStart(city); }}
                                className="flex-1 bg-emerald-500 hover:bg-emerald-600 text-white text-xs py-1 px-2 rounded transition-colors"
                            >
                                Start
                            </button>
                            <button
                                onClick={(e) => { e.stopPropagation(); onSelectDest && onSelectDest(city); }}
                                className="flex-1 bg-red-500 hover:bg-red-600 text-white text-xs py-1 px-2 rounded transition-colors"
                            >
                                End
                            </button>
                        </div>

                        <div
                            className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium text-white"
                            style={{ backgroundColor: ZONE_COLORS[city.zone || city.safety_zone] }}
                        >
                            {ZONE_LABELS[city.zone || city.safety_zone] || 'Unknown'}
                        </div>
                        {city.crime_index !== undefined && (
                            <p className="text-gray-500 text-xs mt-2">
                                Crime Index: {city.crime_index.toFixed(1)}
                            </p>
                        )}
                    </div>
                </Popup>
            </Marker>

            {showCircle && (
                <Circle
                    center={[city.lat || city.latitude, city.lng || city.longitude]}
                    radius={city.radius || 25000}
                    pathOptions={{
                        color: ZONE_COLORS[city.zone || city.safety_zone],
                        fillColor: ZONE_COLORS[city.zone || city.safety_zone],
                        fillOpacity: 0.12,
                        weight: 2,
                    }}
                />
            )}
        </>
    );
});

CityMarker.displayName = 'CityMarker';

// Route polyline component - supports custom colors and animation
const RoutePolyline = React.memo(({ route, index }) => {
    // Get color from route or default based on safety
    const color = route.color || (route.safe ? ZONE_COLORS.green : ZONE_COLORS.red);
    const weight = route.weight || (route.selected ? 6 : 4);
    const opacity = route.opacity || (route.selected ? 0.9 : 0.5);

    // Animation class for pulsing effect
    const animatedStyle = route.animate ? {
        className: 'route-pulse'
    } : {};

    return (
        <Polyline
            key={index}
            positions={route.path}
            pathOptions={{
                color: color,
                weight: weight,
                opacity: opacity,
                dashArray: route.safe === false && !route.selected ? '10, 10' : null,
                lineCap: 'round',
                lineJoin: 'round',
            }}
            {...animatedStyle}
        >
            {route.info && (
                <Popup>
                    <div className="p-2 min-w-[120px]">
                        <p className="font-semibold text-gray-900 capitalize mb-1">
                            {route.info.type?.replace('_', ' ') || 'Route'}
                        </p>
                        <p className="text-gray-600 text-sm">{route.info.distance}</p>
                        <p className="text-gray-500 text-xs">{route.info.duration}</p>
                    </div>
                </Popup>
            )}
        </Polyline>
    );
});

RoutePolyline.displayName = 'RoutePolyline';

// Legend component
const MapLegend = React.memo(() => (
    <div className="absolute bottom-4 left-4 z-[1000] bg-slate-900/90 backdrop-blur-sm border border-slate-700 rounded-xl p-3 text-xs">
        <p className="text-slate-400 font-medium mb-2">Safety Zones</p>
        {Object.entries(ZONE_LABELS).map(([zone, label]) => (
            <div key={zone} className="flex items-center gap-2 mb-1 last:mb-0">
                <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: ZONE_COLORS[zone] }}
                />
                <span className="text-white">{label}</span>
            </div>
        ))}
    </div>
));

MapLegend.displayName = 'MapLegend';

// Main SafetyMap component
export default function SafetyMap({
    center = INDIA_CENTER,
    zoom = DEFAULT_ZOOM,
    cities = [],
    routes = [],
    showZoneCircles = false,
    showLegend = true,
    zoneFilter = 'all',
    onCityClick = () => { },
    onSelectStart = () => { },
    onSelectDest = () => { },
    onMapReady = () => { },
    loading = false,
    error = null,
    height = '500px',
    className = '',
    fitBounds = null,
}) {
    const [mapReady, setMapReady] = useState(false);

    // Create cluster icon function with current zone filter
    const clusterIconFunction = useCallback((cluster) => {
        return createClusterCustomIcon(cluster, zoneFilter);
    }, [zoneFilter]);


    const handleMapReady = useCallback(() => {
        setMapReady(true);
        onMapReady();
    }, [onMapReady]);

    // Loading state
    if (loading) {
        return (
            <div
                className={`rounded-2xl overflow-hidden bg-slate-900/50 border border-slate-800 flex items-center justify-center ${className}`}
                style={{ height }}
            >
                <div className="text-center">
                    <Loader className="animate-spin text-emerald-400 mx-auto mb-3" size={40} />
                    <p className="text-slate-400 text-sm">Loading map...</p>
                </div>
            </div>
        );
    }

    // Error state
    if (error) {
        return (
            <div
                className={`rounded-2xl overflow-hidden bg-slate-900/50 border border-red-500/30 flex items-center justify-center ${className}`}
                style={{ height }}
            >
                <div className="text-center p-6">
                    <AlertCircle className="text-red-400 mx-auto mb-3" size={40} />
                    <p className="text-red-400 font-medium mb-1">Failed to load map</p>
                    <p className="text-slate-500 text-sm">{error}</p>
                </div>
            </div>
        );
    }

    return (
        <div className={`relative rounded-2xl overflow-hidden ${className}`} style={{ height }}>
            <MapContainer
                center={center}
                zoom={zoom}
                style={{ height: '100%', width: '100%' }}
                scrollWheelZoom={true}
                zoomControl={false}
                whenReady={handleMapReady}
            >
                <TileLayer url={TILE_URL} attribution={TILE_ATTRIBUTION} />
                <ZoomControl position="topright" />
                <MapController center={center} zoom={zoom} fitBounds={fitBounds} />

                {/* City markers with clustering */}
                <MarkerClusterGroup
                    chunkedLoading
                    iconCreateFunction={clusterIconFunction}
                    maxClusterRadius={60}
                    spiderfyOnMaxZoom={true}
                    showCoverageOnHover={false}
                    animate={true}
                    animateAddingMarkers={false}
                >
                    {cities.map((city) => (
                        <CityMarker
                            key={city.id}
                            city={city}
                            onClick={onCityClick}
                            showCircle={showZoneCircles}
                            onSelectStart={onSelectStart}
                            onSelectDest={onSelectDest}
                        />
                    ))}
                </MarkerClusterGroup>

                {/* Route polylines */}
                {routes.map((route, index) => (
                    <RoutePolyline key={index} route={route} index={index} />
                ))}
            </MapContainer>

            {/* Legend */}
            {showLegend && mapReady && <MapLegend />}
        </div>
    );
}
