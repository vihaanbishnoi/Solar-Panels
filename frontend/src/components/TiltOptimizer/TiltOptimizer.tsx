import React, { useState, useEffect, useCallback } from 'react';
import {
  Compass,
  Sun,
  Thermometer,
  Wind,
  Droplets,
  Zap,
  TrendingUp,
  RefreshCw,
  MapPin,
  Clock,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  Sparkles,
  Activity,
  Target,
  BarChart3
} from 'lucide-react';
import { optimizeTiltAngle, TiltOptimizationResult } from '../../services/api';

interface TiltConfig {
  latitude: number;
  longitude: number;
  ghi: number;
  hour: number;
  temperature: number;
  humidity: number;
  windSpeed: number;
  voltage: number;
  current: number;
  daysSinceInstallation: number;
  currentTilt: number;
}

const defaultConfig: TiltConfig = {
  latitude: 28.6139, // Delhi
  longitude: 77.2090,
  ghi: 800,
  hour: 12,
  temperature: 30,
  humidity: 50,
  windSpeed: 2.5,
  voltage: 38,
  current: 8,
  daysSinceInstallation: 365,
  currentTilt: 25
};

// Popular city presets
const cityPresets = [
  { name: 'Delhi', lat: 28.6139, lon: 77.2090, temp: 32 },
  { name: 'Mumbai', lat: 19.0760, lon: 72.8777, temp: 30 },
  { name: 'Ahmedabad', lat: 23.0225, lon: 72.5714, temp: 35 },
  { name: 'Chennai', lat: 13.0827, lon: 80.2707, temp: 33 },
  { name: 'Bangalore', lat: 12.9716, lon: 77.5946, temp: 26 },
  { name: 'Jaipur', lat: 26.9124, lon: 75.7873, temp: 34 },
];

export const TiltOptimizer: React.FC = () => {
  const [config, setConfig] = useState<TiltConfig>(defaultConfig);
  const [result, setResult] = useState<TiltOptimizationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoOptimize, setAutoOptimize] = useState(false);

  const runOptimization = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await optimizeTiltAngle({
        latitude: config.latitude,
        longitude: config.longitude,
        ghi: config.ghi,
        hour: config.hour,
        temperature: config.temperature,
        humidity: config.humidity,
        windSpeed: config.windSpeed,
        voltage: config.voltage,
        current: config.current,
        daysSinceInstallation: config.daysSinceInstallation,
        currentTilt: config.currentTilt
      });

      if (response.success && response.data) {
        setResult(response.data);
      } else {
        setError(response.error || 'Optimization failed');
      }
    } catch (err) {
      setError('Failed to connect to ML service');
    } finally {
      setLoading(false);
    }
  }, [config]);

  // Auto-optimize when config changes
  useEffect(() => {
    if (autoOptimize) {
      const timer = setTimeout(runOptimization, 300);
      return () => clearTimeout(timer);
    }
  }, [config, autoOptimize, runOptimization]);

  // Initial optimization
  useEffect(() => {
    runOptimization();
  }, []);

  const handleCitySelect = (city: typeof cityPresets[0]) => {
    setConfig(prev => ({
      ...prev,
      latitude: city.lat,
      longitude: city.lon,
      temperature: city.temp
    }));
  };

  const updateConfig = (key: keyof TiltConfig, value: number) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-3 bg-gradient-to-r from-amber-500/20 to-orange-500/20 px-6 py-3 rounded-full border border-amber-500/30 mb-4">
            <Compass className="w-6 h-6 text-amber-400" />
            <span className="text-amber-300 font-semibold">ML-Powered Optimization</span>
          </div>
          <h1 className="text-4xl font-bold text-white mb-3">
            Solar Panel Tilt Optimizer
          </h1>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Find the optimal tilt angle for maximum energy output using our machine learning model
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Configuration Panel */}
          <div className="lg:col-span-1 space-y-6">
            {/* Location */}
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
              <div className="flex items-center gap-2 mb-4">
                <MapPin className="w-5 h-5 text-blue-400" />
                <h3 className="text-lg font-semibold text-white">Location</h3>
              </div>

              {/* City Presets */}
              <div className="grid grid-cols-3 gap-2 mb-4">
                {cityPresets.map(city => (
                  <button
                    key={city.name}
                    onClick={() => handleCitySelect(city)}
                    className={`px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                      Math.abs(config.latitude - city.lat) < 0.1
                        ? 'bg-blue-500 text-white'
                        : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                    }`}
                  >
                    {city.name}
                  </button>
                ))}
              </div>

              <div className="space-y-4">
                <div>
                  <label className="text-sm text-slate-400 mb-1 block">Latitude</label>
                  <input
                    type="number"
                    value={config.latitude}
                    onChange={e => updateConfig('latitude', parseFloat(e.target.value))}
                    step="0.1"
                    className="w-full bg-slate-700 rounded-lg px-4 py-2 text-white"
                  />
                </div>
                <div>
                  <label className="text-sm text-slate-400 mb-1 block">Longitude</label>
                  <input
                    type="number"
                    value={config.longitude}
                    onChange={e => updateConfig('longitude', parseFloat(e.target.value))}
                    step="0.1"
                    className="w-full bg-slate-700 rounded-lg px-4 py-2 text-white"
                  />
                </div>
              </div>
            </div>

            {/* Environmental Conditions */}
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
              <div className="flex items-center gap-2 mb-4">
                <Sun className="w-5 h-5 text-yellow-400" />
                <h3 className="text-lg font-semibold text-white">Conditions</h3>
              </div>

              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400 flex items-center gap-1">
                      <Sun className="w-4 h-4" /> GHI (W/m²)
                    </span>
                    <span className="text-yellow-400 font-semibold">{config.ghi}</span>
                  </div>
                  <input
                    type="range"
                    min="200"
                    max="1200"
                    value={config.ghi}
                    onChange={e => updateConfig('ghi', parseInt(e.target.value))}
                    className="w-full accent-yellow-500"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400 flex items-center gap-1">
                      <Clock className="w-4 h-4" /> Hour of Day
                    </span>
                    <span className="text-blue-400 font-semibold">{config.hour}:00</span>
                  </div>
                  <input
                    type="range"
                    min="6"
                    max="18"
                    value={config.hour}
                    onChange={e => updateConfig('hour', parseInt(e.target.value))}
                    className="w-full accent-blue-500"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400 flex items-center gap-1">
                      <Thermometer className="w-4 h-4" /> Temperature
                    </span>
                    <span className="text-red-400 font-semibold">{config.temperature}°C</span>
                  </div>
                  <input
                    type="range"
                    min="10"
                    max="50"
                    value={config.temperature}
                    onChange={e => updateConfig('temperature', parseInt(e.target.value))}
                    className="w-full accent-red-500"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400 flex items-center gap-1">
                      <Droplets className="w-4 h-4" /> Humidity
                    </span>
                    <span className="text-cyan-400 font-semibold">{config.humidity}%</span>
                  </div>
                  <input
                    type="range"
                    min="20"
                    max="100"
                    value={config.humidity}
                    onChange={e => updateConfig('humidity', parseInt(e.target.value))}
                    className="w-full accent-cyan-500"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400 flex items-center gap-1">
                      <Wind className="w-4 h-4" /> Wind Speed
                    </span>
                    <span className="text-teal-400 font-semibold">{config.windSpeed} m/s</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="15"
                    step="0.5"
                    value={config.windSpeed}
                    onChange={e => updateConfig('windSpeed', parseFloat(e.target.value))}
                    className="w-full accent-teal-500"
                  />
                </div>
              </div>
            </div>

            {/* Panel Settings */}
            <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
              <div className="flex items-center gap-2 mb-4">
                <Zap className="w-5 h-5 text-purple-400" />
                <h3 className="text-lg font-semibold text-white">Panel</h3>
              </div>

              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-slate-400">Current Tilt</span>
                    <span className="text-purple-400 font-semibold">{config.currentTilt}°</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="60"
                    value={config.currentTilt}
                    onChange={e => updateConfig('currentTilt', parseInt(e.target.value))}
                    className="w-full accent-purple-500"
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-slate-400">Voltage (V)</label>
                    <input
                      type="number"
                      value={config.voltage}
                      onChange={e => updateConfig('voltage', parseFloat(e.target.value))}
                      className="w-full bg-slate-700 rounded px-3 py-1.5 text-white text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-slate-400">Current (A)</label>
                    <input
                      type="number"
                      value={config.current}
                      onChange={e => updateConfig('current', parseFloat(e.target.value))}
                      className="w-full bg-slate-700 rounded px-3 py-1.5 text-white text-sm"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-xs text-slate-400">Days Since Installation</label>
                  <input
                    type="number"
                    value={config.daysSinceInstallation}
                    onChange={e => updateConfig('daysSinceInstallation', parseInt(e.target.value))}
                    className="w-full bg-slate-700 rounded px-3 py-1.5 text-white text-sm"
                  />
                </div>
              </div>
            </div>

            {/* Controls */}
            <div className="flex gap-3">
              <button
                onClick={runOptimization}
                disabled={loading}
                className="flex-1 flex items-center justify-center gap-2 bg-gradient-to-r from-amber-500 to-orange-500 text-white font-semibold py-3 rounded-xl hover:opacity-90 transition-opacity disabled:opacity-50"
              >
                {loading ? (
                  <RefreshCw className="w-5 h-5 animate-spin" />
                ) : (
                  <Target className="w-5 h-5" />
                )}
                Optimize
              </button>
              <button
                onClick={() => setAutoOptimize(!autoOptimize)}
                className={`px-4 py-3 rounded-xl font-semibold transition-all ${
                  autoOptimize
                    ? 'bg-green-500 text-white'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                Auto
              </button>
            </div>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2 space-y-6">
            {error && (
              <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 flex items-center gap-3">
                <AlertTriangle className="w-5 h-5 text-red-400" />
                <span className="text-red-300">{error}</span>
              </div>
            )}

            {result && (
              <>
                {/* Main Result Card */}
                <div className="bg-gradient-to-br from-amber-500/20 to-orange-500/20 rounded-2xl p-6 border border-amber-500/30">
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <h3 className="text-xl font-bold text-white mb-1">Optimal Tilt Angle</h3>
                      <p className="text-slate-400 text-sm">ML Model Recommendation</p>
                    </div>
                    <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/50 rounded-full">
                      <Sparkles className="w-4 h-4 text-amber-400" />
                      <span className="text-xs text-slate-300">
                        {result.source === 'ml-model' ? 'ML Model' : 'Simulation'}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-6">
                    <div className="text-center">
                      <div className="text-6xl font-bold text-amber-400 mb-2">
                        {result.mlOptimization.optimalTilt}°
                      </div>
                      <p className="text-slate-400">Optimal Tilt</p>
                    </div>
                    <div className="text-center">
                      <div className="text-4xl font-bold text-green-400 mb-2">
                        {result.mlOptimization.estimatedEnergy}
                      </div>
                      <p className="text-slate-400">Est. Energy (W/m²)</p>
                    </div>
                    <div className="text-center">
                      <div className="text-4xl font-bold text-blue-400 mb-2">
                        +{result.mlOptimization.improvementPercent}%
                      </div>
                      <p className="text-slate-400">Improvement</p>
                    </div>
                  </div>

                  {/* Comparison */}
                  <div className="mt-6 flex items-center justify-center gap-4 p-4 bg-slate-800/50 rounded-xl">
                    <div className="text-center">
                      <p className="text-2xl font-semibold text-slate-400">{config.currentTilt}°</p>
                      <p className="text-xs text-slate-500">Current</p>
                    </div>
                    <ArrowRight className="w-6 h-6 text-amber-400" />
                    <div className="text-center">
                      <p className="text-2xl font-semibold text-amber-400">{result.mlOptimization.optimalTilt}°</p>
                      <p className="text-xs text-slate-500">Optimal</p>
                    </div>
                    {result.recommendations.adjustmentNeeded ? (
                      <div className="flex items-center gap-2 ml-4 px-3 py-1.5 bg-amber-500/20 rounded-full">
                        <AlertTriangle className="w-4 h-4 text-amber-400" />
                        <span className="text-xs text-amber-300">Adjustment Needed</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-2 ml-4 px-3 py-1.5 bg-green-500/20 rounded-full">
                        <CheckCircle2 className="w-4 h-4 text-green-400" />
                        <span className="text-xs text-green-300">Already Optimal</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                    <div className="flex items-center gap-2 mb-2">
                      <Sun className="w-5 h-5 text-yellow-400" />
                      <span className="text-sm text-slate-400">Solar Elevation</span>
                    </div>
                    <p className="text-2xl font-bold text-white">
                      {result.mlOptimization.solarElevation}°
                    </p>
                  </div>
                  <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                    <div className="flex items-center gap-2 mb-2">
                      <Activity className="w-5 h-5 text-red-400" />
                      <span className="text-sm text-slate-400">Efficiency Loss</span>
                    </div>
                    <p className="text-2xl font-bold text-white">
                      {result.mlOptimization.efficiencyLoss}%
                    </p>
                  </div>
                  <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                    <div className="flex items-center gap-2 mb-2">
                      <Zap className="w-5 h-5 text-green-400" />
                      <span className="text-sm text-slate-400">Efficiency Retained</span>
                    </div>
                    <p className="text-2xl font-bold text-white">
                      {result.mlOptimization.efficiencyRetained}%
                    </p>
                  </div>
                  <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
                    <div className="flex items-center gap-2 mb-2">
                      <Compass className="w-5 h-5 text-blue-400" />
                      <span className="text-sm text-slate-400">Optimal Azimuth</span>
                    </div>
                    <p className="text-2xl font-bold text-white">
                      {result.recommendations.optimalAzimuth}°
                    </p>
                  </div>
                </div>

                {/* Tilt Curve */}
                {result.mlOptimization.tiltCurve && result.mlOptimization.tiltCurve.length > 0 && (
                  <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                    <div className="flex items-center gap-2 mb-4">
                      <BarChart3 className="w-5 h-5 text-purple-400" />
                      <h3 className="text-lg font-semibold text-white">Energy vs Tilt Angle</h3>
                    </div>
                    <div className="flex items-end gap-2 h-48">
                      {result.mlOptimization.tiltCurve.map((point, idx) => {
                        const maxEnergy = Math.max(...result.mlOptimization.tiltCurve!.map(p => p.net_energy));
                        const height = (point.net_energy / maxEnergy) * 100;
                        const isOptimal = point.tilt === result.mlOptimization.optimalTilt;
                        
                        return (
                          <div key={idx} className="flex-1 flex flex-col items-center gap-1">
                            <div
                              className={`w-full rounded-t-lg transition-all ${
                                isOptimal
                                  ? 'bg-gradient-to-t from-amber-500 to-amber-400'
                                  : 'bg-gradient-to-t from-slate-600 to-slate-500'
                              }`}
                              style={{ height: `${height}%` }}
                            />
                            <span className={`text-xs ${isOptimal ? 'text-amber-400 font-bold' : 'text-slate-500'}`}>
                              {point.tilt}°
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Seasonal Schedule */}
                <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700">
                  <div className="flex items-center gap-2 mb-4">
                    <TrendingUp className="w-5 h-5 text-green-400" />
                    <h3 className="text-lg font-semibold text-white">Seasonal Recommendations</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    {result.seasonalSchedule.map((season, idx) => (
                      <div key={idx} className="bg-slate-700/50 rounded-xl p-4">
                        <p className="text-sm text-slate-400 mb-1">{season.months}</p>
                        <p className="text-2xl font-bold text-white">{season.recommendedTilt}°</p>
                      </div>
                    ))}
                  </div>
                  <div className="mt-4 p-4 bg-green-500/10 rounded-xl border border-green-500/30">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="w-5 h-5 text-green-400" />
                      <span className="text-green-300">
                        Potential gain with ML optimization: <strong>{result.estimatedGain.withMLOptimization}</strong>
                      </span>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TiltOptimizer;
