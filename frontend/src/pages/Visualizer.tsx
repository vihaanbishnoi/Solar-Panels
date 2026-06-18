import React, { useState, useEffect } from 'react';
import { 
  Box, Settings, Maximize2, RotateCcw, Sun, Cloud, Thermometer, 
  Droplets, Wind, Gauge, ZoomIn, ZoomOut, Move, Eye, EyeOff,
  BrainCircuit, Activity, AlertCircle, CheckCircle2, Wifi, WifiOff,
  Info, HelpCircle, Layers, Compass, Target, Sparkles
} from 'lucide-react';
import RoofDesigner from '../components/RoofDesigner/RoofDesigner';
import { getMLStatus, MLModelStatus } from '../services/api';

interface EnvironmentConditions {
  temperature: number;
  humidity: number;
  windSpeed: number;
  cloudCover: number;
  irradiance: number;
  timeOfDay: string;
}

export const Visualizer = () => {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [mlStatus, setMlStatus] = useState<MLModelStatus | null>(null);
  const [mlConnected, setMlConnected] = useState(false);
  const [showEnvironment, setShowEnvironment] = useState(true);
  
  // Simulated environment conditions
  const [environment, setEnvironment] = useState<EnvironmentConditions>({
    temperature: 28,
    humidity: 45,
    windSpeed: 3.2,
    cloudCover: 15,
    irradiance: 876,
    timeOfDay: 'Afternoon (14:30)',
  });

  // Check ML API status
  useEffect(() => {
    const checkMLStatus = async () => {
      try {
        const result = await getMLStatus();
        if (result.success && result.data) {
          setMlStatus(result.data);
          setMlConnected(result.data.overallStatus === 'connected');
        }
      } catch (err) {
        setMlConnected(false);
      }
    };
    checkMLStatus();
    
    // Refresh every 30 seconds
    const interval = setInterval(checkMLStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Update environment conditions periodically (simulating real data)
  useEffect(() => {
    const interval = setInterval(() => {
      setEnvironment(prev => ({
        ...prev,
        temperature: prev.temperature + (Math.random() - 0.5) * 0.5,
        humidity: Math.max(20, Math.min(80, prev.humidity + (Math.random() - 0.5) * 2)),
        windSpeed: Math.max(0, prev.windSpeed + (Math.random() - 0.5) * 0.3),
        irradiance: Math.max(0, Math.min(1200, prev.irradiance + (Math.random() - 0.5) * 20)),
      }));
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  return (
    <div className="w-full h-full flex flex-col relative bg-slate-100">
      {/* Top Header Bar */}
      <div className="absolute top-0 left-0 right-0 z-20 bg-gradient-to-b from-slate-900/90 via-slate-900/70 to-transparent p-4">
        <div className="flex items-center justify-between">
          {/* Left: Title and ML Status */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl shadow-lg">
                <Box className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">3D Solar Panel Designer</h1>
                <p className="text-xs text-slate-400">Interactive visualization with ML predictions</p>
              </div>
            </div>
            
            {/* ML Status Badge */}
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
              mlConnected 
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
                : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
            }`}>
              {mlConnected ? <Wifi className="w-3.5 h-3.5" /> : <WifiOff className="w-3.5 h-3.5" />}
              {mlConnected ? 'ML Models Active' : 'Simulation Mode'}
            </div>
          </div>

          {/* Right: Controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowEnvironment(!showEnvironment)}
              className={`p-2 rounded-lg transition-all ${
                showEnvironment 
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' 
                  : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800'
              }`}
              title="Toggle Environment Panel"
            >
              <Layers className="w-5 h-5" />
            </button>
            <button
              onClick={() => setShowHelp(!showHelp)}
              className={`p-2 rounded-lg transition-all ${
                showHelp 
                  ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30' 
                  : 'bg-slate-800/50 text-slate-400 hover:bg-slate-800'
              }`}
              title="Help & Controls"
            >
              <HelpCircle className="w-5 h-5" />
            </button>
            <button
              onClick={toggleFullscreen}
              className="p-2 bg-slate-800/50 hover:bg-slate-800 rounded-lg text-slate-400 transition-all"
              title="Toggle Fullscreen"
            >
              <Maximize2 className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>

      {/* Environment Conditions Panel (Right Side) */}
      {showEnvironment && (
        <div className="absolute top-20 right-4 z-10 w-64 bg-slate-900/95 backdrop-blur-md rounded-2xl border border-slate-700/50 shadow-2xl overflow-hidden">
          <div className="p-4 border-b border-slate-700/50">
            <h3 className="font-semibold text-white flex items-center gap-2">
              <Cloud className="w-4 h-4 text-blue-400" />
              Environment Conditions
            </h3>
            <p className="text-xs text-slate-500 mt-1">Real-time sensor data</p>
          </div>
          
          <div className="p-4 space-y-4">
            {/* Temperature */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-slate-400">
                <Thermometer className="w-4 h-4 text-orange-400" />
                <span className="text-sm">Temperature</span>
              </div>
              <span className="text-white font-semibold">{environment.temperature.toFixed(1)}°C</span>
            </div>
            
            {/* Humidity */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-slate-400">
                <Droplets className="w-4 h-4 text-cyan-400" />
                <span className="text-sm">Humidity</span>
              </div>
              <span className="text-white font-semibold">{environment.humidity.toFixed(0)}%</span>
            </div>
            
            {/* Wind Speed */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-slate-400">
                <Wind className="w-4 h-4 text-teal-400" />
                <span className="text-sm">Wind Speed</span>
              </div>
              <span className="text-white font-semibold">{environment.windSpeed.toFixed(1)} m/s</span>
            </div>
            
            {/* Irradiance */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-slate-400">
                <Sun className="w-4 h-4 text-amber-400" />
                <span className="text-sm">Irradiance</span>
              </div>
              <span className="text-white font-semibold">{environment.irradiance.toFixed(0)} W/m²</span>
            </div>
            
            {/* Cloud Cover */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2 text-slate-400">
                  <Cloud className="w-4 h-4 text-slate-400" />
                  <span className="text-sm">Cloud Cover</span>
                </div>
                <span className="text-white font-semibold">{environment.cloudCover}%</span>
              </div>
              <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-amber-400 to-slate-400 rounded-full transition-all"
                  style={{ width: `${environment.cloudCover}%` }}
                />
              </div>
            </div>
            
            {/* Time of Day */}
            <div className="pt-2 border-t border-slate-700/50">
              <div className="flex items-center gap-2 text-slate-400">
                <Activity className="w-4 h-4 text-purple-400" />
                <span className="text-xs">{environment.timeOfDay}</span>
              </div>
            </div>
          </div>

          {/* ML Model Info */}
          <div className="p-4 bg-slate-800/50 border-t border-slate-700/50">
            <div className="flex items-center gap-2 mb-2">
              <BrainCircuit className="w-4 h-4 text-purple-400" />
              <span className="text-xs font-semibold text-slate-300">Active ML Models</span>
            </div>
            <div className="space-y-1.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-500">Efficiency Predictor</span>
                <span className="text-emerald-400">● Active</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500">Tilt Optimizer</span>
                <span className="text-emerald-400">● Active</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500">Degradation Model</span>
                <span className="text-emerald-400">● Active</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Help Panel */}
      {showHelp && (
        <div className="absolute top-20 left-4 z-10 w-72 bg-slate-900/95 backdrop-blur-md rounded-2xl border border-slate-700/50 shadow-2xl overflow-hidden">
          <div className="p-4 border-b border-slate-700/50 flex items-center justify-between">
            <h3 className="font-semibold text-white flex items-center gap-2">
              <HelpCircle className="w-4 h-4 text-purple-400" />
              Controls Guide
            </h3>
            <button 
              onClick={() => setShowHelp(false)}
              className="text-slate-500 hover:text-white"
            >
              ×
            </button>
          </div>
          
          <div className="p-4 space-y-4 text-sm">
            <div>
              <h4 className="font-medium text-slate-300 mb-2 flex items-center gap-2">
                <Move className="w-4 h-4 text-blue-400" />
                Camera Controls
              </h4>
              <ul className="space-y-1 text-slate-400 text-xs">
                <li>• <span className="text-slate-300">Left Click + Drag</span> - Rotate view</li>
                <li>• <span className="text-slate-300">Right Click + Drag</span> - Pan camera</li>
                <li>• <span className="text-slate-300">Scroll Wheel</span> - Zoom in/out</li>
                <li>• <span className="text-slate-300">Double Click</span> - Reset view</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium text-slate-300 mb-2 flex items-center gap-2">
                <Settings className="w-4 h-4 text-emerald-400" />
                Panel Controls
              </h4>
              <ul className="space-y-1 text-slate-400 text-xs">
                <li>• <span className="text-slate-300">Tilt Slider</span> - Adjust panel angle (0-90°)</li>
                <li>• <span className="text-slate-300">Azimuth</span> - Panel orientation (0-360°)</li>
                <li>• <span className="text-slate-300">Light Controls</span> - Simulate sun position</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium text-slate-300 mb-2 flex items-center gap-2">
                <BrainCircuit className="w-4 h-4 text-purple-400" />
                ML Features
              </h4>
              <ul className="space-y-1 text-slate-400 text-xs">
                <li>• <span className="text-slate-300">AI Performance Analysis</span> - Real-time efficiency</li>
                <li>• <span className="text-slate-300">Tilt Optimization</span> - ML-recommended angle</li>
                <li>• <span className="text-slate-300">Apply Optimal</span> - One-click optimization</li>
              </ul>
            </div>
            
            <div className="pt-2 border-t border-slate-700/50">
              <div className="flex items-center gap-2 text-purple-400 text-xs">
                <Sparkles className="w-3.5 h-3.5" />
                <span>ML predictions update in real-time as you adjust settings</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Quick Stats Bar (Bottom) */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-10">
        <div className="flex items-center gap-3 px-4 py-2.5 bg-slate-900/90 backdrop-blur-md rounded-full border border-slate-700/50 shadow-xl">
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 text-blue-400">
            <Gauge className="w-4 h-4" />
            <span className="text-sm font-medium">Efficiency: 94.2%</span>
          </div>
          <div className="w-px h-6 bg-slate-700" />
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 text-amber-400">
            <Target className="w-4 h-4" />
            <span className="text-sm font-medium">Tilt: 32°</span>
          </div>
          <div className="w-px h-6 bg-slate-700" />
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400">
            <Sun className="w-4 h-4" />
            <span className="text-sm font-medium">{environment.irradiance.toFixed(0)} W/m²</span>
          </div>
          <div className="w-px h-6 bg-slate-700" />
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 text-purple-400">
            <Compass className="w-4 h-4" />
            <span className="text-sm font-medium">180° S</span>
          </div>
        </div>
      </div>

      {/* 3D Canvas */}
      <RoofDesigner />
    </div>
  );
};
