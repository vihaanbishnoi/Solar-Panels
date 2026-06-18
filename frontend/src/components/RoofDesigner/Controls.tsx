import React, { useState } from 'react';
import { Camera, Grid, Ruler, Sun, RotateCw, Settings, ChevronDown, ChevronUp, Layers, Info } from 'lucide-react';
import { SolarPanelConfig, ViewStats } from './types/roof.types';

interface ControlsProps {
  config: SolarPanelConfig;
  onConfigChange: (config: Partial<SolarPanelConfig>) => void;
  stats: ViewStats;
  onResetCamera: () => void;
}

const ControlSection = ({ title, isOpen, onToggle, children }: any) => (
  <div className="border-b border-gray-200 last:border-0">
    <button 
      className="w-full flex items-center justify-between p-3 text-left hover:bg-gray-50 transition-colors"
      onClick={onToggle}
    >
      <span className="text-sm font-semibold text-gray-700">{title}</span>
      {isOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
    </button>
    {isOpen && <div className="p-3 pt-0 space-y-3">{children}</div>}
  </div>
);

export const Controls: React.FC<ControlsProps> = ({
  config,
  onConfigChange,
  stats,
  onResetCamera,
}) => {
  const [openSections, setOpenSections] = useState({
    dimensions: true,
    orientation: true,
    light: true,
    display: false,
  });

  const toggleSection = (section: keyof typeof openSections) => {
    setOpenSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const colors: { [key: string]: string } = {
    warm: '#ffaa44',
    cool: '#ccddff',
    daylight: '#ffffff'
  };

  return (
    <>
      <div className="absolute top-4 right-4 flex flex-col bg-white/90 rounded-xl shadow-lg backdrop-blur-sm pointer-events-auto w-80 max-h-[90vh] overflow-y-auto z-10 transition-all custom-scrollbar">
        <div className="p-4 border-b border-gray-200 flex justify-between items-center bg-gray-50 rounded-t-xl sticky top-0 z-20">
          <h3 className="font-bold text-gray-800 flex items-center gap-2">
            <Settings size={18} />
            Configuration
          </h3>
          <button onClick={onResetCamera} title="Reset Camera" className="p-1 hover:bg-gray-200 rounded">
            <Camera size={18} />
          </button>
        </div>

        <ControlSection 
          title="Panel Dimensions" 
          isOpen={openSections.dimensions} 
          onToggle={() => toggleSection('dimensions')}
        >
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-gray-600">Width</label>
                <span className="text-xs text-gray-500">{config.width}m</span>
              </div>
              <input 
                type="range" min="0.5" max="3.0" step="0.1" 
                value={config.width}
                onChange={(e) => onConfigChange({ width: parseFloat(e.target.value) })}
                className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
              />
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-gray-600">Height</label>
                <span className="text-xs text-gray-500">{config.height}m</span>
              </div>
              <input 
                type="range" min="0.3" max="2.0" step="0.1" 
                value={config.height}
                onChange={(e) => onConfigChange({ height: parseFloat(e.target.value) })}
                className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
              />
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-gray-600">Thickness</label>
                <span className="text-xs text-gray-500">{config.thickness}m</span>
              </div>
              <input 
                type="range" min="0.02" max="0.1" step="0.01" 
                value={config.thickness}
                onChange={(e) => onConfigChange({ thickness: parseFloat(e.target.value) })}
                className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
              />
            </div>
          </div>
        </ControlSection>

        <ControlSection 
          title="Orientation" 
          isOpen={openSections.orientation} 
          onToggle={() => toggleSection('orientation')}
        >
          <div className="space-y-4">
             <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-gray-600">Lock Rotation</label>
                <input 
                  type="checkbox"
                  checked={config.isRotationLocked}
                  onChange={(e) => onConfigChange({ isRotationLocked: e.target.checked })}
                  className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
             </div>
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-gray-600 flex items-center gap-1"><RotateCw size={10} /> Tilt Angle</label>
                <span className="text-xs text-gray-500">{config.tilt}°</span>
              </div>
              <input 
                type="range" min="0" max="90" step="1" 
                value={config.tilt}
                disabled={config.isRotationLocked}
                onChange={(e) => onConfigChange({ tilt: parseInt(e.target.value) })}
                className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600 disabled:opacity-50"
              />
            </div>
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-gray-600">Azimuth Angle</label>
                <span className="text-xs text-gray-500">{config.azimuth}°</span>
              </div>
              <input 
                type="range" min="0" max="360" step="1" 
                value={config.azimuth}
                disabled={config.isRotationLocked}
                onChange={(e) => onConfigChange({ azimuth: parseInt(e.target.value) })}
                className="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600 disabled:opacity-50"
              />
            </div>
          </div>
        </ControlSection>

        <ControlSection 
          title="Light Source" 
          isOpen={openSections.light} 
          onToggle={() => toggleSection('light')}
        >
          <div className="space-y-4">
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-gray-600 flex items-center gap-1"><Sun size={10} /> Azimuth</label>
                <span className="text-xs text-gray-500">{config.lightAzimuth}°</span>
              </div>
              <input 
                type="range" min="0" max="360" step="1" 
                value={config.lightAzimuth}
                onChange={(e) => onConfigChange({ lightAzimuth: parseInt(e.target.value) })}
                className="w-full h-1 bg-orange-200 rounded-lg appearance-none cursor-pointer accent-orange-500"
              />
            </div>
            <div>
               <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-gray-600">Elevation</label>
                <span className="text-xs text-gray-500">{config.lightElevation}°</span>
              </div>
              <input 
                type="range" min="0" max="90" step="1" 
                value={config.lightElevation}
                onChange={(e) => onConfigChange({ lightElevation: parseInt(e.target.value) })}
                className="w-full h-1 bg-orange-200 rounded-lg appearance-none cursor-pointer accent-orange-500"
              />
            </div>
            <div>
               <div className="flex justify-between mb-1">
                <label className="text-xs font-medium text-gray-600">Intensity</label>
                <span className="text-xs text-gray-500">{config.lightIntensity}</span>
              </div>
              <input 
                type="range" min="0.2" max="2.0" step="0.1" 
                value={config.lightIntensity}
                onChange={(e) => onConfigChange({ lightIntensity: parseFloat(e.target.value) })}
                className="w-full h-1 bg-orange-200 rounded-lg appearance-none cursor-pointer accent-orange-500"
              />
            </div>
            <div>
              <label className="text-xs font-medium text-gray-600 block mb-2">Color Temperature</label>
              <div className="flex gap-2">
                {(['warm', 'cool', 'daylight'] as const).map(c => (
                  <button
                    key={c}
                    onClick={() => onConfigChange({ lightColor: c })}
                    className={`flex-1 py-1.5 text-xs rounded border transition-all ${config.lightColor === c ? 'border-orange-500 bg-orange-50 font-medium' : 'border-gray-200 hover:bg-gray-50'}`}
                    style={{ borderBottomWidth: config.lightColor === c ? 3 : 1, borderBottomColor: config.lightColor === c ? colors[c] : undefined}}
                  >
                    {c.charAt(0).toUpperCase() + c.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </ControlSection>

        <ControlSection 
          title="Display Options" 
          isOpen={openSections.display} 
          onToggle={() => toggleSection('display')}
        >
          <div className="space-y-2">
            <button
              onClick={() => onConfigChange({ showGrid: !config.showGrid })}
              className={`w-full flex items-center justify-between px-3 py-2 text-sm rounded transition-colors ${
                config.showGrid ? 'bg-blue-50 text-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              <span className="flex items-center gap-2"><Grid size={16} /> Show Grid</span>
               <div className={`w-8 h-4 rounded-full relative transition-colors ${config.showGrid ? 'bg-blue-500' : 'bg-gray-300'}`}>
                <div className={`absolute w-3 h-3 bg-white rounded-full top-0.5 transition-all shadow-sm`} style={{ left: config.showGrid ? '18px' : '2px'}} />
              </div>
            </button>
            
            <button
              onClick={() => onConfigChange({ showMeasurements: !config.showMeasurements })}
              className={`w-full flex items-center justify-between px-3 py-2 text-sm rounded transition-colors ${
                config.showMeasurements ? 'bg-blue-50 text-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              <span className="flex items-center gap-2"><Ruler size={16} /> Dimensions</span>
               <div className={`w-8 h-4 rounded-full relative transition-colors ${config.showMeasurements ? 'bg-blue-500' : 'bg-gray-300'}`}>
                <div className={`absolute w-3 h-3 bg-white rounded-full top-0.5 transition-all shadow-sm`} style={{ left: config.showMeasurements ? '18px' : '2px'}} />
              </div>
            </button>
            <button
              onClick={() => onConfigChange({ showLightDirection: !config.showLightDirection })}
              className={`w-full flex items-center justify-between px-3 py-2 text-sm rounded transition-colors ${
                config.showLightDirection ? 'bg-blue-50 text-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
               <span className="flex items-center gap-2"><Layers size={16} /> Light Ray</span>
                <div className={`w-8 h-4 rounded-full relative transition-colors ${config.showLightDirection ? 'bg-blue-500' : 'bg-gray-300'}`}>
                <div className={`absolute w-3 h-3 bg-white rounded-full top-0.5 transition-all shadow-sm`} style={{ left: config.showLightDirection ? '18px' : '2px'}} />
              </div>
            </button>
             <button
              onClick={() => onConfigChange({ showNormalVector: !config.showNormalVector })}
              className={`w-full flex items-center justify-between px-3 py-2 text-sm rounded transition-colors ${
                config.showNormalVector ? 'bg-blue-50 text-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
               <span className="flex items-center gap-2"><Layers size={16} /> Normal Vector</span>
                <div className={`w-8 h-4 rounded-full relative transition-colors ${config.showNormalVector ? 'bg-blue-500' : 'bg-gray-300'}`}>
                 <div className={`absolute w-3 h-3 bg-white rounded-full top-0.5 transition-all shadow-sm`} style={{ left: config.showNormalVector ? '18px' : '2px'}} />
              </div>
            </button>
          </div>
        </ControlSection>

         <div className="p-4 bg-gray-50 flex flex-col gap-2 rounded-b-xl border-t border-gray-200">
             <label className="text-xs font-medium text-gray-600">Panel Color</label>
             <input 
                type="color" 
                value={config.panelColor}
                onChange={(e) => onConfigChange({ panelColor: e.target.value })}
                className="w-full h-8 rounded cursor-pointer"
             />
         </div>
      </div>

      {/* Metrics Panel */}
      <div className="absolute bottom-4 left-4 bg-white/90 p-4 rounded-xl shadow-lg backdrop-blur-sm min-w-[280px] pointer-events-auto z-10">
        <div className="flex items-center gap-2 mb-3 text-gray-500 border-b border-gray-100 pb-2">
           <Info size={16} className="text-blue-500" />
           <h3 className="text-sm font-semibold uppercase tracking-wider">Physics Metrics</h3>
        </div>
        
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600">Efficiency</span>
            <div className="flex items-center gap-2">
              <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className={`h-full transition-all duration-300 ${stats.efficiency > 80 ? 'bg-green-500' : stats.efficiency > 40 ? 'bg-yellow-500' : 'bg-red-500'}`}
                  style={{ width: `${stats.efficiency}%` }}
                />
              </div>
              <span className={`text-sm font-mono font-bold transition-colors duration-300 ${stats.efficiency > 80 ? 'text-green-600' : stats.efficiency > 40 ? 'text-yellow-600' : 'text-red-600'}`}>
                {stats.efficiency.toFixed(1)}%
              </span>
            </div>
          </div>
          <div className="flex justify-between items-center border-b border-gray-100 pb-2">
             <span className="text-sm text-gray-600">Incident Angle</span>
             <span className="text-sm font-mono font-bold text-gray-900">{stats.incidentAngle.toFixed(1)}°</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600">Surface Area</span>
            <span className="text-sm font-mono font-bold text-gray-900">{stats.area.toFixed(2)} m²</span>
          </div>
           <div className="flex justify-between items-center">
             <span className="text-xs text-gray-400">View Angle</span>
             <span className="text-xs font-mono text-gray-400">{stats.viewAngle}</span>
          </div>
        </div>
      </div>
    </>
  );
};
