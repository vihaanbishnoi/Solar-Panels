import React from 'react';
import { ShieldCheck, AlertTriangle, Calendar, BrainCircuit, TrendingDown, Info, CheckCircle2, Loader2, Wifi, WifiOff, Target, Zap, ArrowUp, Sparkles } from 'lucide-react';
import { ViewStats } from './types/roof.types';

interface AdvancedStatsProps {
  stats: ViewStats;
  onApplyOptimalTilt?: () => void;
}

export const AdvancedStats: React.FC<AdvancedStatsProps> = ({ stats, onApplyOptimalTilt }) => {
  const efficiencyLoss = (100 - stats.efficiency).toFixed(1);
  const nextCleaning = new Date();
  nextCleaning.setDate(nextCleaning.getDate() + 14);

  // Determine status badge based on ML connection
  const isMLConnected = stats.mlSource === 'ml-model';
  const isLoading = stats.mlLoading;

  // Tilt optimization data
  const tiltOpt = stats.tiltOptimization;
  const optimalTilt = tiltOpt?.mlOptimization?.optimalTilt;
  const currentTilt = stats.panelTilt;
  const tiltDifference = optimalTilt !== undefined ? Math.abs(optimalTilt - currentTilt) : 0;
  const isOptimalTilt = tiltDifference < 2; // Within 2 degrees is considered optimal
  const improvementPercent = tiltOpt?.mlOptimization?.improvementPercent || 0;

  return (
    <div className="absolute bottom-4 left-4 flex flex-col gap-4 w-96 max-h-[60vh] overflow-y-auto z-10 custom-scrollbar pr-1 pb-1">
      {/* Real-time Performance Card */}
      <div className="bg-white/90 backdrop-blur-md rounded-2xl shadow-xl border border-white/50 overflow-hidden">
        <div className="p-4 border-b border-gray-100 flex items-center justify-between">
          <h3 className="font-bold text-gray-800 flex items-center gap-2">
            <BrainCircuit className="w-5 h-5 text-blue-600" />
            AI Performance Analysis
          </h3>
          {isLoading ? (
            <span className="text-xs bg-blue-100 text-blue-700 font-bold px-2 py-1 rounded-full flex items-center gap-1">
              <Loader2 className="w-3 h-3 animate-spin" />
              Loading
            </span>
          ) : isMLConnected ? (
            <span className="text-xs bg-purple-100 text-purple-700 font-bold px-2 py-1 rounded-full flex items-center gap-1">
              <Wifi className="w-3 h-3" />
              ML Model
            </span>
          ) : (
            <span className="text-xs bg-amber-100 text-amber-700 font-bold px-2 py-1 rounded-full flex items-center gap-1">
              <WifiOff className="w-3 h-3" />
              Geometric
            </span>
          )}
        </div>
        
        <div className="p-4 space-y-4">
          {/* Main Efficiency Meter */}
          <div>
            <div className="flex justify-between items-end mb-2">
              <div className="flex flex-col">
                <span className="text-sm font-medium text-gray-600">Current Efficiency</span>
                {isMLConnected && stats.geometricEfficiency !== undefined && (
                  <span className="text-xs text-gray-400">
                    Geometric: {stats.geometricEfficiency.toFixed(1)}%
                  </span>
                )}
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-2xl font-bold text-blue-600">{stats.efficiency.toFixed(1)}%</span>
                {isMLConnected && (
                  <span className="text-xs text-purple-500 font-medium">ML</span>
                )}
              </div>
            </div>
            <div className="relative h-2 bg-gray-200 rounded-full overflow-hidden">
              <div 
                className={`absolute top-0 left-0 h-full rounded-full transition-all duration-1000 ${
                  stats.efficiency > 80 ? 'bg-gradient-to-r from-blue-500 to-green-400' : 
                  stats.efficiency > 50 ? 'bg-gradient-to-r from-yellow-500 to-orange-400' : 'bg-red-500'
                }`}
                style={{ width: `${stats.efficiency}%` }}
              />
            </div>
          </div>

          {/* Loss Breakdown */}
          <div className="bg-slate-50 rounded-xl p-3 border border-slate-100">
            <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-3">
              Loss Breakdown 
              {isMLConnected && <span className="text-purple-500 ml-1">(ML)</span>}
            </h4>
            <div className="space-y-2">
              <div className="flex justify-between items-center text-sm">
                <span className="flex items-center gap-2 text-slate-600">
                  <span className="w-2 h-2 rounded-full bg-yellow-400" />
                  {isMLConnected ? 'Tilt Impact' : 'Incidence Angle'}
                </span>
                <span className="font-medium text-slate-800">
                  {stats.mlPrediction?.factors?.tiltImpact || `${(100 - stats.efficiency).toFixed(1)}%`}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="flex items-center gap-2 text-slate-600">
                  <span className="w-2 h-2 rounded-full bg-orange-400" />
                  {isMLConnected ? 'Temperature Impact' : 'Thermal Drift'}
                </span>
                <span className="font-medium text-slate-800">
                  {stats.mlPrediction?.factors?.temperatureImpact || '2.1%'}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="flex items-center gap-2 text-slate-600">
                  <span className="w-2 h-2 rounded-full bg-blue-400" />
                  {isMLConnected ? 'Cloud Cover' : 'Weather'}
                </span>
                <span className="font-medium text-slate-800">
                  {stats.mlPrediction?.factors?.cloudImpact || '0.0%'}
                </span>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="flex items-center gap-2 text-slate-600">
                  <span className="w-2 h-2 rounded-full bg-gray-400" />
                  Soiling (Dust)
                </span>
                <span className="font-medium text-slate-800">
                  {stats.mlPrediction?.factors?.soilingImpact || '1.4%'}
                </span>
              </div>
            </div>
          </div>

        </div>
      </div>

      {/* ML Tilt Optimization Card */}
      <div className="bg-white/90 backdrop-blur-md rounded-2xl shadow-xl border border-white/50 overflow-hidden">
        <div className="p-4 border-b border-gray-100 flex items-center justify-between">
          <h3 className="font-bold text-gray-800 flex items-center gap-2">
            <Target className="w-5 h-5 text-emerald-600" />
            Tilt Optimization
          </h3>
          {stats.tiltOptLoading ? (
            <span className="text-xs bg-blue-100 text-blue-700 font-bold px-2 py-1 rounded-full flex items-center gap-1">
              <Loader2 className="w-3 h-3 animate-spin" />
              Calculating
            </span>
          ) : optimalTilt !== undefined ? (
            <span className={`text-xs font-bold px-2 py-1 rounded-full flex items-center gap-1 ${
              isOptimalTilt ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'
            }`}>
              {isOptimalTilt ? <CheckCircle2 className="w-3 h-3" /> : <ArrowUp className="w-3 h-3" />}
              {isOptimalTilt ? 'Optimal' : 'Adjust Needed'}
            </span>
          ) : (
            <span className="text-xs bg-gray-100 text-gray-600 font-bold px-2 py-1 rounded-full">
              N/A
            </span>
          )}
        </div>

        <div className="p-4 space-y-4">
          {/* Current vs Optimal Tilt */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-slate-50 rounded-xl p-3 text-center border border-slate-100">
              <p className="text-xs text-slate-500 uppercase tracking-wide font-medium mb-1">Current Tilt</p>
              <p className="text-2xl font-bold text-slate-700">{currentTilt}°</p>
            </div>
            <div className={`rounded-xl p-3 text-center border ${
              optimalTilt !== undefined 
                ? isOptimalTilt 
                  ? 'bg-emerald-50 border-emerald-200' 
                  : 'bg-amber-50 border-amber-200'
                : 'bg-slate-50 border-slate-100'
            }`}>
              <p className="text-xs text-slate-500 uppercase tracking-wide font-medium mb-1">
                <Sparkles className="w-3 h-3 inline mr-1" />
                ML Optimal
              </p>
              <p className={`text-2xl font-bold ${
                optimalTilt !== undefined 
                  ? isOptimalTilt ? 'text-emerald-600' : 'text-amber-600'
                  : 'text-slate-400'
              }`}>
                {optimalTilt !== undefined ? `${Math.round(optimalTilt)}°` : '--'}
              </p>
            </div>
          </div>

          {/* Improvement Potential */}
          {optimalTilt !== undefined && !isOptimalTilt && improvementPercent > 0 && (
            <div className="bg-gradient-to-r from-emerald-50 to-green-50 rounded-xl p-3 border border-emerald-100">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-emerald-600" />
                  <span className="text-sm font-medium text-emerald-800">Potential Improvement</span>
                </div>
                <span className="text-lg font-bold text-emerald-600">+{improvementPercent.toFixed(1)}%</span>
              </div>
              <p className="text-xs text-emerald-600/80 mt-1">
                Adjusting tilt by {tiltDifference.toFixed(0)}° can boost energy output
              </p>
            </div>
          )}

          {/* Apply Optimal Button */}
          {optimalTilt !== undefined && !isOptimalTilt && onApplyOptimalTilt && (
            <button
              onClick={onApplyOptimalTilt}
              className="w-full bg-gradient-to-r from-emerald-500 to-green-500 text-white font-semibold py-2.5 px-4 rounded-xl shadow-md hover:shadow-lg hover:from-emerald-600 hover:to-green-600 transition-all flex items-center justify-center gap-2"
            >
              <Target className="w-4 h-4" />
              Apply Optimal Tilt ({Math.round(optimalTilt)}°)
            </button>
          )}

          {/* Efficiency Retained */}
          {tiltOpt?.mlOptimization?.efficiencyRetained !== undefined && (
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-600">Efficiency at Current Tilt</span>
              <span className="font-bold text-slate-800">{tiltOpt.mlOptimization.efficiencyRetained.toFixed(1)}%</span>
            </div>
          )}

          {/* Solar Elevation Info */}
          {tiltOpt?.mlOptimization?.solarElevation !== undefined && (
            <div className="flex items-center justify-between text-sm">
              <span className="text-slate-600 flex items-center gap-1">
                <Info className="w-3 h-3" />
                Solar Elevation
              </span>
              <span className="font-medium text-slate-700">{tiltOpt.mlOptimization.solarElevation.toFixed(1)}°</span>
            </div>
          )}
        </div>
      </div>

      {/* Degradation & Alerts */}
      <div className="bg-white/90 backdrop-blur-md rounded-2xl shadow-xl border border-white/50 overflow-hidden">
        <div className="p-4 space-y-4">
          {/* Cleaning Schedule */}
          <div className="flex items-start gap-3">
            <div className="p-2 bg-blue-100 rounded-lg text-blue-600">
              <Calendar className="w-5 h-5" />
            </div>
            <div>
              <h4 className="font-semibold text-gray-800 text-sm">Recommended Cleaning</h4>
              <p className="text-xs text-gray-500 mt-1">
                Next scheduled: <span className="font-medium text-gray-700">{nextCleaning.toLocaleDateString()}</span>
              </p>
              <div className="mt-2 text-xs text-blue-600 font-medium bg-blue-50 px-2 py-1 rounded inline-block">
                +3.2% Efficiency gain est.
              </div>
            </div>
          </div>

          <div className="h-px bg-gray-100" />

          {/* Degradation Alert */}
          <div className="flex items-start gap-3">
             <div className="p-2 bg-amber-100 rounded-lg text-amber-600">
               <TrendingDown className="w-5 h-5" />
             </div>
             <div>
               <h4 className="font-semibold text-gray-800 text-sm">Degradation Alert</h4>
               <p className="text-xs text-gray-500 mt-1">
                 Panels degrading at <span className="font-medium text-amber-600">0.05% / month</span> above threshold.
               </p>
               <button className="text-xs text-blue-600 underline mt-1">View detailed report</button>
             </div>
          </div>
          
           <div className="h-px bg-gray-100" />

           {/* Health Check */}
           <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
                <ShieldCheck className="w-4 h-4 text-green-500" />
                Panel Status
              </div>
              <span className={`text-xs font-bold px-2 py-1 rounded-full ${
                stats.mlPrediction?.panelStatus === 'optimal' || stats.efficiency > 80
                  ? 'text-green-600 bg-green-100'
                  : stats.mlPrediction?.panelStatus === 'good' || stats.efficiency > 60
                  ? 'text-blue-600 bg-blue-100'
                  : stats.mlPrediction?.panelStatus === 'fair' || stats.efficiency > 40
                  ? 'text-amber-600 bg-amber-100'
                  : 'text-red-600 bg-red-100'
              }`}>
                {stats.mlPrediction?.panelStatus?.toUpperCase() || (stats.efficiency > 80 ? 'OPTIMAL' : stats.efficiency > 60 ? 'GOOD' : stats.efficiency > 40 ? 'FAIR' : 'POOR')}
              </span>
           </div>

           {/* ML Recommendation */}
           {stats.mlPrediction?.mlRecommendation && (
             <>
               <div className="h-px bg-gray-100" />
               <div className="bg-purple-50 rounded-lg p-3 border border-purple-100">
                 <h4 className="text-xs font-bold text-purple-600 uppercase tracking-widest mb-1">AI Recommendation</h4>
                 <p className="text-sm text-purple-800">{stats.mlPrediction.mlRecommendation}</p>
               </div>
             </>
           )}
        </div>
      </div>
    </div>
  );
};
