import React, { Suspense, useState, useRef, useMemo, useEffect, useCallback } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

import { Ground } from './Ground';
import { Lights } from './Lights';
import { Controls } from './Controls';
import { AdvancedStats } from './AdvancedStats';
import { SolarPanel } from './SolarPanel';
import { SolarPanelConfig } from './types/roof.types';
import { predictEfficiency, EfficiencyPredictionResult, optimizeTiltAngle, TiltOptimizationResult } from '../../services/api';

const SceneController = ({ onStatsUpdate }: { onStatsUpdate: (angle: string) => void }) => {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);

  useFrame(() => {
    if (controlsRef.current) {
      const angle = controlsRef.current.getAzimuthalAngle();
      const degrees = Math.round(((angle * 180) / Math.PI + 360) % 360);
      onStatsUpdate(`${degrees}°`);
    }
  });

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.05}
      minPolarAngle={Math.PI * (5 / 180)} 
      maxPolarAngle={Math.PI * (85 / 180)}
      maxDistance={20}
      minDistance={2}
    />
  );
};

export const RoofDesigner: React.FC = () => {
  const [viewAngle, setViewAngle] = useState('0°');
  const cameraRef = useRef<THREE.PerspectiveCamera>(null);

  // ML Prediction State
  const [mlPrediction, setMlPrediction] = useState<EfficiencyPredictionResult | null>(null);
  const [mlLoading, setMlLoading] = useState(false);
  const [mlError, setMlError] = useState<string | null>(null);

  // Tilt Optimization State
  const [tiltOptimization, setTiltOptimization] = useState<TiltOptimizationResult | null>(null);
  const [tiltOptLoading, setTiltOptLoading] = useState(false);

  // Initial Configuration State
  const [config, setConfig] = useState<SolarPanelConfig>({
    width: 1.7,
    height: 1.0,
    thickness: 0.04,
    tilt: 30,
    azimuth: 180, // Facing South
    isRotationLocked: false,
    lightAzimuth: 180,
    lightElevation: 45,
    lightIntensity: 1.0,
    lightColor: 'daylight',
    showGrid: true,
    gridSize: 0.2,
    panelColor: '#1f2937',
    showMeasurements: true,
    showLightDirection: true,
    showNormalVector: true,
  });

  const handleConfigChange = (newConfig: Partial<SolarPanelConfig>) => {
    setConfig(prev => ({ ...prev, ...newConfig }));
  };

  const handleResetCamera = () => {
    if (cameraRef.current) {
       cameraRef.current.position.set(5, 4, 5);
       cameraRef.current.lookAt(0, 0, 0);
    }
  };

  // Fetch ML prediction when config changes
  const fetchMLPrediction = useCallback(async () => {
    setMlLoading(true);
    setMlError(null);
    
    try {
      // Map config panel values to ML model inputs
      // Using light elevation as a proxy for irradiance (higher sun = more irradiance)
      const irradiance = 200 + (config.lightElevation / 90) * 800; // 200-1000 W/m²
      // Temperature estimate based on light intensity
      const temperature = 15 + config.lightIntensity * 20; // 15-35°C range
      
      const result = await predictEfficiency({
        temperature: temperature,
        humidity: 50, // Default moderate humidity
        windSpeed: 3, // Default moderate wind
        irradiance: irradiance,
        voltage: 35, // Typical panel voltage
        current: 8, // Typical panel current
        daysSinceInstallation: 365, // 1 year old
        tilt: config.tilt,
        cloudCover: config.lightColor === 'cool' ? 60 : config.lightColor === 'warm' ? 10 : 30,
        soilingLevel: 5, // 5% default soiling
      });
      
      if (result.success && result.data) {
        setMlPrediction(result.data);
      } else {
        setMlError(result.error || 'Failed to get prediction');
      }
    } catch (err) {
      setMlError(err instanceof Error ? err.message : 'ML prediction failed');
    } finally {
      setMlLoading(false);
    }
  }, [config]); // React to ALL config changes

  // Fetch Tilt Optimization when config changes
  const fetchTiltOptimization = useCallback(async () => {
    setTiltOptLoading(true);
    
    try {
      // Calculate irradiance based on light elevation (GHI = Global Horizontal Irradiance)
      const ghi = 200 + (config.lightElevation / 90) * 800; // 200-1000 W/m²
      // Estimate hour from light azimuth (180° = noon = 12:00)
      const hour = Math.round(6 + (config.lightAzimuth / 360) * 12); // 6-18 range
      // Temperature based on light intensity
      const temperature = 15 + config.lightIntensity * 20;
      
      const result = await optimizeTiltAngle({
        latitude: 51.5074, // London as default (can be made configurable)
        ghi: ghi,
        hour: hour,
        temperature: temperature,
        humidity: 50,
        windSpeed: 3,
        currentTilt: config.tilt,
      });
      
      if (result.success && result.data) {
        setTiltOptimization(result.data);
      }
    } catch (err) {
      console.error('Tilt optimization error:', err);
    } finally {
      setTiltOptLoading(false);
    }
  }, [config]);

  // Debounce ML predictions - reduced to 100ms for faster response
  useEffect(() => {
    // Show loading immediately
    setMlLoading(true);
    
    const timeoutId = setTimeout(() => {
      fetchMLPrediction();
      fetchTiltOptimization(); // Also fetch tilt optimization
    }, 100); // 100ms debounce - much more responsive
    
    return () => clearTimeout(timeoutId);
  }, [fetchMLPrediction, fetchTiltOptimization]);

  // Physics & Efficiency Calculation
  const stats = useMemo(() => {
    const tiltRad = (config.tilt * Math.PI) / 180;
    const azRad = (config.azimuth * Math.PI) / 180;
    const lightAzRad = (config.lightAzimuth * Math.PI) / 180;
    const lightElRad = (config.lightElevation * Math.PI) / 180;

    // Panel Normal Vector
    // Y is UP. 
    // At Tilt 0, Normal is (0, 1, 0).
    // Rotate around X (Tilt).
    // Rotate around Y (Azimuth).
    // Note: To match common "South = 180" convention where 0 is North (-Z in ThreeJS or +Z??):
    // In ThreeJS: +Z is South. -Z is North. +X is East.
    // If Azimuth 180 is South (+Z), then Azimuth 0 is North (-Z).
    // Rotation logic: 
    // Normal = (0,1,0)
    // Rotate X by Tilt: (0, cos T, sin T) (Tips forward towards +Z)
    // Rotate Y by Azimuth:
    // x' = x cos A + z sin A
    // z' = -x sin A + z cos A
    // With x=0:
    // x' = sin T * sin A
    // z' = sin T * cos A
    // y' = cos T
    
    // Correct for Azimuth definition. If 0 is North (-Z), we need to rotate such that A=0 gives Z negative.
    // Use A + PI to map 0 to -Z if needed.
    // Let's stick to standard math: 0 degrees usually means +X or +Z depending on domain. 
    // In Geography 0 is North. In Math 0 is East (+X).
    // Let's assume standard Navigation: 0=N, 90=E, 180=S.
    // If we want coordinates: N=(0,0,-1)
    
    // Let's simplify:
    // Vector pointing up: (0, 1, 0)
    // Rotate X by Tilt: (0, cos T, sin T) -> Tips towards +Z (South)
    // Rotate Y by (Azimuth - 180) to align 180 (South) to +Z.
    // Rotation Angle = Azimuth - 180 degrees (in radians).
    // Let's try:
    // Az=180 -> Rot=0 -> Normal tips to +Z (South). Correct.
    // Az=0 -> Rot=-180 -> Normal tips to -Z (North). Correct.
    
    const rotY = azRad - Math.PI; // Adjust so 180 is South (+Z)
    
    const Nx_local = 0;
    const Ny_local = Math.cos(tiltRad);
    const Nz_local = Math.sin(tiltRad);
    
    const Nx = Nx_local * Math.cos(rotY) + Nz_local * Math.sin(rotY);
    const Ny = Ny_local;
    const Nz = -Nx_local * Math.sin(rotY) + Nz_local * Math.cos(rotY);
    
    // Light Vector
    // From Source.
    // Elevation E. Azimuth LA.
    // Ly = sin E
    // Horizontal component = cos E.
    // Lx = cos E * sin(LA)
    // Lz = cos E * cos(LA)  No wait.
    // 0 Azimuth (North) -> -Z.
    // 180 Azimuth (South) -> +Z.
    // Lz = -cos E * cos(LA) ? No.
    // Let's align with Normal. 
    // 180 (S) -> +Z.
    // sin(180) = 0. cos(180) = -1. Wait...
    // If we want +Z for South (180):
    // Z = -cos(A) ??
    // A=0 -> Z=-1 (North). A=180 -> Z=1 (South). Yes.
    // X = sin(A). A=90 -> X=1 (East). Yes.
    
    const Ly = Math.sin(lightElRad);
    const hL = Math.cos(lightElRad);
    const Lx = hL * Math.sin(lightAzRad * Math.PI / 180); // Wait lightAzRad is already radians
    
    // Recalculate Light Vector properly
    // 0 deg = North (-Z)
    // 90 deg = East (+X)
    // 180 deg = South (+Z)
    const Lx_final = hL * Math.sin(lightAzRad); 
    const Lz_final = -hL * Math.cos(lightAzRad); // cos(0)=1 -> -1 (North). cos(180)=-1 -> 1 (South).
    
    // But wait, my Normal calc used A-180 logic. 
    // Use consistent Navigation coordinates:
    // X = sin(Azimuth)
    // Z = -cos(Azimuth)
    // Normal:
    // Tips by Tilt to South (+Z) initially. 
    // Horizontal projection is sin(Tilt) long, along +Z.
    // Rotate vector (0, sin T, -cos T)? NO.
    // Start (0, 1, 0).
    // Tip towards North or South? Usually panels face Equator (South in northern hemisphere).
    // Let's assume Tilt is "down from horizontal towards Azimuth".
    // Vertical component: cos(Tilt).
    // Horizontal magnitude: sin(Tilt).
    // Horizontal direction: Azimuth.
    // Nx = sin(Tilt) * sin(Azimuth)
    // Nz = sin(Tilt) * (-cos(Azimuth))
    // Ny = cos(Tilt)
    
    const Nx_final = Math.sin(tiltRad) * Math.sin(azRad);
    const Nz_final = Math.sin(tiltRad) * -Math.cos(azRad);
    const Ny_final = Math.cos(tiltRad);

    const dot = Nx_final * Lx_final + Ny_final * Ly + Nz_final * Lz_final;
    const incidentAngleRad = Math.acos(Math.max(-1, Math.min(1, dot)));
    const incidentAngleDeg = (incidentAngleRad * 180) / Math.PI;
    const geometricEfficiency = Math.max(0, dot) * 100;
    
    // Use ML model efficiency if available, otherwise fall back to geometric calculation
    // ML API returns predictedEfficiency as a string (e.g., "83.2")
    const mlEfficiency = mlPrediction?.predictedEfficiency 
      ? parseFloat(mlPrediction.predictedEfficiency) 
      : null;
    const efficiency = mlEfficiency !== null && !isNaN(mlEfficiency) 
      ? mlEfficiency 
      : geometricEfficiency;

    return {
      area: config.width * config.height,
      availableSpace: 0,
      viewAngle,
      panelCount: 1,
      systemSize: 0.4,
      panelTilt: config.tilt,
      efficiency,
      incidentAngle: incidentAngleDeg,
      // Add ML-specific data
      mlSource: mlPrediction ? 'ml-model' : 'geometric',
      mlLoading,
      mlError,
      mlPrediction,
      geometricEfficiency, // Keep for comparison
      // Tilt optimization data
      tiltOptimization,
      tiltOptLoading,
    };
  }, [config, viewAngle, mlPrediction, mlLoading, mlError, tiltOptimization, tiltOptLoading]);

  // Handler to apply optimal tilt from ML recommendation
  const handleApplyOptimalTilt = useCallback(() => {
    if (tiltOptimization?.mlOptimization?.optimalTilt !== undefined) {
      const optimalTilt = Math.round(tiltOptimization.mlOptimization.optimalTilt);
      handleConfigChange({ tilt: optimalTilt, isRotationLocked: false });
    }
  }, [tiltOptimization]);

  return (
    <div className="relative w-full h-screen bg-gray-50 flex flex-col">
      <div className="flex-1 relative">
        <Canvas
          shadows
          dpr={[1, 2]}
          gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping, outputColorSpace: THREE.SRGBColorSpace }}
          className="bg-gray-100"
        >
          <PerspectiveCamera 
            ref={cameraRef}
            makeDefault 
            position={[3, 3, 3]} 
            fov={50}
          />
          
          <SceneController onStatsUpdate={setViewAngle} />

          <Lights 
             // @ts-ignore
             azimuth={config.lightAzimuth} 
             elevation={config.lightElevation} 
             intensity={config.lightIntensity}
             color={config.lightColor}
          />
          
          <Suspense fallback={null}>
            <group>
              <SolarPanel 
                position={[0, 0, 0]}
                // @ts-ignore
                config={config}
                stats={stats}
              />
              <Ground showGrid={config.showGrid} />
            </group>
          </Suspense>

          <axesHelper args={[2]} />
        </Canvas>

        {/* UI Overlay */}
        <Controls
          // @ts-ignore
          config={config}
          onConfigChange={handleConfigChange}
          stats={stats}
          onResetCamera={handleResetCamera}
        />
        
        <AdvancedStats stats={stats} onApplyOptimalTilt={handleApplyOptimalTilt} />
      </div>
    </div>
  );
};

export default RoofDesigner;
