import React, { useMemo } from 'react';
import { Html, Text, Grid } from '@react-three/drei';
import * as THREE from 'three';
import { SolarPanelConfig, ViewStats } from './types/roof.types';

interface SolarPanelProps {
  position: [number, number, number];
  config: SolarPanelConfig;
  stats: ViewStats;
}

export const SolarPanel: React.FC<SolarPanelProps> = ({ 
  position,
  config,
  stats
}) => {
  const { width, height, thickness, tilt, azimuth, showGrid, showMeasurements, showLightDirection, showNormalVector, panelColor } = config;

  // Convert angles to radians
  const tiltRad = (tilt * Math.PI) / 180;
  // Azimuth 180 (South) corresponds to +Z in our calculation model (RotY = 0)
  // Azimuth 0 (North) corresponds to -Z (RotY = PI)
  // Rotation offset: - (Azimuth - 180) to align with standard compass
  const azRad = ((azimuth - 180) * Math.PI) / 180;

  // Light Vector for visualization
  const lightVec = useMemo(() => {
     const elRad = (config.lightElevation * Math.PI) / 180;
     const azLRad = (config.lightAzimuth * Math.PI) / 180;
     // Light Direction (To Source)
     // Start South (+Z) for Azimuth 180.
     // If Azimuth is 180 -> +Z. 
     // X = sin(Az)
     // Z = -cos(Az)
     // Y = sin(El)
     const y = Math.sin(elRad);
     const h = Math.cos(elRad);
     const x = h * Math.sin(azLRad * Math.PI / 180); // Correct conversion
     // Actually use passed degrees directly if math.sin accepts radians
     const x_ = h * Math.sin(azLRad * Math.PI / 180); 
     const z_ = -h * Math.cos(azLRad * Math.PI / 180);
     return new THREE.Vector3(x_, y, z_);
  }, [config.lightAzimuth, config.lightElevation]);

  return (
    <group position={position}>
      {/* Azimuth Rotation (Y) */}
      <group rotation={[0, -azRad, 0]}> 
        
        {/* Tilt Rotation (X) */}
        <group rotation={[tiltRad, 0, 0]}>
          
          {/* Panel Mesh centered */}
          <group position={[0, thickness/2, 0]}>
              <mesh castShadow receiveShadow>
                <boxGeometry args={[width, thickness, height]} />
                <meshStandardMaterial 
                  color={panelColor} 
                  roughness={0.2} 
                  metalness={0.6}
                />
              </mesh>
              
              {/* Frame */}
              <mesh position={[0, 0, 0]} scale={[1.02, 1.05, 1.02]}>
                  <boxGeometry args={[width, thickness * 0.9, height]} />
                  <meshStandardMaterial color="#d1d5db" roughness={0.8} />
              </mesh>

              {/* Grid Overlay */}
              {showGrid && (
                 <group position={[0, thickness/2 + 0.002, 0]} rotation={[0, 0, 0]}>
                    <Grid 
                        args={[width, height]} 
                        cellSize={config.gridSize} 
                        cellThickness={1} 
                        cellColor="white" 
                        sectionSize={1} 
                        sectionThickness={1.5} 
                        sectionColor="white" 
                        fadeDistance={20} 
                        infiniteGrid={false}
                    />
                 </group>
              )}
              
              {/* Normal Vector Visualization (Green) */}
              {showNormalVector && (
                <group position={[0, thickness, 0]}>
                    <arrowHelper args={[new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), 1.5, 0x00ff00]} />
                    <Html position={[0.2, 1.5, 0]}>
                        <div className="bg-black/70 text-white text-xs px-1 rounded backdrop-blur-sm pointer-events-none">Normal</div>
                    </Html>
                </group>
              )}

               {/* Measurements */}
               {showMeasurements && (
                  <>
                    {/* Width Label */}
                    <Text 
                        position={[0, thickness + 0.1, height/2 + 0.1]} 
                        fontSize={0.2} 
                        color="black"
                        rotation={[-Math.PI/2, 0, 0]}
                    >
                        {width.toFixed(2)}m
                    </Text>
                     {/* Height Label */}
                    <Text 
                        position={[width/2 + 0.1, thickness + 0.1, 0]} 
                        fontSize={0.2} 
                        color="black"
                        rotation={[-Math.PI/2, 0, -Math.PI/2]}
                    >
                        {height.toFixed(2)}m
                    </Text>
                  </>
               )}
          </group>
        </group>
      </group>

      {/* Light Vector Visualization (Yellow) */}
      {showLightDirection && (
         <group>
            <arrowHelper args={[lightVec, new THREE.Vector3(0, 0, 0), 2.5, 0xffaa00]} />
             <Html position={[lightVec.x * 2.6, lightVec.y * 2.6, lightVec.z * 2.6]}>
                <div className="bg-yellow-500/90 text-white text-xs px-2 py-1 rounded shadow-sm whitespace-nowrap backdrop-blur-sm pointer-events-none">
                   Sun <br/> 
                   {Math.round(stats.incidentAngle)}° Inc.
                </div>
            </Html>
         </group>
      )}
    </group>
  );
};
