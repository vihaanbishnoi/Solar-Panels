import React, { useMemo } from 'react';
import { Vector3 } from 'three';

interface LightsProps {
  azimuth?: number;
  elevation?: number;
  intensity?: number;
  color?: string;
}

export const Lights: React.FC<LightsProps> = ({ 
  azimuth = 180, 
  elevation = 45, 
  intensity = 1.0, 
  color = 'daylight' 
}) => {
  
  const lightColor = useMemo(() => {
    switch (color) {
      case 'warm': return '#ffaa44';
      case 'cool': return '#ccddff';
      case 'daylight':
      default: return '#ffffff';
    }
  }, [color]);

  const lightPosition = useMemo(() => {
    const dist = 20;
    const elRad = (elevation * Math.PI) / 180;
    const azRad = (azimuth * Math.PI) / 180;

    const y = dist * Math.sin(elRad);
    const h = dist * Math.cos(elRad);
    const x = h * Math.sin(azRad);
    const z = -h * Math.cos(azRad); // 0=North (-Z), 180=South (+Z) convention

    return new Vector3(x, y, z);
  }, [azimuth, elevation]);

  return (
    <>
      {/* Ambient light for base illumination */}
      <ambientLight intensity={0.4} color="#ffffff" />
      
      {/* Natural sky lighting */}
      <hemisphereLight 
        args={['#ffffff', '#444444', 0.4]} 
        position={[0, 50, 0]} 
      />

      {/* Main directional light representing the sun */}
      <directionalLight
        position={lightPosition}
        intensity={intensity * 1.5}
        color={lightColor}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-far={50}
        shadow-camera-left={-10}
        shadow-camera-right={10}
        shadow-camera-top={10}
        shadow-camera-bottom={-10}
        shadow-bias={-0.0005}
      />
    </>
  );
};
