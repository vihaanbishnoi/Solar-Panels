import React, { useMemo } from 'react';
import { Box, Edges, Html } from '@react-three/drei';
import { ThreeEvent } from '@react-three/fiber';
import * as THREE from 'three';
import { RoofDimensions } from './types/roof.types';

interface RoofProps {
  dimensions: RoofDimensions;
  showMeasurements: boolean;
  onRoofClick?: (position: [number, number, number]) => void;
}

export const Roof: React.FC<RoofProps> = ({ dimensions, showMeasurements, onRoofClick }) => {
  const { width, length, height } = dimensions;

  // Measurement Label Component
  const MeasurementLabel = ({ position, text }: { position: [number, number, number]; text: string }) => (
    <Html position={position} center className="pointer-events-none select-none">
      <div className="bg-black/70 text-white px-2 py-1 rounded text-xs font-mono whitespace-nowrap backdrop-blur-sm">
        {text}
      </div>
    </Html>
  );

  return (
    <group position={[0, height / 2, 0]}>
      <Box 
        args={[width, height, length]} 
        castShadow 
        receiveShadow
        onClick={(e: ThreeEvent<MouseEvent>) => {
          e.stopPropagation();
          onRoofClick?.([e.point.x, e.point.y, e.point.z]);
        }}
      >
        <meshStandardMaterial
          color="#4a4a4a"
          roughness={0.9}
          metalness={0.1}
          map={null} // Placeholder for texture
        />
        
        {/* Edge highlighting */}
        <Edges
          linewidth={2}
          threshold={15} // Display edges only when angle between faces exceeds this value
          color="white"
        />
      </Box>

      {/* Measurement Labels */}
      {showMeasurements && (
        <>
          {/* Width Label */}
          <MeasurementLabel 
            position={[0, height / 2 + 0.5, length / 2]} 
            text={`${width}m`} 
          />
          
          {/* Length Label */}
          <MeasurementLabel 
            position={[width / 2, height / 2 + 0.5, 0]} 
            text={`${length}m`} 
          />
        </>
      )}
    </group>
  );
};
