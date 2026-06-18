import React from 'react';
import { Plane } from '@react-three/drei';

interface GroundProps {
  showGrid?: boolean;
}

export const Ground: React.FC<GroundProps> = ({ showGrid = true }) => {
  return (
    <group position={[0, -0.01, 0]}>
      {/* Ground Plane */}
      <Plane 
        args={[30, 30]} 
        rotation={[-Math.PI / 2, 0, 0]} 
        receiveShadow
      >
        <meshStandardMaterial color="#e5e5e5" roughness={1} metalness={0} />
      </Plane>

      {/* Grid Helper */}
      {showGrid && (
        <gridHelper 
          args={[30, 30, '#888888', '#bbbbbb']} 
          position={[0, 0.01, 0]} 
        />
      )}
    </group>
  );
};
