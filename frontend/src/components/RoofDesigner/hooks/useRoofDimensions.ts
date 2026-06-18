import { useState, useMemo } from 'react';
import { RoofDimensions } from '../types/roof.types';

export const useRoofDimensions = (initialDimensions?: RoofDimensions) => {
  const [dimensions, setDimensions] = useState<RoofDimensions>(
    initialDimensions || {
      width: 12,
      length: 8,
      height: 0.3,
    }
  );

  const stats = useMemo(() => {
    const area = dimensions.width * dimensions.length;
    const availableSpace = area * 0.9;
    
    return {
      area,
      availableSpace,
    };
  }, [dimensions]);

  return {
    dimensions,
    setDimensions,
    stats,
  };
};
