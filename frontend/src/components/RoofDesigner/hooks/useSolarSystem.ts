import { useState, useCallback } from 'react';
import {SolarPanel,  RoofDimensions } from '../types/roof.types';
import * as THREE from 'three';

export const useSolarSystem = (dimensions: RoofDimensions) => {
  const [panels, setPanels] = useState<SolarPanel[]>([]);
  const [tilt, setTilt] = useState(0);

  const addPanel = useCallback((position: [number, number, number]) => {
    const newPanel: SolarPanel = {
      id: crypto.randomUUID(),
      position,
      rotation: [0, 0, 0],
      isValid: true,
    };
    
    // Check boundaries
    const halfWidth = dimensions.width / 2;
    const halfLength = dimensions.length / 2;
    const padding = 0.2; // Margin from edge

    // Panel approx dimensions
    const pWidth = 1.7; 
    const pLength = 1.0;

    const inBounds = 
      Math.abs(position[0]) + pWidth/2 < halfWidth - padding &&
      Math.abs(position[2]) + pLength/2 < halfLength - padding;

    if (inBounds) {
        setPanels(prev => [...prev, newPanel]);
    }
  }, [dimensions]);

  const clearPanels = useCallback(() => {
    setPanels([]);
  }, []);

  const removeLastPanel = useCallback(() => {
    setPanels(prev => prev.slice(0, -1));
  }, []);

  const systemStats = {
    panelCount: panels.length,
    systemSize: panels.length * 0.4, // Assume 400W panels
    panelTilt: tilt,
  };

  return {
    panels,
    addPanel,
    clearPanels,
    removeLastPanel,
    systemStats,
    tilt,
    setTilt,
  };
};
