export interface RoofDimensions {
  width: number;
  length: number;
  height: number;
}

export interface SolarPanelConfig {
  // Dimensions
  width: number;
  height: number;
  thickness: number;
  
  // Orientation
  tilt: number;
  azimuth: number;
  isRotationLocked: boolean;

  // Light Source
  lightAzimuth: number;
  lightElevation: number;
  lightIntensity: number;
  lightColor: string; // 'warm' | 'cool' | 'daylight'

  // Grid & Display
  showGrid: boolean;
  gridSize: number;
  panelColor: string;
  showMeasurements: boolean;
  showLightDirection: boolean;
  showNormalVector: boolean;
}

export interface RoofState {
  dimensions: RoofDimensions;
  config: SolarPanelConfig;
}

export interface ViewStats {
  area: number;
  availableSpace: number;
  viewAngle: string;
  panelCount: number;
  systemSize: number;
  panelTilt: number;
  efficiency: number;
  incidentAngle: number;
  // ML-specific fields
  mlSource?: 'ml-model' | 'geometric';
  mlLoading?: boolean;
  mlError?: string | null;
  mlPrediction?: {
    efficiencyLoss: number;
    panelStatus: string;
    mlRecommendation: string;
    source: 'ml-model' | 'simulation';
    factors?: {
      tiltImpact: string;
      temperatureImpact: string;
      cloudImpact: string;
      soilingImpact: string;
    };
    recommendations?: string[];
  } | null;
  geometricEfficiency?: number;
  // Tilt optimization data
  tiltOptimization?: {
    mlOptimization?: {
      optimalTilt: number;
      estimatedEnergy: number;
      solarElevation: number;
      efficiencyLoss: number;
      efficiencyRetained: number;
      improvementPercent: number;
    };
    recommendations?: {
      mlOptimalTilt: number;
      adjustmentNeeded: boolean;
    };
    estimatedGain?: {
      withMLOptimization: string;
      withTracking: string;
    };
  } | null;
  tiltOptLoading?: boolean;
}

export interface SolarPanel {
  id: string;
  position: [number, number, number];
  rotation: [number, number, number];
  isValid: boolean;
}
