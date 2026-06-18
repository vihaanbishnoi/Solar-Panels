/**
 * API Service - Connects frontend to backend
 * Handles all communication with the Node.js/Express server
 * Includes automatic fallback to mock data when ML API is unavailable
 */

import { 
  mockPredictEfficiency, 
  mockPredictDegradation, 
  mockOptimizeTilt, 
  mockClassifyPanel,
} from './mockData';

const API_BASE_URL = 'http://localhost:3001/api';

// Track API health for intelligent fallback
let apiHealthy = true;
let lastHealthCheck = 0;
const HEALTH_CHECK_INTERVAL = 30000; // 30 seconds

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  source?: 'api' | 'mock-fallback';
}

async function fetchApi<T>(
  endpoint: string,
  options?: RequestInit
): Promise<ApiResponse<T>> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    });

    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'API request failed');
    }

    // Mark API as healthy on successful response
    apiHealthy = true;
    lastHealthCheck = Date.now();

    return { ...data, source: 'api' };
  } catch (error) {
    console.error(`API Error (${endpoint}):`, error);
    apiHealthy = false;
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

/**
 * Check if we should use fallback data
 * Exported for use in components that need to check API health
 */
export function shouldUseFallback(): boolean {
  // If last health check was recent and API was unhealthy, use fallback
  if (!apiHealthy && (Date.now() - lastHealthCheck) < HEALTH_CHECK_INTERVAL) {
    return true;
  }
  return false;
}

/**
 * Get current API health status
 */
export function getApiHealthStatus(): { healthy: boolean; lastCheck: number } {
  return { healthy: apiHealthy, lastCheck: lastHealthCheck };
}

// ==================== Health Check ====================

export async function checkHealth() {
  const result = await fetchApi<{ status: string; version: string }>('/health');
  if (!result.success) {
    apiHealthy = false;
  }
  return result;
}

// ==================== ML Predictions ====================

export interface EfficiencyPredictionInput {
  temperature?: number;
  humidity?: number;
  windSpeed?: number;
  irradiance?: number;
  voltage?: number;
  current?: number;
  daysSinceInstallation?: number;
  tilt?: number;
  cloudCover?: number;
  soilingLevel?: number;
}

export interface EfficiencyPredictionResult {
  predictionId: string;
  efficiencyLoss: number;
  efficiencyLossPercent: string;
  panelStatus: string;
  mlRecommendation: string;
  source: 'ml-model' | 'simulation' | 'mock-fallback';
  predictedEfficiency: string;
  factors: {
    tiltImpact: string;
    temperatureImpact: string;
    cloudImpact: string;
    soilingImpact: string;
  };
  recommendations: string[];
}

export async function predictEfficiency(input: EfficiencyPredictionInput): Promise<ApiResponse<EfficiencyPredictionResult>> {
  // Try API first
  const result = await fetchApi<EfficiencyPredictionResult>('/ml/predict/efficiency', {
    method: 'POST',
    body: JSON.stringify(input),
  });
  
  // If API fails, use fallback mock data
  if (!result.success) {
    console.log('⚠️ ML API unavailable, using physics-based fallback');
    const mockResult = mockPredictEfficiency({
      temperature: input.temperature,
      humidity: input.humidity,
      windSpeed: input.windSpeed,
      irradiance: input.irradiance,
      voltage: input.voltage,
      current: input.current,
      daysSinceInstallation: input.daysSinceInstallation,
      tilt: input.tilt,
      cloudCover: input.cloudCover,
      soilingLevel: input.soilingLevel,
    });
    
    return {
      success: true,
      data: mockResult as unknown as EfficiencyPredictionResult,
      source: 'mock-fallback',
    };
  }
  
  return result;
}

export interface DegradationPredictionInput {
  panelId?: string;
  temperature?: number;
  humidity?: number;
  voltageMax?: number;
  voltageMin?: number;
  currentMax?: number;
  currentMin?: number;
  daysSinceInstallation?: number;
  projectionMonths?: number;
}

export interface DegradationPredictionResult {
  predictionId: string;
  panelId: string;
  degradationIndex: number;
  degradationStatus: string;
  stressFactors: {
    thermal_stress: number;
    humidity_stress: number;
    electrical_stress: number;
    aging_stress: number;
  };
  mlRecommendation: string;
  source: 'ml-model' | 'simulation';
  currentEfficiency: string;
  monthlyDegradationRate: string;
  predictions: Array<{
    month: number;
    date: string;
    predictedEfficiency: string;
  }>;
}

export async function predictDegradation(input: DegradationPredictionInput): Promise<ApiResponse<DegradationPredictionResult>> {
  const result = await fetchApi<DegradationPredictionResult>('/ml/predict/degradation', {
    method: 'POST',
    body: JSON.stringify(input),
  });
  
  // If API fails, use fallback mock data
  if (!result.success) {
    console.log('⚠️ ML API unavailable for degradation, using physics-based fallback');
    const mockResult = mockPredictDegradation({
      panelId: input.panelId,
      temperature: input.temperature,
      humidity: input.humidity,
      voltageMax: input.voltageMax,
      voltageMin: input.voltageMin,
      currentMax: input.currentMax,
      currentMin: input.currentMin,
      daysSinceInstallation: input.daysSinceInstallation,
      projectionMonths: input.projectionMonths,
    });
    
    return {
      success: true,
      data: mockResult as unknown as DegradationPredictionResult,
      source: 'mock-fallback',
    };
  }
  
  return result;
}

// ==================== ML Model Status ====================

export interface MLModelStatus {
  overallStatus: 'connected' | 'simulated';
  message: string;
  mlApiStatus: {
    connected: boolean;
    status: string;
    model_loaded: boolean;
  };
  mlModelInfo: {
    model_type: string;
    n_estimators: number;
    features: string[];
    target: string;
  };
}

export async function getMLStatus() {
  return fetchApi<MLModelStatus>('/ml/models/status');
}

// ==================== Dashboard Stats ====================

export interface DashboardStats {
  totalGeneration: { value: number; unit: string; change: number };
  efficiency: { value: number; change: number };
  activePanels: { active: number; total: number };
  batteryStatus: { percentage: number; remainingMinutes: number };
}

export async function getDashboardStats() {
  return fetchApi<DashboardStats>('/dashboard/stats');
}

// ==================== Panels ====================

export interface Panel {
  id: string;
  name: string;
  status: string;
  efficiency: number;
  temperature: number;
  power: number;
}

export async function getPanels() {
  return fetchApi<Panel[]>('/panels');
}

export async function getPanelById(id: string) {
  return fetchApi<Panel>(`/panels/${id}`);
}

// ==================== Alerts ====================

export interface Alert {
  id: string;
  type: 'warning' | 'error' | 'info' | 'success';
  title: string;
  description: string;
  timestamp: string;
  acknowledged: boolean;
}

export async function getAlerts() {
  return fetchApi<Alert[]>('/alerts');
}

// ==================== Panel Defect Classification ====================

export interface PanelClassificationResult {
  analysisId: string;
  filename: string;
  fileSize: number;
  prediction: {
    label: 'NORMAL' | 'DEFECTIVE';
    status: 'good' | 'bad';
    class_id: number;
    confidence: number;
    confidence_percent: string;
  };
  probabilities: {
    normal: number;
    defective: number;
  };
  analysis: {
    severity: 'none' | 'medium' | 'high' | 'critical';
    action_required: boolean;
    recommendation: string;
  };
  source: 'ml-model' | 'simulation';
  modelInfo: {
    architecture: string;
    classes: string[];
  };
  timestamp: string;
  // Email alert status (only present when defective panel is detected)
  emailAlert?: {
    sent: boolean;
    messageId: string | null;
    error: string | null;
  };
}

export interface BatchClassificationResult {
  batchId: string;
  totalProcessed: number;
  summary: {
    normal_count: number;
    defective_count: number;
    defect_rate: number;
  };
  results: Array<{
    index: number;
    filename: string;
    prediction: 'NORMAL' | 'DEFECTIVE';
    confidence: number;
    is_defective: boolean;
  }>;
  source: 'ml-model' | 'simulation';
  timestamp: string;
}

/**
 * Classify a single panel image for defects using ML model
 */
export async function classifyPanelImage(imageFile: File): Promise<{
  success: boolean;
  data?: PanelClassificationResult;
  error?: string;
  source?: 'api' | 'mock-fallback';
}> {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    const response = await fetch(`${API_BASE_URL}/ml/image/classify-panel`, {
      method: 'POST',
      body: formData,
    });
    
    const result = await response.json();
    
    if (!response.ok) {
      throw new Error(result.error || 'Classification failed');
    }
    
    apiHealthy = true;
    return { ...result, source: 'api' };
  } catch (error) {
    console.error('Panel classification error:', error);
    apiHealthy = false;
    
    // Use mock fallback for image classification
    console.log('⚠️ ML API unavailable for image classification, using mock fallback');
    const mockResult = mockClassifyPanel(imageFile.name, imageFile.size);
    
    return {
      success: true,
      data: mockResult as unknown as PanelClassificationResult,
      source: 'mock-fallback',
    };
  }
}

/**
 * Batch classify multiple panel images
 */
export async function classifyPanelImagesBatch(imageFiles: File[]): Promise<{
  success: boolean;
  data?: BatchClassificationResult;
  error?: string;
}> {
  try {
    const formData = new FormData();
    imageFiles.forEach(file => {
      formData.append('images', file);
    });
    
    const response = await fetch(`${API_BASE_URL}/ml/image/classify-panel/batch`, {
      method: 'POST',
      body: formData,
    });
    
    const result = await response.json();
    
    if (!response.ok) {
      throw new Error(result.error || 'Batch classification failed');
    }
    
    apiHealthy = true;
    return result;
  } catch (error) {
    console.error('Batch classification error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Batch classification failed',
    };
  }
}

// ==================== Tilt Optimization ====================

export interface TiltOptimizationInput {
  latitude: number;
  longitude?: number;
  ghi: number;
  hour: number;
  temperature?: number;
  humidity?: number;
  windSpeed?: number;
  voltage?: number;
  current?: number;
  daysSinceInstallation?: number;
  currentTilt?: number;
}

export interface TiltOptimizationResult {
  optimizationId: string;
  source: 'ml-model' | 'simulation';
  location: {
    latitude: number;
    longitude: number;
  };
  currentSettings: {
    tilt: number;
    azimuth: number;
  };
  mlOptimization: {
    optimalTilt: number;
    estimatedEnergy: number;
    solarElevation: number;
    efficiencyLoss: number;
    efficiencyRetained: number;
    improvementPercent: number;
    defaultTilt: number;
    defaultEnergy: number;
    tiltCurve?: Array<{
      tilt: number;
      effective_irradiance: number;
      net_energy: number;
    }>;
  };
  conditions: {
    ghi: number;
    latitude: number;
    hour: number;
    temperature: number;
  };
  recommendations: {
    mlOptimalTilt: number;
    annualOptimalTilt: number;
    seasonalOptimalTilt: number;
    optimalAzimuth: number;
    adjustmentNeeded: boolean;
  };
  seasonalSchedule: Array<{
    months: string;
    recommendedTilt: string;
  }>;
  estimatedGain: {
    withMLOptimization: string;
    withTracking: string;
  };
}

/**
 * Optimize tilt angle using ML model
 */
export async function optimizeTiltAngle(input: TiltOptimizationInput): Promise<{
  success: boolean;
  data?: TiltOptimizationResult;
  error?: string;
  source?: 'api' | 'mock-fallback';
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/ml/optimize/tilt`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || 'Optimization failed');
    }

    apiHealthy = true;
    return { ...result, source: 'api' };
  } catch (error) {
    console.error('Tilt optimization error:', error);
    apiHealthy = false;
    
    // Use mock fallback for tilt optimization
    console.log('⚠️ ML API unavailable for tilt optimization, using physics-based fallback');
    const mockResult = mockOptimizeTilt({
      latitude: input.latitude,
      ghi: input.ghi,
      hour: input.hour,
      temperature: input.temperature,
      humidity: input.humidity,
      windSpeed: input.windSpeed,
      currentTilt: input.currentTilt,
    });
    
    return {
      success: true,
      data: mockResult as unknown as TiltOptimizationResult,
      source: 'mock-fallback',
    };
  }
}

/**
 * Quick tilt optimization with minimal parameters
 */
export async function quickTiltOptimize(params: {
  ghi?: number;
  latitude?: number;
  hour?: number;
  temperature?: number;
}): Promise<{
  success: boolean;
  data?: {
    optimalTilt: number;
    estimatedEnergy: number;
    improvementPercent: number;
    solarElevation: number;
  };
  error?: string;
}> {
  const searchParams = new URLSearchParams();
  if (params.ghi) searchParams.append('ghi', params.ghi.toString());
  if (params.latitude) searchParams.append('latitude', params.latitude.toString());
  if (params.hour) searchParams.append('hour', params.hour.toString());
  if (params.temperature) searchParams.append('temperature', params.temperature.toString());

  return fetchApi(`/ml/optimize/tilt/quick?${searchParams.toString()}`);
}

// Export all
export const api = {
  checkHealth,
  predictEfficiency,
  predictDegradation,
  getMLStatus,
  getDashboardStats,
  getPanels,
  getPanelById,
  getAlerts,
  classifyPanelImage,
  classifyPanelImagesBatch,
  optimizeTiltAngle,
  quickTiltOptimize,
};

export default api;
