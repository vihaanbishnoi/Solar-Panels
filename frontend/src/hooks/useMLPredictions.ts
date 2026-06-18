import { useState, useEffect, useCallback } from 'react';
import { 
  predictEfficiency, 
  predictDegradation, 
  getMLStatus,
  EfficiencyPredictionInput,
  EfficiencyPredictionResult,
  DegradationPredictionInput,
  DegradationPredictionResult,
  MLModelStatus
} from '../services/api';

interface UseMLPredictionsOptions {
  autoFetch?: boolean;
  refreshInterval?: number; // in milliseconds
}

export function useMLPredictions(options: UseMLPredictionsOptions = {}) {
  const { autoFetch = true, refreshInterval = 0 } = options;
  
  const [mlStatus, setMlStatus] = useState<MLModelStatus | null>(null);
  const [efficiencyResult, setEfficiencyResult] = useState<EfficiencyPredictionResult | null>(null);
  const [degradationResult, setDegradationResult] = useState<DegradationPredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch ML API status
  const fetchMLStatus = useCallback(async () => {
    try {
      const response = await getMLStatus();
      if (response.success && response.data) {
        setMlStatus(response.data);
      }
    } catch (err) {
      console.error('Failed to fetch ML status:', err);
    }
  }, []);

  // Predict efficiency
  const fetchEfficiencyPrediction = useCallback(async (input: EfficiencyPredictionInput) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await predictEfficiency(input);
      if (response.success && response.data) {
        setEfficiencyResult(response.data);
        return response.data;
      } else {
        setError(response.error || 'Failed to get prediction');
        return null;
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMsg);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Predict degradation
  const fetchDegradationPrediction = useCallback(async (input: DegradationPredictionInput) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await predictDegradation(input);
      if (response.success && response.data) {
        setDegradationResult(response.data);
        return response.data;
      } else {
        setError(response.error || 'Failed to get prediction');
        return null;
      }
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMsg);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-fetch on mount
  useEffect(() => {
    if (autoFetch) {
      fetchMLStatus();
    }
  }, [autoFetch, fetchMLStatus]);

  // Refresh interval
  useEffect(() => {
    if (refreshInterval > 0) {
      const interval = setInterval(() => {
        fetchMLStatus();
      }, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [refreshInterval, fetchMLStatus]);

  return {
    // State
    mlStatus,
    efficiencyResult,
    degradationResult,
    loading,
    error,
    
    // Actions
    fetchMLStatus,
    fetchEfficiencyPrediction,
    fetchDegradationPrediction,
    
    // Computed
    isMLConnected: mlStatus?.overallStatus === 'connected',
    isUsingRealModel: mlStatus?.mlApiStatus?.model_loaded === true,
  };
}

export default useMLPredictions;
