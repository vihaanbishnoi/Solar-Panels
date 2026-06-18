/**
 * Mock Data Service - Fallback values when ML API is unavailable
 * Provides realistic, physics-based simulated data that mimics actual ML model outputs
 * 
 * Models Simulated:
 * 1. efficiency_loss_model.pkl (RandomForest) - 7 features
 * 2. pv_classifier.pth (ResNet-18 CNN) - Binary classification
 * 3. Tilt Optimizer - Physics-ML hybrid
 */

// ==================== Efficiency Loss Model Mock ====================
// Based on: temperature, humidity, wind_speed, irradiance, voltage, current, days_since_installation

export interface MockEfficiencyInput {
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

export interface MockEfficiencyResult {
  predictionId: string;
  efficiencyLoss: number;
  efficiencyLossPercent: string;
  panelStatus: 'optimal' | 'good' | 'fair' | 'poor' | 'critical';
  mlRecommendation: string;
  source: 'mock-fallback';
  predictedEfficiency: string;
  factors: {
    tiltImpact: string;
    temperatureImpact: string;
    cloudImpact: string;
    soilingImpact: string;
  };
  recommendations: string[];
}

/**
 * Simulates the RandomForest efficiency_loss_model.pkl behavior
 * Uses physics-based formulas that approximate real model outputs
 */
export function mockPredictEfficiency(input: MockEfficiencyInput): MockEfficiencyResult {
  const temp = input.temperature ?? 25;
  const humidity = input.humidity ?? 50;
  const windSpeed = input.windSpeed ?? 3;
  const irradiance = input.irradiance ?? 800;
  const voltage = input.voltage ?? 35;
  const current = input.current ?? 8;
  const daysInstalled = input.daysSinceInstallation ?? 365;
  const tilt = input.tilt ?? 30;
  const cloudCover = input.cloudCover ?? 20;
  const soilingLevel = input.soilingLevel ?? 5;

  // Physics-based efficiency loss calculations
  // Temperature coefficient: ~-0.4% per °C above 25°C
  const tempCoeff = -0.004;
  const tempLoss = Math.max(0, (temp - 25) * tempCoeff * 100);
  
  // Tilt efficiency: Optimal tilt ≈ latitude (~35° for mid-latitudes)
  const optimalTilt = 35;
  const tiltDiff = Math.abs(tilt - optimalTilt);
  const tiltLoss = (tiltDiff / 90) * 15; // Max ~15% loss at 90° off
  
  // Irradiance factor: Below 200 W/m² causes non-linear losses
  const irradianceFactor = irradiance < 200 ? 0.6 : irradiance < 500 ? 0.85 : 1.0;
  const irradianceLoss = (1 - irradianceFactor) * 20;
  
  // Aging degradation: ~0.5% per year
  const agingLoss = (daysInstalled / 365) * 0.5;
  
  // Humidity stress: High humidity causes micro-corrosion
  const humidityLoss = humidity > 80 ? 2.5 : humidity > 60 ? 1.2 : 0.5;
  
  // Cloud cover impact
  const cloudLoss = cloudCover * 0.12;
  
  // Soiling impact
  const soilingLoss = soilingLevel * 0.8;
  
  // Wind cooling benefit (reduces temperature loss)
  const windBenefit = Math.min(2, windSpeed * 0.3);
  
  // Electrical stress (voltage/current deviations from nominal)
  const nominalVoltage = 35;
  const nominalCurrent = 8.5;
  const voltageDeviation = Math.abs(voltage - nominalVoltage) / nominalVoltage;
  const currentDeviation = Math.abs(current - nominalCurrent) / nominalCurrent;
  const electricalLoss = (voltageDeviation + currentDeviation) * 5;
  
  // Total efficiency loss
  const totalLoss = Math.max(0, Math.min(50, 
    tempLoss + tiltLoss + irradianceLoss + agingLoss + 
    humidityLoss + cloudLoss + soilingLoss + electricalLoss - windBenefit
  ));
  
  const efficiency = Math.max(0, 100 - totalLoss);
  
  // Determine status
  let status: MockEfficiencyResult['panelStatus'];
  let recommendation: string;
  const recommendations: string[] = [];
  
  if (efficiency >= 90) {
    status = 'optimal';
    recommendation = 'Panel performing at peak efficiency. Maintain current setup.';
  } else if (efficiency >= 80) {
    status = 'good';
    recommendation = 'Good performance. Minor optimizations available.';
    if (soilingLevel > 5) recommendations.push('Schedule panel cleaning to reduce soiling losses');
    if (tiltDiff > 15) recommendations.push(`Adjust tilt angle by ${tiltDiff.toFixed(0)}° for optimal solar capture`);
  } else if (efficiency >= 65) {
    status = 'fair';
    recommendation = 'Performance below optimal. Review factors and schedule maintenance.';
    recommendations.push('Consider professional inspection within 30 days');
    if (temp > 35) recommendations.push('Improve panel ventilation to reduce thermal losses');
    if (cloudCover > 50) recommendations.push('Performance limited by cloud cover - normal behavior');
  } else if (efficiency >= 50) {
    status = 'poor';
    recommendation = 'Significant efficiency loss detected. Immediate attention recommended.';
    recommendations.push('Schedule professional inspection within 7 days');
    recommendations.push('Check for physical damage or obstructions');
  } else {
    status = 'critical';
    recommendation = 'Critical efficiency loss! Emergency inspection required.';
    recommendations.push('Immediate system shutdown and inspection recommended');
    recommendations.push('Check electrical connections and inverter status');
  }

  return {
    predictionId: `mock_eff_${Date.now()}`,
    efficiencyLoss: parseFloat(totalLoss.toFixed(2)),
    efficiencyLossPercent: `${totalLoss.toFixed(2)}%`,
    panelStatus: status,
    mlRecommendation: recommendation,
    source: 'mock-fallback',
    predictedEfficiency: efficiency.toFixed(1),
    factors: {
      tiltImpact: `${tiltLoss.toFixed(1)}%`,
      temperatureImpact: `${Math.max(0, tempLoss - windBenefit).toFixed(1)}%`,
      cloudImpact: `${cloudLoss.toFixed(1)}%`,
      soilingImpact: `${soilingLoss.toFixed(1)}%`,
    },
    recommendations,
  };
}

// ==================== Degradation Model Mock ====================

export interface MockDegradationInput {
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

export interface MockDegradationResult {
  predictionId: string;
  panelId: string;
  degradationIndex: number;
  degradationStatus: 'Healthy' | 'Degrading' | 'Critical';
  stressFactors: {
    thermal_stress: number;
    humidity_stress: number;
    electrical_stress: number;
    aging_stress: number;
  };
  mlRecommendation: string;
  source: 'mock-fallback';
  currentEfficiency: string;
  monthlyDegradationRate: string;
  predictions: Array<{
    month: number;
    date: string;
    predictedEfficiency: string;
  }>;
}

/**
 * Simulates degradation prediction with stress factor analysis
 */
export function mockPredictDegradation(input: MockDegradationInput): MockDegradationResult {
  const panelId = input.panelId ?? `panel_${Math.random().toString(36).substr(2, 9)}`;
  const temp = input.temperature ?? 25;
  const humidity = input.humidity ?? 50;
  const vMax = input.voltageMax ?? 38;
  const vMin = input.voltageMin ?? 32;
  const iMax = input.currentMax ?? 9;
  const iMin = input.currentMin ?? 7;
  const daysInstalled = input.daysSinceInstallation ?? 365;
  const projectionMonths = input.projectionMonths ?? 12;
  
  // Calculate stress factors
  const thermalStress = Math.max(0, (temp - 25) / 40);
  const humidityStress = humidity / 100;
  const voltageDrop = vMax > 0 ? (vMax - vMin) / vMax : 0;
  const currentDrop = iMax > 0 ? (iMax - iMin) / iMax : 0;
  const electricalStress = Math.min(1, voltageDrop + currentDrop);
  const agingStress = Math.min(1, daysInstalled / (25 * 365));
  
  // Composite degradation index
  const degradationIndex = Math.min(1, Math.max(0,
    0.30 * thermalStress +
    0.20 * humidityStress +
    0.30 * electricalStress +
    0.20 * agingStress
  ));
  
  // Monthly degradation rate (industry standard: 0.5-1% per year)
  const yearlyRate = 0.5 + (degradationIndex * 0.5);
  const monthlyRate = yearlyRate / 12;
  
  // Current efficiency
  const currentEfficiency = 100 - (daysInstalled / 365) * yearlyRate;
  
  // Generate predictions
  const predictions: MockDegradationResult['predictions']= [];
  const now = new Date();
  for (let m = 1; m <= projectionMonths; m++) {
    const futureDate = new Date(now);
    futureDate.setMonth(futureDate.getMonth() + m);
    const predictedEff = Math.max(0, currentEfficiency - (m * monthlyRate));
    predictions.push({
      month: m,
      date: futureDate.toISOString().split('T')[0],
      predictedEfficiency: predictedEff.toFixed(1),
    });
  }
  
  // Determine status
  let status: MockDegradationResult['degradationStatus'];
  let recommendation: string;
  
  if (degradationIndex < 0.4) {
    status = 'Healthy';
    recommendation = 'Panel aging normally. Continue standard maintenance schedule.';
  } else if (degradationIndex < 0.7) {
    status = 'Degrading';
    recommendation = 'Accelerated degradation detected. Schedule preventive maintenance within 30 days.';
  } else {
    status = 'Critical';
    recommendation = 'Critical degradation levels! Immediate inspection and possible replacement required.';
  }

  return {
    predictionId: `mock_deg_${Date.now()}`,
    panelId,
    degradationIndex: parseFloat(degradationIndex.toFixed(3)),
    degradationStatus: status,
    stressFactors: {
      thermal_stress: parseFloat(thermalStress.toFixed(3)),
      humidity_stress: parseFloat(humidityStress.toFixed(3)),
      electrical_stress: parseFloat(electricalStress.toFixed(3)),
      aging_stress: parseFloat(agingStress.toFixed(3)),
    },
    mlRecommendation: recommendation,
    source: 'mock-fallback',
    currentEfficiency: currentEfficiency.toFixed(1),
    monthlyDegradationRate: `${monthlyRate.toFixed(3)}%`,
    predictions,
  };
}

// ==================== Panel Defect Classification Mock ====================

export interface MockClassificationResult {
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
  source: 'mock-fallback';
  modelInfo: {
    architecture: string;
    classes: string[];
  };
  timestamp: string;
}

/**
 * Simulates ResNet-18 panel defect classification
 * Provides realistic probability distributions based on file characteristics
 */
export function mockClassifyPanel(
  filename: string = 'panel.jpg',
  fileSize: number = 1024000,
  simulateDefect: boolean = false
): MockClassificationResult {
  // Simulate realistic confidence scores
  // Normal panels have 85-98% confidence, defective 75-95%
  const isDefective = simulateDefect || Math.random() < 0.15; // 15% defect rate
  
  const baseConfidence = isDefective ? 
    75 + Math.random() * 20 : // 75-95% for defective
    85 + Math.random() * 13;  // 85-98% for normal
  
  const confidence = parseFloat(baseConfidence.toFixed(2));
  const normalProb = isDefective ? 100 - confidence : confidence;
  const defectiveProb = isDefective ? confidence : 100 - confidence;
  
  let severity: MockClassificationResult['analysis']['severity'];
  let recommendation: string;
  
  if (!isDefective) {
    severity = 'none';
    recommendation = 'Panel appears healthy. Continue regular monitoring schedule.';
  } else if (confidence > 90) {
    severity = 'critical';
    recommendation = 'High confidence defect detected! Schedule immediate inspection and consider replacement.';
  } else if (confidence > 80) {
    severity = 'high';
    recommendation = 'Likely defect detected. Schedule professional inspection within 1 week.';
  } else {
    severity = 'medium';
    recommendation = 'Possible defect detected. Monitor closely and verify with manual inspection.';
  }

  return {
    analysisId: `mock_cls_${Date.now()}`,
    filename,
    fileSize,
    prediction: {
      label: isDefective ? 'DEFECTIVE' : 'NORMAL',
      status: isDefective ? 'bad' : 'good',
      class_id: isDefective ? 0 : 1,
      confidence,
      confidence_percent: `${confidence}%`,
    },
    probabilities: {
      normal: parseFloat(normalProb.toFixed(2)),
      defective: parseFloat(defectiveProb.toFixed(2)),
    },
    analysis: {
      severity,
      action_required: isDefective,
      recommendation,
    },
    source: 'mock-fallback',
    modelInfo: {
      architecture: 'ResNet-18 (Mock)',
      classes: ['DEFECTIVE (BAD)', 'NORMAL (GOOD)'],
    },
    timestamp: new Date().toISOString(),
  };
}

// ==================== Tilt Optimization Mock ====================

export interface MockTiltInput {
  latitude: number;
  ghi: number;
  hour: number;
  temperature?: number;
  humidity?: number;
  windSpeed?: number;
  currentTilt?: number;
}

export interface MockTiltResult {
  optimizationId: string;
  source: 'mock-fallback';
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
}

/**
 * Simulates tilt optimization using physics-based solar geometry
 */
export function mockOptimizeTilt(input: MockTiltInput): MockTiltResult {
  const lat = input.latitude;
  const ghi = input.ghi;
  const hour = input.hour;
  const temp = input.temperature ?? 25;
  const currentTilt = input.currentTilt ?? 30;
  
  // Calculate solar declination (approximation for winter solstice to summer solstice)
  const dayOfYear = Math.floor((new Date().getTime() - new Date(new Date().getFullYear(), 0, 0).getTime()) / 86400000);
  const declination = -23.45 * Math.cos((2 * Math.PI / 365) * (dayOfYear + 10));
  
  // Hour angle (0 at solar noon)
  const hourAngle = (hour - 12) * 15;
  
  // Solar elevation angle
  const latRad = lat * Math.PI / 180;
  const decRad = declination * Math.PI / 180;
  const hourRad = hourAngle * Math.PI / 180;
  
  const sinElevation = Math.sin(latRad) * Math.sin(decRad) + 
                       Math.cos(latRad) * Math.cos(decRad) * Math.cos(hourRad);
  const elevation = Math.asin(Math.max(-1, Math.min(1, sinElevation))) * 180 / Math.PI;
  
  // Optimal tilt based on solar geometry
  // For fixed tilt, optimal ≈ latitude - declination (simplified)
  const baseOptimalTilt = Math.max(0, Math.min(90, lat - declination * 0.5));
  
  // Adjust for time of day (lower tilt for morning/evening)
  const timeAdjustment = Math.abs(hour - 12) * 0.5;
  const optimalTilt = Math.max(0, baseOptimalTilt - timeAdjustment);
  
  // Calculate energy with different tilts
  const tiltRad = currentTilt * Math.PI / 180;
  const optTiltRad = optimalTilt * Math.PI / 180;
  const elevRad = Math.max(0.01, elevation) * Math.PI / 180;
  
  // Simplified POA irradiance calculation
  const currentPOA = ghi * (Math.sin(elevRad + tiltRad) / Math.sin(elevRad));
  const optimalPOA = ghi * (Math.sin(elevRad + optTiltRad) / Math.sin(elevRad));
  
  // Temperature derate (~-0.4% per °C above 25°C)
  const tempDerate = 1 - Math.max(0, temp - 25) * 0.004;
  
  // Calculate energies
  const currentEnergy = currentPOA * tempDerate * 0.2; // 20% panel efficiency
  const optimalEnergy = optimalPOA * tempDerate * 0.2;
  const defaultEnergy = ghi * tempDerate * 0.2; // 0° tilt
  
  // Improvement percentage
  const improvementPercent = currentEnergy > 0 ? 
    ((optimalEnergy - currentEnergy) / currentEnergy) * 100 : 0;
  
  // Generate tilt curve
  const tiltCurve: MockTiltResult['mlOptimization']['tiltCurve'] = [];
  for (let t = 0; t <= 60; t += 5) {
    const tRad = t * Math.PI / 180;
    const poa = ghi * (Math.sin(elevRad + tRad) / Math.sin(Math.max(0.01, elevRad)));
    const energy = poa * tempDerate * 0.2;
    tiltCurve.push({
      tilt: t,
      effective_irradiance: parseFloat(poa.toFixed(1)),
      net_energy: parseFloat(energy.toFixed(2)),
    });
  }
  
  // Efficiency calculations
  const efficiencyLoss = 100 - (currentEnergy / optimalEnergy * 100);
  const efficiencyRetained = 100 - efficiencyLoss;

  return {
    optimizationId: `mock_tilt_${Date.now()}`,
    source: 'mock-fallback',
    location: {
      latitude: lat,
      longitude: 0,
    },
    currentSettings: {
      tilt: currentTilt,
      azimuth: 180,
    },
    mlOptimization: {
      optimalTilt: parseFloat(optimalTilt.toFixed(1)),
      estimatedEnergy: parseFloat(optimalEnergy.toFixed(2)),
      solarElevation: parseFloat(Math.max(0, elevation).toFixed(1)),
      efficiencyLoss: parseFloat(Math.max(0, efficiencyLoss).toFixed(2)),
      efficiencyRetained: parseFloat(Math.min(100, efficiencyRetained).toFixed(2)),
      improvementPercent: parseFloat(Math.max(0, improvementPercent).toFixed(2)),
      defaultTilt: 0,
      defaultEnergy: parseFloat(defaultEnergy.toFixed(2)),
      tiltCurve,
    },
    conditions: {
      ghi,
      latitude: lat,
      hour,
      temperature: temp,
    },
    recommendations: {
      mlOptimalTilt: parseFloat(optimalTilt.toFixed(1)),
      annualOptimalTilt: parseFloat(lat.toFixed(1)),
      seasonalOptimalTilt: parseFloat((lat - declination * 0.5).toFixed(1)),
      optimalAzimuth: lat >= 0 ? 180 : 0, // South in NH, North in SH
      adjustmentNeeded: Math.abs(currentTilt - optimalTilt) > 5,
    },
  };
}

// ==================== Dashboard Metrics Mock ====================

export interface MockDashboardMetrics {
  timestamp: string;
  platformMetrics: {
    totalInstallations: number;
    totalPanels: number;
    predictionsToday: number;
    apiUptime: number;
    avgResponseTime: number;
    arrValue: number;
  };
  customerValue: {
    totalSavings: number;
    energyOptimized: number;
    co2Offset: number;
    defectsDetected: number;
    alertsSent: number;
  };
  modelPerformance: {
    efficiencyModel: {
      accuracy: number;
      avgPredictionTime: number;
      predictionsLast24h: number;
    };
    classifierModel: {
      accuracy: number;
      avgPredictionTime: number;
      imagesProcessed: number;
    };
    tiltOptimizer: {
      avgImprovement: number;
      optimizationsRun: number;
    };
  };
  recentPredictions: Array<{
    id: string;
    type: 'efficiency' | 'classification' | 'tilt';
    timestamp: string;
    result: string;
    confidence: number;
  }>;
}

/**
 * Generates realistic dashboard metrics for demo/fallback scenarios
 */
export function mockGetDashboardMetrics(): MockDashboardMetrics {
  const now = new Date();
  
  // Generate realistic recent predictions
  const recentPredictions = [];
  for (let i = 0; i < 10; i++) {
    const types = ['efficiency', 'classification', 'tilt'] as const;
    const type = types[Math.floor(Math.random() * types.length)];
    const timestamp = new Date(now.getTime() - i * 60000);
    
    let result: string;
    let confidence: number;
    
    switch (type) {
      case 'efficiency':
        const eff = 75 + Math.random() * 20;
        result = `${eff.toFixed(1)}% efficiency`;
        confidence = 92 + Math.random() * 7;
        break;
      case 'classification':
        const isNormal = Math.random() > 0.15;
        result = isNormal ? 'NORMAL' : 'DEFECTIVE';
        confidence = 85 + Math.random() * 13;
        break;
      case 'tilt':
        const optTilt = 25 + Math.random() * 20;
        result = `Optimal: ${optTilt.toFixed(0)}°`;
        confidence = 88 + Math.random() * 10;
        break;
    }
    
    recentPredictions.push({
      id: `pred_${Date.now()}_${i}`,
      type,
      timestamp: timestamp.toISOString(),
      result,
      confidence: parseFloat(confidence.toFixed(1)),
    });
  }

  return {
    timestamp: now.toISOString(),
    platformMetrics: {
      totalInstallations: 2847,
      totalPanels: 1284729,
      predictionsToday: 4823156,
      apiUptime: 99.97,
      avgResponseTime: 47,
      arrValue: 101200000,
    },
    customerValue: {
      totalSavings: 127400000,
      energyOptimized: 847230,
      co2Offset: 423847,
      defectsDetected: 12847,
      alertsSent: 3291,
    },
    modelPerformance: {
      efficiencyModel: {
        accuracy: 94.7,
        avgPredictionTime: 23,
        predictionsLast24h: 3847291,
      },
      classifierModel: {
        accuracy: 97.2,
        avgPredictionTime: 156,
        imagesProcessed: 284719,
      },
      tiltOptimizer: {
        avgImprovement: 8.3,
        optimizationsRun: 691147,
      },
    },
    recentPredictions,
  };
}

// ==================== Report Data Mock ====================

export interface MockReportData {
  id: string;
  generatedAt: string;
  reportType: 'efficiency' | 'degradation' | 'defects' | 'optimization' | 'financial';
  title: string;
  summary: {
    period: string;
    keyMetrics: Record<string, string | number>;
    insights: string[];
    recommendations: string[];
  };
  data: {
    timeSeries?: Array<{ date: string; value: number }>;
    distribution?: Array<{ category: string; count: number; percentage: number }>;
    topIssues?: Array<{ issue: string; severity: string; count: number }>;
  };
}

/**
 * Generates mock report data for different report types
 */
export function mockGenerateReport(reportType: MockReportData['reportType']): MockReportData {
  const now = new Date();
  const monthNames = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December'];
  
  // Generate time series data for the last 30 days
  const timeSeries = [];
  for (let i = 29; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    let value: number;
    switch (reportType) {
      case 'efficiency':
        value = 88 + Math.random() * 8 - Math.random() * 3;
        break;
      case 'degradation':
        value = 0.3 + (i * 0.002) + Math.random() * 0.05;
        break;
      case 'defects':
        value = Math.floor(5 + Math.random() * 10);
        break;
      case 'optimization':
        value = 5 + Math.random() * 10;
        break;
      case 'financial':
        value = 150000 + Math.random() * 50000;
        break;
      default:
        value = Math.random() * 100;
    }
    timeSeries.push({
      date: date.toISOString().split('T')[0],
      value: parseFloat(value.toFixed(2)),
    });
  }

  const reportConfigs: Record<MockReportData['reportType'], Omit<MockReportData, 'id' | 'generatedAt' | 'reportType' | 'data'>> = {
    efficiency: {
      title: `Monthly Efficiency Analysis - ${monthNames[now.getMonth()]} ${now.getFullYear()}`,
      summary: {
        period: `${monthNames[now.getMonth()]} 1-${now.getDate()}, ${now.getFullYear()}`,
        keyMetrics: {
          'Average Efficiency': '91.4%',
          'Peak Efficiency': '96.8%',
          'Lowest Efficiency': '78.2%',
          'Efficiency Trend': '+1.2%',
          'Predictions Made': '4.2M',
          'Model Accuracy': '94.7%',
        },
        insights: [
          'Overall system efficiency improved by 1.2% compared to last month',
          'Temperature-related losses decreased due to improved cooling',
          'Soiling losses increased by 0.8% - cleaning recommended',
          'Peak performance hours shifted 15 minutes earlier due to seasonal change',
        ],
        recommendations: [
          'Schedule comprehensive cleaning for panels in Zone C showing elevated soiling losses',
          'Consider tilt adjustment for installations below 85% efficiency',
          'Upgrade monitoring sensors in 23 installations showing data gaps',
        ],
      },
    },
    degradation: {
      title: `Panel Degradation Forecast - ${monthNames[now.getMonth()]} ${now.getFullYear()}`,
      summary: {
        period: 'Next 12 Months Projection',
        keyMetrics: {
          'Current Avg Degradation': '0.42%/year',
          'Projected 12-Month Loss': '0.45%',
          'Panels at Risk': '127',
          'Critical Panels': '8',
          'Healthy Panels': '98.7%',
          'Model Confidence': '96.1%',
        },
        insights: [
          'Degradation rate within industry standards (0.3-0.8%/year)',
          '8 panels showing accelerated degradation require immediate attention',
          'Humidity-related stress factor elevated in coastal installations',
          'Electrical stress decreasing following voltage regulator upgrades',
        ],
        recommendations: [
          'Replace 8 critically degraded panels within 30 days',
          'Implement enhanced moisture protection for 47 coastal panels',
          'Schedule mid-year inspection for 127 at-risk panels',
        ],
      },
    },
    defects: {
      title: `Defect Detection Report - ${monthNames[now.getMonth()]} ${now.getFullYear()}`,
      summary: {
        period: `${monthNames[now.getMonth()]} 1-${now.getDate()}, ${now.getFullYear()}`,
        keyMetrics: {
          'Total Panels Scanned': '284,719',
          'Defects Detected': '847',
          'Defect Rate': '0.30%',
          'Critical Defects': '23',
          'Model Accuracy': '97.2%',
          'False Positive Rate': '2.1%',
        },
        insights: [
          'Defect rate below industry average of 0.5%',
          'Hot spot defects increased 15% - seasonal pattern expected',
          'Micro-crack detection improved following model update',
          'Cell degradation defects stable month-over-month',
        ],
        recommendations: [
          'Priority replacement for 23 panels with critical defects',
          'Enhanced thermal imaging for hot spot prone installations',
          'Update classification model with new training data by Q2',
        ],
      },
    },
    optimization: {
      title: `Tilt Optimization Report - ${monthNames[now.getMonth()]} ${now.getFullYear()}`,
      summary: {
        period: `${monthNames[now.getMonth()]} 1-${now.getDate()}, ${now.getFullYear()}`,
        keyMetrics: {
          'Optimizations Performed': '691,147',
          'Average Improvement': '+8.3%',
          'Energy Gained': '2.4 GWh',
          'Cost Savings': '$312,000',
          'Panels Adjusted': '45,291',
          'Model Accuracy': '93.8%',
        },
        insights: [
          'Seasonal tilt adjustment yielded 8.3% average improvement',
          'ML-driven optimization outperformed static tilt by 12%',
          'Morning hour optimizations most effective (6-9 AM)',
          'Cloud cover prediction integration improved results by 3.2%',
        ],
        recommendations: [
          'Implement 2-axis tracking for top 100 highest-value installations',
          'Adjust seasonal tilt schedule for winter optimization',
          'Deploy real-time tilt adjustment for 500 priority sites',
        ],
      },
    },
    financial: {
      title: `Financial Impact Report - ${monthNames[now.getMonth()]} ${now.getFullYear()}`,
      summary: {
        period: `${monthNames[now.getMonth()]} ${now.getFullYear()}`,
        keyMetrics: {
          'Monthly ARR': '$8.43M',
          'Customer Savings': '$127.4M',
          'Energy Value Captured': '$42.3M',
          'Maintenance Avoided': '$8.7M',
          'Carbon Credits': '$2.1M',
          'ROI': '847%',
        },
        insights: [
          'Platform ROI exceeds industry benchmarks by 340%',
          'ML predictions saved customers $8.7M in avoided maintenance',
          'Early defect detection prevented $2.3M in panel replacements',
          'Optimization recommendations generated $4.1M in additional revenue',
        ],
        recommendations: [
          'Expand enterprise tier to capture $2.4M additional ARR opportunity',
          'Launch carbon credit monetization service for customers',
          'Introduce predictive maintenance SLA tier at premium pricing',
        ],
      },
    },
  };

  const config = reportConfigs[reportType];

  return {
    id: `report_${reportType}_${Date.now()}`,
    generatedAt: now.toISOString(),
    reportType,
    title: config.title,
    summary: config.summary,
    data: {
      timeSeries,
      distribution: [
        { category: 'Optimal', count: 2341, percentage: 78.2 },
        { category: 'Good', count: 412, percentage: 13.8 },
        { category: 'Fair', count: 189, percentage: 6.3 },
        { category: 'Poor', count: 47, percentage: 1.6 },
        { category: 'Critical', count: 3, percentage: 0.1 },
      ],
      topIssues: [
        { issue: 'Temperature stress', severity: 'medium', count: 234 },
        { issue: 'Soiling accumulation', severity: 'low', count: 189 },
        { issue: 'Suboptimal tilt angle', severity: 'low', count: 156 },
        { issue: 'Electrical degradation', severity: 'high', count: 47 },
        { issue: 'Physical damage', severity: 'critical', count: 12 },
      ],
    },
  };
}

// ==================== Export All ====================

export const mockData = {
  predictEfficiency: mockPredictEfficiency,
  predictDegradation: mockPredictDegradation,
  classifyPanel: mockClassifyPanel,
  optimizeTilt: mockOptimizeTilt,
  getDashboardMetrics: mockGetDashboardMetrics,
  generateReport: mockGenerateReport,
};

export default mockData;
