const { v4: uuidv4 } = require('uuid');
const mlService = require('../services/mlService');
const emailService = require('../services/emailService');

/**
 * ML Controller
 * Handles all machine learning related API endpoints
 * Connected to Flask ML API for real predictions
 */

// ML Model status tracking
const modelStatus = {
  defectDetection: {
    name: 'Defect Detection CNN',
    status: 'simulated',
    version: '1.0.0-sim',
    accuracy: 94.5,
    lastTrained: '2025-12-15',
    endpoint: null
  },
  hotspotDetection: {
    name: 'Thermal Hotspot Detector',
    status: 'simulated',
    version: '1.0.0-sim',
    accuracy: 92.8,
    lastTrained: '2025-12-10',
    endpoint: null
  },
  soilingAnalysis: {
    name: 'Soiling Level Analyzer',
    status: 'simulated',
    version: '1.0.0-sim',
    accuracy: 89.2,
    lastTrained: '2025-11-28',
    endpoint: null
  },
  degradationPredictor: {
    name: 'Degradation LSTM Model',
    status: 'simulated',
    version: '2.0.0-sim',
    accuracy: 91.5,
    lastTrained: '2026-01-05',
    endpoint: null
  },
  efficiencyPredictor: {
    name: 'Efficiency Regression Model',
    status: 'simulated',
    version: '1.5.0-sim',
    accuracy: 93.1,
    lastTrained: '2026-01-10',
    endpoint: null
  },
  tiltOptimizer: {
    name: 'Tilt Angle Optimizer',
    status: 'simulated',
    version: '1.0.0-sim',
    accuracy: 96.2,
    lastTrained: '2025-12-20',
    endpoint: null
  }
};

// POST /api/ml/analyze
exports.analyzeData = (req, res) => {
  try {
    const { data, analysisType } = req.body;

    if (!data) {
      return res.status(400).json({ success: false, error: 'No data provided for analysis' });
    }

    // Simulate ML analysis
    const result = {
      analysisId: uuidv4(),
      type: analysisType || 'general',
      timestamp: new Date().toISOString(),
      results: {
        anomaliesDetected: Math.floor(Math.random() * 5),
        dataQuality: (85 + Math.random() * 15).toFixed(1),
        patterns: [
          { type: 'seasonal', confidence: 0.89, description: 'Detected seasonal production pattern' },
          { type: 'daily', confidence: 0.95, description: 'Normal daily production curve' }
        ],
        recommendations: [
          'Consider adjusting tilt angle for winter months',
          'Schedule cleaning within next 14 days'
        ]
      },
      processingTime: (Math.random() * 2 + 0.5).toFixed(2) + 's',
      modelUsed: 'ensemble-analyzer-v1'
    };

    res.json({ success: true, data: result });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// POST /api/ml/image/defect-detection
exports.detectDefects = (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No image provided' });
    }

    // Simulate defect detection
    const defects = [];
    const numDefects = Math.floor(Math.random() * 3);
    
    const defectTypes = ['crack', 'hotspot', 'discoloration', 'delamination', 'snail_trail'];
    const severities = ['low', 'medium', 'high'];

    for (let i = 0; i < numDefects; i++) {
      defects.push({
        id: uuidv4(),
        type: defectTypes[Math.floor(Math.random() * defectTypes.length)],
        severity: severities[Math.floor(Math.random() * severities.length)],
        confidence: (0.75 + Math.random() * 0.24).toFixed(2),
        boundingBox: {
          x: Math.floor(Math.random() * 800),
          y: Math.floor(Math.random() * 600),
          width: Math.floor(Math.random() * 100) + 20,
          height: Math.floor(Math.random() * 100) + 20
        },
        recommendation: 'Schedule inspection within 30 days'
      });
    }

    res.json({
      success: true,
      data: {
        analysisId: uuidv4(),
        imageSize: req.file.size,
        defectsFound: defects.length,
        defects,
        overallHealth: defects.length === 0 ? 'excellent' : defects.some(d => d.severity === 'high') ? 'poor' : 'fair',
        processingTime: (Math.random() * 3 + 1).toFixed(2) + 's',
        modelVersion: modelStatus.defectDetection.version
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// POST /api/ml/image/classify-panel - Uses real PyTorch model
exports.classifyPanelDefect = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No image provided' });
    }

    // Call the ML service to classify the image
    const result = await mlService.classifyPanelImage(req.file.buffer, req.file.originalname);
    
    if (!result.success) {
      return res.status(500).json({ success: false, error: result.error || 'Classification failed' });
    }

    const analysisId = uuidv4();
    const responseData = {
      analysisId,
      filename: req.file.originalname,
      fileSize: req.file.size,
      prediction: result.prediction,
      probabilities: result.probabilities,
      analysis: result.analysis,
      source: result.source,
      modelInfo: result.model_info || {
        architecture: 'ResNet18',
        classes: ['NORMAL', 'DEFECTIVE']
      },
      timestamp: new Date().toISOString()
    };

    // 🚨 SEND EMAIL ALERT IF DEFECTIVE PANEL DETECTED
    if (result.prediction.label === 'DEFECTIVE') {
      const emailResult = await emailService.sendDefectAlert({
        recipientEmail: req.body.alertEmail || process.env.ALERT_RECIPIENTS,
        panelId: analysisId.substring(0, 8).toUpperCase(),
        prediction: result.prediction.label,
        confidence: result.prediction.confidence,
        severity: result.analysis.severity,
        recommendation: result.analysis.recommendation,
        filename: req.file.originalname,
        imageBase64: req.file.buffer.toString('base64')
      });

      responseData.emailAlert = {
        sent: emailResult.success,
        messageId: emailResult.messageId || null,
        error: emailResult.error || null
      };

      if (emailResult.success) {
        console.log(`📧 Alert email sent for defective panel: ${analysisId}`);
      }
    }

    res.json({
      success: true,
      data: responseData
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// POST /api/ml/image/classify-panel/batch - Batch classification
exports.classifyPanelDefectBatch = async (req, res) => {
  try {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ success: false, error: 'No images provided' });
    }

    const images = req.files.map(file => ({
      buffer: file.buffer,
      filename: file.originalname
    }));

    const result = await mlService.classifyPanelImagesBatch(images);
    
    // 🚨 SEND EMAIL ALERTS FOR ALL DEFECTIVE PANELS IN BATCH
    const defectiveResults = result.results.filter(r => r.prediction?.label === 'DEFECTIVE');
    const emailAlerts = [];

    if (defectiveResults.length > 0) {
      const alertRecipient = req.body.alertEmail || process.env.ALERT_RECIPIENTS;
      
      for (const defect of defectiveResults) {
        const matchingFile = req.files.find(f => f.originalname === defect.filename);
        
        const emailResult = await emailService.sendDefectAlert({
          recipientEmail: alertRecipient,
          panelId: `BATCH-${defect.filename.substring(0, 8).toUpperCase()}`,
          prediction: defect.prediction.label,
          confidence: defect.prediction.confidence,
          severity: defect.analysis?.severity || 'medium',
          recommendation: defect.analysis?.recommendation || 'Inspect panel for defects',
          filename: defect.filename,
          imageBase64: matchingFile ? matchingFile.buffer.toString('base64') : null
        });

        emailAlerts.push({
          filename: defect.filename,
          sent: emailResult.success,
          messageId: emailResult.messageId || null
        });
      }

      console.log(`📧 Sent ${emailAlerts.filter(e => e.sent).length}/${defectiveResults.length} alert emails for batch`);
    }

    res.json({
      success: true,
      data: {
        batchId: uuidv4(),
        totalProcessed: result.total_processed,
        summary: result.summary,
        results: result.results,
        source: result.source,
        emailAlerts: {
          defectivePanels: defectiveResults.length,
          alertsSent: emailAlerts.filter(e => e.sent).length,
          details: emailAlerts
        },
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// POST /api/ml/image/hotspot-detection
exports.detectHotspots = (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No thermal image provided' });
    }

    const hotspots = [];
    const numHotspots = Math.floor(Math.random() * 4);

    for (let i = 0; i < numHotspots; i++) {
      hotspots.push({
        id: uuidv4(),
        temperature: (45 + Math.random() * 25).toFixed(1),
        deltaFromNormal: (5 + Math.random() * 15).toFixed(1),
        location: {
          x: Math.floor(Math.random() * 800),
          y: Math.floor(Math.random() * 600)
        },
        radius: Math.floor(Math.random() * 30) + 10,
        severity: Math.random() > 0.7 ? 'critical' : Math.random() > 0.4 ? 'warning' : 'normal'
      });
    }

    res.json({
      success: true,
      data: {
        analysisId: uuidv4(),
        hotspotsDetected: hotspots.length,
        hotspots,
        averageTemperature: (35 + Math.random() * 10).toFixed(1),
        maxTemperature: hotspots.length > 0 ? Math.max(...hotspots.map(h => parseFloat(h.temperature))).toFixed(1) : '35.0',
        recommendation: hotspots.some(h => h.severity === 'critical') 
          ? 'Immediate inspection required' 
          : 'No immediate action required',
        modelVersion: modelStatus.hotspotDetection.version
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// POST /api/ml/image/soiling-analysis
exports.analyzeSoiling = (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No image provided' });
    }

    const soilingLevel = Math.random() * 25;
    const regions = [];
    const numRegions = Math.floor(Math.random() * 5) + 1;

    for (let i = 0; i < numRegions; i++) {
      regions.push({
        id: `region-${i + 1}`,
        soilingPercentage: (Math.random() * 30).toFixed(1),
        type: ['dust', 'bird_droppings', 'pollen', 'leaves'][Math.floor(Math.random() * 4)],
        area: {
          x: Math.floor(Math.random() * 800),
          y: Math.floor(Math.random() * 600),
          width: Math.floor(Math.random() * 200) + 50,
          height: Math.floor(Math.random() * 200) + 50
        }
      });
    }

    res.json({
      success: true,
      data: {
        analysisId: uuidv4(),
        overallSoilingLevel: soilingLevel.toFixed(1),
        category: soilingLevel < 5 ? 'clean' : soilingLevel < 15 ? 'light' : soilingLevel < 25 ? 'moderate' : 'heavy',
        regions,
        estimatedEfficiencyLoss: (soilingLevel * 0.15).toFixed(2),
        cleaningRecommended: soilingLevel > 10,
        daysUntilCritical: Math.max(1, Math.floor((25 - soilingLevel) / 2)),
        modelVersion: modelStatus.soilingAnalysis.version
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// POST /api/ml/predict/degradation
exports.predictDegradation = async (req, res) => {
  try {
    const { panelId, historicalData, projectionMonths = 60, temperature, humidity, voltageMax, voltageMin, currentMax, currentMin, daysSinceInstallation } = req.body;

    // Use real ML model for degradation prediction
    const mlResult = await mlService.predictDegradation({
      temperature: temperature || 30,
      humidity: humidity || 60,
      voltage_max: voltageMax || 40,
      voltage_min: voltageMin || 35,
      current_max: currentMax || 10,
      current_min: currentMin || 8,
      days_since_installation: daysSinceInstallation || 365
    });

    const currentEfficiency = 94 + Math.random() * 4;
    const monthlyDegradation = 0.03 + Math.random() * 0.04;
    
    const predictions = [];
    for (let month = 1; month <= projectionMonths; month++) {
      predictions.push({
        month,
        date: new Date(Date.now() + month * 30 * 86400000).toISOString().split('T')[0],
        predictedEfficiency: Math.max(70, currentEfficiency - (month * monthlyDegradation)).toFixed(2),
        confidenceInterval: {
          lower: Math.max(65, currentEfficiency - (month * monthlyDegradation) - 2).toFixed(2),
          upper: Math.min(100, currentEfficiency - (month * monthlyDegradation) + 2).toFixed(2)
        }
      });
    }

    res.json({
      success: true,
      data: {
        predictionId: uuidv4(),
        panelId: panelId || 'all',
        // Real ML model results
        degradationIndex: mlResult.degradation_index,
        degradationStatus: mlResult.status,
        stressFactors: mlResult.stress_factors,
        mlRecommendation: mlResult.recommendation,
        source: mlResult.source,
        // Projection data
        currentEfficiency: currentEfficiency.toFixed(2),
        monthlyDegradationRate: monthlyDegradation.toFixed(4),
        yearlyDegradationRate: (monthlyDegradation * 12).toFixed(2),
        predictions,
        endOfLifeEstimate: {
          year: Math.floor(projectionMonths / 12) + new Date().getFullYear(),
          efficiency: predictions[predictions.length - 1].predictedEfficiency
        },
        factors: [
          { name: 'Temperature Cycling', impact: 'medium', contribution: '35%' },
          { name: 'UV Exposure', impact: 'high', contribution: '40%' },
          { name: 'Humidity', impact: 'low', contribution: '15%' },
          { name: 'Other', impact: 'low', contribution: '10%' }
        ],
        modelVersion: modelStatus.degradationPredictor.version
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// POST /api/ml/predict/efficiency
exports.predictEfficiency = async (req, res) => {
  try {
    const { 
      tilt, 
      azimuth, 
      temperature, 
      humidity,
      windSpeed,
      irradiance,
      voltage,
      current,
      daysSinceInstallation,
      cloudCover, 
      soilingLevel,
      latitude,
      date 
    } = req.body;

    // Use ML model for efficiency loss prediction with all parameters
    const mlResult = await mlService.predictEfficiencyLoss({
      temperature: temperature || 30,
      humidity: humidity || 60,
      wind_speed: windSpeed || 5,
      irradiance: irradiance || 500,
      voltage: voltage || 35,
      current: current || 8,
      days_since_installation: daysSinceInstallation || 365
    });

    // Additional geometry-based factors for tilt/cloud/soiling
    const tiltFactor = tilt !== undefined ? Math.cos((Math.abs(tilt - 30) * Math.PI) / 180) : 1;
    const cloudFactor = cloudCover !== undefined ? 1 - (cloudCover / 100) * 0.3 : 1;
    const soilingFactor = soilingLevel !== undefined ? 1 - (soilingLevel / 100) * 0.2 : 1;

    // Use ML predicted efficiency, then apply additional factors
    const mlEfficiency = parseFloat(mlResult.predicted_efficiency || 85);
    const finalEfficiency = (mlEfficiency * tiltFactor * cloudFactor * soilingFactor).toFixed(1);

    res.json({
      success: true,
      data: {
        predictionId: uuidv4(),
        // ML model results - dynamic based on inputs
        efficiencyLoss: mlResult.efficiency_loss,
        efficiencyLossPercent: mlResult.efficiency_loss_percent,
        panelStatus: mlResult.status,
        mlRecommendation: mlResult.recommendation,
        source: mlResult.source,
        modelConfidence: mlResult.model_confidence,
        // Final calculated efficiency with all factors
        predictedEfficiency: finalEfficiency,
        theoreticalMaximum: 98.0,
        // Detailed factor breakdown from ML
        factors: {
          tiltImpact: ((1 - tiltFactor) * 100).toFixed(2) + '% loss',
          temperatureImpact: mlResult.factors?.temperature_impact || '0.00% loss',
          irradianceImpact: mlResult.factors?.irradiance_impact || '0.00% loss',
          cloudImpact: ((1 - cloudFactor) * 100).toFixed(2) + '% loss',
          soilingImpact: ((1 - soilingFactor) * 100).toFixed(2) + '% loss',
          humidityImpact: mlResult.factors?.humidity_impact || '0.00%',
          windBenefit: mlResult.factors?.wind_benefit || '0.00%',
          ageDegradation: mlResult.factors?.age_degradation || '0.00%'
        },
        recommendations: [
          tiltFactor < 0.95 ? `Adjust tilt to 30° for optimal angle` : null,
          soilingFactor < 0.95 ? 'Schedule cleaning to improve output' : null,
          mlResult.recommendation
        ].filter(Boolean),
        modelVersion: modelStatus.efficiencyPredictor.version,
        timestamp: mlResult.timestamp || new Date().toISOString()
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// POST /api/ml/predict/cleaning
exports.predictCleaningSchedule = (req, res) => {
  try {
    const { panelIds, currentSoilingLevel, environmentData } = req.body;

    const soilingRate = 0.5 + Math.random() * 0.5; // % per day
    const currentLevel = currentSoilingLevel || Math.random() * 10;
    const threshold = 15; // Cleaning threshold %
    const daysUntilCleaning = Math.max(1, Math.floor((threshold - currentLevel) / soilingRate));
    
    const nextCleaningDate = new Date(Date.now() + daysUntilCleaning * 86400000);

    res.json({
      success: true,
      data: {
        predictionId: uuidv4(),
        currentSoilingLevel: currentLevel.toFixed(1),
        soilingRatePerDay: soilingRate.toFixed(2),
        cleaningThreshold: threshold,
        daysUntilCleaning,
        recommendedCleaningDate: nextCleaningDate.toISOString().split('T')[0],
        estimatedEfficiencyGain: (currentLevel * 0.12).toFixed(2),
        schedule: [
          { 
            date: nextCleaningDate.toISOString().split('T')[0], 
            type: 'routine',
            priority: 'normal'
          },
          { 
            date: new Date(nextCleaningDate.getTime() + 30 * 86400000).toISOString().split('T')[0], 
            type: 'routine',
            priority: 'normal'
          },
          { 
            date: new Date(nextCleaningDate.getTime() + 60 * 86400000).toISOString().split('T')[0], 
            type: 'routine',
            priority: 'normal'
          }
        ],
        costBenefitAnalysis: {
          cleaningCost: 50,
          estimatedRecovery: (currentLevel * 0.12 * 5).toFixed(2),
          netBenefit: ((currentLevel * 0.12 * 5) - 50).toFixed(2),
          currency: 'USD'
        }
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// POST /api/ml/optimize/tilt - Uses real ML model for tilt optimization
exports.optimizeTilt = async (req, res) => {
  try {
    const { 
      latitude, 
      longitude, 
      ghi,
      hour,
      temperature,
      humidity,
      windSpeed,
      voltage,
      current,
      daysSinceInstallation,
      currentTilt, 
      currentAzimuth,
      tiltMin,
      tiltMax
    } = req.body;

    const lat = latitude || 28.6139; // Default to Delhi
    const currentHour = hour || new Date().getHours();
    const irradiance = ghi || 800;

    // Call ML service for optimization
    const result = await mlService.optimizeTiltAngle({
      ghi: irradiance,
      latitude: lat,
      hour: currentHour,
      temperature: temperature || 25,
      humidity: humidity || 50,
      windSpeed: windSpeed || 2,
      voltage: voltage || 38,
      current: current || 8,
      daysSinceInstallation: daysSinceInstallation || 365,
      tiltMin: tiltMin || 0,
      tiltMax: tiltMax || 60
    });

    if (!result.success) {
      return res.status(500).json({ success: false, error: result.error || 'Optimization failed' });
    }

    const optimization = result.optimization;
    
    // Seasonal calculations for additional context
    const optimalAnnualTilt = Math.abs(lat);
    const optimalSummerTilt = Math.max(0, Math.abs(lat) - 15);
    const optimalWinterTilt = Math.min(90, Math.abs(lat) + 15);
    const currentMonth = new Date().getMonth();
    const isSummer = currentMonth >= 3 && currentMonth <= 8;
    const seasonalOptimal = isSummer ? optimalSummerTilt : optimalWinterTilt;

    res.json({
      success: true,
      data: {
        optimizationId: uuidv4(),
        source: result.source,
        location: { latitude: lat, longitude: longitude || 77.2090 },
        currentSettings: {
          tilt: currentTilt || 30,
          azimuth: currentAzimuth || 180
        },
        mlOptimization: {
          optimalTilt: optimization.optimal_tilt,
          estimatedEnergy: optimization.estimated_energy,
          solarElevation: optimization.solar_elevation,
          efficiencyLoss: optimization.efficiency_loss,
          efficiencyRetained: optimization.efficiency_retained,
          improvementPercent: optimization.improvement_percent,
          defaultTilt: optimization.default_tilt,
          defaultEnergy: optimization.default_energy,
          tiltCurve: optimization.tilt_curve
        },
        conditions: optimization.conditions,
        recommendations: {
          mlOptimalTilt: optimization.optimal_tilt,
          annualOptimalTilt: parseFloat(optimalAnnualTilt.toFixed(1)),
          seasonalOptimalTilt: parseFloat(seasonalOptimal.toFixed(1)),
          optimalAzimuth: lat >= 0 ? 180 : 0,
          adjustmentNeeded: Math.abs((currentTilt || 30) - optimization.optimal_tilt) > 5
        },
        seasonalSchedule: [
          { months: 'Mar-Aug', recommendedTilt: optimalSummerTilt.toFixed(1) },
          { months: 'Sep-Feb', recommendedTilt: optimalWinterTilt.toFixed(1) }
        ],
        estimatedGain: {
          withMLOptimization: `${Math.max(0, optimization.improvement_percent).toFixed(1)}%`,
          withTracking: '25-35%'
        },
        modelVersion: result.source === 'ml-model' ? '2.0.0-ml' : modelStatus.tiltOptimizer.version
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/ml/optimize/tilt/quick - Quick tilt optimization
exports.quickTiltOptimize = async (req, res) => {
  try {
    const result = await mlService.quickTiltOptimize({
      ghi: parseFloat(req.query.ghi) || 800,
      latitude: parseFloat(req.query.latitude) || 28.6,
      hour: parseFloat(req.query.hour) || 12,
      temperature: parseFloat(req.query.temperature) || 25
    });

    res.json({
      success: true,
      source: result.source,
      data: {
        optimalTilt: result.optimal_tilt,
        estimatedEnergy: result.estimated_energy,
        improvementPercent: result.improvement_percent,
        solarElevation: result.solar_elevation
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/ml/models/status
exports.getModelStatus = async (req, res) => {
  try {
    // Check real ML API health
    const mlHealth = await mlService.checkHealth();
    const mlModelInfo = await mlService.getModelInfo();

    res.json({
      success: true,
      data: {
        overallStatus: mlHealth.connected ? 'connected' : 'simulated',
        message: mlHealth.connected 
          ? 'ML API connected - using real model predictions'
          : 'ML API offline - using simulation fallback',
        mlApiStatus: mlHealth,
        mlModelInfo: mlModelInfo,
        mlApiUrl: mlService.ML_API_URL,
        models: modelStatus,
        integrationGuide: {
          step1: 'Start Flask ML API: python ml_api.py',
          step2: 'ML API runs on http://localhost:5001',
          step3: 'Predictions now use trained RandomForest model'
        }
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// POST /api/ml/email/test - Test email configuration
exports.testEmailConfig = async (req, res) => {
  try {
    const { testEmail } = req.body;

    if (!testEmail) {
      return res.status(400).json({ 
        success: false, 
        error: 'Please provide testEmail in request body' 
      });
    }

    const result = await emailService.testEmailConfiguration(testEmail);

    if (result.success) {
      res.json({
        success: true,
        message: 'Test email sent successfully!',
        data: {
          recipient: testEmail,
          messageId: result.messageId,
          timestamp: new Date().toISOString()
        }
      });
    } else {
      res.status(500).json({
        success: false,
        error: result.error,
        hint: 'Check SMTP_USER and SMTP_PASS in your .env file'
      });
    }
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/ml/email/status - Check email service status
exports.getEmailStatus = (req, res) => {
  try {
    const config = emailService.EMAIL_CONFIG;
    const isConfigured = !!(config.auth.user && config.auth.pass);

    res.json({
      success: true,
      data: {
        configured: isConfigured,
        host: config.host,
        port: config.port,
        secure: config.secure,
        user: config.auth.user ? `${config.auth.user.substring(0, 5)}...` : 'Not set',
        alertRecipients: process.env.ALERT_RECIPIENTS ? 'Configured' : 'Not set',
        status: isConfigured ? 'ready' : 'not_configured',
        message: isConfigured 
          ? 'Email service is configured and ready to send alerts'
          : 'Email service not configured. Set SMTP_USER and SMTP_PASS in .env file',
        requiredEnvVars: [
          'SMTP_HOST (default: smtp.gmail.com)',
          'SMTP_PORT (default: 587)',
          'SMTP_USER (your email)',
          'SMTP_PASS (app password)',
          'ALERT_RECIPIENTS (comma-separated emails)'
        ]
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};
