const { v4: uuidv4 } = require('uuid');


exports.getStats = (req, res) => {
  try {
    const stats = {
      totalEnergyGeneration: {
        value: 1.2,
        unit: 'MWh',
        change: 12.5,
        trend: 'up',
        period: 'today'
      },
      systemEfficiency: {
        value: 94.2,
        unit: '%',
        change: -0.8,
        trend: 'down',
        baseline: 95.0
      },
      activePanels: {
        active: 48,
        total: 48,
        percentage: 100
      },
      batteryStatus: {
        level: 87,
        unit: '%',
        estimatedRuntime: 45,
        runtimeUnit: 'minutes',
        charging: true
      },
      weatherConditions: {
        temperature: 28,
        humidity: 45,
        cloudCover: 15,
        uvIndex: 7,
        sunlightHours: 8.5
      },
      lastUpdated: new Date().toISOString()
    };

    res.json({ success: true, data: stats });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/dashboard/energy
exports.getEnergyProduction = (req, res) => {
  try {
    const { period = '24h' } = req.query;
    
    // Simulate hourly energy data
    const hours = period === '24h' ? 24 : period === '7d' ? 168 : 720;
    const data = [];
    const now = new Date();
    
    for (let i = hours; i >= 0; i--) {
      const timestamp = new Date(now - i * 3600000);
      const hour = timestamp.getHours();
      
      // Simulate solar production pattern (peak at noon)
      let production = 0;
      if (hour >= 6 && hour <= 18) {
        const peakFactor = 1 - Math.abs(12 - hour) / 6;
        production = (4.5 * peakFactor * (0.8 + Math.random() * 0.4)).toFixed(2);
      }
      
      data.push({
        timestamp: timestamp.toISOString(),
        production: parseFloat(production),
        unit: 'kWh'
      });
    }

    res.json({ 
      success: true, 
      data: {
        period,
        readings: data,
        totalProduction: data.reduce((sum, d) => sum + d.production, 0).toFixed(2),
        peakProduction: Math.max(...data.map(d => d.production)).toFixed(2)
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/dashboard/performance
exports.getPerformanceMetrics = (req, res) => {
  try {
    const metrics = {
      efficiency: {
        current: 94.2,
        average30d: 93.8,
        optimal: 98.0,
        degradationRate: 0.05 // % per month
      },
      losses: {
        incidenceAngle: 3.2,
        thermalDrift: 2.1,
        soiling: 1.4,
        cabling: 0.8,
        inverter: 1.2,
        total: 8.7
      },
      uptime: {
        current: 99.8,
        average30d: 99.5,
        downtime: {
          planned: 2,
          unplanned: 0.5,
          unit: 'hours'
        }
      },
      alerts: {
        critical: 0,
        warnings: 2,
        info: 5
      }
    };

    res.json({ success: true, data: metrics });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/dashboard/summary
exports.getSummary = (req, res) => {
  try {
    const summary = {
      status: 'operational',
      health: 'good',
      lastMaintenance: '2026-01-10T08:00:00Z',
      nextScheduledMaintenance: '2026-02-10T08:00:00Z',
      nextRecommendedCleaning: '2026-02-03T06:00:00Z',
      estimatedCleaningGain: 3.2,
      financials: {
        todayRevenue: 45.60,
        monthRevenue: 1250.80,
        currency: 'USD',
        savingsVsGrid: 890.50
      },
      carbonOffset: {
        today: 0.8,
        month: 22.5,
        unit: 'tons CO2'
      }
    };

    res.json({ success: true, data: summary });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};
