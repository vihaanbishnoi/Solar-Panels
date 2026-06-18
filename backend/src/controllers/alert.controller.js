const { v4: uuidv4 } = require('uuid');

// Simulated alerts
let alerts = [
  {
    id: 'alert-001',
    type: 'warning',
    category: 'efficiency',
    title: 'Efficiency Drop Detected',
    description: 'Panel group B2 showing 5% deviation from expected output',
    panelId: 'panel-002',
    threshold: { metric: 'efficiency', operator: 'below', value: 90 },
    currentValue: 85.2,
    createdAt: '2026-01-20T08:30:00Z',
    acknowledged: false,
    acknowledgedAt: null,
    resolved: false,
    resolvedAt: null,
    priority: 'medium'
  },
  {
    id: 'alert-002',
    type: 'info',
    category: 'maintenance',
    title: 'Cleaning Scheduled',
    description: 'Automated cleaning sequence will initiate in 2 days',
    panelId: null,
    createdAt: '2026-01-20T05:00:00Z',
    acknowledged: true,
    acknowledgedAt: '2026-01-20T06:15:00Z',
    resolved: false,
    resolvedAt: null,
    priority: 'low'
  },
  {
    id: 'alert-003',
    type: 'success',
    category: 'system',
    title: 'Inverter Sync Complete',
    description: 'Main inverter successfully synchronized with grid',
    panelId: null,
    createdAt: '2026-01-19T18:00:00Z',
    acknowledged: true,
    acknowledgedAt: '2026-01-19T18:05:00Z',
    resolved: true,
    resolvedAt: '2026-01-19T18:05:00Z',
    priority: 'low'
  }
];

// Alert thresholds configuration
let thresholds = {
  efficiency: {
    warning: 90,
    critical: 80,
    unit: '%'
  },
  degradation: {
    warning: 0.05,
    critical: 0.1,
    unit: '% per month'
  },
  temperature: {
    warning: 45,
    critical: 55,
    unit: '°C'
  },
  soiling: {
    warning: 10,
    critical: 20,
    unit: '% coverage'
  },
  voltage: {
    warningLow: 350,
    warningHigh: 450,
    criticalLow: 300,
    criticalHigh: 500,
    unit: 'V'
  }
};

// GET /api/alerts
exports.getAllAlerts = (req, res) => {
  try {
    const { type, category, limit = 50 } = req.query;
    let filteredAlerts = [...alerts];
    
    if (type) {
      filteredAlerts = filteredAlerts.filter(a => a.type === type);
    }
    if (category) {
      filteredAlerts = filteredAlerts.filter(a => a.category === category);
    }

    res.json({ 
      success: true, 
      data: filteredAlerts.slice(0, parseInt(limit)),
      total: filteredAlerts.length,
      summary: {
        critical: alerts.filter(a => a.priority === 'critical' && !a.resolved).length,
        warning: alerts.filter(a => a.priority === 'medium' && !a.resolved).length,
        info: alerts.filter(a => a.priority === 'low' && !a.resolved).length
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/alerts/active
exports.getActiveAlerts = (req, res) => {
  try {
    const activeAlerts = alerts.filter(a => !a.resolved);
    
    res.json({ 
      success: true, 
      data: activeAlerts,
      count: activeAlerts.length
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// PUT /api/alerts/:id/acknowledge
exports.acknowledgeAlert = (req, res) => {
  try {
    const alert = alerts.find(a => a.id === req.params.id);
    
    if (!alert) {
      return res.status(404).json({ success: false, error: 'Alert not found' });
    }

    alert.acknowledged = true;
    alert.acknowledgedAt = new Date().toISOString();

    res.json({ 
      success: true, 
      data: alert,
      message: 'Alert acknowledged'
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// PUT /api/alerts/:id/resolve
exports.resolveAlert = (req, res) => {
  try {
    const { resolution } = req.body;
    const alert = alerts.find(a => a.id === req.params.id);
    
    if (!alert) {
      return res.status(404).json({ success: false, error: 'Alert not found' });
    }

    alert.resolved = true;
    alert.resolvedAt = new Date().toISOString();
    alert.resolution = resolution || 'Manually resolved';

    res.json({ 
      success: true, 
      data: alert,
      message: 'Alert resolved'
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/alerts/thresholds
exports.getThresholds = (req, res) => {
  try {
    res.json({ success: true, data: thresholds });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// PUT /api/alerts/thresholds
exports.updateThresholds = (req, res) => {
  try {
    const updates = req.body;
    
    for (const [key, value] of Object.entries(updates)) {
      if (thresholds[key]) {
        thresholds[key] = { ...thresholds[key], ...value };
      }
    }

    res.json({ 
      success: true, 
      data: thresholds,
      message: 'Thresholds updated successfully'
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};
