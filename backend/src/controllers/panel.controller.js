const { v4: uuidv4 } = require('uuid');

// Simulated panels data
let panels = [
  {
    id: 'panel-001',
    name: 'Panel A1',
    group: 'Array A',
    position: { x: 0, y: 0, z: 0 },
    config: {
      width: 1.7,
      height: 1.0,
      thickness: 0.04,
      tilt: 30,
      azimuth: 180
    },
    status: 'active',
    efficiency: 94.5,
    installDate: '2024-06-15',
    lastInspection: '2025-12-01',
    degradation: {
      rate: 0.04,
      cumulative: 1.2
    }
  },
  {
    id: 'panel-002',
    name: 'Panel A2',
    group: 'Array A',
    position: { x: 2, y: 0, z: 0 },
    config: {
      width: 1.7,
      height: 1.0,
      thickness: 0.04,
      tilt: 30,
      azimuth: 180
    },
    status: 'active',
    efficiency: 93.8,
    installDate: '2024-06-15',
    lastInspection: '2025-12-01',
    degradation: {
      rate: 0.05,
      cumulative: 1.5
    }
  }
];

// GET /api/panels
exports.getAllPanels = (req, res) => {
  try {
    const { status, group } = req.query;
    let filteredPanels = [...panels];
    
    if (status) {
      filteredPanels = filteredPanels.filter(p => p.status === status);
    }
    if (group) {
      filteredPanels = filteredPanels.filter(p => p.group === group);
    }

    res.json({ 
      success: true, 
      data: filteredPanels,
      count: filteredPanels.length 
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/panels/:id
exports.getPanelById = (req, res) => {
  try {
    const panel = panels.find(p => p.id === req.params.id);
    
    if (!panel) {
      return res.status(404).json({ success: false, error: 'Panel not found' });
    }

    res.json({ success: true, data: panel });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// PUT /api/panels/:id/config
exports.updatePanelConfig = (req, res) => {
  try {
    const panelIndex = panels.findIndex(p => p.id === req.params.id);
    
    if (panelIndex === -1) {
      return res.status(404).json({ success: false, error: 'Panel not found' });
    }

    const { tilt, azimuth, width, height } = req.body;
    
    panels[panelIndex].config = {
      ...panels[panelIndex].config,
      ...(tilt !== undefined && { tilt }),
      ...(azimuth !== undefined && { azimuth }),
      ...(width !== undefined && { width }),
      ...(height !== undefined && { height })
    };

    res.json({ 
      success: true, 
      data: panels[panelIndex],
      message: 'Panel configuration updated successfully'
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/panels/:id/efficiency
exports.getPanelEfficiency = (req, res) => {
  try {
    const panel = panels.find(p => p.id === req.params.id);
    
    if (!panel) {
      return res.status(404).json({ success: false, error: 'Panel not found' });
    }

    // Simulate efficiency data over time
    const efficiencyData = {
      current: panel.efficiency,
      optimal: 98.0,
      losses: {
        incidenceAngle: 2.8,
        temperature: 1.5,
        soiling: 1.2,
        aging: panel.degradation.cumulative
      },
      history: Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - (29 - i) * 86400000).toISOString().split('T')[0],
        efficiency: (93 + Math.random() * 3).toFixed(1)
      }))
    };

    res.json({ success: true, data: efficiencyData });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// POST /api/panels/:id/tilt
exports.adjustTilt = (req, res) => {
  try {
    const { tilt, reason } = req.body;
    const panelIndex = panels.findIndex(p => p.id === req.params.id);
    
    if (panelIndex === -1) {
      return res.status(404).json({ success: false, error: 'Panel not found' });
    }

    if (tilt < 0 || tilt > 90) {
      return res.status(400).json({ success: false, error: 'Tilt must be between 0 and 90 degrees' });
    }

    const previousTilt = panels[panelIndex].config.tilt;
    panels[panelIndex].config.tilt = tilt;

    res.json({ 
      success: true, 
      data: {
        panelId: req.params.id,
        previousTilt,
        newTilt: tilt,
        reason: reason || 'Manual adjustment',
        adjustedAt: new Date().toISOString()
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/panels/:id/degradation
exports.getDegradationForecast = (req, res) => {
  try {
    const panel = panels.find(p => p.id === req.params.id);
    
    if (!panel) {
      return res.status(404).json({ success: false, error: 'Panel not found' });
    }

    const currentEfficiency = panel.efficiency;
    const monthlyRate = panel.degradation.rate;
    
    // Project degradation over 5 years
    const forecast = Array.from({ length: 60 }, (_, month) => ({
      month: month + 1,
      date: new Date(Date.now() + month * 30 * 86400000).toISOString().split('T')[0],
      projectedEfficiency: Math.max(70, currentEfficiency - (month * monthlyRate)).toFixed(1),
      confidence: month < 12 ? 95 : month < 36 ? 85 : 75
    }));

    res.json({ 
      success: true, 
      data: {
        panelId: panel.id,
        currentEfficiency,
        degradationRate: monthlyRate,
        cumulativeDegradation: panel.degradation.cumulative,
        forecast,
        recommendation: monthlyRate > 0.05 
          ? 'Above-average degradation detected. Consider inspection.'
          : 'Degradation within normal parameters.'
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};
