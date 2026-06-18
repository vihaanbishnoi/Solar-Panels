const express = require('express');
const router = express.Router();
const panelController = require('../controllers/panel.controller');

// GET /api/panels - Get all panels
router.get('/', panelController.getAllPanels);

// GET /api/panels/:id - Get single panel details
router.get('/:id', panelController.getPanelById);

// PUT /api/panels/:id/config - Update panel configuration
router.put('/:id/config', panelController.updatePanelConfig);

// GET /api/panels/:id/efficiency - Get panel efficiency data
router.get('/:id/efficiency', panelController.getPanelEfficiency);

// POST /api/panels/:id/tilt - Adjust panel tilt angle
router.post('/:id/tilt', panelController.adjustTilt);

// GET /api/panels/:id/degradation - Get degradation forecast
router.get('/:id/degradation', panelController.getDegradationForecast);

module.exports = router;
