const express = require('express');
const router = express.Router();
const alertController = require('../controllers/alert.controller');

// GET /api/alerts - Get all alerts
router.get('/', alertController.getAllAlerts);

// GET /api/alerts/active - Get only active/unresolved alerts
router.get('/active', alertController.getActiveAlerts);

// PUT /api/alerts/:id/acknowledge - Acknowledge an alert
router.put('/:id/acknowledge', alertController.acknowledgeAlert);

// PUT /api/alerts/:id/resolve - Resolve an alert
router.put('/:id/resolve', alertController.resolveAlert);

// GET /api/alerts/thresholds - Get alert threshold configurations
router.get('/thresholds', alertController.getThresholds);

// PUT /api/alerts/thresholds - Update alert thresholds
router.put('/thresholds', alertController.updateThresholds);

module.exports = router;
