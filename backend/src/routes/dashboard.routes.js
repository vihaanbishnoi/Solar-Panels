const express = require('express');
const router = express.Router();
const dashboardController = require('../controllers/dashboard.controller');

// GET /api/dashboard/stats - Get all dashboard statistics
router.get('/stats', dashboardController.getStats);

// GET /api/dashboard/energy - Get energy production data
router.get('/energy', dashboardController.getEnergyProduction);

// GET /api/dashboard/performance - Get performance metrics over time
router.get('/performance', dashboardController.getPerformanceMetrics);

// GET /api/dashboard/summary - Get quick summary for widgets
router.get('/summary', dashboardController.getSummary);

module.exports = router;
