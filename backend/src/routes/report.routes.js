const express = require('express');
const router = express.Router();
const reportController = require('../controllers/report.controller');

// GET /api/reports - Get all generated reports
router.get('/', reportController.getAllReports);

// GET /api/reports/:id - Get specific report
router.get('/:id', reportController.getReportById);

// POST /api/reports/generate - Generate new report
router.post('/generate', reportController.generateReport);

// GET /api/reports/:id/download - Download report file
router.get('/:id/download', reportController.downloadReport);

// DELETE /api/reports/:id - Delete a report
router.delete('/:id', reportController.deleteReport);

// GET /api/reports/templates - Get available report templates
router.get('/templates/list', reportController.getTemplates);

module.exports = router;
