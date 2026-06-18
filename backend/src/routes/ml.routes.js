const express = require('express');
const router = express.Router();
const multer = require('multer');
const mlController = require('../controllers/ml.controller');

// Configure multer for image uploads
const imageStorage = multer.memoryStorage();
const imageUpload = multer({ 
  storage: imageStorage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit for images
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  }
});

// POST /api/ml/analyze - General ML analysis endpoint
router.post('/analyze', mlController.analyzeData);

// ========== IMAGE CLASSIFICATION (Real PyTorch Model) ==========
// POST /api/ml/image/classify-panel - Classify single panel image for defects
router.post('/image/classify-panel', imageUpload.single('image'), mlController.classifyPanelDefect);

// POST /api/ml/image/classify-panel/batch - Batch classification
router.post('/image/classify-panel/batch', imageUpload.array('images', 20), mlController.classifyPanelDefectBatch);

// ========== LEGACY IMAGE ENDPOINTS (Simulated) ==========
// POST /api/ml/image/defect-detection - Detect defects in panel images
router.post('/image/defect-detection', imageUpload.single('image'), mlController.detectDefects);

// POST /api/ml/image/hotspot-detection - Detect hotspots via thermal imaging
router.post('/image/hotspot-detection', imageUpload.single('image'), mlController.detectHotspots);

// POST /api/ml/image/soiling-analysis - Analyze dust/soiling levels
router.post('/image/soiling-analysis', imageUpload.single('image'), mlController.analyzeSoiling);

// POST /api/ml/predict/degradation - Predict degradation over time
router.post('/predict/degradation', mlController.predictDegradation);

// POST /api/ml/predict/efficiency - Predict efficiency based on conditions
router.post('/predict/efficiency', mlController.predictEfficiency);

// POST /api/ml/predict/cleaning - Recommend cleaning schedule
router.post('/predict/cleaning', mlController.predictCleaningSchedule);

// ========== TILT OPTIMIZATION (ML-Powered) ==========
// POST /api/ml/optimize/tilt - Optimize tilt angle using ML model
router.post('/optimize/tilt', mlController.optimizeTilt);

// GET /api/ml/optimize/tilt/quick - Quick tilt optimization
router.get('/optimize/tilt/quick', mlController.quickTiltOptimize);

// GET /api/ml/models/status - Check ML model service status
router.get('/models/status', mlController.getModelStatus);

// ========== EMAIL ALERTS ==========
// POST /api/ml/email/test - Test email configuration
router.post('/email/test', mlController.testEmailConfig);

// GET /api/ml/email/status - Check email service status
router.get('/email/status', mlController.getEmailStatus);

module.exports = router;
