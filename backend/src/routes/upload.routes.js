const express = require('express');
const router = express.Router();
const multer = require('multer');
const uploadController = require('../controllers/upload.controller');

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + '.csv');
  }
});

const upload = multer({ 
  storage: storage,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB limit
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'text/csv' || file.originalname.endsWith('.csv') || file.originalname.endsWith('.json')) {
      cb(null, true);
    } else {
      cb(new Error('Only CSV and JSON files are allowed'));
    }
  }
});

// POST /api/upload/csv - Upload CSV file for processing
router.post('/csv', upload.single('file'), uploadController.uploadCSV);

// POST /api/upload/json - Upload JSON data
router.post('/json', uploadController.uploadJSON);

// GET /api/upload/status/:id - Check upload processing status
router.get('/status/:id', uploadController.getUploadStatus);

// GET /api/upload/history - Get upload history
router.get('/history', uploadController.getUploadHistory);

module.exports = router;
