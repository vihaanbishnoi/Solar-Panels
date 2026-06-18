const { v4: uuidv4 } = require('uuid');
const fs = require('fs');
const path = require('path');

// Simulated upload tracking
let uploads = [];

// POST /api/upload/csv
exports.uploadCSV = (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'No file uploaded' });
    }

    const uploadId = uuidv4();
    const upload = {
      id: uploadId,
      filename: req.file.originalname,
      storedAs: req.file.filename,
      size: req.file.size,
      mimetype: req.file.mimetype,
      status: 'processing',
      uploadedAt: new Date().toISOString(),
      processedAt: null,
      rowsProcessed: 0,
      errors: [],
      mlValidation: {
        status: 'pending',
        anomaliesDetected: 0,
        dataQualityScore: null
      }
    };

    uploads.push(upload);

    // Simulate async processing
    setTimeout(() => {
      const index = uploads.findIndex(u => u.id === uploadId);
      if (index !== -1) {
        uploads[index].status = 'completed';
        uploads[index].processedAt = new Date().toISOString();
        uploads[index].rowsProcessed = Math.floor(Math.random() * 10000) + 500;
        uploads[index].mlValidation = {
          status: 'completed',
          anomaliesDetected: Math.floor(Math.random() * 5),
          dataQualityScore: (85 + Math.random() * 15).toFixed(1)
        };
      }
    }, 3000);

    res.status(202).json({ 
      success: true, 
      data: {
        uploadId,
        message: 'File uploaded successfully. Processing started.',
        statusEndpoint: `/api/upload/status/${uploadId}`
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// POST /api/upload/json
exports.uploadJSON = (req, res) => {
  try {
    const { data, dataType } = req.body;

    if (!data || !Array.isArray(data)) {
      return res.status(400).json({ success: false, error: 'Invalid data format. Expected array.' });
    }

    const uploadId = uuidv4();
    const upload = {
      id: uploadId,
      type: 'json',
      dataType: dataType || 'unknown',
      recordCount: data.length,
      status: 'processing',
      uploadedAt: new Date().toISOString(),
      processedAt: null
    };

    uploads.push(upload);

    // Simulate processing
    setTimeout(() => {
      const index = uploads.findIndex(u => u.id === uploadId);
      if (index !== -1) {
        uploads[index].status = 'completed';
        uploads[index].processedAt = new Date().toISOString();
      }
    }, 1500);

    res.status(202).json({ 
      success: true, 
      data: {
        uploadId,
        recordsReceived: data.length,
        message: 'JSON data received. Processing started.'
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/upload/status/:id
exports.getUploadStatus = (req, res) => {
  try {
    const upload = uploads.find(u => u.id === req.params.id);
    
    if (!upload) {
      return res.status(404).json({ success: false, error: 'Upload not found' });
    }

    res.json({ success: true, data: upload });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/upload/history
exports.getUploadHistory = (req, res) => {
  try {
    const { limit = 20, status } = req.query;
    let history = [...uploads].reverse();
    
    if (status) {
      history = history.filter(u => u.status === status);
    }

    res.json({ 
      success: true, 
      data: history.slice(0, parseInt(limit)),
      total: history.length
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};
