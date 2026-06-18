const { v4: uuidv4 } = require('uuid');

// Simulated reports store
let reports = [
  {
    id: 'rpt-001',
    title: 'Monthly Efficiency Analysis',
    type: 'efficiency',
    format: 'PDF',
    size: '2.4 MB',
    createdAt: '2026-01-15T10:30:00Z',
    period: { start: '2025-12-01', end: '2025-12-31' },
    status: 'ready',
    downloadCount: 12
  },
  {
    id: 'rpt-002',
    title: 'Degradation Forecast Model',
    type: 'degradation',
    format: 'CSV',
    size: '15.1 MB',
    createdAt: '2026-01-10T14:20:00Z',
    period: { start: '2024-01-01', end: '2025-12-31' },
    status: 'ready',
    downloadCount: 8
  },
  {
    id: 'rpt-003',
    title: 'Maintenance & Cleaning Schedule',
    type: 'maintenance',
    format: 'PDF',
    size: '1.2 MB',
    createdAt: '2025-12-20T09:00:00Z',
    period: { start: '2026-01-01', end: '2026-06-30' },
    status: 'ready',
    downloadCount: 25
  },
  {
    id: 'rpt-004',
    title: 'ROI Projection 2026-2030',
    type: 'financial',
    format: 'XLSX',
    size: '3.8 MB',
    createdAt: '2025-12-18T16:45:00Z',
    period: { start: '2026-01-01', end: '2030-12-31' },
    status: 'ready',
    downloadCount: 31
  }
];

const reportTemplates = [
  { id: 'tpl-efficiency', name: 'Efficiency Analysis', description: 'Detailed efficiency metrics and loss breakdown' },
  { id: 'tpl-degradation', name: 'Degradation Forecast', description: 'ML-powered degradation predictions' },
  { id: 'tpl-maintenance', name: 'Maintenance Schedule', description: 'Recommended maintenance and cleaning dates' },
  { id: 'tpl-financial', name: 'Financial ROI', description: 'Revenue projections and savings analysis' },
  { id: 'tpl-alerts', name: 'Alert Summary', description: 'Historical alert data and resolutions' }
];

// GET /api/reports
exports.getAllReports = (req, res) => {
  try {
    const { type, format, limit = 50 } = req.query;
    let filteredReports = [...reports];
    
    if (type) {
      filteredReports = filteredReports.filter(r => r.type === type);
    }
    if (format) {
      filteredReports = filteredReports.filter(r => r.format.toLowerCase() === format.toLowerCase());
    }

    res.json({ 
      success: true, 
      data: filteredReports.slice(0, parseInt(limit)),
      total: filteredReports.length
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/reports/:id
exports.getReportById = (req, res) => {
  try {
    const report = reports.find(r => r.id === req.params.id);
    
    if (!report) {
      return res.status(404).json({ success: false, error: 'Report not found' });
    }

    res.json({ success: true, data: report });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// POST /api/reports/generate
exports.generateReport = (req, res) => {
  try {
    const { templateId, title, period, format = 'PDF' } = req.body;

    if (!templateId || !period) {
      return res.status(400).json({ 
        success: false, 
        error: 'templateId and period are required' 
      });
    }

    const template = reportTemplates.find(t => t.id === templateId);
    if (!template) {
      return res.status(400).json({ success: false, error: 'Invalid template' });
    }

    const reportId = uuidv4();
    const newReport = {
      id: reportId,
      title: title || template.name,
      type: templateId.replace('tpl-', ''),
      format,
      size: 'Generating...',
      createdAt: new Date().toISOString(),
      period,
      status: 'generating',
      downloadCount: 0
    };

    reports.unshift(newReport);

    // Simulate report generation
    setTimeout(() => {
      const index = reports.findIndex(r => r.id === reportId);
      if (index !== -1) {
        reports[index].status = 'ready';
        reports[index].size = (Math.random() * 10 + 0.5).toFixed(1) + ' MB';
      }
    }, 5000);

    res.status(202).json({ 
      success: true, 
      data: {
        reportId,
        message: 'Report generation started',
        estimatedTime: '5-10 seconds'
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/reports/:id/download
exports.downloadReport = (req, res) => {
  try {
    const report = reports.find(r => r.id === req.params.id);
    
    if (!report) {
      return res.status(404).json({ success: false, error: 'Report not found' });
    }

    if (report.status !== 'ready') {
      return res.status(400).json({ success: false, error: 'Report is not ready for download' });
    }

    // Increment download count
    report.downloadCount++;

    // In production, this would stream the actual file
    res.json({ 
      success: true, 
      data: {
        downloadUrl: `/files/reports/${report.id}.${report.format.toLowerCase()}`,
        expiresIn: 3600,
        message: 'Simulated download URL (file generation not implemented)'
      }
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// DELETE /api/reports/:id
exports.deleteReport = (req, res) => {
  try {
    const index = reports.findIndex(r => r.id === req.params.id);
    
    if (index === -1) {
      return res.status(404).json({ success: false, error: 'Report not found' });
    }

    reports.splice(index, 1);
    
    res.json({ success: true, message: 'Report deleted successfully' });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// GET /api/reports/templates/list
exports.getTemplates = (req, res) => {
  try {
    res.json({ success: true, data: reportTemplates });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};
