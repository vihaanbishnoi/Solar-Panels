const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

// Import routes
const dashboardRoutes = require('./routes/dashboard.routes');
const panelRoutes = require('./routes/panel.routes');
const uploadRoutes = require('./routes/upload.routes');
const reportRoutes = require('./routes/report.routes');
const alertRoutes = require('./routes/alert.routes');
const mlRoutes = require('./routes/ml.routes');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:5173',
  credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Health check
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    services: {
      database: 'simulated',
      mlModels: 'pending_integration'
    }
  });
});

// API Routes
app.use('/api/dashboard', dashboardRoutes);
app.use('/api/panels', panelRoutes);
app.use('/api/upload', uploadRoutes);
app.use('/api/reports', reportRoutes);
app.use('/api/alerts', alertRoutes);
app.use('/api/ml', mlRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ 
    error: 'Internal Server Error',
    message: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

app.listen(PORT, () => {
  console.log(`
  ╔═══════════════════════════════════════════════════╗
  ║                                                   ║
  ║     🌞 SolarAI Backend Server                     ║
  ║     Running on http://localhost:${PORT}             ║
  ║                                                   ║
  ║     API Endpoints:                                ║
  ║     • GET  /api/health                            ║
  ║     • GET  /api/dashboard/stats                   ║
  ║     • GET  /api/panels                            ║
  ║     • POST /api/upload/csv                        ║
  ║     • GET  /api/reports                           ║
  ║     • GET  /api/alerts                            ║
  ║     • POST /api/ml/analyze                        ║
  ║                                                   ║
  ╚═══════════════════════════════════════════════════╝
  `);
});

module.exports = app;
