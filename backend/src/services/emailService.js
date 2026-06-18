/**
 * Email Service - NodeMailer Integration
 * Sends alert emails when defective solar panels are detected
 */

const nodemailer = require('nodemailer');

// Email configuration from environment variables
const EMAIL_CONFIG = {
  host: process.env.SMTP_HOST || 'smtp.gmail.com',
  port: parseInt(process.env.SMTP_PORT) || 587,
  secure: process.env.SMTP_SECURE === 'true', // true for 465, false for other ports
  auth: {
    user: process.env.SMTP_USER || '',
    pass: process.env.SMTP_PASS || '' // App password for Gmail
  }
};

// Default recipients for alerts
const ALERT_RECIPIENTS = process.env.ALERT_RECIPIENTS || '';

// Create reusable transporter
let transporter = null;

/**
 * Initialize the email transporter
 */
function initializeTransporter() {
  if (!EMAIL_CONFIG.auth.user || !EMAIL_CONFIG.auth.pass) {
    console.log('⚠️  Email service not configured. Set SMTP_USER and SMTP_PASS in .env');
    return null;
  }

  transporter = nodemailer.createTransport(EMAIL_CONFIG);
  
  // Verify connection
  transporter.verify((error, success) => {
    if (error) {
      console.error('❌ Email service connection failed:', error.message);
    } else {
      console.log('✅ Email service ready to send alerts');
    }
  });

  return transporter;
}

/**
 * Send a defect detection alert email
 * @param {Object} options - Alert options
 * @param {string} options.recipientEmail - Email to send to (optional, uses default if not provided)
 * @param {string} options.panelId - Panel identifier
 * @param {string} options.prediction - DEFECTIVE or NORMAL
 * @param {number} options.confidence - Confidence percentage
 * @param {string} options.severity - none, medium, high, critical
 * @param {string} options.recommendation - Action recommendation
 * @param {string} options.filename - Original image filename
 * @param {string} options.imageBase64 - Base64 encoded image (optional)
 * @returns {Promise<Object>} Send result
 */
async function sendDefectAlert(options) {
  if (!transporter) {
    return {
      success: false,
      error: 'Email service not initialized. Check SMTP configuration.'
    };
  }

  const {
    recipientEmail,
    panelId = 'Unknown',
    prediction,
    confidence,
    severity,
    recommendation,
    filename = 'panel_image.jpg',
    imageBase64 = null
  } = options;

  const recipient = recipientEmail || ALERT_RECIPIENTS;
  
  if (!recipient) {
    return {
      success: false,
      error: 'No recipient email configured'
    };
  }

  // Determine severity color and emoji
  const severityConfig = {
    critical: { color: '#DC2626', emoji: '🚨', label: 'CRITICAL' },
    high: { color: '#EA580C', emoji: '⚠️', label: 'HIGH PRIORITY' },
    medium: { color: '#CA8A04', emoji: '⚡', label: 'MEDIUM PRIORITY' },
    none: { color: '#16A34A', emoji: '✅', label: 'LOW PRIORITY' }
  };

  const sev = severityConfig[severity] || severityConfig.medium;
  const timestamp = new Date().toLocaleString('en-US', {
    timeZone: 'Asia/Kolkata',
    dateStyle: 'full',
    timeStyle: 'medium'
  });

  // Build HTML email
  const htmlContent = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f3f4f6;">
      <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); border-radius: 12px 12px 0 0; padding: 30px; text-align: center;">
          <h1 style="color: white; margin: 0; font-size: 24px;">🌞 SolarAI Alert System</h1>
          <p style="color: rgba(255,255,255,0.8); margin: 10px 0 0 0;">Panel Defect Detection Notification</p>
        </div>
        
        <!-- Alert Badge -->
        <div style="background-color: ${sev.color}; padding: 15px; text-align: center;">
          <span style="color: white; font-size: 18px; font-weight: bold;">
            ${sev.emoji} ${sev.label} - DEFECTIVE PANEL DETECTED
          </span>
        </div>
        
        <!-- Content -->
        <div style="background-color: white; padding: 30px; border-radius: 0 0 12px 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
          <!-- Summary Box -->
          <div style="background-color: #fef2f2; border-left: 4px solid ${sev.color}; padding: 15px; margin-bottom: 20px; border-radius: 0 8px 8px 0;">
            <h3 style="margin: 0 0 10px 0; color: #991b1b;">Detection Summary</h3>
            <p style="margin: 0; color: #7f1d1d;">
              Our AI system has detected a <strong>potentially defective solar panel</strong> that requires attention.
            </p>
          </div>
          
          <!-- Details Table -->
          <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
            <tr>
              <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280; width: 40%;">📋 Panel ID</td>
              <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; font-weight: 600;">${panelId}</td>
            </tr>
            <tr>
              <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">🔍 Classification</td>
              <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                <span style="background-color: #fef2f2; color: #dc2626; padding: 4px 12px; border-radius: 20px; font-weight: 600;">
                  ${prediction}
                </span>
              </td>
            </tr>
            <tr>
              <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">📊 Confidence</td>
              <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; font-weight: 600;">${confidence}%</td>
            </tr>
            <tr>
              <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">⚡ Severity</td>
              <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">
                <span style="background-color: ${sev.color}; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600;">
                  ${severity.toUpperCase()}
                </span>
              </td>
            </tr>
            <tr>
              <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">📁 Image File</td>
              <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">${filename}</td>
            </tr>
            <tr>
              <td style="padding: 12px; border-bottom: 1px solid #e5e7eb; color: #6b7280;">🕐 Detected At</td>
              <td style="padding: 12px; border-bottom: 1px solid #e5e7eb;">${timestamp}</td>
            </tr>
          </table>
          
          <!-- Recommendation Box -->
          <div style="background-color: #eff6ff; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #1e40af;">💡 Recommended Action</h4>
            <p style="margin: 0; color: #1e3a8a;">${recommendation}</p>
          </div>
          
          <!-- CTA Button -->
          <div style="text-align: center; margin: 25px 0;">
            <a href="#" style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-weight: 600; display: inline-block;">
              View in Dashboard →
            </a>
          </div>
          
          <!-- Footer -->
          <div style="border-top: 1px solid #e5e7eb; padding-top: 20px; margin-top: 20px; text-align: center; color: #9ca3af; font-size: 12px;">
            <p style="margin: 0;">This is an automated alert from SolarAI Panel Monitoring System</p>
            <p style="margin: 5px 0 0 0;">Powered by ResNet18 Deep Learning Model</p>
          </div>
        </div>
      </div>
    </body>
    </html>
  `;

  // Plain text fallback
  const textContent = `
🌞 SolarAI Alert - Defective Panel Detected

${sev.emoji} ${sev.label}

Panel ID: ${panelId}
Classification: ${prediction}
Confidence: ${confidence}%
Severity: ${severity.toUpperCase()}
Image File: ${filename}
Detected At: ${timestamp}

Recommended Action:
${recommendation}

---
This is an automated alert from SolarAI Panel Monitoring System
  `.trim();

  // Build attachments if image provided
  const attachments = [];
  if (imageBase64) {
    attachments.push({
      filename: filename,
      content: imageBase64,
      encoding: 'base64',
      cid: 'panelImage'
    });
  }

  try {
    const result = await transporter.sendMail({
      from: `"SolarAI Alerts" <${EMAIL_CONFIG.auth.user}>`,
      to: recipient,
      subject: `${sev.emoji} [${sev.label}] Defective Solar Panel Detected - ${panelId}`,
      text: textContent,
      html: htmlContent,
      attachments
    });

    console.log(`📧 Alert email sent to ${recipient} (Message ID: ${result.messageId})`);
    
    return {
      success: true,
      messageId: result.messageId,
      recipient
    };
  } catch (error) {
    console.error('❌ Failed to send alert email:', error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Send a daily summary email
 * @param {Object} options - Summary options
 * @param {string} options.recipientEmail - Email to send to
 * @param {number} options.totalAnalyzed - Total panels analyzed
 * @param {number} options.defectiveCount - Number of defective panels
 * @param {number} options.normalCount - Number of normal panels
 * @param {Array} options.criticalAlerts - List of critical alerts
 * @returns {Promise<Object>} Send result
 */
async function sendDailySummary(options) {
  if (!transporter) {
    return {
      success: false,
      error: 'Email service not initialized'
    };
  }

  const {
    recipientEmail,
    totalAnalyzed = 0,
    defectiveCount = 0,
    normalCount = 0,
    criticalAlerts = []
  } = options;

  const recipient = recipientEmail || ALERT_RECIPIENTS;
  const defectRate = totalAnalyzed > 0 ? ((defectiveCount / totalAnalyzed) * 100).toFixed(1) : 0;
  const date = new Date().toLocaleDateString('en-US', { dateStyle: 'full' });

  const htmlContent = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
    </head>
    <body style="margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f3f4f6;">
      <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); border-radius: 12px 12px 0 0; padding: 30px; text-align: center;">
          <h1 style="color: white; margin: 0;">📊 Daily Analysis Report</h1>
          <p style="color: rgba(255,255,255,0.8); margin: 10px 0 0 0;">${date}</p>
        </div>
        
        <div style="background-color: white; padding: 30px; border-radius: 0 0 12px 12px;">
          <!-- Stats Grid -->
          <div style="display: flex; gap: 15px; margin-bottom: 20px;">
            <div style="flex: 1; background-color: #f0fdf4; padding: 20px; border-radius: 8px; text-align: center;">
              <div style="font-size: 32px; font-weight: bold; color: #16a34a;">${normalCount}</div>
              <div style="color: #166534;">Normal</div>
            </div>
            <div style="flex: 1; background-color: #fef2f2; padding: 20px; border-radius: 8px; text-align: center;">
              <div style="font-size: 32px; font-weight: bold; color: #dc2626;">${defectiveCount}</div>
              <div style="color: #991b1b;">Defective</div>
            </div>
            <div style="flex: 1; background-color: #eff6ff; padding: 20px; border-radius: 8px; text-align: center;">
              <div style="font-size: 32px; font-weight: bold; color: #2563eb;">${totalAnalyzed}</div>
              <div style="color: #1e40af;">Total</div>
            </div>
          </div>
          
          <p style="text-align: center; color: #6b7280;">
            Defect Rate: <strong>${defectRate}%</strong>
          </p>
          
          ${criticalAlerts.length > 0 ? `
            <div style="background-color: #fef2f2; border-radius: 8px; padding: 15px; margin-top: 20px;">
              <h3 style="margin: 0 0 10px 0; color: #991b1b;">🚨 Critical Alerts (${criticalAlerts.length})</h3>
              <ul style="margin: 0; padding-left: 20px; color: #7f1d1d;">
                ${criticalAlerts.map(alert => `<li>${alert.panelId} - ${alert.confidence}% confidence</li>`).join('')}
              </ul>
            </div>
          ` : ''}
        </div>
      </div>
    </body>
    </html>
  `;

  try {
    const result = await transporter.sendMail({
      from: `"SolarAI Reports" <${EMAIL_CONFIG.auth.user}>`,
      to: recipient,
      subject: `📊 SolarAI Daily Report - ${defectiveCount} Defects Detected`,
      html: htmlContent
    });

    return { success: true, messageId: result.messageId };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

/**
 * Test email configuration
 * @param {string} testEmail - Email to send test to
 * @returns {Promise<Object>} Test result
 */
async function testEmailConfiguration(testEmail) {
  if (!transporter) {
    initializeTransporter();
  }
  
  if (!transporter) {
    return {
      success: false,
      error: 'Email service not configured. Check SMTP credentials.'
    };
  }

  try {
    const result = await transporter.sendMail({
      from: `"SolarAI Test" <${EMAIL_CONFIG.auth.user}>`,
      to: testEmail,
      subject: '✅ SolarAI Email Configuration Test',
      html: `
        <div style="font-family: sans-serif; padding: 20px;">
          <h2>🌞 Email Configuration Successful!</h2>
          <p>Your SolarAI alert system is now configured to send email notifications.</p>
          <p>You will receive alerts when defective solar panels are detected.</p>
          <hr>
          <p style="color: #666; font-size: 12px;">Sent at ${new Date().toISOString()}</p>
        </div>
      `
    });

    return {
      success: true,
      message: 'Test email sent successfully',
      messageId: result.messageId
    };
  } catch (error) {
    return {
      success: false,
      error: error.message
    };
  }
}

// Initialize on module load
initializeTransporter();

module.exports = {
  initializeTransporter,
  sendDefectAlert,
  sendDailySummary,
  testEmailConfiguration,
  EMAIL_CONFIG
};
