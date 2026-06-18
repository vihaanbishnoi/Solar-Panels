import React, { useState, useEffect, useCallback } from 'react';
import { 
  FileText, Download, Share2, Calendar, BarChart3, PieChart, TrendingUp, 
  TrendingDown, Activity, Zap, Shield, AlertTriangle, CheckCircle2, 
  RefreshCw, Loader2, Eye, ChevronRight, BrainCircuit, Target, 
  ThermometerSun, Droplets, Gauge, Clock, DollarSign, Leaf, FileSpreadsheet,
  ArrowUpRight, ArrowDownRight, Sparkles, Filter, Plus
} from 'lucide-react';
import { mockGenerateReport, MockReportData } from '../services/mockData';
import { predictDegradation, getMLStatus, MLModelStatus } from '../services/api';

// Report type configuration
const reportTypes = [
  { id: 'efficiency', label: 'Efficiency Analysis', icon: Gauge, color: 'blue', description: 'ML-powered efficiency predictions and loss analysis' },
  { id: 'degradation', label: 'Degradation Forecast', icon: TrendingDown, color: 'amber', description: 'Panel aging analysis and predictive maintenance' },
  { id: 'defects', label: 'Defect Detection', icon: Shield, color: 'red', description: 'CNN-based visual inspection results' },
  { id: 'optimization', label: 'Tilt Optimization', icon: Target, color: 'emerald', description: 'Solar geometry and tilt angle recommendations' },
  { id: 'financial', label: 'Financial Impact', icon: DollarSign, color: 'purple', description: 'ROI analysis and cost savings metrics' },
] as const;

// Color mappings for different report types
const colorMap: Record<string, { bg: string; border: string; text: string; gradient: string }> = {
  blue: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', text: 'text-blue-400', gradient: 'from-blue-600 to-cyan-500' },
  amber: { bg: 'bg-amber-500/10', border: 'border-amber-500/30', text: 'text-amber-400', gradient: 'from-amber-500 to-orange-500' },
  red: { bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', gradient: 'from-red-500 to-pink-500' },
  emerald: { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', gradient: 'from-emerald-500 to-green-500' },
  purple: { bg: 'bg-purple-500/10', border: 'border-purple-500/30', text: 'text-purple-400', gradient: 'from-purple-500 to-violet-500' },
};

export const Reports = () => {
  const [selectedReport, setSelectedReport] = useState<MockReportData | null>(null);
  const [activeReportType, setActiveReportType] = useState<MockReportData['reportType']>('efficiency');
  const [isGenerating, setIsGenerating] = useState(false);
  const [mlStatus, setMlStatus] = useState<MLModelStatus | null>(null);
  const [mlConnected, setMlConnected] = useState(false);
  const [recentReports, setRecentReports] = useState<MockReportData[]>([]);

  // Check ML API status on mount
  useEffect(() => {
    const checkMLStatus = async () => {
      try {
        const result = await getMLStatus();
        if (result.success && result.data) {
          setMlStatus(result.data);
          setMlConnected(result.data.overallStatus === 'connected');
        }
      } catch (err) {
        setMlConnected(false);
      }
    };
    checkMLStatus();
    
    // Generate initial reports for demo
    const initialReports = reportTypes.map(rt => mockGenerateReport(rt.id as MockReportData['reportType']));
    setRecentReports(initialReports);
    setSelectedReport(initialReports[0]);
  }, []);

  // Generate new report
  const handleGenerateReport = useCallback(async () => {
    setIsGenerating(true);
    
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const newReport = mockGenerateReport(activeReportType);
    setSelectedReport(newReport);
    setRecentReports(prev => [newReport, ...prev.filter(r => r.reportType !== activeReportType)]);
    
    setIsGenerating(false);
  }, [activeReportType]);

  // Get color config for current report type
  const getColorConfig = (type: string) => {
    const reportConfig = reportTypes.find(rt => rt.id === type);
    return colorMap[reportConfig?.color || 'blue'];
  };

  // Format date for display
  const formatDate = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Sparkline chart component
  const SparklineChart: React.FC<{ data: Array<{ value: number }>; color: string }> = ({ data, color }) => {
    const max = Math.max(...data.map(d => d.value));
    const min = Math.min(...data.map(d => d.value));
    const range = max - min || 1;
    
    const points = data.map((d, i) => {
      const x = (i / (data.length - 1)) * 100;
      const y = 100 - ((d.value - min) / range) * 80;
      return `${x},${y}`;
    }).join(' ');

    return (
      <svg viewBox="0 0 100 100" className="w-full h-16" preserveAspectRatio="none">
        <defs>
          <linearGradient id={`sparkline-${color}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={`var(--tw-gradient-from)`} stopOpacity="0.3" />
            <stop offset="100%" stopColor={`var(--tw-gradient-to)`} stopOpacity="0" />
          </linearGradient>
        </defs>
        <polyline
          points={points}
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`${colorMap[color]?.text || 'text-blue-400'}`}
        />
        <polygon
          points={`0,100 ${points} 100,100`}
          className={`fill-current ${colorMap[color]?.text || 'text-blue-400'} opacity-20`}
        />
      </svg>
    );
  };

  return (
    <div className="p-6 lg:p-8 min-h-full bg-slate-950 text-white">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl">
              <FileText className="w-6 h-6" />
            </div>
            ML Intelligence Reports
          </h1>
          <p className="text-slate-400 mt-2 flex items-center gap-2">
            <BrainCircuit className="w-4 h-4" />
            AI-driven insights from production ML models
            <span className={`ml-2 text-xs px-2 py-0.5 rounded-full ${mlConnected ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'}`}>
              {mlConnected ? '● Live ML' : '● Simulation Mode'}
            </span>
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <button 
            onClick={handleGenerateReport}
            disabled={isGenerating}
            className="flex items-center gap-2 px-5 py-2.5 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 rounded-xl font-semibold transition-all shadow-lg shadow-purple-900/30 disabled:opacity-50"
          >
            {isGenerating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Plus className="w-4 h-4" />
                Generate Report
              </>
            )}
          </button>
        </div>
      </div>

      {/* Report Type Selector */}
      <div className="flex flex-wrap gap-2 mb-8">
        {reportTypes.map((rt) => {
          const Icon = rt.icon;
          const isActive = activeReportType === rt.id;
          const colors = colorMap[rt.color];
          
          return (
            <button
              key={rt.id}
              onClick={() => {
                setActiveReportType(rt.id as MockReportData['reportType']);
                const existingReport = recentReports.find(r => r.reportType === rt.id);
                if (existingReport) setSelectedReport(existingReport);
              }}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-medium transition-all ${
                isActive 
                  ? `bg-gradient-to-r ${colors.gradient} text-white shadow-lg` 
                  : `${colors.bg} ${colors.border} border ${colors.text} hover:bg-opacity-20`
              }`}
            >
              <Icon className="w-4 h-4" />
              {rt.label}
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Main Report View */}
        <div className="xl:col-span-2 space-y-6">
          {selectedReport ? (
            <>
              {/* Report Header Card */}
              <div className={`bg-gradient-to-br ${getColorConfig(selectedReport.reportType).gradient} p-6 rounded-2xl shadow-xl`}>
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2 text-white/80 text-sm mb-2">
                      <Calendar className="w-4 h-4" />
                      {formatDate(selectedReport.generatedAt)}
                    </div>
                    <h2 className="text-2xl font-bold text-white mb-2">{selectedReport.title}</h2>
                    <p className="text-white/70 text-sm">{selectedReport.summary.period}</p>
                  </div>
                  <div className="flex gap-2">
                    <button className="p-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors" title="Download PDF">
                      <Download className="w-5 h-5" />
                    </button>
                    <button className="p-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors" title="Share">
                      <Share2 className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              </div>

              {/* Key Metrics Grid */}
              <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded-2xl p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-blue-400" />
                  Key Performance Metrics
                </h3>
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(selectedReport.summary.keyMetrics).map(([key, value]) => (
                    <div key={key} className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
                      <p className="text-xs text-slate-400 uppercase tracking-wide mb-1">{key}</p>
                      <p className="text-xl font-bold text-white">{value}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Trend Chart */}
              {selectedReport.data.timeSeries && (
                <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded-2xl p-6">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-emerald-400" />
                    30-Day Trend Analysis
                  </h3>
                  <div className="h-48">
                    <SparklineChart 
                      data={selectedReport.data.timeSeries} 
                      color={reportTypes.find(rt => rt.id === selectedReport.reportType)?.color || 'blue'} 
                    />
                  </div>
                  <div className="flex items-center justify-between mt-4 text-sm text-slate-400">
                    <span>{selectedReport.data.timeSeries[0]?.date}</span>
                    <span>{selectedReport.data.timeSeries[selectedReport.data.timeSeries.length - 1]?.date}</span>
                  </div>
                </div>
              )}

              {/* AI Insights */}
              <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded-2xl p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-purple-400" />
                  AI-Generated Insights
                </h3>
                <div className="space-y-3">
                  {selectedReport.summary.insights.map((insight, i) => (
                    <div key={i} className="flex items-start gap-3 p-3 bg-purple-500/5 border border-purple-500/20 rounded-xl">
                      <div className="p-1.5 bg-purple-500/20 rounded-lg">
                        <BrainCircuit className="w-4 h-4 text-purple-400" />
                      </div>
                      <p className="text-sm text-slate-300 leading-relaxed">{insight}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Recommendations */}
              <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded-2xl p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  Action Recommendations
                </h3>
                <div className="space-y-3">
                  {selectedReport.summary.recommendations.map((rec, i) => (
                    <div key={i} className="flex items-start gap-3 p-3 bg-emerald-500/5 border border-emerald-500/20 rounded-xl group hover:bg-emerald-500/10 transition-colors cursor-pointer">
                      <div className="p-1.5 bg-emerald-500/20 rounded-lg">
                        <Target className="w-4 h-4 text-emerald-400" />
                      </div>
                      <p className="text-sm text-slate-300 leading-relaxed flex-1">{rec}</p>
                      <ChevronRight className="w-4 h-4 text-slate-500 group-hover:text-emerald-400 transition-colors" />
                    </div>
                  ))}
                </div>
              </div>

              {/* Distribution Chart */}
              {selectedReport.data.distribution && (
                <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded-2xl p-6">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <PieChart className="w-5 h-5 text-cyan-400" />
                    Status Distribution
                  </h3>
                  <div className="space-y-3">
                    {selectedReport.data.distribution.map((item, i) => (
                      <div key={i}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-slate-300">{item.category}</span>
                          <span className="text-slate-400">{item.count.toLocaleString()} ({item.percentage}%)</span>
                        </div>
                        <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full transition-all ${
                              item.category === 'Optimal' ? 'bg-emerald-500' :
                              item.category === 'Good' ? 'bg-blue-500' :
                              item.category === 'Fair' ? 'bg-amber-500' :
                              item.category === 'Poor' ? 'bg-orange-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${item.percentage}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Top Issues */}
              {selectedReport.data.topIssues && (
                <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded-2xl p-6">
                  <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-amber-400" />
                    Top Issues Detected
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="text-left text-xs text-slate-500 uppercase tracking-wider">
                          <th className="pb-3">Issue</th>
                          <th className="pb-3">Severity</th>
                          <th className="pb-3 text-right">Count</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800">
                        {selectedReport.data.topIssues.map((issue, i) => (
                          <tr key={i} className="hover:bg-slate-800/30">
                            <td className="py-3 text-sm text-slate-300">{issue.issue}</td>
                            <td className="py-3">
                              <span className={`text-xs px-2 py-1 rounded-full ${
                                issue.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                                issue.severity === 'high' ? 'bg-orange-500/20 text-orange-400' :
                                issue.severity === 'medium' ? 'bg-amber-500/20 text-amber-400' :
                                'bg-slate-500/20 text-slate-400'
                              }`}>
                                {issue.severity}
                              </span>
                            </td>
                            <td className="py-3 text-sm text-slate-400 text-right font-mono">{issue.count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded-2xl p-12 text-center">
              <FileText className="w-16 h-16 text-slate-600 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-slate-400 mb-2">No Report Selected</h3>
              <p className="text-slate-500 mb-6">Select a report type and generate a new report</p>
              <button 
                onClick={handleGenerateReport}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-xl font-medium transition-colors"
              >
                Generate Your First Report
              </button>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* ML Model Status */}
          <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded-2xl p-5">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <BrainCircuit className="w-5 h-5 text-purple-400" />
              ML Models Status
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-xl">
                <div className="flex items-center gap-2">
                  <Gauge className="w-4 h-4 text-blue-400" />
                  <span className="text-sm">Efficiency Model</span>
                </div>
                <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded-full">
                  94.7% acc
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-xl">
                <div className="flex items-center gap-2">
                  <Shield className="w-4 h-4 text-red-400" />
                  <span className="text-sm">Defect Classifier</span>
                </div>
                <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-1 rounded-full">
                  97.2% acc
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-xl">
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4 text-emerald-400" />
                  <span className="text-sm">Tilt Optimizer</span>
                </div>
                <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded-full">
                  +8.3% avg
                </span>
              </div>
            </div>
          </div>

          {/* Recent Reports */}
          <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded-2xl p-5">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Clock className="w-5 h-5 text-slate-400" />
              Recent Reports
            </h3>
            <div className="space-y-2">
              {recentReports.slice(0, 5).map((report, i) => {
                const rtConfig = reportTypes.find(rt => rt.id === report.reportType);
                const Icon = rtConfig?.icon || FileText;
                const colors = getColorConfig(report.reportType);
                
                return (
                  <button
                    key={report.id}
                    onClick={() => {
                      setSelectedReport(report);
                      setActiveReportType(report.reportType);
                    }}
                    className={`w-full flex items-center gap-3 p-3 rounded-xl transition-all hover:bg-slate-800/50 ${
                      selectedReport?.id === report.id ? 'bg-slate-800/70 border border-slate-700' : ''
                    }`}
                  >
                    <div className={`p-2 rounded-lg ${colors.bg}`}>
                      <Icon className={`w-4 h-4 ${colors.text}`} />
                    </div>
                    <div className="flex-1 text-left">
                      <p className="text-sm font-medium text-slate-200 truncate">{rtConfig?.label}</p>
                      <p className="text-xs text-slate-500">{formatDate(report.generatedAt)}</p>
                    </div>
                    <Eye className="w-4 h-4 text-slate-500" />
                  </button>
                );
              })}
            </div>
          </div>

          {/* Quick Stats */}
          <div className="bg-gradient-to-br from-purple-900/30 to-blue-900/30 border border-purple-500/20 rounded-2xl p-5">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-amber-400" />
              Report Analytics
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-900/50 p-3 rounded-xl text-center">
                <p className="text-2xl font-bold text-white">127</p>
                <p className="text-xs text-slate-400">Reports MTD</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-xl text-center">
                <p className="text-2xl font-bold text-emerald-400">+23%</p>
                <p className="text-xs text-slate-400">vs Last Month</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-xl text-center">
                <p className="text-2xl font-bold text-white">4.8M</p>
                <p className="text-xs text-slate-400">Data Points</p>
              </div>
              <div className="bg-slate-900/50 p-3 rounded-xl text-center">
                <p className="text-2xl font-bold text-blue-400">847</p>
                <p className="text-xs text-slate-400">Insights</p>
              </div>
            </div>
          </div>

          {/* Export Options */}
          <div className="bg-slate-900/80 backdrop-blur-sm border border-slate-800 rounded-2xl p-5">
            <h3 className="text-lg font-semibold mb-4">Export Options</h3>
            <div className="space-y-2">
              <button className="w-full flex items-center justify-center gap-2 p-3 bg-slate-800 hover:bg-slate-700 rounded-xl text-sm font-medium transition-colors">
                <Download className="w-4 h-4" />
                Download as PDF
              </button>
              <button className="w-full flex items-center justify-center gap-2 p-3 bg-slate-800 hover:bg-slate-700 rounded-xl text-sm font-medium transition-colors">
                <FileSpreadsheet className="w-4 h-4" />
                Export to Excel
              </button>
              <button className="w-full flex items-center justify-center gap-2 p-3 bg-slate-800 hover:bg-slate-700 rounded-xl text-sm font-medium transition-colors">
                <Share2 className="w-4 h-4" />
                Share Report Link
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
