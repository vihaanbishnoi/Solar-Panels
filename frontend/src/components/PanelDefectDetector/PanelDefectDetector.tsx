import React, { useState, useCallback } from 'react';
import { 
  Upload, 
  Camera, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  Loader2, 
  Image as ImageIcon,
  Trash2,
  BrainCircuit,
  Shield,
  AlertCircle,
  FileImage,
  Sparkles,
  MailCheck,
  MailX
} from 'lucide-react';
import { classifyPanelImage, PanelClassificationResult } from '../../services/api';

interface AnalysisResult {
  file: File;
  preview: string;
  result: PanelClassificationResult | null;
  loading: boolean;
  error: string | null;
}

export const PanelDefectDetector: React.FC = () => {
  const [analyses, setAnalyses] = useState<AnalysisResult[]>([]);
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = useCallback(async (files: FileList | File[]) => {
    const fileArray = Array.from(files).filter(f => f.type.startsWith('image/'));
    
    if (fileArray.length === 0) return;

    // Create preview entries
    const newAnalyses: AnalysisResult[] = fileArray.map(file => ({
      file,
      preview: URL.createObjectURL(file),
      result: null,
      loading: true,
      error: null,
    }));

    setAnalyses(prev => [...prev, ...newAnalyses]);

    // Process each image
    for (let i = 0; i < fileArray.length; i++) {
      const file = fileArray[i];
      try {
        const response = await classifyPanelImage(file);
        
        setAnalyses(prev => 
          prev.map(a => 
            a.file === file 
              ? { 
                  ...a, 
                  result: response.data || null, 
                  loading: false,
                  error: response.success ? null : (response.error || 'Classification failed')
                }
              : a
          )
        );
      } catch (err) {
        setAnalyses(prev => 
          prev.map(a => 
            a.file === file 
              ? { ...a, loading: false, error: 'Failed to analyze image' }
              : a
          )
        );
      }
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      handleFiles(e.target.files);
    }
  };

  const removeAnalysis = (file: File) => {
    setAnalyses(prev => {
      const analysis = prev.find(a => a.file === file);
      if (analysis) {
        URL.revokeObjectURL(analysis.preview);
      }
      return prev.filter(a => a.file !== file);
    });
  };

  const clearAll = () => {
    analyses.forEach(a => URL.revokeObjectURL(a.preview));
    setAnalyses([]);
  };

  const stats = {
    total: analyses.length,
    normal: analyses.filter(a => a.result?.prediction.label === 'NORMAL').length,
    defective: analyses.filter(a => a.result?.prediction.label === 'DEFECTIVE').length,
    processing: analyses.filter(a => a.loading).length,
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-3 bg-gradient-to-r from-purple-500/20 to-blue-500/20 px-6 py-3 rounded-full border border-purple-500/30 mb-4">
            <BrainCircuit className="w-6 h-6 text-purple-400" />
            <span className="text-purple-300 font-semibold">AI-Powered Analysis</span>
          </div>
          <h1 className="text-4xl font-bold text-white mb-3">
            Panel Defect Detection
          </h1>
          <p className="text-slate-400 text-lg max-w-2xl mx-auto">
            Upload solar panel images for instant AI classification. Our ResNet18 model
            identifies defects with high accuracy.
          </p>
        </div>

        {/* Stats Bar */}
        {analyses.length > 0 && (
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
              <div className="flex items-center gap-3">
                <FileImage className="w-8 h-8 text-blue-400" />
                <div>
                  <p className="text-2xl font-bold text-white">{stats.total}</p>
                  <p className="text-xs text-slate-400">Total Images</p>
                </div>
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4 border border-green-500/30">
              <div className="flex items-center gap-3">
                <CheckCircle2 className="w-8 h-8 text-green-400" />
                <div>
                  <p className="text-2xl font-bold text-green-400">{stats.normal}</p>
                  <p className="text-xs text-slate-400">Normal</p>
                </div>
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4 border border-red-500/30">
              <div className="flex items-center gap-3">
                <XCircle className="w-8 h-8 text-red-400" />
                <div>
                  <p className="text-2xl font-bold text-red-400">{stats.defective}</p>
                  <p className="text-xs text-slate-400">Defective</p>
                </div>
              </div>
            </div>
            <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700">
              <div className="flex items-center gap-3">
                <Loader2 className={`w-8 h-8 text-yellow-400 ${stats.processing > 0 ? 'animate-spin' : ''}`} />
                <div>
                  <p className="text-2xl font-bold text-yellow-400">{stats.processing}</p>
                  <p className="text-xs text-slate-400">Processing</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Upload Zone */}
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 mb-8 ${
            isDragging 
              ? 'border-purple-500 bg-purple-500/10 scale-[1.02]' 
              : 'border-slate-600 bg-slate-800/30 hover:border-slate-500 hover:bg-slate-800/50'
          }`}
        >
          <input
            type="file"
            accept="image/*"
            multiple
            onChange={handleFileInput}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          <div className="flex flex-col items-center">
            <div className={`w-20 h-20 rounded-full flex items-center justify-center mb-6 transition-all ${
              isDragging ? 'bg-purple-500/30' : 'bg-slate-700'
            }`}>
              <Upload className={`w-10 h-10 ${isDragging ? 'text-purple-400' : 'text-slate-400'}`} />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">
              Drop panel images here
            </h3>
            <p className="text-slate-400 mb-4">or click to browse</p>
            <div className="flex items-center gap-4 text-sm text-slate-500">
              <span className="flex items-center gap-2">
                <ImageIcon className="w-4 h-4" /> JPG, PNG, WebP
              </span>
              <span className="flex items-center gap-2">
                <Camera className="w-4 h-4" /> Up to 10MB each
              </span>
            </div>
          </div>
        </div>

        {/* Clear Button */}
        {analyses.length > 0 && (
          <div className="flex justify-end mb-4">
            <button
              onClick={clearAll}
              className="flex items-center gap-2 px-4 py-2 bg-red-500/20 text-red-400 rounded-lg hover:bg-red-500/30 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              Clear All
            </button>
          </div>
        )}

        {/* Results Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {analyses.map((analysis, idx) => (
            <div
              key={idx}
              className={`bg-slate-800/50 rounded-2xl overflow-hidden border transition-all ${
                analysis.result?.prediction.label === 'DEFECTIVE'
                  ? 'border-red-500/50 shadow-lg shadow-red-500/10'
                  : analysis.result?.prediction.label === 'NORMAL'
                  ? 'border-green-500/50 shadow-lg shadow-green-500/10'
                  : 'border-slate-700'
              }`}
            >
              {/* Image Preview */}
              <div className="relative aspect-video bg-slate-900">
                <img
                  src={analysis.preview}
                  alt={analysis.file.name}
                  className="w-full h-full object-cover"
                />
                
                {/* Loading Overlay */}
                {analysis.loading && (
                  <div className="absolute inset-0 bg-slate-900/80 flex flex-col items-center justify-center">
                    <Loader2 className="w-12 h-12 text-purple-500 animate-spin mb-3" />
                    <p className="text-sm text-slate-300">Analyzing...</p>
                  </div>
                )}

                {/* Status Badge */}
                {analysis.result && (
                  <div className={`absolute top-3 right-3 px-3 py-1.5 rounded-full font-bold text-sm flex items-center gap-2 ${
                    analysis.result.prediction.label === 'DEFECTIVE'
                      ? 'bg-red-500 text-white'
                      : 'bg-green-500 text-white'
                  }`}>
                    {analysis.result.prediction.label === 'DEFECTIVE' ? (
                      <XCircle className="w-4 h-4" />
                    ) : (
                      <CheckCircle2 className="w-4 h-4" />
                    )}
                    {analysis.result.prediction.label}
                  </div>
                )}

                {/* Remove Button */}
                <button
                  onClick={() => removeAnalysis(analysis.file)}
                  className="absolute top-3 left-3 p-2 bg-slate-900/80 rounded-full hover:bg-red-500 transition-colors"
                >
                  <Trash2 className="w-4 h-4 text-white" />
                </button>
              </div>

              {/* Result Details */}
              <div className="p-4">
                <p className="text-sm text-slate-400 truncate mb-3">{analysis.file.name}</p>
                
                {analysis.error && (
                  <div className="flex items-center gap-2 text-red-400 text-sm">
                    <AlertCircle className="w-4 h-4" />
                    {analysis.error}
                  </div>
                )}

                {analysis.result && (
                  <div className="space-y-3">
                    {/* Confidence Bar */}
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-400">Confidence</span>
                        <span className="text-white font-semibold">
                          {analysis.result.prediction.confidence_percent}
                        </span>
                      </div>
                      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${
                            analysis.result.prediction.label === 'DEFECTIVE'
                              ? 'bg-gradient-to-r from-red-500 to-orange-500'
                              : 'bg-gradient-to-r from-green-500 to-emerald-400'
                          }`}
                          style={{ width: `${analysis.result.prediction.confidence}%` }}
                        />
                      </div>
                    </div>

                    {/* Probabilities */}
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="bg-green-500/10 rounded-lg p-2 text-center">
                        <p className="text-green-400 font-semibold">{analysis.result.probabilities.normal}%</p>
                        <p className="text-xs text-slate-400">Normal</p>
                      </div>
                      <div className="bg-red-500/10 rounded-lg p-2 text-center">
                        <p className="text-red-400 font-semibold">{analysis.result.probabilities.defective}%</p>
                        <p className="text-xs text-slate-400">Defective</p>
                      </div>
                    </div>

                    {/* Severity & Recommendation */}
                    {analysis.result.analysis.action_required && (
                      <div className={`rounded-lg p-3 ${
                        analysis.result.analysis.severity === 'critical'
                          ? 'bg-red-500/20 border border-red-500/30'
                          : analysis.result.analysis.severity === 'high'
                          ? 'bg-orange-500/20 border border-orange-500/30'
                          : 'bg-yellow-500/20 border border-yellow-500/30'
                      }`}>
                        <div className="flex items-center gap-2 mb-1">
                          <AlertTriangle className={`w-4 h-4 ${
                            analysis.result.analysis.severity === 'critical'
                              ? 'text-red-400'
                              : analysis.result.analysis.severity === 'high'
                              ? 'text-orange-400'
                              : 'text-yellow-400'
                          }`} />
                          <span className="text-sm font-semibold text-white capitalize">
                            {analysis.result.analysis.severity} Priority
                          </span>
                        </div>
                        <p className="text-xs text-slate-300">
                          {analysis.result.analysis.recommendation}
                        </p>
                      </div>
                    )}

                    {/* ML Source Badge */}
                    <div className="flex items-center justify-between pt-2 border-t border-slate-700">
                      <div className="flex items-center gap-2">
                        <Sparkles className="w-4 h-4 text-purple-400" />
                        <span className="text-xs text-slate-400">
                          {analysis.result.source === 'ml-model' ? 'ML Model' : 'Simulation'}
                        </span>
                      </div>
                      <span className="text-xs text-slate-500">
                        {analysis.result.modelInfo.architecture}
                      </span>
                    </div>

                    {/* Email Alert Status - Only for defective panels */}
                    {analysis.result.prediction.label === 'DEFECTIVE' && analysis.result.emailAlert && (
                      <div className={`flex items-center gap-2 p-2 rounded-lg mt-2 ${
                        analysis.result.emailAlert.sent 
                          ? 'bg-green-500/10 border border-green-500/30' 
                          : 'bg-orange-500/10 border border-orange-500/30'
                      }`}>
                        {analysis.result.emailAlert.sent ? (
                          <>
                            <MailCheck className="w-4 h-4 text-green-400" />
                            <span className="text-xs text-green-400">Alert email sent!</span>
                          </>
                        ) : (
                          <>
                            <MailX className="w-4 h-4 text-orange-400" />
                            <span className="text-xs text-orange-400">
                              Email not configured
                            </span>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Empty State */}
        {analyses.length === 0 && (
          <div className="text-center py-16">
            <Shield className="w-16 h-16 text-slate-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-slate-400 mb-2">
              No images analyzed yet
            </h3>
            <p className="text-slate-500">
              Upload panel images to start defect detection
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PanelDefectDetector;
