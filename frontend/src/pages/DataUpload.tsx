import React, { useState } from 'react';
import { Upload, FileType, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';

export const DataUpload = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setUploadStatus('uploading');
    // Simulate upload
    setTimeout(() => setUploadStatus('success'), 2000);
  };

  return (
    <div className="p-8 min-h-full bg-slate-950 text-white flex flex-col items-center justify-center max-w-4xl mx-auto">
      <div className="text-center mb-10">
        <h1 className="text-3xl font-bold mb-2">Import System Data</h1>
        <p className="text-slate-400">Upload your CSV or JSON files for deep learning analysis</p>
      </div>

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`w-full max-w-2xl aspect-video rounded-3xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center p-12 cursor-pointer
          ${isDragging 
            ? 'border-blue-500 bg-blue-500/10 scale-[1.02]' 
            : 'border-slate-700 bg-slate-900/50 hover:border-slate-600 hover:bg-slate-900'
          }`}
      >
        {uploadStatus === 'idle' && (
          <>
            <div className="w-20 h-20 bg-slate-800 rounded-full flex items-center justify-center mb-6 shadow-xl">
              <Upload className="w-10 h-10 text-blue-400" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Drag & Drop files here</h3>
            <p className="text-slate-400 mb-6">or click to browse from your computer</p>
            <div className="flex items-center gap-4 text-sm text-slate-500">
              <span className="flex items-center gap-2"><FileType className="w-4 h-4" /> CSV files supported</span>
              <span className="flex items-center gap-2"><FileType className="w-4 h-4" /> Max size 50MB</span>
            </div>
          </>
        )}

        {uploadStatus === 'uploading' && (
          <div className="text-center">
            <RefreshCw className="w-16 h-16 text-blue-500 animate-spin mb-6 mx-auto" />
            <h3 className="text-xl font-semibold">Processing Data...</h3>
            <p className="text-slate-400 mt-2">Running ML validation models</p>
          </div>
        )}

        {uploadStatus === 'success' && (
          <div className="text-center">
            <div className="w-20 h-20 bg-green-500/20 rounded-full flex items-center justify-center mb-6 mx-auto">
              <CheckCircle className="w-10 h-10 text-green-500" />
            </div>
            <h3 className="text-xl font-semibold">Upload Complete</h3>
            <p className="text-slate-400 mt-2 mb-6">Data has been successfully integrated</p>
            <button 
              onClick={(e) => { e.stopPropagation(); setUploadStatus('idle'); }}
              className="px-6 py-2 bg-slate-800 rounded-lg text-white hover:bg-slate-700 transition-colors"
            >
              Upload Another
            </button>
          </div>
        )}
      </div>

      {/* Feature List */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12 w-full">
        {[
          { title: 'Automated Parsing', desc: 'Smart column detection and formatting' },
          { title: 'ML Validation', desc: 'Deep learning anomaly detection on upload' },
          { title: 'Instant Visualization', desc: 'Generate 3D models from coordinates' }
        ].map((feat, i) => (
          <div key={i} className="bg-slate-900 p-6 rounded-xl border border-slate-800">
            <h4 className="font-semibold text-blue-400 mb-2">{feat.title}</h4>
            <p className="text-sm text-slate-400">{feat.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
