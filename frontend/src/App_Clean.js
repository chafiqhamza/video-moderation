import React, { useState, useEffect } from 'react';
import ComprehensiveReportDisplay from './ComprehensiveReportDisplay';
import VideoUpload from './VideoUpload';
import RagAnalysisPage from './RagAnalysisPage';
import SettingsPage from './SettingsPage';
import UploadHistoryPage from './UploadHistoryPage';
import FrameDetailsPage from './FrameDetailsPage';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  Alert,
  LinearProgress,
  Paper
} from '@mui/material';
import CloudUpload from '@mui/icons-material/CloudUpload';
import VideoLibrary from '@mui/icons-material/VideoLibrary';
import CheckCircle from '@mui/icons-material/CheckCircle';
import Analytics from '@mui/icons-material/Analytics';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [progress] = useState(0);
  const [showRagPage, setShowRagPage] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [ragAnalysis, setRagAnalysis] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [showFrameDetails, setShowFrameDetails] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [parsedReport, setParsedReport] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('testing');
  const [frameSettings, setFrameSettings] = useState({
    frameCount: 1,
    frameInterval: 0.5,
    frameResolution: 'auto',
    frameFormat: 'jpg',
    frameStart: 0,
    frameEnd: null,
    frameSampling: 'interval',
  });

  useEffect(() => {
    testConnection();
  }, []);

  const testConnection = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (response.ok) {
        setConnectionStatus('connected');
      } else {
        setConnectionStatus('disconnected');
      }
    } catch (err) {
      setConnectionStatus('disconnected');
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type.startsWith('video/')) {
        setSelectedFile(file);
        setError('');
        setAnalysis(null);
      } else {
        setError('Please select a valid video file');
        setSelectedFile(null);
      }
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select a video file');
      return;
    }
    setLoading(true);
    setError('');
    setAnalysis(null);
    try {
      const params = new URLSearchParams();
      if (frameSettings.frameSampling === 'count') {
        params.append('frame_count', frameSettings.frameCount);
        params.append('sampling_method', 'count');
      } else {
        params.append('frame_interval', frameSettings.frameInterval);
        params.append('sampling_method', 'interval');
      }
      params.append('resolution', frameSettings.frameResolution);
      params.append('format', frameSettings.frameFormat);
      params.append('start_time', frameSettings.frameStart);
      params.append('end_time', frameSettings.frameEnd !== null ? frameSettings.frameEnd : -1);
      const formData = new FormData();
      formData.append('file', selectedFile);
      const response = await fetch(`${API_BASE_URL}/upload-video?${params.toString()}`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const analysisRaw = await response.text();
      let analysisObj = {};
      try {
        analysisObj = JSON.parse(analysisRaw);
      } catch (e) {
        analysisObj = { full_report: analysisRaw };
      }
      setAnalysis(analysisObj);
    } catch (err) {
      setError(`Analysis failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Always use frameSettings for RAG and frame analysis
    if (analysis && (analysis.rag_explanations || (analysis.analysis_json && analysis.analysis_json.rag_explanations))) {
      const ragFrames = analysis.rag_explanations || analysis.analysis_json.rag_explanations;
      setRagAnalysis({
        decision: analysis.analysis_json?.overall_assessment?.status || 'N/A',
        confidence: analysis.analysis_json?.frame_analysis?.average_confidence || 'N/A',
        reasoning: 'Detailed frame-level policy violation explanations below.',
        retrieved_docs: ragFrames,
        frames_analyzed: analysis.analysis_json?.frame_analysis?.total_frames || ragFrames.length
      });
    } else if (analysis && analysis.full_report) {
      const ragSectionMatch = analysis.full_report.match(/DETAILED POLICY VIOLATION EXPLANATIONS \(RAG\):([\s\S]*?)(?=\n\s*\n|$)/i);
      if (ragSectionMatch) {
        const ragText = ragSectionMatch[1];
        const frameRegex = /- Frame (\d+) \(t=([\d\.]+)s\): ([^\n]+)\n([\s\S]*?)(?=(?:- Frame \d+ \(t=|$))/g;
        const frames = [];
        let match;
        while ((match = frameRegex.exec(ragText)) !== null) {
          const frameIdx = match[1];
          const timestamp = match[2];
          const category = match[3] && match[3].trim() !== '' ? match[3].trim() : 'safe_content';
          frames.push({
            title: `Frame ${frameIdx} (t=${timestamp}s): ${category}`,
            category,
            reasoning: category === 'safe_content' ? 'No violation detected. Content is safe.' : undefined
          });
        }
        const limitedFrames = frameSettings && frameSettings.frameCount && frames.length > frameSettings.frameCount
          ? frames.slice(0, frameSettings.frameCount)
          : frames;
        setRagAnalysis({
          decision: parsedReport?.overall || 'N/A',
          confidence: parsedReport?.image_score || 'N/A',
          reasoning: 'Detailed frame-level policy violation explanations below.',
          retrieved_docs: limitedFrames
        });
      } else {
        setRagAnalysis(null);
      }
    }
  }, [analysis, parsedReport, frameSettings]);

  // Use frames array from backend for accurate compliance
  const framesArr = Array.isArray(analysis?.frames) ? analysis.frames : [];
  const totalFrames = framesArr.length;
  const safeCount = framesArr.filter(f => String(f.category).toLowerCase().includes('safe')).length;
  const unsafeCount = framesArr.filter(f => !String(f.category).toLowerCase().includes('safe')).length;

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const sidebarOptions = [
    { label: 'SETTINGS', color: 'secondary', icon: <Analytics />, tooltip: 'Configure frame extraction and analysis settings', onClick: () => { setShowSettings(true); setShowHistory(false); setShowFrameDetails(false); setShowRagPage(false); } },
    { label: 'RAG ANALYSIS', color: 'info', icon: <CheckCircle />, tooltip: 'View policy violation explanations', onClick: () => { setShowRagPage(true); setShowSettings(false); setShowHistory(false); setShowFrameDetails(false); } },
    { label: 'UPLOAD HISTORY', color: 'primary', icon: <VideoLibrary />, tooltip: 'See previous uploads and results', onClick: () => { setShowHistory(true); setShowFrameDetails(false); setShowRagPage(false); setShowSettings(false); } },
    { label: 'FRAME DETAILS', color: 'info', icon: <CloudUpload />, tooltip: 'Detailed report of analyzed frames', onClick: () => { setShowFrameDetails(true); setShowHistory(false); setShowRagPage(false); setShowSettings(false); } },
    { label: 'CUSTOMIZE', color: 'warning', icon: <Analytics />, tooltip: 'Customize moderation options', onClick: () => alert('Customize feature coming soon!') },
  ];

  let mainContent = null;
  if (showSettings) {
    mainContent = <SettingsPage settings={frameSettings} onSave={settings => { setFrameSettings(settings); setShowSettings(false); }} />;
  } else if (showRagPage) {
    mainContent = (
      <Box sx={{ width: '100%', height: '100%', p: 2, background: 'linear-gradient(120deg, #e3f2fd 60%, #fffde7 100%)', borderRadius: 4, boxShadow: 3 }}>
        <Box sx={{ mb: 2 }}>
          <Typography variant="h5" color="success.main" sx={{ fontWeight: 'bold', mb: 1 }}>
            Safe Content Frames: {safeCount} / {totalFrames}
          </Typography>
          <LinearProgress variant="determinate" value={totalFrames ? (safeCount / totalFrames) * 100 : 0} sx={{ height: 10, borderRadius: 5, mt: 1 }} color="success" />
          {parsedReport && (
            <Card sx={{ mt: 2, boxShadow: 2, borderRadius: 2, background: '#fff' }}>
              <CardContent>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1 }} color="primary">
                  Model Analysis Summary
                </Typography>
                <Typography variant="body2">Score: <b>{parsedReport?.image_score || 'N/A'}%</b></Typography>
                <Typography variant="body2">Status: <b>{parsedReport?.overall || 'N/A'}</b></Typography>
                <Typography variant="body2">Frames Analyzed: <b>{parsedReport?.frames_analyzed || 'N/A'}</b></Typography>
              </CardContent>
            </Card>
          )}
        </Box>
        <RagAnalysisPage ragAnalysis={ragAnalysis} onBack={() => setShowRagPage(false)} />
      </Box>
    );
  } else if (showHistory) {
    mainContent = <UploadHistoryPage />;
  } else if (showFrameDetails) {
    mainContent = <FrameDetailsPage frames={ragAnalysis?.retrieved_docs || []} onBack={() => setShowFrameDetails(false)} />;
  } else {
    mainContent = (
      <Box sx={{ width: '100vw', height: '100vh', p: 0, m: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column', alignItems: 'stretch', justifyContent: 'flex-start', background: 'linear-gradient(120deg, #e3f2fd 60%, #fffde7 100%)' }}>
        {!analysis && (
          <Box sx={{ width: '100vw', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', pt: 6 }}>
            <VideoUpload
              selectedFile={selectedFile}
              onFileChange={handleFileChange}
              onAnalyze={handleAnalyze}
              loading={loading}
              error={error}
              frameSettings={frameSettings}
              connectionStatus={connectionStatus}
            />
            <Typography variant="h4" color="primary" sx={{ fontWeight: 'bold', mt: 4 }}>
              Please upload a video to start analysis.
            </Typography>
          </Box>
        )}
        {loading && (
          <Paper sx={{ p: 1, mb: 2, width: '100%', background: '#fffde7', borderRadius: 2 }}>
            <Typography variant="body2" gutterBottom>
              Analyzing video content...
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Box sx={{ width: '100%', mr: 1 }}>
                <LinearProgress variant="determinate" value={progress} />
              </Box>
              <Typography variant="caption" sx={{ minWidth: 30 }}>{progress}%</Typography>
            </Box>
          </Paper>
        )}
        {analysis && (
          <Box sx={{ width: '100vw', height: '100vh', p: 0, m: 0 }}>
            <ComprehensiveReportDisplay report={{
              ...analysis,
              analysis_json: analysis.analysis_json || analysis,
              copyright_check: analysis.copyright_check || (analysis.analysis_json && analysis.analysis_json.copyright_check) || null,
              // Pass model scores and issues directly if present
              text_score: analysis.analysis_json?.text_score ?? analysis.text_score,
              audio_score: analysis.analysis_json?.audio_score ?? analysis.audio_score,
              image_score: analysis.analysis_json?.image_score ?? analysis.image_score,
              text_issues: analysis.analysis_json?.text_issues ?? analysis.text_issues,
              audio_issues: analysis.analysis_json?.audio_issues ?? analysis.audio_issues,
              image_issues: analysis.analysis_json?.image_issues ?? analysis.image_issues
            }} />
          </Box>
        )}
        {connectionStatus === 'disconnected' && (
          <Alert severity="error" sx={{ mb: 2 }} action={
            <Button color="inherit" size="small" onClick={testConnection}>
              Retry
            </Button>
          }>
            Cannot connect to backend server. Make sure the server is running on http://localhost:8000
          </Alert>
        )}
        {connectionStatus === 'connected' && (
          <Alert severity="success" sx={{ mb: 2 }}>
            ✅ AI Video Analyzer ready! Upload and analyze your video files.
          </Alert>
        )}
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', width: '100vw', height: '100vh', background: 'linear-gradient(120deg, #e3f2fd 60%, #fffde7 100%)' }}>
      {/* Sidebar */}
      <Box sx={{ width: sidebarOpen ? 180 : 60, transition: 'width 0.2s', background: 'linear-gradient(180deg,#1976d2 0%,#21cbf3 100%)', color: '#fff', display: { xs: 'none', md: 'flex' }, flexDirection: 'column', alignItems: 'center', py: 2, boxShadow: 2, position: 'relative', minHeight: '100vh' }}>
        <Box sx={{ mb: 3 }}>
          <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="Menu" style={{ width: 56, borderRadius: '50%' }} />
        </Box>
        <Button variant="text" color="inherit" sx={{ mb: 2, minWidth: 0, p: 0, fontSize: '1.5rem', borderRadius: 2 }} onClick={() => setSidebarOpen(!sidebarOpen)} aria-label={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}>
          {sidebarOpen ? '<' : '>'}
        </Button>
        <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1, letterSpacing: 1, display: sidebarOpen ? 'block' : 'none' }}>
          Menu
        </Typography>
        {sidebarOptions.map(opt => (
          <Box key={opt.label} sx={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 1 }}>
            <Button
              variant="contained"
              color={opt.color}
              sx={{ width: sidebarOpen ? '90%' : 48, fontWeight: 'bold', fontSize: '0.95rem', borderRadius: 2, boxShadow: 1, textTransform: 'none', py: 1, px: sidebarOpen ? 2 : 0, minWidth: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
              onClick={opt.onClick}
              aria-label={opt.label}
              title={opt.tooltip}
            >
              {opt.icon}
              {sidebarOpen && <span style={{ marginLeft: 8 }}>{opt.label}</span>}
            </Button>
          </Box>
        ))}
      </Box>
      {/* Main content */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', width: '100%', height: '100%', background: 'transparent', overflow: 'auto' }}>
        <AppBar position="static" sx={{ background: 'linear-gradient(90deg,#1976d2 60%,#21cbf3 100%)', boxShadow: 2 }}>
          <Toolbar>
            <Typography variant="h4" sx={{ fontWeight: 'bold', flexGrow: 1, letterSpacing: 1, color: '#fff' }}>
              AI Video Content Analyzer
            </Typography>
            {/* Summary widgets */}
            {ragAnalysis && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, ml: 4 }}>
                <Typography variant="body1" sx={{ color: '#fff', fontWeight: 'bold' }}>
                  Videos Analyzed: <span style={{ color: '#e3f2fd' }}>{totalFrames}</span>
                </Typography>
                <Typography variant="body1" sx={{ color: '#fff', fontWeight: 'bold' }}>
                  Safe Frames: <span style={{ color: '#a5d6a7' }}>{safeCount}</span>
                </Typography>
                <Typography variant="body1" sx={{ color: '#fff', fontWeight: 'bold' }}>
                  Compliance Rate: <span style={{ color: '#ffd54f' }}>{totalFrames ? ((safeCount / totalFrames) * 100).toFixed(1) : '0'}%</span>
                </Typography>
              </Box>
            )}
          </Toolbar>
        </AppBar>
        <Box sx={{ flex: 1, overflow: 'auto', p: 0, m: 0 }}>
          {mainContent}
        </Box>
      </Box>
    </Box>
  );
}

export default App;