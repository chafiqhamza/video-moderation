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
  Button,
  Alert,
  LinearProgress,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Tooltip
} from '@mui/material';
import CloudUpload from '@mui/icons-material/CloudUpload';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DownloadIcon from '@mui/icons-material/Download';
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
  const [parsedReport] = useState({});
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

  // --- recommendation generator (frontend) ---
  function generateRecommendationsFromAnalysis(analysis) {
    if (!analysis) return [];
    const retrieved_docs = analysis.retrieved_docs || [];
    const complianceRate = Math.round((analysis.compliance_rate || analysis.complianceRate || 0) * 100);
    const pacingScore = analysis.pacing_rate || analysis.pacingRate || analysis.pacingScore || null;
    const pacingDeviation = typeof pacingScore === 'number' ? Math.abs(pacingScore - 1) : null;
    const pacingWarning = pacingDeviation !== null && pacingDeviation > 0.15;

    const positiveCategories = new Set([
      'safe content','educational content','entertainment content','news content',
      'tutorials & how-to','community & family-friendly','artistic & creative expression','positive social impact'
    ]);

    const groups = {};
    retrieved_docs.forEach(doc => {
      const cat = String(doc.category || '').toLowerCase();
      if (!cat) return;
      if (!groups[cat]) groups[cat] = { count: 0, samples: [] };
      groups[cat].count++;
      if (groups[cat].samples.length < 3) groups[cat].samples.push({ frame: doc.frame, ts: doc.timestamp, text: doc.blip || doc.transcript || doc.ocr });
    });

    const violationCats = Object.keys(groups).filter(c => !positiveCategories.has(c));
    const recommendations = [];
    recommendations.push({ severity: 'info', text: `Overview: ${complianceRate}% compliance · ${retrieved_docs.filter(d => !positiveCategories.has(String(d.category||'').toLowerCase())).length} flagged frames · ${violationCats.length} issue categories` });
    if (pacingWarning) recommendations.push({ severity: 'warning', text: `Pacing: detected unusual pacing (${typeof pacingScore === 'number' ? pacingScore.toFixed(2) : 'N/A'}). Consider adding short pauses, trimming dense speech, or adding captions.` });
    violationCats.forEach(cat => {
      const info = groups[cat];
      let action = 'Review flagged frames and add clarifying text or remove/blur content.';
      if (cat.includes('suggest') || cat.includes('sexual') || cat.includes('adult')) action = 'Blur or replace explicit visuals; add contextual intro or age gate.';
      else if (cat.includes('misinform') || cat.includes('false')) action = 'Add on-screen corrections and cite authoritative sources in description/overlay.';
      else if (cat.includes('bad') || cat.includes('profan')) action = 'Bleep, re-record or add a content warning; provide a cleaned transcript.';
      else if (cat.includes('violence')) action = 'Blur graphic details, avoid close-ups, add a trigger warning/age gate.';
      recommendations.push({ severity: 'high', text: `${cat.replace(/_/g,' ')}: ${action} (Detected in ${info.count} frame${info.count>1?'s':''})`, samples: info.samples });
    });
    recommendations.push({ severity: 'info', text: 'Quick wins: trim or replace top 1–2 high-impact flagged segments, add captions/subtitles, update video description with clarifying notes.' });
    const order = { high: 0, warning: 1, info: 2 };
    recommendations.sort((a,b) => (order[a.severity] - order[b.severity]));
    return recommendations;
  }

  // helper removed — recommendations dialog is opened inline where used

  const closeRecommendationsDialog = () => setRecDialogOpen(false);

  const copyRecommendationsToClipboard = () => {
    const text = dashboardRecommendations.map(r => `- ${r.text}`).join('\n');
    if (navigator.clipboard) navigator.clipboard.writeText(text);
  };

  const downloadRecommendationsJSON = () => {
    const blob = new Blob([JSON.stringify(dashboardRecommendations, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    const vid = (ragAnalysis && ragAnalysis.video_id) || 'video';
    a.download = `recommendations_${vid}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const saveRecommendationsToBackend = async () => {
    try {
      const vid = (ragAnalysis && ragAnalysis.video_id);
      if (!vid) return;
    await fetch(`${API_BASE_URL}/api/videos/${vid}/recommendations`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ recommendations: dashboardRecommendations }) });
      setRecDialogOpen(false);
    } catch (err) { console.error('Save failed', err); }
  };

  const [applyResult, setApplyResult] = useState(null);

  const applyRecommendations = async () => {
    try {
      const vid = (ragAnalysis && ragAnalysis.video_id);
      if (!vid) return;
      const payload = { recommendations: dashboardRecommendations };
      const resp = await fetch(`${API_BASE_URL}/api/videos/${vid}/apply`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!resp.ok) throw new Error('Apply failed');
      const data = await resp.json();
      setApplyResult(data);
    } catch (err) {
      console.error('Apply failed', err);
      setApplyResult({ error: err.message });
    }
  };

  // note: openRecommendationsDialog removed (used inline in UI) to avoid unused-variable lint

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
  const frameRegex = /- Frame (\d+) \(t=([\d.]+)s\): ([^\n]+)\n([\s\S]*?)(?=(?:- Frame \d+ \(t=|$))/g;
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
  // Rounded compliance percent for dashboard and color bands
  const compliancePct = totalFrames ? Math.round((safeCount / totalFrames) * 100) : 0;
  let complianceColor = '#ffd54f'; // default (yellow)
  if (compliancePct < 50) complianceColor = '#e53935'; // red
  else if (compliancePct >= 50 && compliancePct < 70) complianceColor = '#fb8c00'; // orange (barely passing)
  else if (compliancePct >= 90) complianceColor = '#a5d6a7'; // green
  const [sidebarOpen, setSidebarOpen] = useState(true);
  // Recommendations dialog state
  const [recDialogOpen, setRecDialogOpen] = useState(false);
  const [dashboardRecommendations, setDashboardRecommendations] = useState([]);
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
        {/* Only show grouped policy explanations and summary, no frame-by-frame cards */}
        <RagAnalysisPage ragAnalysis={ragAnalysis} onBack={() => setShowRagPage(false)} />
      </Box>
    );
  } else if (showHistory) {
    mainContent = <UploadHistoryPage />;
  } else if (showFrameDetails) {
  // Prefer the rich `frame_details` produced by the backend (contains blip_description, ocr_text, etc.)
  // Fallback order:
  // 1) analysis.analysis_json.frame_details (newer backends)
  // 2) analysis.frames (some backends return frames directly when video is safe)
  // 3) ragAnalysis.retrieved_docs (legacy RAG-style payloads)
  const frameSource = analysis?.analysis_json?.frame_details || analysis?.frames || ragAnalysis?.retrieved_docs || [];
    mainContent = <FrameDetailsPage frames={frameSource} onBack={() => setShowFrameDetails(false)} />;
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

  // recommendations dialog is rendered inside the returned JSX below
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
                      Compliance Rate: <span style={{ color: complianceColor }}>{compliancePct}%</span>
                    </Typography>
                    <Button variant="contained" color="secondary" sx={{ ml: 2 }} onClick={() => {
                      // generate recommendations from current ragAnalysis
                      const recs = generateRecommendationsFromAnalysis(ragAnalysis || analysis);
                      setDashboardRecommendations(recs);
                      setRecDialogOpen(true);
                    }}>
                      Generate Fixes & Recommendations
                    </Button>
                  </Box>
                )}
          </Toolbar>
        </AppBar>
        <Box sx={{ flex: 1, overflow: 'auto', p: 0, m: 0 }}>
          {mainContent}
        </Box>
        {/* Recommendations Dialog */}
        <Dialog open={recDialogOpen} onClose={closeRecommendationsDialog} maxWidth="md" fullWidth>
          <DialogTitle>Personalized Fixes & Recommendations</DialogTitle>
          <DialogContent dividers>
            <List>
              {dashboardRecommendations && dashboardRecommendations.length > 0 ? dashboardRecommendations.map((r, i) => (
                <ListItem key={i} alignItems="flex-start">
                  <ListItemText primary={r.text} secondary={r.samples && r.samples.length > 0 ? `Examples: ${r.samples.map(s => (s.ts !== undefined ? `${s.ts}s` : (s.frame!==undefined ? `#${s.frame}` : ''))).join(', ')}` : null} />
                </ListItem>
              )) : <ListItem><ListItemText primary="No recommendations available." /></ListItem>}
            </List>
          </DialogContent>
            <DialogActions>
            <Tooltip title="Copy recommendations to clipboard">
              <IconButton onClick={copyRecommendationsToClipboard}><ContentCopyIcon /></IconButton>
            </Tooltip>
            <Tooltip title="Download JSON">
              <IconButton onClick={downloadRecommendationsJSON}><DownloadIcon /></IconButton>
            </Tooltip>
            <Button onClick={applyRecommendations} variant="contained" color="warning">Apply to Video</Button>
            <Button onClick={saveRecommendationsToBackend} variant="contained" color="primary">Save to Backend</Button>
            <Button onClick={closeRecommendationsDialog}>Close</Button>
          </DialogActions>
        </Dialog>
        {applyResult && (
          <Box sx={{ p: 2 }}>
            {applyResult.error ? (
              <Typography color="error">Apply failed: {applyResult.error}</Typography>
            ) : (
              <>
                <Typography>Applied successfully. Result: </Typography>
                <a href={applyResult.result_path} target="_blank" rel="noreferrer">{applyResult.result_path}</a>
              </>
            )}
          </Box>
        )}
      </Box>
    </Box>
  );
}

export default App;