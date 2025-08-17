
import React, { useState, useEffect } from 'react';
import RagAnalysisPage from './RagAnalysisPage';
import SettingsPage from './SettingsPage';
import UploadHistoryPage from './UploadHistoryPage';
import FrameDetailsPage from './FrameDetailsPage';
import {
  Container,
  AppBar,
  Toolbar,
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  CircularProgress,
  Alert,
  LinearProgress,
  Paper,
  Grid
} from '@mui/material';
import CloudUpload from '@mui/icons-material/CloudUpload';
import VideoLibrary from '@mui/icons-material/VideoLibrary';
import CheckCircle from '@mui/icons-material/CheckCircle';
import Analytics from '@mui/icons-material/Analytics';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [progress, setProgress] = useState(0);
  const [showRagPage, setShowRagPage] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [ragAnalysis, setRagAnalysis] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [showFrameDetails, setShowFrameDetails] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [parsedReport, setParsedReport] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('testing');
  const [frameSettings, setFrameSettings] = useState({
    frameCount: 20,
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
      const formData = new FormData();
      formData.append('file', selectedFile);
      Object.entries(frameSettings).forEach(([key, value]) => {
        if (value !== null && value !== undefined) formData.append(key, value);
      });
      const params = new URLSearchParams();
      Object.entries(frameSettings).forEach(([key, value]) => {
        if (value !== null && value !== undefined) params.append(key, value);
      });
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
      if (analysisObj.analysis_json && Object.keys(analysisObj.analysis_json).length > 0) {
        const aj = analysisObj.analysis_json;
        setParsedReport({
          duration: aj.video_info?.duration || aj.video_info?.duration_formatted || 'N/A',
          resolution: aj.video_info?.resolution || 'N/A',
          fps: aj.video_info?.fps || 'N/A',
          processing_time: aj.video_info?.analysis_time || aj.processing_time || 'N/A',
          text_score: aj.frame_analysis?.average_confidence ? (aj.frame_analysis.average_confidence * 100).toFixed(1) : 'N/A',
          text_status: aj.overall_assessment?.status || 'N/A',
          text_issues: aj.recommendations ? aj.recommendations.join('\n') : '',
          audio_score: aj.overall_assessment?.audio_compliance ? aj.overall_assessment.audio_compliance.toFixed(1) : 'N/A',
          audio_status: aj.overall_assessment?.status || 'N/A',
          audio_issues: aj.audio_analysis?.policy_flags ? Object.keys(aj.audio_analysis.policy_flags).join(', ') : '',
          image_score: aj.overall_assessment?.visual_compliance ? aj.overall_assessment.visual_compliance.toFixed(1) : 'N/A',
          image_status: aj.overall_assessment?.status || 'N/A',
          image_issues: aj.frame_analysis?.violation_categories ? aj.frame_analysis.violation_categories.join(', ') : '',
          frames_analyzed: aj.frame_analysis?.total_frames || 0,
          overall: aj.overall_assessment?.status || 'N/A',
          overall_scores: {
            text: aj.overall_assessment?.visual_compliance ? aj.overall_assessment.visual_compliance.toFixed(1) : 'N/A',
            audio: aj.overall_assessment?.audio_compliance ? aj.overall_assessment.audio_compliance.toFixed(1) : 'N/A',
            image: aj.overall_assessment?.overall_compliance ? aj.overall_assessment.overall_compliance.toFixed(1) : 'N/A'
          }
        });
      } else {
        setParsedReport(parseReportFields(analysisObj.full_report));
      }
    } catch (err) {
      setError(`Analysis failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Utility function for parsing report fields (restored to previous working version)
  function parseReportFields(report) {
    if (!report) return {};
    const get = (block, regex) => {
      const m = block.match(regex);
      return m ? m[1].trim() : undefined;
    };
    const getIssues = (block) => {
      const m = block.match(/Results:\s*([\s\S]*?)(?=\n\w|\n$|$)/i);
      if (!m) return [];
      const issues = m[1];
      return issues.split(/\n|•/).map(i => i.trim()).filter(i => i);
    };
    const extractBlock = (title) => {
      const regex = new RegExp(`${title}[\s\S]*?(?=(?:\n[A-Z][a-zA-Z ]+:|\n$|$))`, 'i');
      const m = report.match(regex);
      return m ? m[0] : '';
    };
    const textBlock = extractBlock('Text Analysis');
    const audioBlock = extractBlock('Audio Analysis');
    const imageBlock = extractBlock('Video Frame Analysis');
    const overallBlock = extractBlock('Overall YouTube Compliance Report');
    return {
      duration: get(report, /Duration\s*:?\s*([\d\.]+)/i),
      resolution: get(report, /Resolution\s*:?\s*([\dx]+)/i),
      fps: get(report, /FPS\s*:?\s*([\d\.]+)/i),
      processing_time: get(report, /Processing Time\s*:?\s*([\d\.]+)/i),
      text_score: get(textBlock, /Score\s*:?\s*(\d+\.?\d*)%/i) || get(textBlock, /(\d+\.?\d*)%/i),
      text_status: get(textBlock, /Status\s*:?\s*(conforme|attention|non_conforme|N\/A|MINOR_VIOLATIONS)/i) || get(textBlock, /(conforme|attention|non_conforme|N\/A|MINOR_VIOLATIONS)/i),
      text_issues: getIssues(textBlock),
      audio_score: get(audioBlock, /Score\s*:?\s*(\d+\.?\d*)%/i) || get(audioBlock, /(\d+\.?\d*)%/i),
      audio_status: get(audioBlock, /Status\s*:?\s*(conforme|attention|non_conforme|N\/A|MINOR_VIOLATIONS)/i) || get(audioBlock, /(conforme|attention|non_conforme|N\/A|MINOR_VIOLATIONS)/i),
      audio_issues: getIssues(audioBlock),
      image_score: get(imageBlock, /Score\s*:?\s*(\d+\.?\d*)%/i) || get(imageBlock, /(\d+\.?\d*)%/i),
      image_status: get(imageBlock, /Status\s*:?\s*(conforme|attention|non_conforme|N\/A|MINOR_VIOLATIONS)/i) || get(imageBlock, /(conforme|attention|non_conforme|N\/A|MINOR_VIOLATIONS)/i),
      image_issues: getIssues(imageBlock),
      frames_analyzed: get(imageBlock, /Frames analyzed\s*:?\s*(\d+)/i),
      overall: get(overallBlock, /Overall Status\s*:?\s*(conforme|attention|non_conforme|N\/A|MINOR_VIOLATIONS)/i) || get(overallBlock, /(conforme|attention|non_conforme|N\/A|MINOR_VIOLATIONS)/i),
      overall_scores: {
        text: get(overallBlock, /Text Content \(20%\)\s*(\d+\.?\d*)%/i),
        audio: get(overallBlock, /Audio Content \(35%\)\s*(\d+\.?\d*)%/i),
        image: get(overallBlock, /Video Frames \(45%\)\s*(\d+\.?\d*)%/i)
      }
    };
  }

  useEffect(() => {
    // Always use frameSettings for RAG and frame analysis
    if (analysis && (analysis.rag_explanations || (analysis.analysis_json && analysis.analysis_json.rag_explanations))) {
      let ragFrames = analysis.rag_explanations || analysis.analysis_json.rag_explanations;
      ragFrames = ragFrames.map((frame) => {
        if (!frame.category || frame.category === '' || frame.category === 'N/A') {
          return { ...frame, category: 'safe_content', reasoning: 'No violation detected. Content is safe.' };
        }
        return frame;
      });
      if (frameSettings && frameSettings.frameCount && ragFrames.length > frameSettings.frameCount) {
        ragFrames = ragFrames.slice(0, frameSettings.frameCount);
      }
      setRagAnalysis({
        decision: analysis.analysis_json?.overall_assessment?.status || 'N/A',
        confidence: analysis.analysis_json?.frame_analysis?.average_confidence || 'N/A',
        reasoning: 'Detailed frame-level policy violation explanations below.',
        retrieved_docs: ragFrames.map((frame) => ({
          title: frame.type === 'visual'
            ? `Frame ${frame.frame} (t=${frame.timestamp}s): ${frame.category} detected by model (confidence ${frame.confidence})`
            : `Audio: ${frame.category} detected in transcript (severity: ${frame.severity})`,
          category: frame.category,
          confidence: frame.confidence,
          reasoning: frame.reasoning
        }))
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

  const safeCount = ragAnalysis && ragAnalysis.retrieved_docs
    ? ragAnalysis.retrieved_docs.filter(doc => doc.category === 'safe_content').length
    : 0;
  const totalFrames = ragAnalysis && ragAnalysis.retrieved_docs ? ragAnalysis.retrieved_docs.length : 0;

  const sidebarOptions = [
    { label: 'SETTINGS', color: 'secondary', onClick: () => { setShowSettings(true); setShowHistory(false); setShowFrameDetails(false); setShowRagPage(false); } },
    { label: 'RAG ANALYSIS', color: 'info', onClick: () => { setShowRagPage(true); setShowSettings(false); setShowHistory(false); setShowFrameDetails(false); } },
    { label: 'UPLOAD HISTORY', color: 'primary', onClick: () => { setShowHistory(true); setShowFrameDetails(false); setShowRagPage(false); setShowSettings(false); } },
    { label: 'FRAME DETAILS', color: 'info', onClick: () => { setShowFrameDetails(true); setShowHistory(false); setShowRagPage(false); setShowSettings(false); } },
    { label: 'CUSTOMIZE', color: 'warning', onClick: () => alert('Customize feature coming soon!') },
  ];

  if (showRagPage) {
    return (
      <>
        <Box sx={{ p: 3, mb: 2 }}>
          <Typography variant="h5" color="success.main">
            Safe Content Frames: {safeCount} / {totalFrames}
          </Typography>
          <LinearProgress variant="determinate" value={totalFrames ? (safeCount / totalFrames) * 100 : 0} sx={{ height: 10, borderRadius: 5, mt: 1 }} color="success" />
        </Box>
        <RagAnalysisPage ragAnalysis={ragAnalysis} onBack={() => setShowRagPage(false)} />
      </>
    );
  }
  if (showSettings) {
    return <SettingsPage settings={frameSettings} onSave={settings => { setFrameSettings(settings); setShowSettings(false); }} />;
  }
  if (showHistory) {
    return <UploadHistoryPage />;
  }
  if (showFrameDetails) {
    return <FrameDetailsPage frames={ragAnalysis?.retrieved_docs || []} />;
  }
  // Main page: preserve analysis and ragAnalysis state

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i]);
  };

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', backgroundColor: '#f5f5f5' }}>
      {/* Sidebar */}
      <Box sx={{ width: 160, background: 'linear-gradient(180deg,#1976d2 0%,#21cbf3 100%)', color: '#fff', display: { xs: 'none', md: 'flex' }, flexDirection: 'column', alignItems: 'center', py: 2, boxShadow: 2 }}>
        <Box sx={{ mb: 3 }}>
          <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="Menu" style={{ width: 56, borderRadius: '50%' }} />
        </Box>
        <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1, letterSpacing: 1 }}>
          Menu
        </Typography>
        {sidebarOptions.map(opt => (
          <Button key={opt.label} variant="contained" color={opt.color} sx={{ mb: 1, width: '95%', fontWeight: 'bold', fontSize: '0.95rem', borderRadius: 2, boxShadow: 1, textTransform: 'none', py: 1 }} onClick={opt.onClick}>
            {opt.label}
          </Button>
        ))}
      </Box>
      {/* Main content */}
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: '100vh', background: '#f5f5f5', px: 1 }}>
        <AppBar position="static" sx={{ background: 'linear-gradient(90deg,#1976d2 60%,#21cbf3 100%)', boxShadow: 2 }}>
          <Toolbar>
            <Typography variant="h4" sx={{ fontWeight: 'bold', flexGrow: 1, letterSpacing: 1, color: '#fff' }}>
              AI Video Content Analyzer
            </Typography>
          </Toolbar>
        </AppBar>
        <Container maxWidth="md" sx={{ mt: 2, pb: 2 }}>
          <Grid container spacing={2} alignItems="stretch" justifyContent="center">
            {/* Welcome/Info Section */}
            <Grid item xs={12} md={6} lg={6}>
              <Box sx={{
                minHeight: 180,
                width: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(120deg, #1976d2 60%, #21cbf3 100%)',
                borderRadius: 4,
                boxShadow: '0 4px 16px rgba(33,203,243,0.10)',
                px: 3,
                py: 3,
                mt: 1,
              }}>
                <Box display="flex" alignItems="center" justifyContent="center" mb={2}>
                  <VideoLibrary sx={{ fontSize: 72, mr: 2, color: '#fff' }} />
                  <Analytics sx={{ fontSize: 56, mr: 2, color: '#fffde7' }} />
                  <CheckCircle sx={{ fontSize: 56, color: '#e8f5e9' }} />
                </Box>
                <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#fff', mb: 1, textShadow: '1px 1px 8px #1976d2', letterSpacing: 1 }}>
                  Welcome to AI Video Content Analyzer
                </Typography>
                <Typography variant="subtitle1" sx={{ color: '#e3f2fd', mb: 1, fontWeight: 500 }}>
                  Analyze your videos for YouTube policy compliance in seconds!
                </Typography>
                <Typography variant="body2" sx={{ color: '#fff', mb: 2, maxWidth: 400, textAlign: 'center', fontSize: '1rem' }}>
                  <b>How it works:</b> Upload a video file and get instant feedback on compliance, violations, and recommendations.<br />
                  Powered by advanced AI models for text, audio, and image analysis.
                </Typography>
                <Typography variant="body2" sx={{ color: '#fffde7', mb: 2, fontSize: '0.95rem' }}>
                  <b>Supported formats:</b> MP4, AVI, MOV, MKV
                </Typography>
                <Button
                  variant="contained"
                  color="secondary"
                  size="medium"
                  sx={{ fontWeight: 'bold', px: 3, py: 1, fontSize: '1rem', boxShadow: 2, borderRadius: 2, background: 'linear-gradient(90deg,#8e24aa 60%,#d1c4e9 100%)', ':hover': { background: 'linear-gradient(90deg,#6a1b9a 60%,#b39ddb 100%)', transform: 'scale(1.03)' } }}
                  onClick={() => document.getElementById('video-upload').click()}
                >
                  <CloudUpload sx={{ mr: 1, fontSize: 24 }} /> START ANALYSIS
                </Button>
              </Box>
            </Grid>
            {/* Upload Section */}
            <Grid item xs={12} md={6} lg={6}>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%' }}>
                <Card sx={{ width: '100%', maxWidth: 320, mx: 'auto', p: 2, boxShadow: '0 2px 8px rgba(33,203,243,0.08)', borderRadius: 3, background: '#fff' }}>
                  <CardContent>
                    <Typography variant="subtitle1" align="center" sx={{ mb: 2, fontWeight: 'bold', color: '#1976d2', letterSpacing: 1 }}>
                      Full Video Moderation (Upload)
                    </Typography>
                    {/* Display chosen frame settings */}
                    <Box sx={{ mb: 1, p: 1, background: '#e3f2fd', borderRadius: 1, fontSize: '0.95rem', color: '#1976d2' }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>Current Frame Extraction Settings:</Typography>
                      <ul style={{ margin: 0, paddingLeft: 20 }}>
                        <li>Frames: <b>{frameSettings.frameCount}</b></li>
                        <li>Interval: <b>{frameSettings.frameInterval}</b> sec</li>
                        <li>Resolution: <b>{frameSettings.frameResolution}</b></li>
                        <li>Format: <b>{frameSettings.frameFormat}</b></li>
                        <li>Start: <b>{frameSettings.frameStart}</b> sec</li>
                        <li>End: <b>{frameSettings.frameEnd !== null ? frameSettings.frameEnd : 'Full video'}</b></li>
                        <li>Sampling: <b>{frameSettings.frameSampling}</b></li>
                      </ul>
                    </Box>
                    <Box
                      sx={{
                        border: '2px dashed #1976d2',
                        borderRadius: 2,
                        p: 2,
                        mb: 2,
                        textAlign: 'center',
                        background: '#e3f2fd',
                        cursor: 'pointer',
                        transition: '0.2s',
                        ':hover': { background: '#bbdefb', borderColor: '#1565c0' },
                      }}
                      onClick={() => document.getElementById('video-upload').click()}
                      onDragOver={e => e.preventDefault()}
                      onDrop={e => {
                        e.preventDefault();
                        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                          handleFileChange({ target: { files: e.dataTransfer.files } });
                        }
                      }}
                    >
                      <input
                        accept="video/*"
                        style={{ display: 'none' }}
                        id="video-upload"
                        type="file"
                        onChange={handleFileChange}
                      />
                      {selectedFile ? (
                        <Typography variant="body2" color="primary" sx={{ fontWeight: 'bold', fontSize: '1rem' }}>
                          {selectedFile.name} ({formatFileSize(selectedFile.size)})
                        </Typography>
                      ) : (
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: '1rem' }}>
                          Drag & Drop a video here, or click to upload
                        </Typography>
                      )}
                    </Box>
                    <Button
                      variant="contained"
                      color="primary"
                      fullWidth
                      sx={{ fontWeight: 'bold', py: 1, fontSize: '1rem', borderRadius: 2, boxShadow: 1, background: 'linear-gradient(90deg,#1976d2 60%,#21cbf3 100%)', ':hover': { background: 'linear-gradient(90deg,#1565c0 60%,#039be5 100%)', transform: 'scale(1.01)' } }}
                      onClick={handleAnalyze}
                      disabled={loading || !selectedFile || connectionStatus !== 'connected'}
                    >
                      {loading ? 'Uploading...' : 'UPLOAD VIDEO'}
                    </Button>
                    {error && (
                      <Alert severity="error" sx={{ mt: 3 }}>
                        {error}
                      </Alert>
                    )}
                  </CardContent>
                </Card>
                {/* Progress Bar with Percentage */}
                {loading && (
                  <Paper sx={{ p: 1, mb: 2, width: '100%' }}>
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
                {/* Simple Model Report (Analysis Results) */}
                {analysis && (
                  <Card sx={{ mt: 2, boxShadow: 2, borderRadius: 2 }}>
                    <CardContent>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1 }} color="primary">
                        Model Analysis Summary
                      </Typography>
                      <Typography variant="body2">Score: <b>{parsedReport?.image_score || 'N/A'}%</b></Typography>
                      <Typography variant="body2">Status: <b>{parsedReport?.overall || 'N/A'}</b></Typography>
                      <Typography variant="body2">Frames Analyzed: <b>{parsedReport?.frames_analyzed || 'N/A'}</b></Typography>
                      <Button variant="outlined" color="info" sx={{ mt: 1, fontSize: '0.95rem', py: 0.5 }} onClick={() => setShowFrameDetails(true)}>
                        View Detailed Frame Report
                      </Button>
                    </CardContent>
                  </Card>
                )}
              </Box>
            </Grid>
          </Grid>
          {connectionStatus === 'disconnected' && (
            <Alert severity="error" sx={{ mb: 4 }} action={
              <Button color="inherit" size="small" onClick={testConnection}>
                Retry
              </Button>
            }>
              Cannot connect to backend server. Make sure the server is running on http://localhost:8000
            </Alert>
          )}
          {connectionStatus === 'connected' && (
            <Alert severity="success" sx={{ mb: 4 }}>
              ✅ AI Video Analyzer ready! Upload and analyze your video files.
            </Alert>
          )}
        </Container>
      </Box>
    </Box>
  );

}

export default App;