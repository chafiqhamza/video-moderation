
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
  const [copyrightStatus, setCopyrightStatus] = useState(null);
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
    setCopyrightStatus(null);
    try {
      // Build params based on extraction mode
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
      // Run main analysis (now includes copyright check)
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
      // Set copyright status from main analysis response
      if (analysisObj.copyright_check) {
        setCopyrightStatus(analysisObj.copyright_check);
      } else {
        setCopyrightStatus(null);
      }
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
      // Show RAG explanations if present
      if (analysisObj.rag_explanations && analysisObj.rag_explanations.length > 0) {
        setRagAnalysis({
          decision: analysisObj.analysis_json?.overall_assessment?.status || '',
          confidence: analysisObj.analysis_json?.overall_assessment?.overall_compliance || '',
          reasoning: analysisObj.analysis_json?.recommendations?.join('\n') || '',
          retrieved_docs: analysisObj.rag_explanations
        });
        setShowRagPage(true);
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
      // Use backend's rag_explanations array directly for full details
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

  const safeCount = ragAnalysis && ragAnalysis.retrieved_docs
    ? ragAnalysis.retrieved_docs.filter(doc => doc.category === 'safe_content').length
    : 0;
  const totalFrames = ragAnalysis && ragAnalysis.retrieved_docs ? ragAnalysis.retrieved_docs.length : 0;

  const [sidebarOpen, setSidebarOpen] = useState(true);
  const sidebarOptions = [
    { label: 'SETTINGS', color: 'secondary', icon: <Analytics />, tooltip: 'Configure frame extraction and analysis settings', onClick: () => { setShowSettings(true); setShowHistory(false); setShowFrameDetails(false); setShowRagPage(false); } },
    { label: 'RAG ANALYSIS', color: 'info', icon: <CheckCircle />, tooltip: 'View policy violation explanations', onClick: () => { setShowRagPage(true); setShowSettings(false); setShowHistory(false); setShowFrameDetails(false); } },
    { label: 'UPLOAD HISTORY', color: 'primary', icon: <VideoLibrary />, tooltip: 'See previous uploads and results', onClick: () => { setShowHistory(true); setShowFrameDetails(false); setShowRagPage(false); setShowSettings(false); } },
    { label: 'FRAME DETAILS', color: 'info', icon: <CloudUpload />, tooltip: 'Detailed report of analyzed frames', onClick: () => { setShowFrameDetails(true); setShowHistory(false); setShowRagPage(false); setShowSettings(false); } },
    { label: 'CUSTOMIZE', color: 'warning', icon: <Analytics />, tooltip: 'Customize moderation options', onClick: () => alert('Customize feature coming soon!') },
  ];

  // Instead of returning early, render all pages conditionally inside main JSX so state is always preserved
  // Main page: preserve analysis and ragAnalysis state

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i]);
  };

  // Conditional rendering for all pages
  let mainContent = null;
  if (showSettings) {
    mainContent = <SettingsPage settings={frameSettings} onSave={settings => { setFrameSettings(settings); setShowSettings(false); }} />;
  } else if (showRagPage) {
    mainContent = <><Box sx={{ p: 3, mb: 2 }}>
      <Typography variant="h5" color="success.main">
        Safe Content Frames: {safeCount} / {totalFrames}
      </Typography>
      <LinearProgress variant="determinate" value={totalFrames ? (safeCount / totalFrames) * 100 : 0} sx={{ height: 10, borderRadius: 5, mt: 1 }} color="success" />
      {/* Always show model analysis summary above RAG */}
      {parsedReport && (
        <Card sx={{ mt: 2, boxShadow: 2, borderRadius: 2 }}>
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
      <RagAnalysisPage ragAnalysis={ragAnalysis} onBack={() => setShowRagPage(false)} /></>;
  } else if (showHistory) {
    mainContent = <UploadHistoryPage />;
  } else if (showFrameDetails) {
    mainContent = <FrameDetailsPage frames={ragAnalysis?.retrieved_docs || []} onBack={() => setShowFrameDetails(false)} />;
  } else {
    mainContent = (
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
                    {/* Copyright Violation Status - always show a clear message */}
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="body2" color={copyrightStatus?.violation ? 'error' : 'success.main'}>
                        Copyright Violation: <b>{copyrightStatus?.violation ? 'Yes' : 'No'}</b>
                      </Typography>
                      {/* Show details if present */}
                      {copyrightStatus && (
                        <Box sx={{ mt: 1 }}>
                          {copyrightStatus?.result && typeof copyrightStatus.result === 'string' && (
                            <Typography variant="body2" sx={{ mt: 1 }}>
                              {copyrightStatus.result}
                            </Typography>
                          )}
                          {copyrightStatus?.result && typeof copyrightStatus.result === 'object' && (
                            <Box sx={{ mt: 1, maxHeight: 120, overflow: 'auto', background: '#f5f5f5', borderRadius: 1, p: 1 }}>
                              <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>AudD API Details:</Typography>
                              <pre style={{ fontSize: '0.85rem', margin: 0 }}>{JSON.stringify(copyrightStatus.result, null, 2)}</pre>
                            </Box>
                          )}
                          {copyrightStatus?.error && (
                            <Typography variant="body2" color="error" sx={{ mt: 1 }}>
                              {typeof copyrightStatus.error === 'object' ? JSON.stringify(copyrightStatus.error) : copyrightStatus.error}
                            </Typography>
                          )}
                        </Box>
                      )}
                    </Box>
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
    );
  }

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', backgroundColor: '#f5f5f5' }}>
      {/* Sidebar */}
      <Box sx={{ width: sidebarOpen ? 180 : 60, transition: 'width 0.2s', background: 'linear-gradient(180deg,#1976d2 0%,#21cbf3 100%)', color: '#fff', display: { xs: 'none', md: 'flex' }, flexDirection: 'column', alignItems: 'center', py: 2, boxShadow: 2, position: 'relative' }}>
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
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: '100vh', background: '#f5f5f5', px: 1 }}>
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
        {mainContent}
      </Box>
    </Box>
  );

}

export default App;