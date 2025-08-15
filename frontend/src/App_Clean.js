import React, { useState, useEffect } from 'react';
import RagAnalysisPage from './RagAnalysisPage';
import {
  Container,
  AppBar,
  Toolbar,
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  Grid,
  CircularProgress,
  Alert,
  Chip,
  LinearProgress,
  Paper
} from '@mui/material';
import CloudUpload from '@mui/icons-material/CloudUpload';
import VideoLibrary from '@mui/icons-material/VideoLibrary';
import CheckCircle from '@mui/icons-material/CheckCircle';
import Warning from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
import Analytics from '@mui/icons-material/Analytics';
import Image from '@mui/icons-material/Image';
import RecordVoiceOver from '@mui/icons-material/RecordVoiceOver';
import TextFields from '@mui/icons-material/TextFields';

const API_BASE_URL = 'http://localhost:8000';

// Helper component to parse and display the comprehensive report interactively
function ComprehensiveReportDisplay({ report }) {
  const [expandedSections, setExpandedSections] = useState({});

  if (!report) return null;

  // Split report into sections by headers (e.g., '====', '----', etc.)
  const sectionRegex = /(^|\n)(={4,}|-{4,}|\*{4,})\s*(.+?)\s*(={4,}|-{4,}|\*{4,})/g;
  let sections = [];
  let sectionHeaders = [];
  let match;
  let lastIndex = 0;

  // Reset regex lastIndex for multiple executions
  sectionRegex.lastIndex = 0;

  while ((match = sectionRegex.exec(report)) !== null) {
    const header = match[3].trim();
    sectionHeaders.push({ header, start: match.index });
  }

  for (let i = 0; i < sectionHeaders.length; i++) {
    const start = sectionHeaders[i].start;
    const end = i + 1 < sectionHeaders.length ? sectionHeaders[i + 1].start : report.length;
    const header = sectionHeaders[i].header;
    const content = report.slice(start, end).trim();
    sections.push({ header, content });
  }

  // If no sections found, fallback to whole report
  if (sections.length === 0) {
    sections = [{ header: 'Full Report', content: report }];
  }

  // Helper to color-code section headers
  const getSectionColor = (header) => {
    if (/violation|non[-_ ]?conforme|policy violation/i.test(header)) return 'error';
    if (/safe|conforme|approved/i.test(header)) return 'success';
    if (/suggestive|attention|warning/i.test(header)) return 'warning';
    return 'default';
  };

  // Helper to get icon
  const getSectionIcon = (header) => {
    if (/violation|non[-_ ]?conforme|policy violation/i.test(header)) return <ErrorIcon />;
    if (/safe|conforme|approved/i.test(header)) return <CheckCircle />;
    if (/suggestive|attention|warning/i.test(header)) return <Warning />;
    return <Analytics />;
  };

  return (
    <Box>
      {sections.map((section, idx) => {
        const expanded = expandedSections[idx] || false;
        const color = getSectionColor(section.header);
        const icon = getSectionIcon(section.header);
        return (
          <Card key={idx} sx={{ mb: 2, backgroundColor: color === 'error' ? '#ffebee' : color === 'success' ? '#e8f5e9' : color === 'warning' ? '#fffde7' : '#f5f5f5', boxShadow: 1 }}>
            <CardContent>
              <Box display="flex" alignItems="center" justifyContent="space-between">
                <Box display="flex" alignItems="center">
                  <Chip icon={icon} label={section.header} color={color} variant="filled" sx={{ fontWeight: 'bold', fontSize: '1rem', mr: 2 }} />
                </Box>
                <Button size="small" onClick={() => setExpandedSections(prev => ({ ...prev, [idx]: !expanded }))}>
                  {expanded ? 'Hide Details' : 'Show Details'}
                </Button>
              </Box>
              <Box sx={{ mt: 1, p: 1, background: '#fff', borderRadius: 1, fontFamily: 'monospace', whiteSpace: 'pre-wrap', fontSize: '0.95rem', color: '#333', maxHeight: expanded ? 600 : 120, overflow: 'auto' }}>
                {expanded ? section.content : section.content.slice(0, 300) + (section.content.length > 300 ? '... (truncated)' : '')}
              </Box>
            </CardContent>
          </Card>
        );
      })}
    </Box>
  );
}

function App() {
  // Progress bar state
  const [progress, setProgress] = useState(0);
  // RAG analysis state
  const [showRagPage, setShowRagPage] = useState(false);
  const [ragAnalysis, setRagAnalysis] = useState(null);
  // Helper to extract main status/category from report text
  function extractCategory(report) {
    if (!report) return 'Unknown';
    if (report.includes('Policy Violation Detected')) return 'Violation';
    if (report.includes('safe_content')) return 'Safe';
    if (report.includes('suggestive_content')) return 'Suggestive';
    return 'Other';
  }

  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  // Track expanded state for each history item by id
  const [expandedHistory, setExpandedHistory] = useState({});

  const fetchHistory = async () => {
    setHistoryLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/videos`);
      if (!response.ok) throw new Error('Failed to fetch history');
      const data = await response.json();
      setHistory(data);
      // Reset expanded state when history is fetched
      setExpandedHistory({});
    } catch (err) {
      setError('Could not load history');
    } finally {
      setHistoryLoading(false);
    }
  };

  const [selectedFile, setSelectedFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  // Parsed fields from the raw report
  const [parsedReport, setParsedReport] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('testing');

  // Test connection on startup
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
        setAnalysis(null); // Clear previous results
      } else {
        setError('Please select a valid video file');
        setSelectedFile(null);
      }
    }
  };

  useEffect(() => {
    let timer;
    if (loading) {
      setProgress(0);
      timer = setInterval(() => {
        setProgress((old) => (old < 95 ? old + 5 : old));
      }, 500);
    } else {
      setProgress(100);
    }
    return () => clearInterval(timer);
  }, [loading]);

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select a video file');
      return;
    }

    setLoading(true);
    setError('');
    setAnalysis(null);

    // Helper to robustly parse the raw report string into structured fields
    function parseReportFields(report) {
      if (!report) return {};
      // Helper to extract a block by header (case-insensitive, flexible)
      const extractBlock = (header) => {
        const regex = new RegExp(`${header}[\s\S]*?(?=\n\n|$|Text Analysis|Audio Analysis|Video Frame Analysis|Overall YouTube Compliance Report)`, 'i');
        const match = report.match(regex);
        return match ? match[0] : '';
      };
      // Helper to extract a value by regex from a block
      const get = (block, regex) => {
        const m = block.match(regex);
        return m ? m[1].trim() : undefined;
      };
      // Helper to extract issues/results (multi-line, after 'Results:')
      const getIssues = (block) => {
        const m = block.match(/Results:\s*([\s\S]*?)(?=\n\w|\n$|$)/i);
        if (m) {
          // Remove any trailing section headers or extra whitespace
          let issues = m[1].replace(/(={2,}|-{2,}|\*{2,}|Text Analysis|Audio Analysis|Video Frame Analysis|Overall YouTube Compliance Report)[\s\S]*$/i, '').trim();
          // Split by bullet or newline, filter empty
          return issues.split(/\n|•/).map(i => i.trim()).filter(i => i);
        }
        return [];
      };
      // Extract blocks
      const textBlock = extractBlock('Text Analysis');
      const audioBlock = extractBlock('Audio Analysis');
      const imageBlock = extractBlock('Video Frame Analysis');
      const overallBlock = extractBlock('Overall YouTube Compliance Report');
      // Main fields
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

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      console.log('Uploading file:', selectedFile.name);

      const response = await fetch(`${API_BASE_URL}/upload-video`, {
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

      // Use structured JSON if available, else fallback to raw parsing
      if (analysisObj.analysis_json && Object.keys(analysisObj.analysis_json).length > 0) {
        const aj = analysisObj.analysis_json;
        // Map backend JSON to frontend fields
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
        // Fallback to raw report parsing
        setParsedReport(parseReportFields(analysisObj.full_report));
      }

    } catch (err) {
      console.error('Analysis error:', err);
      setError(`Analysis failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'conforme': return 'success';
      case 'attention': return 'warning';
      case 'non_conforme': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'conforme': return <CheckCircle />;
      case 'attention': return <Warning />;
      case 'non_conforme': return <ErrorIcon />;
      default: return null;
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i]);
  };

  // Parse RAG analysis from backend report JSON
  useEffect(() => {
    if (analysis && (analysis.rag_explanations || (analysis.analysis_json && analysis.analysis_json.rag_explanations))) {
      // Prefer top-level rag_explanations, fallback to analysis_json
      let ragFrames = analysis.rag_explanations || analysis.analysis_json.rag_explanations;
      // Fallback: if no violation detected, mark as safe_content
      ragFrames = ragFrames.map((frame) => {
        if (!frame.category || frame.category === '' || frame.category === 'N/A') {
          return { ...frame, category: 'safe_content', reasoning: 'No violation detected. Content is safe.' };
        }
        return frame;
      });
      setRagAnalysis({
        decision: analysis.analysis_json?.overall_assessment?.status || 'N/A',
        confidence: analysis.analysis_json?.frame_analysis?.average_confidence || 'N/A',
        reasoning: 'Detailed frame-level policy violation explanations below.',
        retrieved_docs: ragFrames.map((frame, idx) => ({
          title: frame.type === 'visual'
            ? `Frame ${frame.frame} (t=${frame.timestamp}s): ${frame.category} detected by model (confidence ${frame.confidence})`
            : `Audio: ${frame.category} detected in transcript (severity: ${frame.severity})`,
          category: frame.category,
          confidence: frame.confidence,
          blip: frame.blip,
          ocr: frame.ocr,
          policy: frame.policy,
          examples: frame.examples,
          action_required: frame.action_required,
          severity_indicators: frame.severity_indicators,
          context_factors: frame.context_factors,
          reasoning: frame.reasoning,
          transcript: frame.transcript,
          keywords: frame.keywords
        }))
      });
    } else if (analysis && analysis.full_report) {
      // Fallback: parse DETAILED POLICY VIOLATION EXPLANATIONS (RAG) from raw report text
      const ragSectionMatch = analysis.full_report.match(/DETAILED POLICY VIOLATION EXPLANATIONS \(RAG\):([\s\S]*?)(?=\n\s*\n|$)/i);
      if (ragSectionMatch) {
        const ragText = ragSectionMatch[1];
        // Split by frame
        const frameRegex = /- Frame (\d+) \(t=([\d\.]+)s\): ([^\n]+)\n([\s\S]*?)(?=(?:- Frame \d+ \(t=|$))/g;
        let frames = [];
        let match;
        while ((match = frameRegex.exec(ragText)) !== null) {
          const frameIdx = match[1];
          const timestamp = match[2];
          const category = match[3] && match[3] !== '' ? match[3] : 'safe_content';
          const details = match[4];
          frames.push({
            title: `Frame ${frameIdx} (t=${timestamp}s): ${category}`,
            category,
            confidence: undefined,
            blip: undefined,
            ocr: undefined,
            policy: undefined,
            examples: undefined,
            action_required: undefined,
            severity_indicators: undefined,
            context_factors: undefined,
            reasoning: category === 'safe_content' ? 'No violation detected. Content is safe.' : undefined,
            transcript: undefined,
            keywords: undefined
          });
        }
        setRagAnalysis({
          decision: parsedReport?.overall || 'N/A',
          confidence: parsedReport?.image_score || 'N/A',
          reasoning: 'Detailed frame-level policy violation explanations below.',
          retrieved_docs: frames
        });
      } else {
        setRagAnalysis(null);
      }
    }
  }, [analysis, parsedReport]);

  // Count safe_content frames for summary
  const safeCount = ragAnalysis && ragAnalysis.retrieved_docs
    ? ragAnalysis.retrieved_docs.filter(doc => doc.category === 'safe_content').length
    : 0;
  const totalFrames = ragAnalysis && ragAnalysis.retrieved_docs ? ragAnalysis.retrieved_docs.length : 0;

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

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f5f5f5' }}>
      <Box sx={{ position: 'fixed', top: 16, right: 16, zIndex: 1000 }}>
        <Button variant="contained" color="secondary" onClick={() => { setShowHistory(!showHistory); if (!showHistory) fetchHistory(); }}>
          {showHistory ? 'Hide Upload History' : 'Show Upload History'}
        </Button>
      </Box>
      <AppBar position="static" sx={{ backgroundColor: '#1976d2' }}>
        <Toolbar>
          <VideoLibrary sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            AI Video Content Analyzer
          </Typography>
          <Chip
            icon={connectionStatus === 'connected' ? <CheckCircle /> : <ErrorIcon />}
            label={connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
            color={connectionStatus === 'connected' ? 'success' : 'error'}
            variant="filled"
            size="small"
          />
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, pb: 4 }}>
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
          <Button variant="contained" color="secondary" onClick={() => setShowRagPage(true)}>
            View RAG Analysis
          </Button>
        </Box>
        {/* Enhanced Welcome Banner */}
        <Box sx={{
          minHeight: 320,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'linear-gradient(120deg, #1976d2 60%, #21cbf3 100%)',
          borderRadius: 4,
          boxShadow: 4,
          mb: 5,
          px: 4,
          py: 5
        }}>
          <Box display="flex" alignItems="center" justifyContent="center" mb={2}>
            <VideoLibrary sx={{ fontSize: 64, mr: 2, color: '#fff' }} />
            <Analytics sx={{ fontSize: 48, mr: 2, color: '#fffde7' }} />
            <CheckCircle sx={{ fontSize: 48, color: '#e8f5e9' }} />
          </Box>
          <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#fff', mb: 1, textShadow: '1px 1px 4px #1976d2' }}>
            Welcome to AI Video Content Analyzer
          </Typography>
          <Typography variant="h5" sx={{ color: '#e3f2fd', mb: 2 }}>
            Analyze your videos for YouTube policy compliance in seconds!
          </Typography>
          <Typography variant="body1" sx={{ color: '#fff', mb: 2, maxWidth: 600, textAlign: 'center' }}>
            <b>How it works:</b> Upload a video file and get instant feedback on compliance, violations, and recommendations.<br />
            Powered by advanced AI models for text, audio, and image analysis.
          </Typography>
          <Typography variant="body2" sx={{ color: '#fffde7', mb: 3 }}>
            <b>Supported formats:</b> MP4, AVI, MOV, MKV
          </Typography>
          <Button
            variant="contained"
            color="secondary"
            size="large"
            sx={{ fontWeight: 'bold', px: 4, py: 1, fontSize: '1.2rem', boxShadow: 2 }}
            onClick={() => document.getElementById('video-upload').click()}
          >
            <CloudUpload sx={{ mr: 1 }} /> Start Analysis
          </Button>
        </Box>
        {showHistory && (
          <Card sx={{ mb: 4 }}>
            <CardContent>
              <Typography variant="h5" gutterBottom color="secondary">
                📜 Upload History
              </Typography>
              {historyLoading ? (
                <CircularProgress />
              ) : history.length === 0 ? (
                <Typography>No uploads found.</Typography>
              ) : (
                <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                  {history.map((item) => {
                    const category = extractCategory(item.report);
                    let color = 'default';
                    let icon = null;
                    if (category === 'Safe') { color = 'success'; icon = <CheckCircle />; }
                    else if (category === 'Violation') { color = 'error'; icon = <ErrorIcon />; }
                    else if (category === 'Suggestive') { color = 'warning'; icon = <Warning />; }
                    const expanded = expandedHistory[item.id] || false;
                    return (
                      <Card key={item.id} sx={{ mb: 2, backgroundColor: '#f9fbe7', boxShadow: 2 }}>
                        <CardContent>
                          <Box display="flex" alignItems="center" justifyContent="space-between">
                            <Box>
                              <Typography variant="subtitle1"><b>Filename:</b> {item.filename}</Typography>
                              <Typography variant="body2" color="text.secondary"><b>Uploaded:</b> {item.upload_time}</Typography>
                            </Box>
                            <Chip icon={icon} label={category} color={color} variant="filled" sx={{ fontWeight: 'bold', fontSize: '1rem' }} />
                          </Box>
                          <Box sx={{ mt: 1, p: 1, background: '#fffde7', borderRadius: 1, fontFamily: 'monospace', whiteSpace: 'pre-wrap', fontSize: '0.95rem', color: '#333' }}>
                            {expanded ? item.report : item.report.slice(0, 300) + (item.report.length > 300 ? '... (truncated)' : '')}
                          </Box>
                          <Button size="small" sx={{ mt: 1 }} onClick={() => setExpandedHistory(prev => ({ ...prev, [item.id]: !expanded }))}>
                            {expanded ? 'Hide Details' : 'Show Full Report'}
                          </Button>
                        </CardContent>
                      </Card>
                    );
                  })}
                </Box>
              )}
            </CardContent>
          </Card>
        )}
        {/* Connection Status */}
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

        {/* Upload Section */}
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Typography variant="h5" gutterBottom color="primary">
              📹 Upload Video for Analysis
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Upload your video file to analyze content for YouTube policy compliance
            </Typography>

            <Box sx={{ mt: 3 }}>
              <input
                accept="video/*"
                style={{ display: 'none' }}
                id="video-upload"
                type="file"
                onChange={handleFileChange}
              />
              <label htmlFor="video-upload">
                <Button
                  variant="outlined"
                  component="span"
                  startIcon={<CloudUpload />}
                  sx={{ mr: 2 }}
                  disabled={connectionStatus !== 'connected'}
                >
                  Choose Video File
                </Button>
              </label>

              {selectedFile && (
                <Chip
                  label={`${selectedFile.name} (${formatFileSize(selectedFile.size)})`}
                  color="primary"
                  variant="outlined"
                  sx={{ ml: 1 }}
                />
              )}

              <Box sx={{ mt: 2 }}>
                <Button
                  variant="contained"
                  startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <Analytics />}
                  onClick={handleAnalyze}
                  disabled={loading || !selectedFile || connectionStatus !== 'connected'}
                  size="large"
                  sx={{
                    background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                    color: 'white'
                  }}
                >
                  {loading ? 'Analyzing...' : 'Analyze Video'}
                </Button>
              </Box>

              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {error}
                </Alert>
              )}
            </Box>
          </CardContent>
        </Card>

        {/* Progress Bar with Percentage */}
        {loading && (
          <Paper sx={{ p: 2, mb: 4 }}>
            <Typography variant="h6" gutterBottom>
              Analyzing video content...
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Box sx={{ width: '100%', mr: 2 }}>
                <LinearProgress variant="determinate" value={progress} />
              </Box>
              <Typography variant="body2" color="text.secondary" sx={{ minWidth: 40 }}>
                {`${progress}%`}
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Processing video frames, audio, and text content with AI...
            </Typography>
          </Paper>
        )}

        {/* Analysis Results */}
        {analysis && (
          <>
            {/* Raw Model Output */}
            {analysis.full_report && (
              <Card sx={{ mb: 3, backgroundColor: '#fffbe6', border: '1px solid #ffe082' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom color="warning.main">
                    📝 Comprehensive Model Report
                  </Typography>
                  <ComprehensiveReportDisplay report={analysis.full_report} />
                </CardContent>
              </Card>
            )}

            {/* Analysis Summary Cards */}
            <Grid container spacing={3}>
              {/* Text Analysis */}
              <Grid item xs={12} md={4}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <TextFields color="primary" sx={{ mr: 1 }} />
                      <Typography variant="h6">Text Analysis</Typography>
                    </Box>
                    <Box display="flex" alignItems="center" mb={2}>
                      <Chip
                        icon={getStatusIcon(parsedReport.text_status)}
                        label={(parsedReport.text_status || 'N/A').toUpperCase()}
                        color={getStatusColor(parsedReport.text_status)}
                        variant="filled"
                      />
                      <Typography variant="h4" sx={{ ml: 2 }}>
                        {parsedReport.text_score || 'N/A'}%
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Results:
                    </Typography>
                    {parsedReport.text_issues
                      ? parsedReport.text_issues.split('\n').map((issue, index) => (
                        <Typography key={index} variant="body2" sx={{ mt: 0.5 }}>
                          {issue.trim().startsWith('•') ? issue.trim() : `• ${issue.trim()}`}
                        </Typography>
                      ))
                      : <Typography variant="body2">No issues found.</Typography>
                    }
                  </CardContent>
                </Card>
              </Grid>

              {/* Audio Analysis */}
              <Grid item xs={12} md={4}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <RecordVoiceOver color="primary" sx={{ mr: 1 }} />
                      <Typography variant="h6">Audio Analysis</Typography>
                    </Box>
                    <Box display="flex" alignItems="center" mb={2}>
                      <Chip
                        icon={getStatusIcon(parsedReport.audio_status)}
                        label={(parsedReport.audio_status || 'N/A').toUpperCase()}
                        color={getStatusColor(parsedReport.audio_status)}
                        variant="filled"
                      />
                      <Typography variant="h4" sx={{ ml: 2 }}>
                        {parsedReport.audio_score || 'N/A'}%
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Results:
                    </Typography>
                    {parsedReport.audio_issues
                      ? parsedReport.audio_issues.split('\n').map((issue, index) => (
                        <Typography key={index} variant="body2" sx={{ mt: 0.5 }}>
                          {issue.trim().startsWith('•') ? issue.trim() : `• ${issue.trim()}`}
                        </Typography>
                      ))
                      : <Typography variant="body2">No issues found.</Typography>
                    }
                  </CardContent>
                </Card>
              </Grid>

              {/* Video/Image Analysis */}
              <Grid item xs={12} md={4}>
                <Card sx={{ height: '100%' }}>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <Image color="primary" sx={{ mr: 1 }} />
                      <Typography variant="h6">Video Frame Analysis</Typography>
                    </Box>

                    <Box display="flex" alignItems="center" mb={2}>
                      <Chip
                        icon={getStatusIcon(parsedReport.image_status)}
                        label={(parsedReport.image_status || 'N/A').toUpperCase()}
                        color={getStatusColor(parsedReport.image_status)}
                        variant="filled"
                      />
                      <Typography variant="h4" sx={{ ml: 2 }}>
                        {parsedReport.image_score || 'N/A'}%
                        <Typography variant="body2" color="text.secondary" sx={{ ml: 2 }}>
                          Image compliance score (higher is better)
                        </Typography>
                      </Typography>
                    </Box>

                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Frames analyzed: {parsedReport.frames_analyzed || 0} (more frames = more accurate)
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Results (issues detected):
                    </Typography>
                    {parsedReport.image_issues
                      ? parsedReport.image_issues.split('\n').map((issue, index) => (
                        <Typography key={index} variant="body2" sx={{ mt: 0.5, color: '#d32f2f' }}>
                          • {issue.trim()}
                        </Typography>
                      ))
                      : <Typography variant="body2" color="success.main">No issues found. Content is compliant.</Typography>
                    }
                  </CardContent>
                </Card>
              </Grid>

              {/* Overall Results */}
              <Grid item xs={12}>
                {/* Large summary verdict card */}
                <Card sx={{ mb: 4, p: 3, backgroundColor: parsedReport.overall === 'conforme' ? '#e8f5e9' : parsedReport.overall === 'attention' ? '#fffde7' : parsedReport.overall === 'non_conforme' ? '#ffebee' : '#f5f5f5', borderRadius: 3, boxShadow: 3 }}>
                  <CardContent>
                    <Box display="flex" alignItems="center" justifyContent="center" mb={2}>
                      <Typography variant="h2" sx={{ mr: 2 }}>
                        {parsedReport.overall === 'conforme' && '✅'}
                        {parsedReport.overall === 'attention' && '⚠️'}
                        {parsedReport.overall === 'non_conforme' && '❌'}
                        {!["conforme","attention","non_conforme"].includes(parsedReport.overall) && '❓'}
                      </Typography>
                      <Box>
                        <Typography variant="h3" sx={{ mr: 2, fontWeight: 'bold', color: '#1976d2' }}>
                          {parsedReport.overall?.toUpperCase() || 'N/A'}
                        </Typography>
                        <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
                          Overall Compliance: {parsedReport.overall_scores?.image || parsedReport.image_score || 'N/A'}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                          Status: {parsedReport.overall || 'N/A'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                          Total Violations: {Array.isArray(parsedReport.image_issues) ? parsedReport.image_issues.length : typeof parsedReport.image_issues === 'string' ? parsedReport.image_issues.split('\n').length : 'N/A'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          Violation Categories: {Array.isArray(parsedReport.image_issues) ? parsedReport.image_issues.join(', ') : typeof parsedReport.image_issues === 'string' ? parsedReport.image_issues : 'N/A'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                          📊 Analysis completed in {parsedReport.processing_time || 'N/A'}s • {parsedReport.frames_analyzed || 0} frames analyzed • AI-powered content detection
                        </Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
                {/* Content type cards */}
                <Grid container spacing={3} sx={{ mb: 2 }}>
                  <Grid item xs={12} sm={4}>
                    <Card sx={{ p: 3, borderRadius: 3, boxShadow: 3, backgroundColor: '#e3f2fd' }}>
                      <CardContent>
                        <Box display="flex" alignItems="center" mb={1}>
                          <TextFields color="primary" sx={{ mr: 1 }} />
                          <Typography variant="h5" color="primary" sx={{ fontWeight: 'bold' }}>Text Content</Typography>
                        </Box>
                        <Typography variant="h3" color="primary" sx={{ fontWeight: 'bold' }}>{parsedReport.overall_scores?.text || 'N/A'}%</Typography>
                        <Typography variant="body2" color="text.secondary" title="Text detected in video (higher = more compliant)">Text detected in video (higher = more compliant)</Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Card sx={{ p: 3, borderRadius: 3, boxShadow: 3, backgroundColor: '#fffde7' }}>
                      <CardContent>
                        <Box display="flex" alignItems="center" mb={1}>
                          <RecordVoiceOver color="warning" sx={{ mr: 1 }} />
                          <Typography variant="h5" color="warning.main" sx={{ fontWeight: 'bold' }}>Audio Content</Typography>
                        </Box>
                        <Typography variant="h3" color="warning.main" sx={{ fontWeight: 'bold' }}>{parsedReport.overall_scores?.audio || 'N/A'}%</Typography>
                        <Typography variant="body2" color="text.secondary" title="Speech and sound compliance (higher = better)">Speech and sound compliance (higher = better)</Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Card sx={{ p: 3, borderRadius: 3, boxShadow: 3, backgroundColor: '#fce4ec' }}>
                      <CardContent>
                        <Box display="flex" alignItems="center" mb={1}>
                          <Image color="error" sx={{ mr: 1 }} />
                          <Typography variant="h5" color="error" sx={{ fontWeight: 'bold' }}>Video Frames</Typography>
                        </Box>
                        <Typography variant="h3" color="error" sx={{ fontWeight: 'bold' }}>{parsedReport.overall_scores?.image || 'N/A'}%</Typography>
                        <Typography variant="body2" color="text.secondary" title="Visual content compliance (higher = better)">Visual content compliance (higher = better)</Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
                {/* What does this mean section */}
                <Card sx={{ p: 2, backgroundColor: '#f5f5f5', borderRadius: 2, boxShadow: 1 }}>
                  <CardContent>
                    <Typography variant="subtitle2" color="primary">What does this mean?</Typography>
                    <ul style={{ margin: 0, paddingLeft: 20 }}>
                      <li><b>Compliance:</b> Percentage of content following policy. Higher is better.</li>
                      <li><b>Status:</b> Final decision for YouTube upload. See verdict above.</li>
                      <li><b>Violations:</b> Issues detected in video. Lower is better.</li>
                      <li><b>Categories:</b> Types of violations (e.g., violence, nudity, copyright).</li>
                    </ul>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </>
        )}
      </Container>
    </div>
  );
}

export default App;