import React, { useState, useEffect } from 'react';
import {
  Container,
  AppBar,
  Toolbar,
  Typography,
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Grid,
  Paper,
  CircularProgress,
  Alert,
  Chip,
  LinearProgress,
  Divider
} from '@mui/material';
import './App.css';

// Configuration de l'API
const API_BASE_URL = 'http://localhost:8000';

// FrameCaptionsGrid: displays frame previews and captions
// FrameAnalysisGrid: displays frame previews and full per-frame analysis
function FrameAnalysisGrid({ frameAnalysis }) {
  // Defensive: sort by frame_index and ensure preview_path is present
  const sortedFrames = [...frameAnalysis].sort((a, b) => (a.frame_index ?? 0) - (b.frame_index ?? 0));
  return (
    <Grid container spacing={2}>
      {sortedFrames.map((frame, idx) => (
        <Grid item xs={12} sm={6} md={4} lg={3} key={frame.preview_path || idx}>
          <Card sx={{ mb: 2 }}>
            {frame.preview_path && (
              <img
                src={`http://localhost:8000${frame.preview_path}`}
                alt={`Frame ${frame.frame_index}`}
                style={{ width: '100%', borderRadius: 4, marginBottom: 8 }}
              />
            )}
            <CardContent>
              <Typography variant="body2" color="text.secondary" sx={{ minHeight: 40, textAlign: 'left', fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
                {frame.frame_report || 'No analysis available for this frame.'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  );
}

function App() {
  // State hooks
  const [selectedFile, setSelectedFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('testing');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [frameCaptions, setFrameCaptions] = useState([]);
  const [frameCaptionsSummary, setFrameCaptionsSummary] = useState('');

  // Fetch all frame captions for report when analysis changes
  useEffect(() => {
    let isMounted = true;
    const fetchAllCaptions = async () => {
      if (!analysis?.image?.details?.frame_preview_paths || analysis.image.details.frame_preview_paths.length === 0) {
        if (isMounted) {
          setFrameCaptions([]);
          setFrameCaptionsSummary('');
        }
        return;
      }
      const paths = [...analysis.image.details.frame_preview_paths];
      const captionsArr = [];
      await Promise.all(paths.map(async (framePath) => {
        const frameId = framePath.split(/[\\/]/).pop();
        try {
          const res = await fetch(`${API_BASE_URL}/frame-describe/${frameId}`);
          const data = await res.json();
          captionsArr.push(data.caption || data.error || 'Erreur de légende');
        } catch (e) {
          captionsArr.push('Erreur de requête');
        }
      }));
      if (isMounted) {
        setFrameCaptions(captionsArr);
        // Simple summary: most common words (excluding stopwords)
        const stopwords = ['the', 'a', 'an', 'of', 'and', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'it', 'this', 'that', 'these', 'those', 'be', 'or', 'but', 'if', 'then', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'about', 'into', 'over', 'after', 'before', 'under', 'above', 'between', 'out', 'up', 'down', 'off', 'not', 'no', 'yes', 'you', 'i', 'he', 'she', 'they', 'we', 'my', 'your', 'his', 'her', 'their', 'our', 'its', 'me', 'him', 'them', 'us', 'do', 'does', 'did', 'have', 'has', 'had', 'which', 'who', 'whom', 'whose', 'what', 'when', 'where', 'why', 'how', 'all', 'any', 'some', 'each', 'few', 'more', 'most', 'other', 'such'];
        const wordCounts = {};
        captionsArr.forEach(caption => {
          caption.split(/\W+/).forEach(word => {
            const w = word.toLowerCase();
            if (w && !stopwords.includes(w)) {
              wordCounts[w] = (wordCounts[w] || 0) + 1;
            }
          });
        });
        const sortedWords = Object.entries(wordCounts).sort((a, b) => b[1] - a[1]);
        const summary = sortedWords.slice(0, 5).map(([w, c]) => `${w} (${c})`).join(', ');
        setFrameCaptionsSummary(summary);
      }
    };
    if (analysis) fetchAllCaptions();
    else {
      setFrameCaptions([]);
      setFrameCaptionsSummary('');
    }
    return () => { isMounted = false; };
  }, [analysis]);

  // Test backend connection on mount
  useEffect(() => {
    testConnection();
  }, []);

  // Connection test function
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

  // File change handler
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type.startsWith('video/')) {
        setSelectedFile(file);
        setError('');
      } else {
        setError('Veuillez sélectionner un fichier vidéo valide');
        setSelectedFile(null);
      }
    }
  };

  // Analyze handler
  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Veuillez sélectionner un fichier vidéo');
      return;
    }
    setLoading(true);
    setError('');
    setUploadProgress(0);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      const response = await fetch(`${API_BASE_URL}/upload-video`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error(`Erreur HTTP: ${response.status}`);
      }
      const result = await response.text();
      setAnalysis({ full_report: result });
    } catch (err) {
      setError(`Erreur lors de l'analyse: ${err.message}`);
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  // Status color helper
  const getStatusColor = (status) => {
    switch (status) {
      case 'conforme': return 'success';
      case 'attention': return 'warning';
      case 'non-conforme': return 'error';
      default: return 'default';
    }
  };

  // Status icon helper
  const getStatusIcon = (status) => {
    switch (status) {
      case 'conforme':
      case 'excellent':
      case 'good':
        return '✅';
      case 'attention':
      case 'ok':
        return '⚠️';
      case 'non-conforme':
        return '❌';
      default:
        return '❓';
    }
  };

  // File size formatter
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="app">
      <AppBar position="static" sx={{ backgroundColor: '#2c3e50' }}>
        <Toolbar>
          🎬
          <Typography variant="h6" component="div" sx={{ flexGrow: 1, ml: 2 }}>
            Analyseur de Contenu Vidéo Local
          </Typography>
          {/* Indicateur de connexion */}
          <Box display="flex" alignItems="center">
            {connectionStatus === 'connected' ? (
              <Chip
                icon={'🟢'}
                label="Connecté"
                color="success"
                variant="filled"
                size="small"
              />
            ) : connectionStatus === 'disconnected' ? (
              <Chip
                icon={'🔴'}
                label="Déconnecté"
                color="error"
                variant="filled"
                size="small"
              />
            ) : (
              <CircularProgress size={20} color="inherit" />
            )}
          </Box>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg" sx={{ mt: 4, pb: 4 }}>
        {/* Test de connexion */}
        {connectionStatus === 'disconnected' && (
          <Alert severity="error" sx={{ mb: 4 }} action={
            <Button color="inherit" size="small" onClick={testConnection}>
              Reconnect
            </Button>
          }>
            Impossible de se connecter au backend FastAPI. Assurez-vous que le serveur est démarré sur http://localhost:8000
          </Alert>
        )}
        {connectionStatus === 'connected' && (
          <Alert severity="success" sx={{ mb: 4 }}>
            ✅ Système d'analyse vidéo prêt ! Upload et analysez vos fichiers vidéo locaux.
          </Alert>
        )}
        {/* Section d'upload */}
        <Card sx={{ mb: 4, backgroundColor: 'rgba(255,255,255,0.95)' }}>
          <CardContent>
            <Typography variant="h5" gutterBottom color="primary">
              🔍 Analyser un fichier vidéo local
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Uploadez votre fichier vidéo pour analyser son contenu : audio, images et texte
            </Typography>
            <Box sx={{ mt: 3 }}>
              {/* Input file */}
              <Box sx={{ mb: 2 }}>
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
                    startIcon={'📤'}
                    sx={{ mr: 2 }}
                    disabled={connectionStatus !== 'connected'}
                  >
                    Choisir un fichier vidéo
                  </Button>
                </label>
                {selectedFile && (
                  <Chip
                    label={`${selectedFile.name} (${formatFileSize(selectedFile.size)})`}
                    color="primary"
                    variant="outlined"
                  />
                )}
              </Box>
              <Button
                variant="contained"
                startIcon={loading ? <CircularProgress size={20} /> : '📊'}
                onClick={handleAnalyze}
                disabled={loading || !selectedFile || connectionStatus !== 'connected'}
                size="large"
                sx={{
                  background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)',
                  color: 'white'
                }}
              >
                {loading ? 'Analyse en cours...' : 'Analyser la vidéo'}
              </Button>
            </Box>
            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </CardContent>
        </Card>
        {/* Barre de progression */}
        {loading && (
          <Paper sx={{ p: 2, mb: 4, backgroundColor: 'rgba(255,255,255,0.95)' }}>
            <Typography variant="h6" gutterBottom>
              Analyse en cours...
            </Typography>
            <LinearProgress sx={{ mb: 1 }} />
            <Typography variant="body2" color="text.secondary">
              Extraction et analyse : audio, images, texte et métadonnées...
            </Typography>
          </Paper>
        )}
        {/* Résultats d'analyse */}
        {analysis && (
          <>
            {/* Full Model Report (human-readable, CLI-style) */}
            {analysis.full_report && (
              <Card sx={{ mb: 3, backgroundColor: 'rgba(30,30,30,0.97)', color: '#fff', border: '2px solid #1976d2' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ color: '#90caf9' }}>
                    📝 Rapport Complet du Modèle
                  </Typography>
                  <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: 15, background: 'none', color: '#fff', margin: 0, padding: 0 }}>
                    {analysis.full_report}
                  </pre>
                </CardContent>
              </Card>
            )}
            {/* Display extracted frames if available in the report (look for 'Frame' in the text) */}
            {analysis.full_report && analysis.full_report.includes('Frame') && (
              <Card sx={{ mb: 3, backgroundColor: 'rgba(240,248,255,0.95)', border: '2px solid #1976d2' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ color: '#1976d2' }}>
                    🖼️ Frames Extracted
                  </Typography>
                  <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: 14, background: 'none', color: '#222', margin: 0, padding: 0 }}>
                    {/* Show only lines containing 'Frame' and the next few lines for context */}
                    {analysis.full_report.split('\n').filter((line, idx, arr) => {
                      if (line.startsWith('Frame ')) return true;
                      // Show next 3 lines after 'Frame'
                      if (idx > 0 && arr[idx - 1].startsWith('Frame ')) return true;
                      if (idx > 1 && arr[idx - 2].startsWith('Frame ')) return true;
                      if (idx > 2 && arr[idx - 3].startsWith('Frame ')) return true;
                      return false;
                    }).join('\n')}
                  </pre>
                </CardContent>
              </Card>
            )}
            {/* ...existing code... */}
            {/* Video Summary/Resume Section - Simplified */}
            {analysis.video_summary && (
              <Card sx={{ mb: 3, backgroundColor: 'rgba(240,248,255,0.95)', border: '2px solid #1976d2' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ color: '#1976d2' }}>
                    📋 Résumé Complet de la Vidéo
                  </Typography>

                  {/* Compliance Assessment */}
                  {analysis.video_summary.compliance_assessment && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="h6" gutterBottom>
                        {analysis.video_summary.compliance_assessment.status_icon} {analysis.video_summary.compliance_assessment.overall_status}
                      </Typography>
                      <Typography variant="body1" sx={{ mb: 2 }}>
                        Score moyen: {analysis.video_summary.compliance_assessment.average_score}%
                      </Typography>
                    </Box>
                  )}

                  {/* Key Insights */}
                  {analysis.video_summary.content_analysis?.key_insights && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="h6" gutterBottom>🎯 Principales Découvertes:</Typography>
                      {analysis.video_summary.content_analysis.key_insights.map((insight, index) => (
                        <Typography key={index} variant="body2" sx={{ mb: 1, pl: 2 }}>
                          • {insight}
                        </Typography>
                      ))}
                    </Box>
                  )}

                  {/* Recommendations */}
                  {analysis.video_summary.recommendations && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="h6" gutterBottom>💡 Recommandations:</Typography>
                      {analysis.video_summary.recommendations.map((rec, index) => (
                        <Typography key={index} variant="body2" sx={{ mb: 1, pl: 2 }}>
                          • {rec}
                        </Typography>
                      ))}
                    </Box>
                  )}

                  {/* Transcript Preview */}
                  {analysis.video_summary.transcript_preview && (
                    <Box sx={{ mb: 3 }}>
                      <Typography variant="h6" gutterBottom>📝 Aperçu de la Transcription:</Typography>
                      <Paper elevation={1} sx={{ p: 2, backgroundColor: 'rgba(240,240,240,0.5)' }}>
                        <Typography variant="body2" sx={{ fontStyle: 'italic' }}>
                          "{analysis.video_summary.transcript_preview}"
                        </Typography>
                      </Paper>
                    </Box>
                  )}
                </CardContent>
              </Card>
            )}

            <Grid container spacing={3}>
              {/* Analyse Audio/Voix */}
              <Grid item xs={12} md={6}>
                <Card sx={{ backgroundColor: 'rgba(255,255,255,0.95)', height: '100%' }}>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      🎤
                      <Typography variant="h6" sx={{ ml: 1 }}>Analyse Audio/Voix</Typography>
                    </Box>
                    <Box display="flex" alignItems="center" mb={2}>
                      <Chip
                        label={`${getStatusIcon(analysis.voice?.status || 'attention')} ${(analysis.voice?.status || 'N/A').toUpperCase()}`}
                        color={getStatusColor(analysis.voice?.status || 'attention')}
                        variant="filled"
                      />
                      <Typography variant="h4" sx={{ ml: 2 }}>
                        {analysis.voice?.score || 'N/A'}%
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      Résultats :
                    </Typography>
                    {analysis.voice?.issues?.map((issue, index) => (
                      <Typography key={index} variant="body2" sx={{ mt: 1 }}>
                        • {issue}
                      </Typography>
                    ))}
                    {/* Audio Player, Transcript, and Bad Words */}
                    {analysis.voice?.details?.audio_path && (
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>🔊 Audio extrait :</Typography>
                        <audio controls style={{ width: '100%' }}>
                          <source src={`${API_BASE_URL}/audio/${analysis.voice.details.audio_path}`} type="audio/wav" />
                          Votre navigateur ne supporte pas l'audio.
                        </audio>
                      </Box>
                    )}
                    {analysis.voice?.details?.transcript && (
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>📝 Transcription complète :</Typography>
                        <Paper sx={{ p: 1, backgroundColor: '#f5f5f5', maxHeight: 150, overflow: 'auto' }}>
                          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>
                            {analysis.voice.details.transcript}
                          </Typography>
                        </Paper>
                      </Box>
                    )}
                    {analysis.voice?.details?.bad_words && analysis.voice.details.bad_words.length > 0 && (
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#d32f2f' }}>🤬 Mots inappropriés détectés :</Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
                          {analysis.voice.details.bad_words.map((word, idx) => (
                            <Chip key={idx} label={word} color="error" variant="outlined" />
                          ))}
                        </Box>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Analyse Images */}
              <Grid item xs={12} md={6}>
                <Card sx={{ backgroundColor: 'rgba(255,255,255,0.95)', height: '100%' }}>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      🖼️
                      <Typography variant="h6" sx={{ ml: 1 }}>Analyse des Images</Typography>
                    </Box>
                    <Box display="flex" alignItems="center" mb={2}>
                      <Chip
                        label={`${getStatusIcon(analysis.image?.status || 'attention')} ${(analysis.image?.status || 'N/A').toUpperCase()}`}
                        color={getStatusColor(analysis.image?.status || 'attention')}
                        variant="filled"
                      />
                      <Typography variant="h4" sx={{ ml: 2 }}>
                        {analysis.image?.score || 'N/A'}%
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      Frames analysées: {analysis.image?.details?.frames_analyzed || 0}
                    </Typography>
                    {analysis.image?.issues?.map((issue, index) => (
                      <Typography key={index} variant="body2" sx={{ mt: 1 }}>
                        • {issue}
                      </Typography>
                    ))}
                    {/* --- Frame Captions Summary and List --- */}
                    {frameCaptions.length > 0 && (
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>Résumé des descriptions IA&nbsp;:</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                          {frameCaptionsSummary ? `Mots fréquents : ${frameCaptionsSummary}` : 'Résumé indisponible'}
                        </Typography>
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mt: 2 }}>Descriptions de chaque frame :</Typography>
                        <ul style={{ margin: 0, paddingLeft: 18 }}>
                          {frameCaptions.map((caption, idx) => (
                            <li key={idx} style={{ fontSize: '0.95em', color: '#333', marginBottom: 2 }}>{caption}</li>
                          ))}
                        </ul>
                      </Box>
                    )}
                  </CardContent>
                </Card>
              </Grid>

              {/* Frame Previews and Full Analysis */}
              {analysis.image?.details?.frame_analysis && analysis.image.details.frame_analysis.length > 0 && (
                <Grid item xs={12}>
                  <Card sx={{ backgroundColor: 'rgba(255,255,255,0.98)', mt: 2 }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>🖼️ Aperçu des Images & Analyse IA</Typography>
                      <FrameAnalysisGrid frameAnalysis={analysis.image.details.frame_analysis} />
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {/* Analyse Texte */}
              <Grid item xs={12} md={6}>
                <Card sx={{ backgroundColor: 'rgba(255,255,255,0.95)', height: '100%' }}>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      📝
                      <Typography variant="h6" sx={{ ml: 1 }}>Analyse du Texte</Typography>
                    </Box>

                    <Box display="flex" alignItems="center" mb={2}>
                      <Chip
                        label={`${getStatusIcon(analysis.title?.status || 'attention')} ${(analysis.title?.status || 'N/A').toUpperCase()}`}
                        color={getStatusColor(analysis.title?.status || 'attention')}
                        variant="filled"
                      />
                      <Typography variant="h4" sx={{ ml: 2 }}>
                        {analysis.title?.score || 'N/A'}%
                      </Typography>
                    </Box>

                    <Typography variant="body2" color="text.secondary">
                      Résultats :
                    </Typography>
                    {analysis.title?.issues?.map((issue, index) => (
                      <Typography key={index} variant="body2" sx={{ mt: 1 }}>
                        • {issue}
                      </Typography>
                    ))}
                  </CardContent>
                </Card>
              </Grid>

              {/* Résultat global et Compliance Report */}
              <Grid item xs={12}>
                <Card sx={{ backgroundColor: 'rgba(255,255,255,0.95)' }}>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      📈
                      <Typography variant="h6" sx={{ ml: 1 }}>Rapport de Conformité YouTube</Typography>
                    </Box>

                    {/* Score global pondéré */}
                    <Box display="flex" alignItems="center" mb={3}>
                      <Typography variant="h4" sx={{ mr: 2 }}>
                        Score Global: {analysis.compliance_report?.weighted_score || 'N/A'}%
                      </Typography>
                      <Chip
                        label={`${getStatusIcon((analysis.overall && analysis.overall.includes('CONFORME')) ? 'conforme' :
                          (analysis.overall && analysis.overall.includes('ATTENTION')) ? 'attention' : 'non-conforme')} ${analysis.overall || 'N/A'}`}
                        color={getStatusColor((analysis.overall && analysis.overall.includes('CONFORME')) ? 'conforme' :
                          (analysis.overall && analysis.overall.includes('ATTENTION')) ? 'attention' : 'non-conforme')}
                        variant="filled"
                        sx={{ fontSize: '1rem', padding: '8px 16px' }}
                      />
                    </Box>

                    {/* Scores individuels */}
                    <Typography variant="h6" sx={{ mb: 2 }}>Scores par Catégorie:</Typography>
                    <Grid container spacing={2} sx={{ mb: 3 }}>
                      <Grid item xs={4}>
                        <Box textAlign="center" p={2} border={1} borderColor="grey.300" borderRadius={2}>
                          <Typography variant="h5" color="primary">
                            {analysis.compliance_report?.individual_scores?.audio || analysis.voice?.score || 'N/A'}%
                          </Typography>
                          <Typography variant="body2">Audio/Voix (35%)</Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={4}>
                        <Box textAlign="center" p={2} border={1} borderColor="grey.300" borderRadius={2}>
                          <Typography variant="h5" color="primary">
                            {analysis.compliance_report?.individual_scores?.image || analysis.image?.score || 'N/A'}%
                          </Typography>
                          <Typography variant="body2">Images (40%)</Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={4}>
                        <Box textAlign="center" p={2} border={1} borderColor="grey.300" borderRadius={2}>
                          <Typography variant="h5" color="primary">
                            {analysis.compliance_report?.individual_scores?.text || analysis.title?.score || 'N/A'}%
                          </Typography>
                          <Typography variant="body2">Texte (25%)</Typography>
                        </Box>
                      </Grid>
                    </Grid>

                    {/* Violations critiques */}
                    {analysis.compliance_report?.critical_violations?.length > 0 && (
                      <Box sx={{ mb: 3 }}>
                        <Typography variant="h6" color="error" sx={{ mb: 1 }}>
                          ⚠️ Violations Critiques:
                        </Typography>
                        {analysis.compliance_report.critical_violations.map((violation, index) => (
                          <Alert severity="error" key={index} sx={{ mb: 1 }}>
                            {violation}
                          </Alert>
                        ))}
                      </Box>
                    )}

                    {/* Recommandations */}
                    {analysis.compliance_report?.recommendations?.length > 0 && (
                      <Box sx={{ mb: 3 }}>
                        <Typography variant="h6" color="warning.main" sx={{ mb: 1 }}>
                          📋 Recommandations:
                        </Typography>
                        {analysis.compliance_report.recommendations.map((rec, index) => (
                          <Alert severity="warning" key={index} sx={{ mb: 1 }}>
                            {rec}
                          </Alert>
                        ))}
                      </Box>
                    )}

                    {/* Informations détaillées */}
                    <Typography variant="h6" sx={{ mb: 2 }}>Détails de l'Analyse:</Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body2" color="text.secondary">
                          <strong>Durée:</strong> {analysis.video_info?.duration || 'N/A'}s
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          <strong>Résolution:</strong> {analysis.video_info?.resolution || 'N/A'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          <strong>FPS:</strong> {analysis.video_info?.fps || 'N/A'}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body2" color="text.secondary">
                          <strong>Frames analysées:</strong> {analysis.image?.details?.frames_analyzed || 'N/A'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          <strong>Audio analysé:</strong> {analysis.voice?.details?.has_audio ? 'Oui' : 'Non'}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          <strong>Temps de traitement:</strong> {analysis.processing_time?.toFixed(2) || 'N/A'}s
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>

              {/* Détails par catégorie */}
              <Grid item xs={12}>
                <Typography variant="h5" sx={{ mb: 3, textAlign: 'center' }}>
                  Analyse Détaillée par Catégorie
                </Typography>
              </Grid>

              {/* Résultat global */}
              <Grid item xs={12} md={6}>
                <Card sx={{ backgroundColor: 'rgba(255,255,255,0.95)', height: '100%' }}>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      🎯 Résultat Global
                    </Typography>
                    <Box display="flex" alignItems="center" mb={2}>
                      <Chip
                        label={`${getStatusIcon((analysis.overall && analysis.overall.includes('CONFORME')) ? 'conforme' :
                          (analysis.overall && analysis.overall.includes('ATTENTION')) ? 'attention' : 'non-conforme')} ${analysis.overall || 'N/A'}`}
                        color={getStatusColor((analysis.overall && analysis.overall.includes('CONFORME')) ? 'conforme' :
                          (analysis.overall && analysis.overall.includes('ATTENTION')) ? 'attention' : 'non-conforme')}
                        variant="filled"
                        size="large"
                      />
                    </Box>
                    <Typography variant="body1" sx={{ mb: 2 }}>
                      Score global pondéré: {analysis.compliance_report?.weighted_score || 'N/A'}%
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      {analysis.overall || 'N/A'}
                    </Typography>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="body2" color="text.secondary">
                      Scores détaillés :
                    </Typography>
                    <Typography variant="body2">Audio: {analysis.voice?.score || 'N/A'}% (35%)</Typography>
                    <Typography variant="body2">Images: {analysis.image?.score || 'N/A'}% (40%)</Typography>
                    <Typography variant="body2">Texte: {analysis.title?.score || 'N/A'}% (25%)</Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Full Conversation Section */}
            {analysis && analysis.full_conversation && analysis.full_conversation.conversation_available && (
              <Card sx={{ mt: 3, backgroundColor: 'rgba(255,255,255,0.95)' }}>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    💬
                    <Typography variant="h6" sx={{ ml: 1 }}>Conversation Complète / Full Audio Transcript</Typography>
                  </Box>

                  <Box mb={2}>
                    <Chip label={`${analysis.full_conversation.word_count} mots`} size="small" sx={{ mr: 1 }} />
                    <Chip label={`${analysis.full_conversation.character_count} caractères`} size="small" sx={{ mr: 1 }} />
                    <Chip label={`Durée estimée: ${analysis.full_conversation.estimated_speaking_time}`} size="small" />
                  </Box>

                  <Paper sx={{ p: 2, backgroundColor: '#f5f5f5', maxHeight: 300, overflow: 'auto' }}>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      📝 Transcription complète de l'audio :
                    </Typography>
                    <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                      {analysis.full_conversation.complete_transcript}
                    </Typography>
                  </Paper>

                  {!analysis.full_conversation.complete_transcript && (
                    <Alert severity="info">
                      Aucune conversation détectée dans cette vidéo ou la reconnaissance vocale n'a pas pu extraire le contenu audio.
                    </Alert>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Profanity & Speech Analysis Section */}
            {analysis && analysis.audio_details && (analysis.audio_details.speech_transcription || analysis.audio_details.profanity_analysis) && (
              <Card sx={{ mt: 3, backgroundColor: 'rgba(255,255,255,0.95)' }}>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    🗣️
                    <Typography variant="h6" sx={{ ml: 1 }}>Analyse de la Parole et Contenu</Typography>
                  </Box>

                  {/* Speech Transcription */}
                  {analysis.audio_details.speech_transcription && (
                    <Box mb={3}>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1, color: '#1976d2' }}>
                        📝 Transcription de la Conversation
                      </Typography>
                      <Box mb={2}>
                        <Chip label={`${analysis.audio_details.word_count || 0} mots`} size="small" sx={{ mr: 1 }} />
                        <Chip label={`${analysis.audio_details.character_count || 0} caractères`} size="small" sx={{ mr: 1 }} />
                        <Chip label={`Durée: ${Math.round((analysis.audio_details.duration || 0) * 100) / 100}s`} size="small" />
                      </Box>
                      <Paper sx={{ p: 2, backgroundColor: '#f5f5f5', maxHeight: 250, overflow: 'auto' }}>
                        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                          {analysis.audio_details.speech_transcription}
                        </Typography>
                      </Paper>
                    </Box>
                  )}

                  {/* Profanity Analysis */}
                  {analysis.audio_details.profanity_analysis && (
                    <Box>
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 2, color: '#d32f2f' }}>
                        🚨 Analyse du Langage Inapproprié
                      </Typography>

                      <Grid container spacing={2}>
                        <Grid item xs={12} sm={6}>
                          <Card sx={{ backgroundColor: '#ffebee' }}>
                            <CardContent>
                              <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                                🤬 Mots Grossiers Détectés
                              </Typography>
                              <Typography variant="h4" color="error">
                                {analysis.audio_details.profanity_analysis.curse_count || 0}
                              </Typography>
                              {analysis.audio_details.profanity_analysis.curse_words && analysis.audio_details.profanity_analysis.curse_words.length > 0 ? (
                                <Box sx={{ mt: 2, maxHeight: 150, overflow: 'auto' }}>
                                  {analysis.audio_details.profanity_analysis.curse_words.map((curse, index) => (
                                    <Box key={index} sx={{ mb: 1, p: 1, backgroundColor: 'rgba(244,67,54,0.1)', borderRadius: 1 }}>
                                      <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                        "{curse.word}"
                                      </Typography>
                                      <Typography variant="caption" sx={{ fontStyle: 'italic' }}>
                                        Contexte: "{curse.context}"
                                      </Typography>
                                    </Box>
                                  ))}
                                </Box>
                              ) : (
                                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                                  Aucun mot grossier détecté
                                </Typography>
                              )}
                            </CardContent>
                          </Card>
                        </Grid>

                        <Grid item xs={12} sm={6}>
                          <Card sx={{ backgroundColor: '#fff3e0' }}>
                            <CardContent>
                              <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                                ⚔️ Contenu Violent
                              </Typography>
                              <Typography variant="h4" color="warning.main">
                                {analysis.audio_details.profanity_analysis.violence_count || 0}
                              </Typography>
                              {analysis.audio_details.profanity_analysis.violence_words && analysis.audio_details.profanity_analysis.violence_words.length > 0 ? (
                                <Box sx={{ mt: 2, maxHeight: 150, overflow: 'auto' }}>
                                  {analysis.audio_details.profanity_analysis.violence_words.map((violence, index) => (
                                    <Box key={index} sx={{ mb: 1, p: 1, backgroundColor: 'rgba(255,152,0,0.1)', borderRadius: 1 }}>
                                      <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                                        "{violence.word}"
                                      </Typography>
                                      <Typography variant="caption" sx={{ fontStyle: 'italic' }}>
                                        Contexte: "{violence.context}"
                                      </Typography>
                                    </Box>
                                  ))}
                                </Box>
                              ) : (
                                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                                  Aucun contenu violent détecté
                                </Typography>
                              )}
                            </CardContent>
                          </Card>
                        </Grid>

                        <Grid item xs={12}>
                          <Alert severity={
                            (analysis.audio_details.profanity_analysis.total_problematic || 0) > 5 ? "error" :
                              (analysis.audio_details.profanity_analysis.total_problematic || 0) > 0 ? "warning" : "success"
                          }>
                            <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                              Total de contenu problématique: {analysis.audio_details.profanity_analysis.total_problematic || 0}
                            </Typography>
                            <Typography variant="body2">
                              {(analysis.audio_details.profanity_analysis.total_problematic || 0) === 0
                                ? "✅ Aucun langage inapproprié détecté dans cette vidéo"
                                : (analysis.audio_details.profanity_analysis.total_problematic || 0) > 5
                                  ? "🚨 Contenu fortement inapproprié détecté - Non recommandé pour YouTube"
                                  : "⚠️ Contenu modérément inapproprié détecté - Peut nécessiter une révision"}
                            </Typography>
                          </Alert>
                        </Grid>
                      </Grid>
                    </Box>
                  )}

                  {!analysis.audio_details.speech_transcription && !analysis.audio_details.profanity_analysis && (
                    <Alert severity="info">
                      Impossible d'extraire l'audio de cette vidéo pour l'analyse de la parole.
                    </Alert>
                  )}
                </CardContent>
              </Card>
            )}

            {/* AI Video Summary Section */}
            {analysis && analysis.ai_video_summary && (
              <Card sx={{ mt: 3, backgroundColor: 'rgba(255,255,255,0.95)' }}>
                <CardContent>
                  <Box display="flex" alignItems="center" mb={2}>
                    🤖
                    <Typography variant="h6" sx={{ ml: 1 }}>Résumé Intelligent de la Vidéo (IA)</Typography>
                  </Box>

                  <Grid container spacing={2}>
                    {/* Content Type & Summary */}
                    <Grid item xs={12}>
                      <Paper sx={{ p: 2, backgroundColor: '#e3f2fd', mb: 2 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1 }}>
                          🎯 {analysis.ai_video_summary.ai_analysis.content_type}
                        </Typography>
                        <Typography variant="body1" sx={{ mb: 2 }}>
                          {analysis.ai_video_summary.ai_analysis.summary_text}
                        </Typography>

                        {analysis.ai_video_summary.ai_analysis.main_topics.length > 0 && (
                          <Box>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>Sujets principaux :</Typography>
                            {analysis.ai_video_summary.ai_analysis.main_topics.map((topic, index) => (
                              <Chip key={index} label={topic} size="small" sx={{ mr: 1, mb: 1 }} />
                            ))}
                          </Box>
                        )}

                        {analysis.ai_video_summary.ai_analysis.content_themes.length > 0 && (
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>Thèmes :</Typography>
                            {analysis.ai_video_summary.ai_analysis.content_themes.map((theme, index) => (
                              <Chip key={index} label={theme} size="small" variant="outlined" sx={{ mr: 1, mb: 1 }} />
                            ))}
                          </Box>
                        )}
                      </Paper>
                    </Grid>

                    {/* Speech Analysis */}
                    {analysis.ai_video_summary.speech_analysis.has_conversation && (
                      <Grid item xs={12} md={6}>
                        <Card sx={{ backgroundColor: '#f3e5f5' }}>
                          <CardContent>
                            <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1 }}>
                              🗣️ Analyse de la Parole
                            </Typography>
                            <Typography variant="body2">
                              • Mots détectés: {analysis.ai_video_summary.speech_analysis.word_count}
                            </Typography>
                            <Typography variant="body2">
                              • Durée estimée: {analysis.ai_video_summary.speech_analysis.estimated_duration}
                            </Typography>
                            <Typography variant="body2">
                              • Qualité du langage: {analysis.ai_video_summary.speech_analysis.language_quality === 'appropriate' ? '✅ Approprié' : '⚠️ Langage inapproprié détecté'}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    )}

                    {/* Visual Analysis Summary */}
                    <Grid item xs={12} md={6}>
                      <Card sx={{ backgroundColor: '#e8f5e8' }}>
                        <CardContent>
                          <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1 }}>
                            🎬 Analyse Visuelle
                          </Typography>
                          <Typography variant="body2">
                            • Images analysées: {analysis.ai_video_summary.visual_analysis.total_frames_analyzed}
                          </Typography>
                          <Typography variant="body2">
                            • Score qualité: {analysis.ai_video_summary.visual_analysis.content_quality_score}%
                          </Typography>
                          <Typography variant="body2">
                            • Contenu problématique: {analysis.ai_video_summary.visual_analysis.problematic_content_detected ? '⚠️ Détecté' : '✅ Aucun'}
                          </Typography>

                          {analysis.ai_video_summary.visual_analysis.main_visual_issues.length > 0 && (
                            <Box sx={{ mt: 1 }}>
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>Problèmes détectés :</Typography>
                              {analysis.ai_video_summary.visual_analysis.main_visual_issues.map((issue, index) => (
                                <Typography key={index} variant="body2" sx={{ fontSize: '0.8rem' }}>
                                  • {issue}
                                </Typography>
                              ))}
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>

                    {/* Overall Recommendation */}
                    <Grid item xs={12}>
                      <Alert
                        severity={analysis.ai_video_summary.overall_recommendation.suitable_for_youtube ? 'success' : 'warning'}
                        sx={{ mt: 1 }}
                      >
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                          {analysis.ai_video_summary.overall_recommendation.suitable_for_youtube ? '🟢 Recommandation IA: Adapté pour YouTube' : '🟡 Recommandation IA: Nécessite des améliorations'}
                        </Typography>
                        <Typography variant="body2">
                          Évaluation IA: {analysis.ai_video_summary.ai_analysis.content_assessment.appropriateness === 'appropriate' ? 'Contenu approprié' : 'Contenu questionnable'},
                          Valeur éducative: {analysis.ai_video_summary.ai_analysis.content_assessment.educational_value},
                          Qualité technique: {analysis.ai_video_summary.ai_analysis.content_assessment.technical_quality}
                        </Typography>
                      </Alert>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            )}
          </>
        )}
      </Container>
    </div>
  );
}

export default App;