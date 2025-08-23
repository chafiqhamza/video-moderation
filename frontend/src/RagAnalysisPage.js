import React from 'react';
import { Card, CardContent, Typography, Box, Button, Chip } from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';

function RagAnalysisPage({ ragAnalysis, onBack }) {
  if (!ragAnalysis) {
    return (
      <Box sx={{ p: 4 }}>
        <Typography variant="h5">No RAG analysis data available.</Typography>
        <Button variant="contained" color="primary" onClick={onBack} sx={{ mt: 2 }}>
          <ArrowBackIcon sx={{ mr: 1 }} /> Back
        </Button>
      </Box>
    );
  }

  const { decision, confidence, reasoning, retrieved_docs, frames_analyzed } = ragAnalysis;

  // Compute total frames analyzed and displayed
  const totalAnalyzed = frames_analyzed || (retrieved_docs ? retrieved_docs.length : 0);
  const totalDisplayed = retrieved_docs ? retrieved_docs.length : 0;

  return (
    <Box sx={{ p: 5, background: 'linear-gradient(120deg, #e3f2fd 60%, #fffde7 100%)', minHeight: '100vh' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <Button variant="outlined" color="secondary" sx={{ mr: 2, fontWeight: 'bold', borderRadius: 3 }} onClick={onBack}>
          ← Back
        </Button>
        <Typography variant="h4" color="primary" sx={{ fontWeight: 'bold' }}>
          RAG Policy Violation Explanations
        </Typography>
      </Box>
      {/* Summary Card */}
      <Card sx={{ mb: 4, background: 'linear-gradient(90deg,#1976d2 60%,#21cbf3 100%)', color: '#fff', boxShadow: 3, borderRadius: 4 }}>
        <CardContent>
          <Typography variant="h5" sx={{ fontWeight: 'bold', mb: 2 }}>
            RAG Analysis Result
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
            <Chip label={`Decision: ${decision || 'N/A'}`} color="primary" sx={{ fontSize: '1.1rem' }} />
            <Chip label={`Confidence: ${confidence !== undefined ? confidence : 'N/A'}`} color="success" sx={{ fontSize: '1.1rem' }} />
          </Box>
          <Typography variant="body1"><b>Reasoning:</b> {reasoning || 'N/A'}</Typography>
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="info.main">
              <b>Total Frames Analyzed by Model:</b> {totalAnalyzed}
            </Typography>
            <Typography variant="body2" color="info.main">
              <b>Frames Displayed:</b> {totalDisplayed}
            </Typography>
            {totalAnalyzed !== totalDisplayed && (
              <Typography variant="body2" color="warning.main">
                (Only the first {totalDisplayed} frames are shown based on your settings)
              </Typography>
            )}
          </Box>
        </CardContent>
      </Card>
      {/* Documents Section */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="h5" color="primary" sx={{ fontWeight: 'bold', mb: 2 }}>
          Frame & Audio Details
        </Typography>
        {retrieved_docs && retrieved_docs.length > 0 ? (
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: 3 }}>
            {retrieved_docs.map((doc, idx) => (
              <Card key={idx} sx={{ background: typeof doc.title === 'string' && doc.title.startsWith('Audio') ? '#e3f2fd' : '#f9fbe7', borderRadius: 3, boxShadow: 2 }}>
                <CardContent>
                  <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1, color: typeof doc.title === 'string' && doc.title.startsWith('Audio') ? '#1976d2' : 'success.main' }}>
                    {doc.title || 'N/A'}
                  </Typography>
                  {/* Context: frame number, timestamp, etc. */}
                  {doc.frame !== undefined && (
                    <Typography variant="body2" sx={{ mb: 0.5 }}>
                      <b>Frame #:</b> {doc.frame}
                    </Typography>
                  )}
                  {doc.timestamp !== undefined && (
                    <Typography variant="body2" sx={{ mb: 0.5 }}>
                      <b>Timestamp:</b> {doc.timestamp}s
                    </Typography>
                  )}
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 1 }}>
                    {doc.category && (
                      <Chip label={`Category: ${doc.category}`} color={doc.category === 'safe_content' ? 'success' : 'primary'} />
                    )}
                    {doc.confidence !== undefined && (
                      <Chip label={`Confidence: ${doc.confidence}`} color="success" />
                    )}
                  </Box>
                  {doc.reasoning && <Typography variant="body2" sx={{ mb: 0.5 }}><b>Reasoning:</b> {doc.reasoning}</Typography>}
                  {doc.blip && <Typography variant="body2" sx={{ mb: 0.5 }}><b>BLIP:</b> {doc.blip}</Typography>}
                  {doc.ocr && <Typography variant="body2" sx={{ mb: 0.5 }}><b>OCR:</b> {doc.ocr}</Typography>}
                  {doc.policy && <Typography variant="body2" sx={{ mb: 0.5 }}><b>Policy:</b> {doc.policy.description}</Typography>}
                  {doc.examples && doc.examples.length > 0 && <Typography variant="body2" sx={{ mb: 0.5 }}><b>Examples:</b> {doc.examples.join(', ')}</Typography>}
                  {doc.action_required && <Typography variant="body2" sx={{ mb: 0.5 }}><b>Action Required:</b> {doc.action_required}</Typography>}
                  {doc.severity_indicators && doc.severity_indicators.length > 0 && <Typography variant="body2" sx={{ mb: 0.5 }}><b>Severity Indicators:</b> {doc.severity_indicators.join(', ')}</Typography>}
                  {doc.context_factors && doc.context_factors.length > 0 && <Typography variant="body2" sx={{ mb: 0.5 }}><b>Context Factors:</b> {doc.context_factors.join(', ')}</Typography>}
                  {doc.transcript && <Typography variant="body2" sx={{ mb: 0.5 }}><b>Transcript Excerpt:</b> {doc.transcript}</Typography>}
                  {doc.keywords && doc.keywords.length > 0 && <Typography variant="body2" sx={{ mb: 0.5 }}><b>Keywords:</b> {doc.keywords.join(', ')}</Typography>}
                </CardContent>
              </Card>
            ))}
          </Box>
        ) : (
          <Typography>No RAG explanations found.</Typography>
        )}
      </Box>
    </Box>
  );
}

export default RagAnalysisPage;
