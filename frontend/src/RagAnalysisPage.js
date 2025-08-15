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

  const { decision, confidence, reasoning, retrieved_docs } = ragAnalysis;

  return (
    <Box sx={{ p: 4 }}>
      <Button variant="contained" color="primary" onClick={onBack} sx={{ mb: 3 }}>
        <ArrowBackIcon sx={{ mr: 1 }} /> Back
      </Button>
      <Card sx={{ mb: 3, boxShadow: 3 }}>
        <CardContent>
          <Typography variant="h4" gutterBottom>RAG Analysis Result</Typography>
          <Chip label={`Decision: ${decision || 'N/A'}`} color="primary" sx={{ mr: 2, fontSize: '1.1rem' }} />
          <Chip label={`Confidence: ${confidence !== undefined ? confidence : 'N/A'}`} color="success" sx={{ fontSize: '1.1rem' }} />
          <Typography variant="body1" sx={{ mt: 2 }}><b>Reasoning:</b> {reasoning || 'N/A'}</Typography>
        </CardContent>
      </Card>
      <Card sx={{ boxShadow: 2 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>Retrieved Documents</Typography>
          {retrieved_docs && retrieved_docs.length > 0 ? (
            <Box>
              {retrieved_docs.map((doc, idx) => (
                <Card key={idx} sx={{ mb: 2, background: '#f5f5f5', borderRadius: 2, boxShadow: 1 }}>
                  <CardContent>
                    <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1 }}>
                      {doc.title || 'N/A'}
                    </Typography>
                    {doc.category && (
                      <Chip label={`Category: ${doc.category}`} color="primary" sx={{ mr: 1, mb: 1 }} />
                    )}
                    {doc.confidence !== undefined && (
                      <Chip label={`Confidence: ${doc.confidence}`} color="success" sx={{ mr: 1, mb: 1 }} />
                    )}
                    {doc.blip && (
                      <Typography variant="body2" sx={{ mb: 0.5 }}><b>BLIP:</b> {doc.blip}</Typography>
                    )}
                    {doc.ocr && (
                      <Typography variant="body2" sx={{ mb: 0.5 }}><b>OCR:</b> {doc.ocr}</Typography>
                    )}
                    {doc.policy && (
                      <Typography variant="body2" sx={{ mb: 0.5 }}><b>Policy:</b> {doc.policy.description}</Typography>
                    )}
                    {doc.examples && doc.examples.length > 0 && (
                      <Typography variant="body2" sx={{ mb: 0.5 }}><b>Examples:</b> {doc.examples.join(', ')}</Typography>
                    )}
                    {doc.action_required && (
                      <Typography variant="body2" sx={{ mb: 0.5 }}><b>Action Required:</b> {doc.action_required}</Typography>
                    )}
                    {doc.severity_indicators && doc.severity_indicators.length > 0 && (
                      <Typography variant="body2" sx={{ mb: 0.5 }}><b>Severity Indicators:</b> {doc.severity_indicators.join(', ')}</Typography>
                    )}
                    {doc.context_factors && doc.context_factors.length > 0 && (
                      <Typography variant="body2" sx={{ mb: 0.5 }}><b>Context Factors:</b> {doc.context_factors.join(', ')}</Typography>
                    )}
                    {doc.reasoning && (
                      <Typography variant="body2" sx={{ mb: 0.5 }}><b>Reasoning:</b> {doc.reasoning}</Typography>
                    )}
                    {doc.transcript && (
                      <Typography variant="body2" sx={{ mb: 0.5 }}><b>Transcript Excerpt:</b> {doc.transcript}</Typography>
                    )}
                    {doc.keywords && doc.keywords.length > 0 && (
                      <Typography variant="body2" sx={{ mb: 0.5 }}><b>Keywords:</b> {doc.keywords.join(', ')}</Typography>
                    )}
                  </CardContent>
                </Card>
              ))}
            </Box>
          ) : (
            <Typography>No RAG explanations found.</Typography>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}

export default RagAnalysisPage;
