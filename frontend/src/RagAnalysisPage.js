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

  const { retrieved_docs } = ragAnalysis;

  // Helper: get icon and color for frame type
  const getFrameIcon = (category) => {
    if (!category) return null;
    const cat = String(category).replace('_', ' ').toLowerCase();
    if ([
      'safe content',
      'educational content',
      'entertainment content',
      'news content',
      'tutorials & how-to',
      'community & family-friendly',
      'artistic & creative expression',
      'positive social impact'
    ].includes(cat)) {
      return <span style={{ color: '#43a047', fontWeight: 'bold', fontSize: 22 }}>✔️</span>;
    }
    return <span style={{ color: '#e53935', fontWeight: 'bold', fontSize: 22 }}>⚠️</span>;
  };

  // Define positive categories
  const positiveCategories = [
    'safe content',
    'educational content',
    'entertainment content',
    'news content',
    'tutorials & how-to',
    'community & family-friendly',
    'artistic & creative expression',
    'positive social impact'
  ];

  // Group violations by type for summary, but skip positive categories
  const violationTypes = {};
  if (retrieved_docs && retrieved_docs.length > 0) {
    retrieved_docs.forEach(doc => {
      const cat = String(doc.category || '').toLowerCase();
      if (positiveCategories.includes(cat)) return; // skip positive
      if (!violationTypes[cat]) violationTypes[cat] = { count: 0, frames: [], actions: new Set(), reasons: new Set() };
      violationTypes[cat].count++;
      violationTypes[cat].frames.push(doc);
      if (doc.action_required) violationTypes[cat].actions.add(doc.action_required);
      if (doc.policy && doc.policy.description) violationTypes[cat].reasons.add(doc.policy.description);
      if (doc.reasoning) violationTypes[cat].reasons.add(doc.reasoning);
    });
  }
  const totalFrames = retrieved_docs ? retrieved_docs.length : 0;
  // Count safe frames and violation frames by their actual detected categories
  let safeFrames = 0;
  let violationFrames = 0;
  if (retrieved_docs && retrieved_docs.length > 0) {
    retrieved_docs.forEach(doc => {
      const cat = String(doc.category || '').toLowerCase();
      if (cat === 'safe content' || positiveCategories.includes(cat)) {
        safeFrames++;
      } else {
        violationFrames++;
      }
    });
  }
  const complianceRate = totalFrames > 0 ? Math.round((safeFrames / totalFrames) * 100) : 0;

  // Build grouped summary lines and collect all unique explanations for compliant videos
  let guidelineSummary = '';
  let guidelineReasons = [];
  let allExplanations = [];
  if (violationFrames === 0) {
    guidelineSummary = 'This video follows YouTube guidelines.';
    guidelineReasons = ['No violations detected. Content is safe and suitable for YouTube.'];
    // Collect all unique policy descriptions and reasoning from positive frames
    if (retrieved_docs && retrieved_docs.length > 0) {
      const explanationsSet = new Set();
      retrieved_docs.forEach(doc => {
        if (doc.policy && doc.policy.description) explanationsSet.add(doc.policy.description);
        if (doc.reasoning) explanationsSet.add(doc.reasoning);
      });
      allExplanations = Array.from(explanationsSet);
    }
  } else {
    guidelineSummary = 'This video does NOT follow YouTube guidelines.';
    // Use RAG explanations (scraped policy data from backend/database) for summary
    if (ragAnalysis && ragAnalysis.rag_explanations && Array.isArray(ragAnalysis.rag_explanations)) {
      ragAnalysis.rag_explanations.forEach((exp) => {
        if (exp.policy) {
          let examples = '';
          if (Array.isArray(exp.policy.examples) && exp.policy.examples.length > 0) {
            examples = '\nExamples:';
            exp.policy.examples.forEach((ex, i) => {
              examples += `\n  ${i+1}. ${ex}`;
            });
          }
          let actionRequired = exp.policy.action_required ? `\nAction Required: ${exp.policy.action_required}` : '';
          let line = '';
          // Only use official YouTube policy description and examples from DB
          line = `${exp.category}:\n${exp.policy.description || ''}${examples}${actionRequired}`;
          guidelineReasons.push(line);
        }
      });
    } else {
      // Fallback: previous logic if no RAG explanations
      Object.entries(violationTypes).forEach(([cat, info]) => {
        const doc = info.frames[0] || {};
        let details = '';
        if (doc.policy && doc.policy.description) details += doc.policy.description + ' ';
        if (doc.personalized_reason) details += doc.personalized_reason + ' ';
        guidelineReasons.push(
          `${cat.replace('_', ' ')}: Detected in ${info.count} frame${info.count > 1 ? 's' : ''}. ${details.trim()}`
        );
      });
      if (safeFrames > 0) {
        const safeDoc = retrieved_docs.find(doc => {
          const cat = String(doc.category || '').toLowerCase();
          return cat === 'safe content' || positiveCategories.includes(cat);
        }) || {};
        let details = '';
        if (safeDoc.policy && safeDoc.policy.description) details += safeDoc.policy.description + ' ';
        guidelineReasons.push(
          `Safe: Detected in ${safeFrames} frame${safeFrames > 1 ? 's' : ''}. ${details.trim()}`
        );
      }
    }
    if (guidelineReasons.length === 0) {
      guidelineReasons = ['Violations detected, but no detailed policy explanation available.'];
    }
  }

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
      {/* YouTube-style guideline summary */}
      <Card sx={{ maxWidth: 900, mb: 3, boxShadow: 3, borderRadius: 3, p: 2, background: violationFrames === 0 ? '#e8f5e9' : '#ffebee', border: violationFrames === 0 ? undefined : '2px solid #e53935' }}>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 'bold', color: violationFrames === 0 ? '#43a047' : '#e53935', mb: 1 }}>
            {guidelineSummary}
          </Typography>
          {guidelineReasons.map((reason, idx) => (
            <Typography key={idx} variant="body2" sx={{ mb: 0.5, color: violationFrames === 0 ? '#43a047' : '#e53935', fontWeight: 'bold' }}>{reason}</Typography>
          ))}
          {/* Prominent explanation section for compliant videos */}
          {violationFrames === 0 && allExplanations.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 'bold', color: '#1976d2', mb: 1 }}>
                Content Explanations:
              </Typography>
              {allExplanations.map((exp, idx) => (
                <Typography key={idx} variant="body2" sx={{ mb: 0.5, color: '#1976d2' }}>{exp}</Typography>
              ))}
            </Box>
          )}
          <Box sx={{ display: 'flex', gap: 2, mt: 2 }}>
            <Chip label={`Compliance Rate: ${complianceRate}%`} color={violationFrames === 0 ? 'success' : 'error'} sx={{ fontWeight: 'bold' }} />
            <Chip label={`Safe Frames: ${safeFrames}`} color="success" sx={{ fontWeight: 'bold' }} />
            <Chip label={`Violations: ${violationFrames}`} color="error" sx={{ fontWeight: 'bold' }} />
            <Chip label={`Total Frames: ${totalFrames}`} color="primary" sx={{ fontWeight: 'bold' }} />
          </Box>
        </CardContent>
      </Card>
      {/* Frame-by-frame cards */}
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: 3 }}>
        {retrieved_docs && retrieved_docs.length > 0 ? (
          <>
            {retrieved_docs.map((doc, idx) => {
              const cat = String(doc.category || '').toLowerCase();
              const isPositive = positiveCategories.includes(cat);
              // Build a summary of what the frame contains
              let frameSummary = [];
              if (doc.blip) frameSummary.push(`BLIP: ${doc.blip}`);
              if (doc.ocr) frameSummary.push(`OCR: ${doc.ocr}`);
              if (doc.transcript) frameSummary.push(`Transcript: ${doc.transcript}`);
              if (doc.keywords && doc.keywords.length > 0) frameSummary.push(`Keywords: ${doc.keywords.join(', ')}`);
              if (doc.policy && doc.policy.description) frameSummary.push(`Policy: ${doc.policy.description}`);
              if (doc.reasoning) frameSummary.push(`Explication: ${doc.reasoning}`);
              if (doc.rag) frameSummary.push(`RAG Explanation: ${typeof doc.rag === 'string' ? doc.rag : JSON.stringify(doc.rag)}`);
              if (doc.personalized_reason) frameSummary.push(`Personalized Reason: ${doc.personalized_reason}`);
              if (doc.explanation) frameSummary.push(`Explanation: ${doc.explanation}`);
              if (doc.examples && doc.examples.length > 0) frameSummary.push(`Examples: ${doc.examples.join(', ')}`);
              if (doc.action_required) frameSummary.push(`Action Required: ${doc.action_required}`);
              if (doc.severity_indicators && doc.severity_indicators.length > 0) frameSummary.push(`Severity Indicators: ${doc.severity_indicators.join(', ')}`);
              if (doc.context_factors && doc.context_factors.length > 0) frameSummary.push(`Context Factors: ${doc.context_factors.join(', ')}`);

              return (
                <Card key={idx} sx={{ background: isPositive ? '#f9fbe7' : '#ffebee', borderRadius: 3, boxShadow: 2 }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      {getFrameIcon(doc.category)}
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', ml: 1, color: isPositive ? 'success.main' : 'error.main', background: isPositive ? '#1976d2' : undefined, color: isPositive ? '#fff' : undefined, px: 1, borderRadius: 1 }}>
                        {doc.title || doc.category || 'Frame'}
                      </Typography>
                    </Box>
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
                        <Chip label={`Category: ${doc.category}`} color={isPositive ? 'success' : 'primary'} sx={{ fontWeight: 'bold', background: isPositive ? '#388e3c' : undefined, color: isPositive ? '#fff' : undefined }} />
                      )}
                      {doc.confidence !== undefined && (
                        <Chip label={`Confidence: ${doc.confidence}`} color="success" />
                      )}
                    </Box>
                    {/* Frame summary section */}
                    {frameSummary.length > 0 && (
                      <Box sx={{ mt: 1, mb: 1, background: '#e3f2fd', borderRadius: 2, p: 1 }}>
                        <Typography variant="body2" sx={{ fontWeight: 'bold', color: '#1976d2', mb: 0.5 }}>Frame Details:</Typography>
                        {frameSummary.map((line, i) => (
                          <Typography key={i} variant="body2" sx={{ color: '#1976d2', mb: 0.5 }}>{line}</Typography>
                        ))}
                      </Box>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </>
        ) : (
          <Typography>No RAG explanations found.</Typography>
        )}
      </Box>
    </Box>
  );
}

export default RagAnalysisPage;






