import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, Box, Button, Chip, List, ListItem, ListItemText, IconButton, Collapse, Divider, Tooltip, Dialog, DialogTitle, DialogContent, DialogActions, Stack, CssBaseline, TextField, CircularProgress, Snackbar, Alert } from '@mui/material';
import { ThemeProvider, createTheme, responsiveFontSizes } from '@mui/material/styles';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

function RagAnalysisPage({ ragAnalysis, onBack }) {
  const [serverRecs, setServerRecs] = useState(null);
  const [llmRaw, setLlmRaw] = useState(null);
  const [showRaw, setShowRaw] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [expandedFrames, setExpandedFrames] = useState({});
  const [recExpanded, setRecExpanded] = useState({});
  const [copiedCase, setCopiedCase] = useState(false);
  const [openDoc, setOpenDoc] = useState(null);
  const [openPolicy, setOpenPolicy] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [previewing, setPreviewing] = useState(false);
  // Local fallback policies loaded from a committed JSON file when DB helper returns no entries
  const [localFallbackPolicies, setLocalFallbackPolicies] = useState(null);
  const [localFallbackTried, setLocalFallbackTried] = useState(false);
  // If no DB-backed policies were found, try loading a local fallback JSON (read-only).
  // We only attempt this once per page load to avoid noisy network activity.
  useEffect(() => {
    if (localFallbackTried) return;
    setLocalFallbackTried(true);
    const tryUrls = [
      '/api/static/youtube_policy_blocks.json', // backend-served static path (preferred)
      '/youtube_policy_blocks.json', // public root (create-react-app style)
      '/static/youtube_policy_blocks.json'
    ];
    (async () => {
      for (const u of tryUrls) {
        try {
          const resp = await fetch(u, { method: 'GET', headers: { 'Accept': 'application/json' } });
          if (!resp.ok) continue;
          const j = await resp.json();
          if (Array.isArray(j) && j.length > 0) {
            setLocalFallbackPolicies(j);
            return;
          }
        } catch (e) {
          // ignore and try next
        }
      }
      // if none found, leave localFallbackPolicies as null
    })();
  }, []);
  const [toast, setToast] = useState({ open: false, severity: 'success', message: '' });
  const [blurStrength, setBlurStrength] = useState(10);
  // guideline details toggle removed - DB content always shown
  // showDbEvidence state removed — DB evidence will always be displayed
  // showAllSnippets state removed (personalized snippets UI removed)
  // Removed LLM availability banner (no setup button)
  const API_BASE = 'http://localhost:8000';

  // Professional theme for this page (keeps styles consistent and modern)
  // Defined inside module so it's stable across renders
  const baseTheme = createTheme({
    palette: {
      primary: { main: '#1e88e5' },
      secondary: { main: '#6b7280' },
      background: { default: '#f6f7fb', paper: '#ffffff' },
      success: { main: '#43a047' },
      error: { main: '#e53935' },
      info: { main: '#0288d1' }
    },
    typography: {
      fontFamily: 'Inter, Roboto, Arial, sans-serif',
      h5: { fontWeight: 700 },
      h6: { fontWeight: 700 }
    },
    shape: { borderRadius: 10 },
    spacing: 8,
    components: {
      MuiCard: {
        defaultProps: { elevation: 2 },
        styleOverrides: {
          root: {
            border: '1px solid rgba(16,24,40,0.06)',
            borderRadius: 12,
            // reduce default overflow and improve visual separation
            overflow: 'visible'
          }
        }
      },
      MuiCardContent: {
        styleOverrides: {
          root: {
            padding: '14px 16px'
          }
        }
      },
      MuiChip: {
        defaultProps: { variant: 'filled', size: 'small' },
        styleOverrides: {
          root: {
            borderRadius: 8
          }
        }
      },
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 10,
            textTransform: 'none'
          }
        }
      },
      MuiListItem: {
        styleOverrides: {
          root: {
            paddingTop: 6,
            paddingBottom: 6
          }
        }
      },
      MuiTypography: {
        styleOverrides: {
          subtitle1: { fontWeight: 600 }
        }
      }
    }
  });
  const theme = responsiveFontSizes(baseTheme);

  // Note: LLM availability banner and ping removed to simplify UI
  // Helper to append debug flag when showRaw is enabled
  const withDebug = (url) => (showRaw ? `${url}${url.includes('?') ? '&' : '?'}debug=true` : url);

  // Helper: pick the best retrieved policy (highest similarity) for a doc or top-level
  const pickBestPolicy = (policies) => {
    if (!policies || !Array.isArray(policies) || policies.length === 0) return null;
    try {
      return policies.slice().sort((a, b) => (Number(b.similarity || 0) - Number(a.similarity || 0)))[0];
    } catch (e) { return policies[0]; }
  };

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
  // If the backend returned a structured RAG explanation at the top-level, prefer it
  // for the banner and guideline reasons. This ensures DB-backed, authoritative text
  // is shown instead of heuristics when available.
  const topLevelExplanation = (ragAnalysis && ragAnalysis.rag_decision && ragAnalysis.rag_decision.explanation) ? ragAnalysis.rag_decision.explanation : null;

  const applyFrameFix = async (doc, action, opts = {}) => {
    try {
      const vid = ragAnalysis.video_id;
      // Guard: require a valid integer video id before calling apply endpoint
      let realVid = vid;
      if (!realVid || Number.isNaN(Number(realVid)) || !Number.isInteger(Number(realVid))) {
        console.warn('No valid video id available for apply, attempting to save report first:', realVid);
        try {
          // Try stable save endpoint first
          let saveResp = await fetch(`${API_BASE}/api/videos/save-or-id`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragAnalysis) });
          if (!saveResp.ok && (saveResp.status === 404 || saveResp.status === 405)) {
            saveResp = await fetch(`${API_BASE}/videos/save-or-id`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragAnalysis) });
          }
          if (!saveResp.ok) {
            const txt = await saveResp.text().catch(() => '');
            throw new Error('Save before apply failed: ' + (txt || saveResp.statusText || saveResp.status));
          }
          const saveData = await saveResp.json().catch(() => null);
          realVid = saveData && saveData.video_id;
          if (!realVid) throw new Error('Save did not return video_id');
          // reflect saved id locally so subsequent actions use it
          try { ragAnalysis.video_id = realVid; } catch (e) { /* ignore if read-only */ }
        } catch (e) {
          console.error('Auto-save before apply failed', e);
          // eslint-disable-next-line no-alert
          alert('Cannot apply fixes: failed to save report first. ' + (e.message || e));
          return;
        }
      }
      const payload = {
        type: 'frame_fix',
        frame: doc.frame,
        timestamp: doc.timestamp,
        action: action,
        options: opts || {},
        note: `Applied fix '${action}' to frame ${doc.frame}`,
        // If frontend knows a local source path, include it to allow ffmpeg processing server-side
        source_path: ragAnalysis.source_path || ragAnalysis.video_path || null
      };
  const resp = await fetch(`${API_BASE}/api/videos/${realVid}/apply`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!resp.ok) {
        const txt = await resp.text().catch(() => '');
        throw new Error('Apply request failed: ' + txt);
      }
      const data = await resp.json();
      // If queued, poll the applied status endpoint until result_path available or timeout
  if (data && (data.status === 'queued' || data.status === 'ok') && data.applied_id) {
        const appliedId = data.applied_id;
        // Inform user that processing is queued
        // eslint-disable-next-line no-alert
        alert(`Apply job queued (id=${appliedId}). Waiting for server to finish processing...`);
        const maxTries = 30; // ~45s
        let tries = 0;
        while (tries < maxTries) {
          // wait
          // eslint-disable-next-line no-await-in-loop
          await new Promise(r => setTimeout(r, 1500));
          // eslint-disable-next-line no-await-in-loop
          const st = await fetch(`${API_BASE}/api/videos/${realVid}/applied/${appliedId}`);
          if (st.ok) {
            // eslint-disable-next-line no-await-in-loop
            const sdat = await st.json();
            if (sdat && sdat.status === 'ok' && sdat.applied) {
              const rec = sdat.applied;
              if (rec.result_path) {
                try { setPreviewUrl(rec.result_path); setToast({ open: true, severity: 'success', message: 'Processing complete — preview available' }); } catch (e) { /* ignore */ }
                return rec.result_path;
              }
              if (rec.status === 'error') {
                // eslint-disable-next-line no-alert
                alert('Processing failed: ' + (rec.error_message || JSON.stringify(rec)));
                throw new Error('Processing failed: ' + (rec.error_message || JSON.stringify(rec)));
              }
            }
          }
          tries++;
        }
  // timed out
  // eslint-disable-next-line no-alert
  alert('Processing timed out. You can check the Applied Actions endpoint later.');
  return null;
      } else if (data && data.result_path) {
        try { setPreviewUrl(data.result_path); setToast({ open: true, severity: 'success', message: 'Apply finished — preview ready' }); } catch (e) { /* ignore */ }
        return data.result_path;
      } else {
        // eslint-disable-next-line no-alert
        alert('Apply response: ' + JSON.stringify(data));
  return null;
      }
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('Apply failed', err);
      // eslint-disable-next-line no-alert
      alert('Apply failed: ' + (err.message || err));
      return null;
    }
  };

  // Helper to render examples robustly (handles strings and objects)
  const formatExamples = (examples) => {
    if (!examples || !Array.isArray(examples) || examples.length === 0) return '';
    return examples.map(e => {
      if (!e && e !== 0) return '';
      if (typeof e === 'string') return e;
      if (typeof e === 'number') return String(e);
      if (typeof e === 'object') {
        const ts = (e.timestamp !== undefined && e.timestamp !== null) ? (typeof e.timestamp === 'number' ? `${e.timestamp.toFixed ? e.timestamp.toFixed(1) : Number(e.timestamp)}s` : String(e.timestamp)) : null;
        const cat = e.category || e.cat || e.category_name || null;
        if (ts && cat) return `${cat}@${ts}`;
        if (ts) return `${ts}`;
        if (cat) return `${cat}`;
        // Fallback to a short JSON summary
        try { return JSON.stringify(e); } catch (err) { return String(e); }
      }
      return String(e);
    }).filter(Boolean).join(', ');
  };

  // Toggle per-frame expanded detailed view
  const toggleFrame = (i) => setExpandedFrames(prev => ({ ...prev, [i]: !prev[i] }));

  // Render a compact, slide-friendly "PowerPoint-like" detailed analysis for a single frame
  const renderDetailedAnalysis = (doc, idx) => {
    if (!doc) return null;
    // Collect structured pieces from the doc (use multiple fields when available)
    const why = doc.policy && doc.policy.description ? doc.policy.description : (doc.personalized_reason || doc.reasoning || doc.explanation || 'No detailed policy text available.');
    const severity = (doc.severity_indicators && doc.severity_indicators.length > 0) ? doc.severity_indicators.join(', ') : (doc.severity || 'Not specified');
    const context = (doc.context_factors && doc.context_factors.length > 0) ? doc.context_factors.join(', ') : (doc.context || 'Not specified');
    const examples = (doc.examples && doc.examples.length > 0) ? doc.examples : (doc.samples || []);
    const blip = doc.blip ? doc.blip : null;
    const ocr = doc.ocr ? doc.ocr : null;
    const transcript = doc.transcript ? doc.transcript : null;
    const suggested = doc.suggested_action || doc.suggestedAction || doc.action || (doc.policy && doc.policy.action_required) || 'No suggested action provided';
    const steps = Array.isArray(doc.suggested_steps) ? doc.suggested_steps : (doc.suggestedSteps || doc.suggested || []);
    const effort = doc.estimated_effort_minutes || doc.estimated_effort || null;
    const quick = Array.isArray(doc.quick_wins) ? doc.quick_wins : (doc.quickWins || []);

    return (
      <Box sx={{ mt: 1 }}>
        <Divider sx={{ my: 1 }} />
        <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>Detailed Analysis (slide-style bullets)</Typography>
        <List dense>
          <ListItem>
            <ListItemText primary={<><strong>Summary:</strong> {why}</>} />
          </ListItem>
          <ListItem>
            <ListItemText primary={<><strong>Detected Evidence:</strong> {blip ? (`BLIP: ${blip}`) : ''} {ocr ? (`OCR: ${ocr}`) : ''} {transcript ? (`Transcript excerpt: ${transcript}`) : ''}</>} />
          </ListItem>
          <ListItem>
            <ListItemText primary={<><strong>Severity indicators:</strong> {severity}</>} />
          </ListItem>
          <ListItem>
            <ListItemText primary={<><strong>Context factors:</strong> {context}</>} />
          </ListItem>
          <ListItem>
            <ListItemText primary={<><strong>Examples / frames:</strong> {examples && examples.length > 0 ? formatExamples(examples) : `Frame ${doc.frame}${doc.timestamp ? ` @ ${doc.timestamp}s` : ''}`}</>} />
          </ListItem>
          <ListItem>
            <ListItemText primary={<><strong>Suggested action:</strong> {suggested}</>} />
          </ListItem>
          {steps && steps.length > 0 && (
            <ListItem>
              <ListItemText primary={<><strong>Suggested steps:</strong> {Array.isArray(steps) ? steps.join(' · ') : String(steps)}</>} />
            </ListItem>
          )}
          {effort && (
            <ListItem>
              <ListItemText primary={<><strong>Estimated effort:</strong> {effort} minutes</>} />
            </ListItem>
          )}
          {quick && quick.length > 0 && (
            <ListItem>
              <ListItemText primary={<><strong>Quick wins:</strong> {quick.join(' · ')}</>} />
            </ListItem>
          )}
          {/* Retrieved policies for this frame (if any) */}
          {doc.retrieved_policies && doc.retrieved_policies.length > 0 && (
            <ListItem>
              <ListItemText primary={<><strong>Retrieved policies (DB evidence):</strong></>} />
              <Box component="ul" sx={{ pl: 2, mt: 1 }}>
                {doc.retrieved_policies.map((p, pi) => (
                  <li key={pi} style={{ cursor: 'pointer' }} onClick={() => setOpenPolicy(p)}>
                    <Typography variant="body2"><strong>{p.policy_info?.category || p.category || 'policy'}</strong> — {p.policy_info?.title || (p.policy_info && (p.policy_info.description||p.policy_text)) || p.policy_info?.description || p.description || 'No description available'}</Typography>
                    <Typography variant="caption" color="text.secondary">Source: {p.policy_info?.source || p.source || 'unknown'} | sim: {p.similarity ? Number(p.similarity).toFixed(3) : 'N/A'}</Typography>
                  </li>
                ))}
              </Box>
            </ListItem>
          )}
        </List>
      </Box>
    );
  };

  // Render an aggregated, slide-friendly policy breakdown for all violation types
  const renderPolicyBreakdown = () => {
    const entries = Object.entries(violationTypes || {});
    if (!entries || entries.length === 0) return null;
    return (
      <Card sx={{ maxWidth: 900, mb: 3, boxShadow: 3, borderRadius: 3, p: 2 }}>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 'bold', mb: 1 }}>Detailed Policy Breakdown (slide bullets)</Typography>
          {entries.map(([cat, info], i) => {
            const sample = info.frames && info.frames[0] ? info.frames[0] : {};
            // Prefer DB policy_info when available for authoritative text
            let why = null;
            let actionReq = null;
            let reasons = '';
            if (Array.isArray(sample.retrieved_policies) && sample.retrieved_policies.length > 0) {
              const pp = sample.retrieved_policies[0];
              const infoPolicy = pp.policy_info || pp;
              why = infoPolicy && (infoPolicy.description || infoPolicy.policy_text || infoPolicy.title);
              actionReq = (info.actions && Array.from(info.actions).join(', ')) || (infoPolicy && (infoPolicy.action_required || infoPolicy.action)) || (sample.action_required || (sample.policy && sample.policy.action_required));
              // collect reason snippets from retrieved policies if present
              reasons = (info.reasons && Array.from(info.reasons).join(' · ')) || (infoPolicy && (infoPolicy.description ? infoPolicy.description.split('\n')[0] : '')) || '';
            }
            if (!why) why = (sample.policy && sample.policy.description) || sample.personalized_reason || sample.reasoning || 'No detailed policy description available.';
            if (!actionReq) actionReq = (info.actions && Array.from(info.actions).join(', ')) || (sample.action_required || (sample.policy && sample.policy.action_required)) || 'No action required specified.';
            return (
              <Box key={i} sx={{ mb: 1 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>{i+1}. {(cat||'').replace(/_/g,' ')}</Typography>
                <List dense>
                  <ListItem>
                    <ListItemText primary={<><strong>Why this is a concern:</strong> {why}</>} />
                  </ListItem>
                  {reasons && (
                    <ListItem>
                      <ListItemText primary={<><strong>Policy excerpts / reasoning:</strong> {reasons}</>} />
                    </ListItem>
                  )}
                  <ListItem>
                    <ListItemText primary={<><strong>Detected frames:</strong> {info.count} frame{info.count>1?'s':''} (examples: {info.frames.slice(0,3).map(f => `#${f.frame}${f.timestamp?`@${f.timestamp}s`:''}`).join(', ')})</>} />
                  </ListItem>
                  <ListItem>
                    <ListItemText primary={<><strong>Action Required:</strong> {actionReq}</>} />
                  </ListItem>
                </List>
              </Box>
            );
          })}
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>Tip: Click individual frames below to expand slide-style detailed analysis for each flagged frame.</Typography>
        </CardContent>
      </Card>
    );
  };

  // Render RAG decision and DB evidence if available
  const renderRagDbSummary = () => {
    const caseId = ragAnalysis.case_id || ragAnalysis.caseId || ragAnalysis.case || null;
    const ragDecision = ragAnalysis.rag_decision || ragAnalysis.ragDecision || ragAnalysis.rag_decision_summary || null;
    const relevantPolicies = ragAnalysis.relevant_policies || ragAnalysis.relevantPolicies || ragAnalysis.retrieved_policies || null;
    const metadata = ragAnalysis.metadata || ragAnalysis.db_metadata || ragAnalysis.moderation_case || null;

    if (!caseId && !ragDecision && !(relevantPolicies && relevantPolicies.length) && !metadata) return null;

    return (
      <Card sx={{ maxWidth: 900, mb: 3, boxShadow: 3, borderRadius: 3, p: 2 }}>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 'bold', mb: 1 }}>RAG Decision & Database Evidence</Typography>
          {caseId && <Typography variant="body2" sx={{ mb: 1 }}><strong>Case ID:</strong> {caseId}</Typography>}
          {caseId && (
            <Tooltip title={copiedCase ? 'Copied' : 'Copy Case ID'}>
              <IconButton size="small" onClick={() => { navigator.clipboard && navigator.clipboard.writeText(String(caseId)); setCopiedCase(true); setTimeout(() => setCopiedCase(false), 2200); }}>
                <ContentCopyIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
          {ragDecision && (
            <Box sx={{ mb: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>RAG Decision</Typography>
              <Typography variant="body2">Decision: {ragDecision.decision || ragDecision.action || String(ragDecision)}</Typography>
              {ragDecision.reasoning && <Typography variant="body2">Reasoning: {ragDecision.reasoning}</Typography>}
              {typeof ragDecision.confidence !== 'undefined' && <Typography variant="body2">Confidence: {Number(ragDecision.confidence).toFixed(2)}</Typography>}
              {ragDecision.relevant_policy && <Typography variant="body2">Relevant policy: {ragDecision.relevant_policy} ({ragDecision.policy_similarity ? Number(ragDecision.policy_similarity).toFixed(3) : 'N/A'})</Typography>}
            </Box>
          )}
          {relevantPolicies && relevantPolicies.length > 0 && (
            <Box sx={{ mb: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>Retrieved policies (RAG)</Typography>
              <List dense>
                {relevantPolicies.map((p, i) => (
                  <ListItem key={i}>
                    <ListItemText primary={<><strong>{p.category || p.policy_info?.category || 'policy'}</strong> — {p.source || p.policy_info?.source || 'source unknown'}</>} secondary={p.policy_info?.description || (p.policy_info && JSON.stringify(p.policy_info)) || (p.description || '')} />
                    <Chip label={p.similarity ? `sim:${Number(p.similarity).toFixed(3)}` : 'sim:N/A'} size="small" />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}
          {metadata && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>Stored case metadata</Typography>
              <Box component="pre" sx={{ whiteSpace: 'pre-wrap', p: 1, bgcolor: '#f5f5f5', borderRadius: 1, maxHeight: 200, overflow: 'auto' }}>
                {typeof metadata === 'string' ? metadata : JSON.stringify(metadata, null, 2)}
              </Box>
            </Box>
          )}
        </CardContent>
      </Card>
    );
  };

  // Try to locate phi3 model analysis in several possible payload locations
  const findPhi3 = () => {
    if (!ragAnalysis) return null;
    const candidates = [
      ragAnalysis.phi3,
      ragAnalysis.phi_3,
      ragAnalysis.phi_three,
      ragAnalysis.model_outputs && ragAnalysis.model_outputs.phi3,
      ragAnalysis.model_outputs && ragAnalysis.model_outputs.phi_3,
      ragAnalysis.model_outputs && ragAnalysis.model_outputs.phi3_result,
      ragAnalysis.phi3_result,
      ragAnalysis.phi3_analysis,
      ragAnalysis.phi3_output,
    ];
    for (let c of candidates) {
      if (c && (typeof c === 'object' || typeof c === 'string')) return c;
    }
    // also check nested keys under rag_decision or rag_explanations
    if (ragAnalysis.rag_decision && ragAnalysis.rag_decision.phi3) return ragAnalysis.rag_decision.phi3;
    if (Array.isArray(ragAnalysis.rag_explanations)) {
      const found = ragAnalysis.rag_explanations.find(e => e && (e.phi3 || e.phi_3 || e.phi3_result));
      if (found) return found.phi3 || found.phi_3 || found.phi3_result;
    }
    return null;
  };

  const renderPhi3Card = () => {
    const phi = findPhi3();
    if (!phi) return null;
    // phi may be string or object; normalize for display
    const phiObj = (typeof phi === 'string') ? { summary: phi } : (typeof phi === 'object' ? phi : { value: String(phi) });
    const summary = phiObj.summary || phiObj.text || phiObj.excerpt || phiObj.description || null;
    const conf = phiObj.confidence || phiObj.conf || phiObj.score || null;
    return (
      <Card sx={{ maxWidth: 900, mb: 3, boxShadow: 3, borderRadius: 3, p: 2 }}>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 'bold', mb: 1 }}>phi3 Model Analysis</Typography>
          {summary && <Typography variant="body2" sx={{ mb: 1 }}>{summary}</Typography>}
          {typeof conf !== 'undefined' && <Typography variant="caption" sx={{ display: 'block', mb: 1 }}>Confidence: {Number(conf).toFixed ? Number(conf).toFixed(3) : String(conf)}</Typography>}
          <Box component="pre" sx={{ whiteSpace: 'pre-wrap', maxHeight: 220, overflow: 'auto', background: '#f5f5f5', p: 1, borderRadius: 1 }}>
            {JSON.stringify(phiObj, null, 2)}
          </Box>
        </CardContent>
      </Card>
    );
  };

  // Collect DB policies that appear political in nature (heuristic)
  const getPoliticalPolicies = () => {
    if (!ragAnalysis) return [];
    const allPolicies = [];
    // gather from top-level retrieved_policies and per-frame retrieved_policies
    const pushIfPolitical = (p) => {
      if (!p) return;
      const cat = String(p.policy_info?.category || p.category || '').toLowerCase();
      const title = String(p.policy_info?.title || p.policy_info?.description || p.description || '');
      if (/politic|election|gov|campaign|public policy|political/i.test(cat + ' ' + title)) {
        allPolicies.push(p);
      }
    };
    if (Array.isArray(ragAnalysis.retrieved_policies)) ragAnalysis.retrieved_policies.forEach(pushIfPolitical);
    if (Array.isArray(retrieved_docs)) {
      retrieved_docs.forEach(d => {
        if (Array.isArray(d.retrieved_policies)) d.retrieved_policies.forEach(pushIfPolitical);
      });
    }
    // dedupe by description+category
    const map = new Map();
    allPolicies.forEach(p => {
      const key = (p.policy_info?.title || p.policy_info?.description || p.description || '') + '|' + (p.policy_info?.category || p.category || '');
      if (!map.has(key)) map.set(key, p);
    });
    return Array.from(map.values());
  };

  const renderPoliticalViolations = () => {
    const policies = getPoliticalPolicies();
    if (!policies || policies.length === 0) return null;
    return (
      <Card sx={{ maxWidth: 1000, mb: 3, boxShadow: 3, borderRadius: 3, p: 2 }}>
        <CardContent>
          <Typography variant="h6" sx={{ fontWeight: 'bold', mb: 1 }}>Political Policy Violations (DB evidence)</Typography>
          <Typography variant="body2" sx={{ mb: 1 }}>Below are retrieved database policies and matching evidence that the RAG system considered relevant to political/policy content; each item shows the full DB text, actions, examples and the frames where it was matched.</Typography>
          {policies.map((p, i) => {
            const title = p.policy_info?.title || p.policy_info?.category || p.category || `Policy ${i+1}`;
            const desc = p.policy_info?.description || p.description || 'No description available.';
            const source = p.policy_info?.source || p.source || 'unknown';
            const sim = p.similarity ? Number(p.similarity).toFixed(3) : 'N/A';
            // find example frames that reference this policy (by exact description match or category)
            const exampleFrames = [];
            if (retrieved_docs && Array.isArray(retrieved_docs)) {
              retrieved_docs.forEach(doc => {
                if (Array.isArray(doc.retrieved_policies)) {
                  const matched = doc.retrieved_policies.find(rp => {
                    const a = (rp.policy_info?.description || rp.description || '').trim();
                    const b = (p.policy_info?.description || p.description || '').trim();
                    return a && b && a === b;
                  });
                  if (matched) exampleFrames.push({ frame: doc.frame || doc.frame_index, ts: doc.timestamp });
                }
              });
            }
            return (
              <Box key={i} sx={{ mb: 2 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: '700' }}>{title}</Typography>
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', mb: 0.5 }}>{desc}</Typography>
                <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', mb: 1 }}>
                  <Typography variant="caption" color="text.secondary">Source: {source}</Typography>
                  <Typography variant="caption" color="text.secondary">similarity: {sim}</Typography>
                </Box>
                {p.policy_info && p.policy_info.examples && p.policy_info.examples.length > 0 && (
                  <Box sx={{ mb: 1 }}>
                    <Typography variant="subtitle2">Database examples</Typography>
                    <Box component="ul" sx={{ pl: 2 }}>
                      {p.policy_info.examples.map((ex, ei) => <li key={ei}><Typography variant="body2">{ex}</Typography></li>)}
                    </Box>
                  </Box>
                )}
                {exampleFrames.length > 0 && (
                  <Box sx={{ mb: 1 }}>
                    <Typography variant="subtitle2">Example frames</Typography>
                    <Box component="ul" sx={{ pl: 2 }}>
                      {exampleFrames.map((ef, ei) => <li key={ei}><Typography variant="body2">Frame {ef.frame}{ef.ts ? ` @ ${ef.ts}s` : ''}</Typography></li>)}
                    </Box>
                  </Box>
                )}
              </Box>
            );
          })}
        </CardContent>
      </Card>
    );
  };

  // Normalize recommendations returned from server so the UI can display them
  const normalizeRecs = (recs) => {
    if (!recs) return [];
    if (!Array.isArray(recs)) return [];
    return recs.map((r) => {
      if (r === null || r === undefined) return null;
      if (typeof r === 'string' || typeof r === 'number') {
        return { category: 'Recommendation', description: String(r), examples: [] };
      }
      if (typeof r === 'object') {
        const cat = r.category || r.cat || r.category_name || r.type || null;
        const suggested = r.suggested_action || r.suggestedAction || r.action || r.suggested || null;
        const desc = r.description || r.text || r.excerpt || r.summary || suggested || '';
        return {
          category: cat ? String(cat) : 'Recommendation',
          description: desc || 'No description provided. Click Show details for raw output.',
          suggested_action: suggested,
          examples: Array.isArray(r.examples) ? r.examples : (r.examples ? [r.examples] : []),
          quick_wins: Array.isArray(r.quick_wins) ? r.quick_wins : (r.quick_wins ? [r.quick_wins] : (r.quickWins || [])),
          // keep other fields for advanced UIs
          ...r
        };
      }
      return { category: 'unknown', description: String(r) };
    }).filter(Boolean);
  };

  // Helper: extract ffmpeg command lines from a block of text
  const extractFfmpegCommands = (text) => {
    if (!text || typeof text !== 'string') return [];
    const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    const cmds = lines.filter(l => /(^|\s)ffmpeg(\s|$)/i.test(l) || /^--?\w+/.test(l));
    // also include multi-line commands that contain 'ffmpeg' somewhere
    if (cmds.length === 0) {
      const matchAll = text.match(/(^|\n)([^\n]*ffmpeg[^\n]*)/ig);
      if (matchAll) cmds.push(...matchAll.map(m => m.replace(/^[\n\r]+/, '').trim()));
    }
    return Array.from(new Set(cmds));
  };

  const toggleRec = (i) => setRecExpanded(prev => ({ ...prev, [i]: !prev[i] }));

  // Helper: get icon for frame type. Accept a boolean `isPositiveFlag` so we can override
  // frames that are labeled 'safe' but contain DB evidence or actions and treat them as violations.
  const getFrameIcon = (isPositiveFlag) => {
    if (isPositiveFlag) {
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

  // policySummaries not required in this view; backend may supply if needed

  // Group violations by type for summary, but skip positive categories
  const violationTypes = {};
  // helper: detect whether a policy entry indicates action required
  const policyIndicatesAction = (p) => {
    if (!p) return false;
    const info = p.policy_info || p;
    const act = info?.action_required || info?.action || p.action_required || p.action;
    if (act && String(act).toLowerCase() !== 'none') return true;
    // consider importance/severity hints
    const sev = info?.severity || info?.importance || p.importance;
    if (typeof sev === 'string' && sev.toLowerCase() !== 'low') return true;
    if (typeof sev === 'number' && sev > 0.5) return true;
    return false;
  };
  if (retrieved_docs && retrieved_docs.length > 0) {
    retrieved_docs.forEach(doc => {
  // Prefer DB evidence: if top retrieved policy indicates action, treat as violation
  const topPolicy = pickBestPolicy(doc.retrieved_policies);
  const topPolicySim = topPolicy ? Number(topPolicy.similarity || 0) : 0;
  const topPolicyAction = topPolicy ? policyIndicatesAction(topPolicy) : false;
      const cat = String(doc.category || '').toLowerCase();
      // determine whether DB evidence is relevant (action/severity or non-safe category)
      const hasDbEvidence = Array.isArray(doc.retrieved_policies) && doc.retrieved_policies.length > 0;
  const dbEvidenceRelevant = hasDbEvidence && (doc.retrieved_policies.some(p => {
        const pcat = String(p.policy_info?.category || p.category || '').toLowerCase().replace(/\s+/g, '_');
        const action = p.policy_info?.action_required || p.action_required || p.action;
        const severity = p.policy_info?.severity || p.severity || p.importance;
        if (action && String(action).toLowerCase() !== 'none') return true;
        if (severity && String(severity).toLowerCase() !== 'low') return true;
        if (pcat === 'safe' || pcat === 'safe_content') return false;
        return pcat !== '';
  }) || (topPolicy && topPolicyAction && topPolicySim > 0.45));
      // treat as violation if category is not positive or DB evidence is relevant
      const asViolation = !positiveCategories.includes(cat) || dbEvidenceRelevant;
      if (!asViolation) return; // skip positive
      const key = cat || 'unknown';
      if (!violationTypes[key]) violationTypes[key] = { count: 0, frames: [], actions: new Set(), reasons: new Set() };
      violationTypes[key].count++;
      violationTypes[key].frames.push(doc);
      if (doc.action_required) violationTypes[key].actions.add(doc.action_required);
      if (doc.policy && doc.policy.description) violationTypes[key].reasons.add(doc.policy.description);
      if (doc.reasoning) violationTypes[key].reasons.add(doc.reasoning);
    });
  }
  const totalFrames = retrieved_docs ? retrieved_docs.length : 0;
  // Count safe frames and violation frames by their actual detected categories
  let safeFrames = 0;
  let violationFrames = 0;
  if (retrieved_docs && retrieved_docs.length > 0) {
    retrieved_docs.forEach(doc => {
      const cat = String(doc.category || '').toLowerCase();
      const hasDbEvidence = Array.isArray(doc.retrieved_policies) && doc.retrieved_policies.length > 0;
      const dbEvidenceRelevant = hasDbEvidence && doc.retrieved_policies.some(p => {
        const pcat = String(p.policy_info?.category || p.category || '').toLowerCase().replace(/\s+/g, '_');
        const action = p.policy_info?.action_required || p.action_required || p.action;
        const severity = p.policy_info?.severity || p.severity || p.importance;
        if (action && String(action).toLowerCase() !== 'none') return true;
        if (severity && String(severity).toLowerCase() !== 'low') return true;
        if (pcat === 'safe' || pcat === 'safe_content') return false;
        return pcat !== '';
      });
      if (dbEvidenceRelevant) {
        violationFrames++;
      } else if (cat === 'safe content' || positiveCategories.includes(cat)) {
        safeFrames++;
      } else {
        violationFrames++;
      }
    });
  }
  // Compute complianceRate robustly:
  // Accept fields in either 0..1 range (fraction) or 0..100 percent. If not provided, derive from frame counts.
  let complianceRate = null;
  const rawCompliance = (typeof ragAnalysis.compliance_rate !== 'undefined') ? ragAnalysis.compliance_rate : (typeof ragAnalysis.complianceRate !== 'undefined' ? ragAnalysis.complianceRate : null);
  if (rawCompliance !== null && rawCompliance !== undefined && !Number.isNaN(Number(rawCompliance))) {
    const num = Number(rawCompliance);
    if (num >= 0 && num <= 1) {
      complianceRate = num * 100;
    } else {
      // assume already a percentage (0..100)
      complianceRate = num;
    }
  } else if (totalFrames > 0) {
    // fallback: percentage of safe frames
    complianceRate = (safeFrames / totalFrames) * 100;
  } else {
    complianceRate = 0;
  }

  // Pacing metrics are available in ragAnalysis (pacing_rate / pacingRate / pacingScore)
  // but are not used in the current UI.

  // Build grouped summary lines and collect all unique explanations for compliant videos
  let guidelineSummary = '';
  let guidelineReasons = [];
  let allExplanations = [];


  // Collect reasons and explanations from frames and RAG data (same as before)
  if (retrieved_docs && retrieved_docs.length > 0) {
    // If there are zero violation frames, give a short positive reason; otherwise gather violation reasons
    if (violationFrames === 0) {
      guidelineReasons = ['No violations detected. Content is safe and suitable for YouTube.'];
      const explanationsSet = new Set();
      retrieved_docs.forEach(doc => {
        if (doc.policy && doc.policy.description) explanationsSet.add(doc.policy.description);
        if (doc.reasoning) explanationsSet.add(doc.reasoning);
      });
      allExplanations = Array.from(explanationsSet);
    } else {
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
            let line = `${exp.category}:\n${exp.policy.description || ''}${examples}${actionRequired}`;
            guidelineReasons.push(line);
          }
        });
      } else {
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
  }

  // Prefer authoritative DB-backed explanations when available.
  // Collect top-level retrieved_policies and per-frame retrieved_policies into a single list
  const collectDbPolicies = () => {
    const top = ragAnalysis.retrieved_policies || ragAnalysis.relevant_policies || [];
    const perFrame = Array.isArray(retrieved_docs) ? retrieved_docs.flatMap(d => (Array.isArray(d.retrieved_policies) ? d.retrieved_policies.map(rp => ({...rp, _frame: d.frame || d.frame_index, _timestamp: d.timestamp})) : [])) : [];
    // unify by policy description or category
    const all = [...top, ...perFrame];
    if (!all || all.length === 0) return null;
    const map = new Map();
    all.forEach(p => {
      const info = p.policy_info || p;
      const cat = (info && (info.category || info.title)) || p.category || 'unknown';
      const key = String(cat).toLowerCase();
      if (!map.has(key)) map.set(key, { category: cat, count: 0, description: info.description || info.policy_text || '', examples: [], severity: [], context: [], action: info.action_required || info.action || '', visual_examples: new Set(), frames: new Set() });
      const entry = map.get(key);
      // increment count if this was attached to a frame, otherwise treat as 1 occurrence
      if (p._frame !== undefined && p._frame !== null) {
        entry.frames.add(p._frame + (p._timestamp ? `@${p._timestamp}s` : ''));
        entry.count += 1;
      } else {
        // top-level retrieved policy likely indicates multiple matches; leave count as-is but ensure at least 1
        entry.count = Math.max(entry.count, 1);
      }
      // gather fields
      if (info.examples) {
        try { entry.examples = entry.examples.concat(Array.isArray(info.examples) ? info.examples : JSON.parse(info.examples)); } catch (e) { entry.examples = entry.examples.concat(String(info.examples).split(/[;\n\,]/).map(s=>s.trim()).filter(Boolean)); }
      }
      if (info.severity_indicators) entry.severity = entry.severity.concat(info.severity_indicators);
      if (info.context_matters) entry.context = entry.context.concat(info.context_matters);
      if (p._frame !== undefined && p._frame !== null) {
        // try to extract a short visual description from the doc that referenced this policy
    const doc = retrieved_docs.find(d => ((d.frame === p._frame || d.frame_index === p._frame) && Array.isArray(d.retrieved_policies) && d.retrieved_policies.some(rp => ((rp.policy_info && rp.policy_info.description) || rp.description) === (info.description || ''))));
        if (doc) {
          const visual = doc.blip || doc.blip_description || doc.visual || (doc.policy && doc.policy.visual_example) || (doc.preview_url || doc.preview_path) || '';
          if (visual) entry.visual_examples.add(visual);
        }
      }
      map.set(key, entry);
    });
    // convert map to array with counts
    const out = Array.from(map.values()).map(e => ({
      category: e.category,
      count: e.count,
      description: e.description,
      examples: Array.from(new Set(e.examples)).slice(0,5),
      severity_indicators: Array.from(new Set(e.severity)).slice(0,5),
      context_matters: Array.from(new Set(e.context)).slice(0,5),
      action_required: e.action,
      visual_examples: Array.from(e.visual_examples).slice(0,3),
      frames: Array.from(e.frames).slice(0,10)
    }));
    return out;
  };

  const dbPolicies = collectDbPolicies();
  // When DB policy entries are present, build structured YouTube-style blocks.
  // We produce `youtubePolicyBlocks` (structured objects) and keep `guidelineReasons`
  // as a quick-copy textual fallback (first policy + others condensed).
  let youtubePolicyBlocks = null;
  if (dbPolicies && dbPolicies.length > 0) {
    youtubePolicyBlocks = dbPolicies.map(p => {
      const description = (p.description || '').trim();
      const examples = Array.isArray(p.examples) ? p.examples : (p.examples ? [String(p.examples)] : []);
      const severity = Array.isArray(p.severity_indicators) ? p.severity_indicators : (p.severity_indicators ? [String(p.severity_indicators)] : []);
      const context = Array.isArray(p.context_matters) ? p.context_matters : (p.context_matters ? [String(p.context_matters)] : []);
      const action = p.action_required || p.action_required === '' ? String(p.action_required) : (p.action_required || p.action || '');
      const visual = Array.isArray(p.visual_examples) ? p.visual_examples : (p.visual_examples ? [String(p.visual_examples)] : []);
      return {
        category: p.category || 'policy',
        count: p.count || 1,
        description,
        severity_indicators: severity.slice(0,5),
        context_matters: context.slice(0,5),
        examples: examples.slice(0,5),
        action_required: action,
        visual_examples: visual.slice(0,3),
        frames: p.frames || []
      };
    });

    // Build guidelineReasons as plain text fallback (first policy primary line + others brief)
    guidelineReasons = youtubePolicyBlocks.map((p, i) => {
      const base = `${p.category}: Detected in ${p.count} frame${p.count>1?'s':''}. ${p.description}`.trim();
      const extras = [];
      if (p.severity_indicators && p.severity_indicators.length) extras.push(`Severity indicators: ${p.severity_indicators.join(', ')}`);
      if (p.context_matters && p.context_matters.length) extras.push(`Context considerations: ${p.context_matters.join(', ')}`);
      if (p.examples && p.examples.length) extras.push(`YouTube examples: ${p.examples.join('; ')}`);
      if (p.action_required) extras.push(`Recommended action: ${String(p.action_required).replace(/_/g,' ')}`);
      if (p.visual_examples && p.visual_examples.length) extras.push(`Visual content: ${p.visual_examples[0]}`);
      return extras.length ? `${base} | ${extras.join(' | ')}` : base;
    });

    // Decide top-level summary based on presence of non-safe categories
    const anyViolation = youtubePolicyBlocks.some(p => !(String(p.category||'').toLowerCase().startsWith('safe')));
    guidelineSummary = anyViolation ? 'This video does NOT follow YouTube guidelines.' : 'This video follows YouTube guidelines.';
  }

  

  // If DB policies are empty, prefer the local fallback policies (if found)
  if ((!dbPolicies || dbPolicies.length === 0) && Array.isArray(localFallbackPolicies) && localFallbackPolicies.length > 0) {
    youtubePolicyBlocks = localFallbackPolicies.map(p => ({
      category: p.category || 'policy',
      count: p.count || 0,
      description: p.description || p.policy_text || '',
      severity_indicators: p.severity_indicators || p.severity || [],
      context_matters: p.context_matters || p.context || [],
      examples: p.examples || [],
      action_required: p.action_required || p.action || '',
      visual_examples: p.visual_examples || p.visuals || [],
      frames: p.frames || []
    }));
    // rebuild guidelineReasons from the fallback so UI remains consistent
    guidelineReasons = youtubePolicyBlocks.map((p, i) => {
      const base = `${p.category}: Detected in ${p.count} frame${p.count>1?'s':''}. ${p.description}`.trim();
      const extras = [];
      if (p.severity_indicators && p.severity_indicators.length) extras.push(`Severity indicators: ${p.severity_indicators.join(', ')}`);
      if (p.context_matters && p.context_matters.length) extras.push(`Context considerations: ${p.context_matters.join(', ')}`);
      if (p.examples && p.examples.length) extras.push(`YouTube examples: ${p.examples.join('; ')}`);
      if (p.action_required) extras.push(`Recommended action: ${String(p.action_required).replace(/_/g,' ')}`);
      if (p.visual_examples && p.visual_examples.length) extras.push(`Visual content: ${p.visual_examples.join('; ')}`);
      return extras.length ? `${base} | ${extras.join(' | ')}` : base;
    });
  }

  // If we built youtubePolicyBlocks override the backend supplied structured summaries for rendering
  if (typeof youtubePolicyBlocks !== 'undefined' && youtubePolicyBlocks && youtubePolicyBlocks.length > 0) {
    // expose to rendering section via a local var; we'll prefer these blocks when rendering the guideline card
  }

  // personalized per-frame snippets have been removed to reduce noisy UI.

  // Determine top-level guideline summary: prioritize detected violations.
  // If any violation frames are present, mark as not following guidelines.
  // If the top-level retrieved policies indicate a clear safe policy with high similarity, prefer that.
  // If the server provided a clear verdict explanation, use that as the canonical summary.
  if (topLevelExplanation && topLevelExplanation.verdict_summary) {
    guidelineSummary = topLevelExplanation.verdict_summary;
    // If the explanation contains a primary policy, surface a concise reason in guidelineReasons
    if (topLevelExplanation.primary_policy) {
      const pp = topLevelExplanation.primary_policy;
      const sim = pp.similarity ? ` (sim:${Number(pp.similarity).toFixed(3)})` : '';
      const action = topLevelExplanation.recommended_action?.action || '';
      guidelineReasons = [`Primary policy: ${pp.category}${sim} — ${pp.policy_text_snippet || ''}`];
      if (action) guidelineReasons.push(`Recommended action: ${action}`);
    }
  } else {
    const topLevelPolicy = pickBestPolicy(ragAnalysis.retrieved_policies || ragAnalysis.relevant_policies || ragAnalysis.retrieved_policies || []);
    const topLevelIsSafe = topLevelPolicy && (String(topLevelPolicy.policy_info?.category || topLevelPolicy.category || '').toLowerCase().includes('safe')) && Number(topLevelPolicy.similarity || 0) >= 0.60;
    const topLevelRequiresAction = topLevelPolicy && policyIndicatesAction(topLevelPolicy) && Number(topLevelPolicy.similarity || 0) >= 0.45;

    if (topLevelIsSafe && violationFrames === 0) {
      guidelineSummary = 'This video follows YouTube guidelines (DB evidence).';
    } else if (topLevelRequiresAction || violationFrames > 0) {
      guidelineSummary = 'This video does NOT follow YouTube guidelines.';
    } else if (complianceRate >= 90) {
      guidelineSummary = 'This video follows YouTube guidelines.';
    } else if (complianceRate >= 70) {
      guidelineSummary = 'This video follows YouTube guidelines, but with warnings.';
    } else if (complianceRate >= 50) {
      guidelineSummary = 'This video barely follows YouTube guidelines.';
    } else {
      guidelineSummary = 'This video does NOT follow YouTube guidelines.';
    }
  }

  // Compute compliance rate and round for display
  const compliancePct = Math.round(complianceRate);

  // Prefer backend-provided YouTube-style explanation when available
  const youtubeText = ragAnalysis.youtube_explanation || ragAnalysis.youtube_explanation_text || null;

  // No client-side synthesis: prefer DB-backed `youtubePolicyBlocks` or backend `youtube_explanation`.
  // If neither exists, show an explicit absence message to avoid any synthesized text.
  const youtubeSynthText = null;

  // define compliance chip color (used by banner) — keep near compliancePct to avoid TDZ
  const complianceChipColor = compliancePct >= 90 ? 'success' : (compliancePct < 50 ? 'error' : 'default');

  // Consider a video 'good' only when there are no violation frames and compliance is high
  const isGoodVideo = (violationFrames === 0) && (complianceRate >= 90);

  // Banner logic (prioritize detected violations so the final banner never claims 'safe' when violations exist)
  let bannerElement = null;
  // Simplified banner with lighter visuals to reduce UI noise
  // IMPORTANT: if any violation frames were detected, show NOT following message regardless of complianceRate.
  let bannerTitle;
  if (violationFrames > 0) {
    bannerTitle = '⛔ This video does NOT follow YouTube guidelines';
  } else {
    bannerTitle = complianceRate < 50 ? '⛔ This video does NOT follow YouTube guidelines' :
      (complianceRate < 70 ? '⚠️ This video passes YouTube guidelines but only barely' :
        (complianceRate < 90 ? '⚠️ This video follows YouTube guidelines, but with warnings' : '✅ This video follows YouTube guidelines'));
  }
  bannerElement = (
    <Card sx={{ mb: 3, borderRadius: 2, boxShadow: 1 }}>
      <CardContent>
        <Typography variant="h6" sx={{ fontWeight: 'bold', mb: 1 }}>{bannerTitle}</Typography>
        <Typography variant="body2" sx={{ mb: 1 }}>{guidelineSummary}</Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          <Chip label={`Compliance Rate: ${compliancePct}%`} color={complianceChipColor} sx={{ fontWeight: 'bold' }} />
          <Chip label={`Safe Frames: ${safeFrames}`} color="success" sx={{ fontWeight: 'bold' }} />
          <Chip label={`Violations: ${violationFrames}`} color="error" sx={{ fontWeight: 'bold' }} />
          <Chip label={`Total Frames: ${totalFrames}`} color="primary" sx={{ fontWeight: 'bold' }} />
        </Box>
      </CardContent>
    </Card>
  );

  // Determine summary card colors from compliance rate bands (keeps header and summary visuals consistent)
  let summaryTextColor = '#43a047';
  if (compliancePct < 50) {
    summaryTextColor = '#e53935';
  } else if (compliancePct >= 50 && compliancePct < 70) {
    summaryTextColor = '#7a4300';
  } else if (compliancePct >= 70 && compliancePct < 90) {
    summaryTextColor = '#92400e';
  }

  // Accent band color for the guideline summary card (derived from summary text color)
  const bannerBandColor = summaryTextColor;

  // define compliance chip color (moved earlier to avoid TDZ when used in banner)

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ p: 4, background: theme.palette.background.default, minHeight: '100vh' }}>
      <Box sx={{ display: 'flex', justifyContent: 'center' }}>
        <Box sx={{ width: '100%', maxWidth: 1100 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Button variant="outlined" color="secondary" sx={{ mr: 2, fontWeight: 'bold', borderRadius: 2 }} onClick={onBack}>
              ← Back
            </Button>
            {/* DB evidence toggle removed — evidence always visible */}
            <Box>
              <Typography variant="h5" color="text.primary" sx={{ fontWeight: 700 }}>
                RAG Policy Violation Explanations
              </Typography>
              <Typography variant="caption" color="text.secondary">Clear, actionable explanations with database evidence and per-frame detail</Typography>
            </Box>
          </Box>
  {/* YouTube-style guideline summary */}
      {/* Guideline summary: neutral paper with left accent + wrapping actions to avoid overflow */}
      <Card sx={{ maxWidth: 900, mb: 3, boxShadow: 1, borderRadius: 3, p: 2, background: theme.palette.background.paper, borderLeft: `6px solid ${bannerBandColor}` }}>
        <CardContent>
          {/* If DB-backed youtubePolicyBlocks exist, render them in YouTube-style format */}
          {youtubePolicyBlocks && youtubePolicyBlocks.length > 0 ? (
            // If no violation frames detected, show a short positive-only explanation;
            // otherwise render the detailed DB-backed policy blocks.
            violationFrames === 0 ? (
              <Box>
                <Typography variant="h6" sx={{ fontWeight: '700', color: 'text.primary', mb: 1 }}>{bannerTitle}</Typography>
                <Box sx={{ borderRadius: 1, p: 1, background: '#f6fffa', border: '1px solid rgba(67,160,71,0.08)' }}>
                  <Typography variant="body1" sx={{ fontWeight: 700, color: '#1976d2', mb: 0.5 }}>This video follows YouTube guidelines.</Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>It is family-friendly, informative, and entertaining. No DB-flagged disallowed material was detected.</Typography>
                  <Box sx={{ mt: 1 }}>
                    <Chip label={`Compliance Rate: ${compliancePct}%`} color={complianceChipColor} size="small" />
                    <Chip label={`Safe Frames: ${safeFrames}`} color="success" size="small" sx={{ ml: 1 }} />
                    <Chip label={`Total Frames: ${totalFrames}`} color="primary" size="small" sx={{ ml: 1 }} />
                  </Box>
                </Box>
              </Box>
            ) : (
              <Box>
                <Typography variant="h6" sx={{ fontWeight: '700', color: 'text.primary', mb: 1 }}>{bannerTitle}</Typography>
                {/* Primary policy highlighted area (red background) */}
                {youtubePolicyBlocks.map((p, idx) => {
                  const isPrimary = idx === 0;
                  const bg = isPrimary ? '#fff5f5' : '#ffffff';
                  const border = isPrimary ? '1px solid rgba(229,57,53,0.08)' : '1px solid rgba(16,24,40,0.04)';
                  const textColor = isPrimary ? summaryTextColor : 'text.secondary';
                  const extras = [];
                  if (p.severity_indicators && p.severity_indicators.length) extras.push(`Severity indicators: ${p.severity_indicators.join(', ')}`);
                  if (p.context_matters && p.context_matters.length) extras.push(`Context considerations: ${p.context_matters.join(', ')}`);
                  if (p.examples && p.examples.length) extras.push(`YouTube examples: ${p.examples.join('; ')}`);
                  if (p.action_required) extras.push(`Recommended action: ${String(p.action_required).replace(/_/g,' ')}`);
                  if (p.visual_examples && p.visual_examples.length) extras.push(`Visual content: ${p.visual_examples.join('; ')}`);
                  const baseLine = `${p.category}: Detected in ${p.count} frame${p.count>1?'s':''}. ${p.description}`.trim();
                  const fullLine = extras.length ? `${baseLine} | ${extras.join(' | ')}` : baseLine;
                  return (
                    <Box key={idx} sx={{ borderRadius: 1, p: 1, background: bg, border, mb: 1 }}>
                      <Typography variant="body1" sx={{ fontWeight: isPrimary ? 700 : 600, color: textColor, mb: 0.5 }}>{fullLine}</Typography>
                      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mt: 0.5, flexWrap: 'wrap' }}>
                        {p.frames && p.frames.length > 0 && <Chip label={`Frames: ${p.frames.slice(0,5).join(', ')}`} size="small" />}
                        {p.examples && p.examples.length > 0 && <Chip label={`DB examples: ${p.examples.slice(0,2).join('; ')}`} size="small" />}
                      </Box>
                    </Box>
                  );
                })}
              </Box>
            )
          ) : youtubeText ? (
            // Backend provided DB-authoritative YouTube-style text
            <>
              <Typography variant="h6" sx={{ fontWeight: '700', color: 'text.primary', mb: 1 }}>{bannerTitle}</Typography>
              <Box sx={{ borderRadius: 1, p: 1, background: '#fff5f5', border: '1px solid rgba(229,57,53,0.08)' }}>
                {String(youtubeText).split('\n').map((line, i) => (
                  <Typography key={i} variant={i===0? 'body1' : 'body2'} sx={{ color: i===0 ? summaryTextColor : 'text.secondary', mb: i===0? 1 : 0.5 }}>{line}</Typography>
                ))}
              </Box>
            </>
          ) : (
            // Explicit: no DB-backed YouTube-style explanation available. Do not synthesize.
            <>
              <Typography variant="h6" sx={{ fontWeight: '700', color: 'text.primary', mb: 1 }}>{bannerTitle}</Typography>
              <Box sx={{ borderRadius: 1, p: 2, background: '#fff', border: '1px dashed rgba(0,0,0,0.08)' }}>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>No DB-backed YouTube-style explanation available for this video.</Typography>
                <Typography variant="caption" sx={{ display: 'block', mt: 1, color: 'text.secondary' }}>The UI will only display authoritative text sourced directly from the policy database.</Typography>
              </Box>
            </>
          )}
          {/* personalized per-frame snippets removed to reduce noise */}
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
          <Box sx={{ display: 'flex', gap: 2, mt: 2, flexWrap: 'wrap', alignItems: 'center' }}>
            <Chip label={`Compliance Rate: ${compliancePct}%`} color={complianceChipColor} sx={{ fontWeight: 'bold' }} />
            <Chip label={`Safe Frames: ${safeFrames}`} color="success" sx={{ fontWeight: 'bold' }} />
            <Chip label={`Violations: ${violationFrames}`} color="error" sx={{ fontWeight: 'bold' }} />
            <Chip label={`Total Frames: ${totalFrames}`} color="primary" sx={{ fontWeight: 'bold' }} />
            { !isGoodVideo ? (
              <Box sx={{ display: 'flex', gap: 1, ml: 2, flexWrap: 'wrap' }}>
                <Button size="small" variant="contained" color="primary" onClick={async () => {
                  if (generating) return;
                  setGenerating(true);
                  try {
                    // Try no-id heuristic endpoint first
                    let endpoint = `${API_BASE}/api/recommendations/generate`;
                    let resp = await fetch(withDebug(endpoint), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragAnalysis) });
                    if (resp.status === 404) {
                      // Fallback: save report to create a video_id, then call video-scoped endpoint
                      // Try both /api/videos and /videos; also treat 405 as a signal to try the alternate
                      // Use a stable save endpoint to reduce routing errors
                      let saveResp = await fetch(`${API_BASE}/api/videos/save-or-id`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragAnalysis) });
                      if (!saveResp.ok && (saveResp.status === 404 || saveResp.status === 405)) {
                        // try alternative path without /api
                        saveResp = await fetch(`${API_BASE}/videos/save-or-id`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragAnalysis) });
                      }
                        if (!saveResp.ok) {
                          // If saving fails (405/404/etc) try a best-effort no-id LLM generate so the user still sees recommendations.
                          const status = saveResp.status;
                          const txt = await saveResp.text().catch(() => '');
                          console.warn('Save before generate failed:', status, txt);
                          // If server refuses the POST, fallback to calling no-id LLM endpoint to keep UX working
                          try {
                            const fallbackResp = await fetch(withDebug(`${API_BASE}/api/recommendations/llm`), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragAnalysis) });
                              if (fallbackResp.ok) {
                              const fbdata = await fallbackResp.json();
                              if (fbdata && fbdata.recommendations) setServerRecs(normalizeRecs(fbdata.recommendations));
                              // stop the normal flow since we've shown results
                              setGenerating(false);
                              return;
                            }
                          } catch (fe) {
                            console.debug('Fallback no-id LLM call failed', fe);
                          }
                          throw new Error('Failed to save report before generate: ' + (txt || saveResp.statusText || saveResp.status));
                        }
                      let saveData = null;
                      try { saveData = await saveResp.json(); } catch(e) { saveData = null; }
                      const vid = saveData && saveData.video_id;
                      if (!vid) throw new Error('Save did not return video_id');
                      endpoint = `${API_BASE}/api/videos/${vid}/recommendations/generate`;
                      resp = await fetch(withDebug(endpoint), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragAnalysis) });
                    }
                    if (!resp.ok) throw new Error(await resp.text());
                    const data = await resp.json();
                    if (data && data.recommendations) setServerRecs(normalizeRecs(data.recommendations));
                    if (data && (data.status === 'llm_unparsable' || data.status === 'llm_validation_failed')) {
                      setLlmRaw(data.llm_raw || data.llm_raw_preview || data.message || JSON.stringify(data));
                    } else {
                      // hide raw output by default when parsing/validation succeeded
                      setLlmRaw(null);
                    }
                  } catch (e) {
                    // eslint-disable-next-line no-alert
                    alert('Failed to generate recommendations: ' + (e.message || e));
                  }
                  setGenerating(false);
                }}>{generating ? 'Generating...' : 'Generate Personalized Recommendations'}</Button>

                <Button size="small" variant="outlined" color="secondary" onClick={async () => {
                  if (generating) return;
                  setGenerating(true);
                  try {
                    // Try no-id LLM endpoint first
                    let endpoint = `${API_BASE}/api/recommendations/llm`;
                    let resp = await fetch(withDebug(endpoint), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragAnalysis) });
                    if (resp.status === 404) {
                      // Fallback: save report to create a video_id, then call video-scoped LLM endpoint
                      // Use stable save endpoint first
                      let saveResp = await fetch(`${API_BASE}/api/videos/save-or-id`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragAnalysis) });
                      if (!saveResp.ok && (saveResp.status === 404 || saveResp.status === 405)) {
                        saveResp = await fetch(`${API_BASE}/videos/save-or-id`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragAnalysis) });
                      }
                      if (!saveResp.ok) {
                        const status = saveResp.status;
                        const txt = await saveResp.text().catch(() => '');
                        console.warn('Save before LLM generate failed:', status, txt);
                        // fallback: try no-id LLM endpoint to keep UX responsive
                        try {
                          const fallbackResp = await fetch(withDebug(`${API_BASE}/api/recommendations/llm`), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragAnalysis) });
                              if (fallbackResp.ok) {
                            const fbdata = await fallbackResp.json();
                            if (fbdata && fbdata.recommendations) setServerRecs(normalizeRecs(fbdata.recommendations));
                            setGenerating(false);
                            return;
                          }
                        } catch (fe) {
                          console.debug('Fallback no-id LLM call failed', fe);
                        }
                        throw new Error('Failed to save report before LLM generate: ' + (txt || saveResp.statusText || saveResp.status));
                      }
                      let saveData = null;
                      try { saveData = await saveResp.json(); } catch(e) { saveData = null; }
                      const vid = saveData && saveData.video_id;
                      if (!vid) throw new Error('Save did not return video_id');
                      endpoint = `${API_BASE}/api/videos/${vid}/recommendations/llm`;
                      resp = await fetch(withDebug(endpoint), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragAnalysis) });
                    }
                    if (!resp.ok) throw new Error(await resp.text());
                    const data = await resp.json();
                    if (data) {
                      if (data.status === 'llm_unavailable') {
                        setServerRecs(normalizeRecs(data.recommendations || []));
                        console.info('LLM unavailable:', data.message || 'Local LLM unavailable', data.error || null);
                        // don't show raw output for unavailability; reserve raw display for unparsable content
                        setLlmRaw(null);
                      } else if (data.status === 'llm_unparsable' || data.status === 'llm_validation_failed') {
                        setServerRecs(normalizeRecs(data.recommendations || []));
                        setLlmRaw(data.llm_raw || data.llm_raw_preview || data.message || JSON.stringify(data));
                      } else if (data.recommendations) {
                        setServerRecs(normalizeRecs(data.recommendations));
                        setLlmRaw(null);
                      }
                    }
                  } catch (e) {
                    // eslint-disable-next-line no-alert
                    alert('LLM request failed: ' + (e.message || e));
                  }
                  setGenerating(false);
                }}>Generate via Local LLM</Button>
                <Button size="small" variant="outlined" color="info" onClick={async () => {
                  // Re-run recommendations using existing serverRecs flow
                  if (generating) return;
                  setGenerating(true);
                  try {
                    const resp = await fetch(withDebug(`${API_BASE}/api/recommendations/llm`), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragAnalysis) });
                    if (!resp.ok) throw new Error(await resp.text());
                    const data = await resp.json();
                    if (data && data.recommendations) setServerRecs(normalizeRecs(data.recommendations));
                    if (data && (data.status === 'llm_unparsable' || data.status === 'llm_validation_failed')) setLlmRaw(data.llm_raw || data.llm_raw_preview || null);
                    else setLlmRaw(null);
                  } catch (e) {
                    alert('Re-run failed: ' + (e.message || e));
                  }
                  setGenerating(false);
                }}>Re-run Recommendations</Button>
                <Button size="small" variant="text" color="inherit" onClick={() => setShowRaw(s => !s)}>{showRaw ? 'Hide raw' : 'Show raw'}</Button>
                <Button size="small" variant="outlined" onClick={() => {
                  try {
                    const txt = (guidelineReasons && guidelineReasons.length > 0) ? guidelineReasons.join('\n\n') : guidelineSummary;
                    navigator.clipboard && navigator.clipboard.writeText(txt);
                    // temporary UX feedback
                    // eslint-disable-next-line no-alert
                    alert('Summary copied to clipboard');
                  } catch (e) { console.debug('copy failed', e); }
                }}>Copy Summary</Button>
              </Box>
            ) : (
              <Box sx={{ ml: 2, display: 'flex', alignItems: 'center' }}>
                <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'text.secondary' }}>Video is compliant — no recommendations or fixes suggested.</Typography>
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>
  {/* Detailed policy breakdown (slide-style bullets) */}
  { renderPolicyBreakdown() }
  {/* RAG decision and DB evidence (if any) */}
  { renderRagDbSummary() }
  { renderPhi3Card() }
  { renderPoliticalViolations() }
      {/* LLM availability notice (shown when backend indicates LLM unreachable) */}
  {/* LLM banner removed */}
      {/* Server-generated recommendations display (compact cards) */}
  { !isGoodVideo && serverRecs && (
        <Box sx={{ maxWidth: 900, mb: 3 }}>
          <Typography variant="h6" sx={{ fontWeight: 'bold', mb: 1 }}>Personalized Fixes & Recommendations</Typography>
          <Box sx={{ display: 'grid', gap: 2 }}>
            {serverRecs.map((r, i) => {
              const ffcmds = extractFfmpegCommands(r.description || (r.text || ''));
              return (
                <Card key={i} sx={{ p: 1 }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <Box sx={{ flex: 1 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: '700' }}>{(r.category || '').toUpperCase()}</Typography>
                        <Typography variant="body2" sx={{ color: 'text.primary' }}>{(r.description || '').split('\n')[0].slice(0, 220)}{(r.description||'').length>220? '…' : ''}</Typography>
                        {r.examples && r.examples.length > 0 && <Typography variant="caption" display="block">Examples: {formatExamples(r.examples)}</Typography>}
                      </Box>
                      <Box>
                        <Button size="small" onClick={() => toggleRec(i)}>{recExpanded[i] ? 'Hide details' : 'Show details'}</Button>
                        <Tooltip title="Copy JSON"><IconButton size="small" onClick={() => { navigator.clipboard && navigator.clipboard.writeText(JSON.stringify(r, null, 2)); }}><ContentCopyIcon fontSize="small" /></IconButton></Tooltip>
                      </Box>
                    </Box>
                    <Collapse in={Boolean(recExpanded[i])} timeout="auto" unmountOnExit>
                      <Box sx={{ mt: 1 }}>
                        <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>{r.description}</Typography>
                        {ffcmds && ffcmds.length > 0 && (
                          <Box component="pre" sx={{ mt: 1, p: 1, bgcolor: '#f5f5f5', borderRadius: 1, fontFamily: 'Monaco, Menlo, monospace', overflow: 'auto' }}>
                            {ffcmds.join('\n')}
                          </Box>
                        )}
                        {r.quick_wins && r.quick_wins.length > 0 && (
                          <Box sx={{ mt: 1 }}>
                            <Typography variant="subtitle2">Quick wins:</Typography>
                            {r.quick_wins.map((q, qi) => <Typography key={qi} variant="body2">• {q}</Typography>)}
                          </Box>
                        )}
                      </Box>
                    </Collapse>
                  </CardContent>
                </Card>
              );
            })}
          </Box>
        </Box>
      )}
      {/* Show raw LLM output when backend returns unparsable output for debugging */}
  { llmRaw && showRaw && (
        <Card sx={{ maxWidth: 900, mb: 3, boxShadow: 1, borderRadius: 2 }}>
          <CardContent>
            <Typography variant="subtitle1" sx={{ fontWeight: 'bold', mb: 1 }}>LLM raw output (truncated)</Typography>
            <Box component="pre" sx={{ whiteSpace: 'pre-wrap', maxHeight: 240, overflow: 'auto', background: '#f5f5f5', p: 1, borderRadius: 1 }}>
              {String(llmRaw).length > 2000 ? String(llmRaw).slice(0, 2000) + '\n\n... (truncated) ...' : String(llmRaw)}
            </Box>
          </CardContent>
        </Card>
      )}
      {bannerElement}
      {/* Frame-by-frame cards */}
  <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 2 }}>
        {retrieved_docs && retrieved_docs.length > 0 ? (
          <>
            {retrieved_docs.map((doc, idx) => {
              const cat = String(doc.category || '').toLowerCase();
              const hasDbEvidence = Array.isArray(doc.retrieved_policies) && doc.retrieved_policies.length > 0;
              const dbEvidenceRelevant = hasDbEvidence && doc.retrieved_policies.some(p => {
                const pcat = String(p.policy_info?.category || p.category || '').toLowerCase().replace(/\s+/g, '_');
                const action = p.policy_info?.action_required || p.action_required || p.action;
                const severity = p.policy_info?.severity || p.severity || p.importance;
                if (action && String(action).toLowerCase() !== 'none') return true;
                if (severity && String(severity).toLowerCase() !== 'low') return true;
                if (pcat === 'safe' || pcat === 'safe_content') return false;
                return pcat !== '';
              });
              const hasActionOrSeverity = Boolean(doc.action_required) || (doc.severity_indicators && doc.severity_indicators.length > 0);
              const isViolation = dbEvidenceRelevant || hasActionOrSeverity || !positiveCategories.includes(cat);
              const isPositive = !isViolation;
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
                <Card key={idx} sx={{ background: isPositive ? '#f9fbe7' : '#ffebee', borderRadius: 3, boxShadow: 2, transition: 'box-shadow 180ms ease, transform 120ms ease', '&:hover': { boxShadow: 8, transform: 'translateY(-6px)' }, border: '1px solid rgba(16,24,40,0.04)' }}>
                  <CardContent sx={{ p: 1.5 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      {getFrameIcon(isPositive)}
                      <Typography variant="subtitle1" sx={{ fontWeight: 'bold', ml: 1, color: isPositive ? '#1976d2' : 'error.main', px: 1, borderRadius: 1 }}>
                        {isPositive ? (doc.title || doc.category || 'Frame') : 'Violation detected'}
                      </Typography>
                      <Box sx={{ ml: 'auto', display: 'flex', alignItems: 'center' }}>
                        <IconButton size="small" onClick={() => toggleFrame(idx)} aria-label={expandedFrames[idx] ? 'Collapse' : 'Expand'}>
                          {expandedFrames[idx] ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                        </IconButton>
                      </Box>
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
                        <Chip label={`Category: ${isPositive ? doc.category : 'Violation detected'}`} variant="outlined" size="small" sx={{ fontWeight: 'bold' }} />
                      )}
                      {doc.confidence !== undefined && (
                        <Chip label={`Confidence: ${Number(doc.confidence).toFixed(2)}`} variant="outlined" size="small" />
                      )}
                      {/* show small chips for top retrieved policies (if present) */}
                      {doc.retrieved_policies && doc.retrieved_policies.slice(0,2).map((p, pi) => (
                        <Chip key={pi} label={p.policy_info?.category || p.category || 'policy'} size="small" variant="outlined" />
                      ))}
                    </Box>
                    {/* Frame summary section */}
                    {frameSummary.length > 0 && (
                      <Box sx={{ mt: 1, mb: 1, background: '#ffffff', borderRadius: 2, p: 1, boxShadow: 1 }}>
                        <Typography variant="body2" sx={{ fontWeight: 'bold', color: '#1976d2', mb: 0.5 }}>Frame Details:</Typography>
                              {frameSummary.slice(0,3).map((line, i) => (
                                <Typography key={i} variant="body2" sx={{ color: '#1976d2', mb: 0.5 }}>{line}</Typography>
                              ))}
                              {frameSummary.length > 3 && (
                                <Typography variant="caption" sx={{ color: 'text.secondary' }}>... and {frameSummary.length - 3} more</Typography>
                              )}
                        {/* only offer per-frame fixes when video is not already compliant */}
                        { !isGoodVideo && (
                          <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
                            <Button size="small" variant="text" onClick={() => setOpenDoc(doc)}>View Details</Button>
                          </Box>
                        )}
                      </Box>
                    )}
                    {/* Expandable detailed slide-style analysis */}
                    <Collapse in={Boolean(expandedFrames[idx])} timeout="auto" unmountOnExit>
                      {renderDetailedAnalysis(doc, idx)}
                    </Collapse>
                  </CardContent>
                </Card>
              );
            })}
          </>
        ) : (
          <Typography>No RAG explanations found.</Typography>
        )}
      </Box>
      {/* Details dialog for a selected frame */}
      <Dialog open={Boolean(openDoc)} onClose={() => setOpenDoc(null)} maxWidth="md" fullWidth>
        <DialogTitle>Frame Details</DialogTitle>
        <DialogContent>
            {openDoc && (
              <Box>
                <Typography variant="subtitle1"><strong>Frame #{openDoc.frame} @ {openDoc.timestamp}s</strong></Typography>
                {/* Preview area: image thumbnail or short clip when available from backend (thumbnail, preview_url, frame_image, preview_clip) */}
                {(openDoc.preview_clip || openDoc.preview_url || openDoc.thumbnail || openDoc.frame_image) && (
                  <Box sx={{ mb: 2, display: 'flex', justifyContent: 'center' }}>
                    {openDoc.preview_clip ? (
                      <Box
                        component="video"
                        src={openDoc.preview_clip}
                        controls
                        sx={{ maxWidth: '100%', borderRadius: 2, boxShadow: 1, border: '1px solid rgba(16,24,40,0.06)', maxHeight: 480 }}
                      />
                    ) : (
                      <Box
                        component="img"
                        src={openDoc.preview_url || openDoc.thumbnail || openDoc.frame_image}
                        alt={`frame-${openDoc.frame}`}
                        sx={{ maxWidth: '100%', borderRadius: 2, boxShadow: 1, border: '1px solid rgba(16,24,40,0.06)', maxHeight: 480, objectFit: 'contain' }}
                      />
                    )}
                  </Box>
                )}
                {/* Inline preview for server-processed blurred video */}
                {previewUrl && (
                  <Box sx={{ mb: 2, display: 'flex', justifyContent: 'center' }}>
                    <Box component="video" src={previewUrl} controls sx={{ maxWidth: '100%', borderRadius: 2, boxShadow: 1, border: '1px solid rgba(16,24,40,0.06)' }} />
                  </Box>
                )}
                <Stack spacing={1} sx={{ mt: 1 }}>
                  <Typography variant="body2"><strong>Category:</strong> {openDoc.category}</Typography>
                <Typography variant="body2"><strong>Confidence:</strong> {typeof openDoc.confidence !== 'undefined' ? Number(openDoc.confidence).toFixed(3) : 'N/A'}</Typography>
                {openDoc.blip && <Typography variant="body2"><strong>BLIP:</strong> {openDoc.blip}</Typography>}
                {openDoc.ocr && <Typography variant="body2"><strong>OCR:</strong> {openDoc.ocr}</Typography>}
                {openDoc.transcript && <Typography variant="body2"><strong>Transcript:</strong> {openDoc.transcript}</Typography>}
                {openDoc.policy && openDoc.policy.description && <Typography variant="body2"><strong>Policy excerpt:</strong> {openDoc.policy.description}</Typography>}
                {openDoc.personalized_reason && <Typography variant="body2"><strong>Personalized reason:</strong> {openDoc.personalized_reason}</Typography>}
                {openDoc.reasoning && <Typography variant="body2"><strong>Reasoning:</strong> {openDoc.reasoning}</Typography>}
                {openDoc.examples && openDoc.examples.length > 0 && <Typography variant="body2"><strong>Examples:</strong> {formatExamples(openDoc.examples)}</Typography>}
                {openDoc.severity_indicators && openDoc.severity_indicators.length > 0 && <Typography variant="body2"><strong>Severity Indicators:</strong> {openDoc.severity_indicators.join(', ')}</Typography>}
                {openDoc.context_factors && openDoc.context_factors.length > 0 && <Typography variant="body2"><strong>Context Factors:</strong> {openDoc.context_factors.join(', ')}</Typography>}
                {openDoc.retrieved_policies && openDoc.retrieved_policies.length > 0 && (
                  <Box>
                    <Typography variant="subtitle2" sx={{ mt: 1 }}><strong>Retrieved policies (DB evidence)</strong></Typography>
                    <List dense>
                      {openDoc.retrieved_policies.map((p, i) => (
                        <ListItem key={i}>
                          <ListItemText primary={<strong>{p.policy_info?.category || p.category || 'policy'}</strong>} secondary={p.policy_info?.description || p.description || ''} />
                          <Chip label={p.similarity ? `sim:${Number(p.similarity).toFixed(3)}` : 'sim:N/A'} size="small" />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}
              </Stack>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          {/* Blur strength control */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mr: 'auto' }}>
            <TextField size="small" label="Blur strength" type="number" value={blurStrength} onChange={(e) => setBlurStrength(Number(e.target.value) || 0)} sx={{ width: 140 }} />
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>px</Typography>
          </Box>
          {previewing ? <CircularProgress size={20} /> : (
            <Button color="primary" onClick={async () => {
              if (!openDoc) return;
              setPreviewing(true);
              try {
                const result = await applyFrameFix(openDoc, 'blur', { strength: blurStrength });
                if (result) {
                  // If applyFrameFix returned a result path, open preview in dialog
                  setPreviewUrl(result);
                }
              } catch (e) {
                console.error('Preview failed', e);
              }
              setPreviewing(false);
            }}>Apply Blur & Preview</Button>
          )}
          <Button onClick={() => setOpenDoc(null)}>Close</Button>
        </DialogActions>
      </Dialog>
      {/* Dialog for showing full DB policy details when a retrieved policy is clicked */}
      <Dialog open={Boolean(openPolicy)} onClose={() => setOpenPolicy(null)} maxWidth="sm" fullWidth>
        <DialogTitle>{openPolicy?.policy_info?.title || openPolicy?.policy_info?.category || 'Policy detail'}</DialogTitle>
        <DialogContent dividers>
          <Typography variant="subtitle2" sx={{ mb: 1 }}>{openPolicy?.policy_info?.category || openPolicy?.category}</Typography>
          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>{openPolicy?.policy_info?.description || openPolicy?.description || openPolicy?.policy_text || 'No description available.'}</Typography>
          {openPolicy?.policy_info?.examples && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="subtitle2">Examples</Typography>
              {(() => {
                let ex = openPolicy.policy_info.examples;
                try { if (typeof ex === 'string') ex = JSON.parse(ex); } catch (e) { }
                if (Array.isArray(ex)) return ex.map((e, i) => <Typography key={i} variant="body2">• {e}</Typography>);
                return <Typography variant="body2">{String(ex)}</Typography>;
              })()}
            </Box>
          )}
          {(openPolicy?.policy_info?.severity || openPolicy?.severity) && <Typography variant="body2" sx={{ mt: 1 }}><strong>Severity:</strong> {openPolicy?.policy_info?.severity || openPolicy?.severity}</Typography>}
          {(openPolicy?.policy_info?.action_required || openPolicy?.action_required) && <Typography variant="body2" sx={{ mt: 1 }}><strong>Action required:</strong> {openPolicy?.policy_info?.action_required || openPolicy?.action_required}</Typography>}
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>Source: {openPolicy?.policy_info?.source || openPolicy?.source || 'database'}</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenPolicy(null)}>Close</Button>
        </DialogActions>
      </Dialog>
        </Box>
      </Box>
    </Box>
      <Snackbar open={toast.open} autoHideDuration={4000} onClose={() => setToast(t => ({ ...t, open: false }))}>
        <Alert onClose={() => setToast(t => ({ ...t, open: false }))} severity={toast.severity} sx={{ width: '100%' }}>
          {toast.message}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

export default RagAnalysisPage;






