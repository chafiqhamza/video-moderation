import React, { useState } from 'react';
import {
  Box,
  Typography,
  Divider,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Checkbox,
  FormControlLabel,
  Chip,
  Snackbar,
  Alert,
  Card,
  CardContent
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

const FrameDetailsPage = ({ frames = [], onBack = () => {} }) => {
  const [previewOpen, setPreviewOpen] = useState(false);
  const [previewSrc, setPreviewSrc] = useState('');
  const [toast, setToast] = useState({ open: false, severity: 'success', message: '' });
  const [blurLoading, setBlurLoading] = useState({});
  const [blurResults, setBlurResults] = useState({});
  const [exportLoading, setExportLoading] = useState({});
  const [exportResults, setExportResults] = useState({});
  const [selectedFrames, setSelectedFrames] = useState({});
  const [exportSelectedLoading, setExportSelectedLoading] = useState(false);
  const [exportSelectedResult, setExportSelectedResult] = useState(null);
  const [savedVideoId, setSavedVideoId] = useState(null);
  const [exportConfirmOpen, setExportConfirmOpen] = useState(false);
  const [exportPending, setExportPending] = useState(null);
  const [exportDownloadAfter, setExportDownloadAfter] = useState(false);

  const formatTs = (s) => {
    if (s === undefined || s === null || isNaN(Number(s))) return 'N/A';
    const total = Number(s);
    const ms = Math.floor((total % 1) * 1000);
    const sec = Math.floor(total % 60);
    const min = Math.floor((total / 60) % 60);
    const hr = Math.floor(total / 3600);
    return `${String(hr).padStart(2,'0')}:${String(min).padStart(2,'0')}:${String(sec).padStart(2,'0')}.${String(ms).padStart(3,'0')}`;
  };

  const handleBlur = async (frame, idx) => {
    try {
      const vid = frame.video_id || frame.videoId || frame.video || null;
      const preview = frame.preview_path || frame.preview || null;
      if (!preview) return setToast({ open: true, severity: 'warning', message: 'No preview path available to blur' });
      setBlurLoading(prev => ({ ...prev, [idx]: true }));
      const payload = { preview_path: preview, options: { strength: 12 } };
      // include video id if present
      if (vid) payload.video_id = vid;
      // Try to request an MP4 generation first so downloads produce a video instead of an image.
      let data = null;
      try {
        const mpResp = await fetch(`/api/videos/${vid || '0'}/apply-frame-blur-to-video`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        if (mpResp.ok) {
          data = await mpResp.json();
        } else {
          // if MP4 endpoint returns an error, capture text for debugging and fall back to image endpoint
          const t = await mpResp.text();
          console.warn('apply-frame-blur-to-video failed, falling back to image endpoint:', t || `HTTP ${mpResp.status}`);
        }
      } catch (err) {
        console.warn('Error calling MP4 endpoint, falling back to image endpoint:', err);
      }

      // If MP4 attempt didn't yield a successful payload, fall back to the legacy image-only blur endpoint
      if (!data) {
        const resp = await fetch(`/api/videos/${vid || '0'}/apply-frame-blur`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        if (!resp.ok) {
          const t = await resp.text();
          throw new Error(t || `HTTP ${resp.status}`);
        }
        data = await resp.json();
      }
      if (data && data.result_path) {
        setBlurResults(prev => ({ ...prev, [idx]: data.result_path }));
        setToast({ open: true, severity: 'success', message: 'Blur applied' });
      } else if (data && data.result_path === undefined && data.result_path === null && data.status === 'ok' && data.result_path) {
        setToast({ open: true, severity: 'success', message: 'Blur job queued' });
      } else if (data && data.result_path) {
        setBlurResults(prev => ({ ...prev, [idx]: data.result_path }));
        setToast({ open: true, severity: 'success', message: 'Blur applied' });
      } else {
        // Try common field names
        const rp = data && (data.result_path || data.result || data.path);
        if (rp) {
          setBlurResults(prev => ({ ...prev, [idx]: rp }));
          setToast({ open: true, severity: 'success', message: 'Blur applied' });
        } else {
          setToast({ open: true, severity: 'info', message: 'Blur request completed' });
        }
      }
    } catch (e) {
      setToast({ open: true, severity: 'error', message: `Blur failed: ${e.message || e}` });
    } finally {
      setBlurLoading(prev => ({ ...prev, [idx]: false }));
    }
  };

  // ...existing queued export handler now implemented later as a function declaration that accepts (videoId, frame, downloadAfter)

  const openExportConfirm = (frame, idx) => {
    setExportPending({ frame, idx });
  // reset download flag to false by default when opening
  setExportDownloadAfter(false);
  setExportConfirmOpen(true);
  };

  const closeExportConfirm = () => {
    setExportPending(null);
    setExportConfirmOpen(false);
  };

  const saveReport = async () => {
    try {
      setToast({ open: true, severity: 'info', message: 'Saving report...' });
      // Try to detect an original source file path from frames (video upload path, source_path, or similar)
      let source = null;
      for (const f of frames) {
        if (f.source_path) { source = f.source_path; break; }
        if (f.source) { source = f.source; break; }
        if (f.video_path) { source = f.video_path; break; }
        if (f.preview_path && f.preview_path.includes('/static/')) { /* no-op */ }
      }
      const payload = { filename: 'ui_saved_report', report: { frame_details: frames }, source_path: source };
      const resp = await fetch('/api/videos/save-or-id', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!resp.ok) {
        const t = await resp.text();
        throw new Error(t || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      if (data && (data.video_id || data.videoId)) {
        const vid = data.video_id || data.videoId;
        setSavedVideoId(vid);
        setToast({ open: true, severity: 'success', message: `Report saved (video_id=${vid})` });
        return vid;
      }
      setToast({ open: true, severity: 'info', message: 'Report saved' });
    } catch (e) {
      setToast({ open: true, severity: 'error', message: `Save failed: ${e.message || e}` });
    }
  };

  const openPreview = async (src) => {
    try {
      if (!src) return setPreviewOpen(true);
      const isVideo = /\.(mp4|webm|mov|mkv|avi|flv)(\?|$)/i.test(String(src));
      if (isVideo) {
        // request raw extracted frame from backend
        const encoded = encodePreviewPath(src);
        const url = `/api/frame/blur?preview=${encoded}&raw=1&_=${Date.now()}`;
        // attempt fetch to ensure 200 before opening modal
        try {
          const r = await fetch(url, { method: 'GET', cache: 'no-store' });
          if (r.ok) {
            setPreviewSrc(url);
            setPreviewOpen(true);
            return;
          }
        } catch (e) {
          console.warn('Failed to fetch extracted preview image', e);
        }
        // fallback to showing the original src if extraction fails
        setPreviewSrc(src);
        setPreviewOpen(true);
        return;
      }
      setPreviewSrc(src || ''); setPreviewOpen(true);
    } catch (e) {
      console.warn('openPreview error', e);
      setPreviewSrc(src || ''); setPreviewOpen(true);
    }
  };
  const closePreview = () => { setPreviewOpen(false); setPreviewSrc(''); };

  // Open the preview dialog but request the server-generated blurred PNG first.
  // This hits /api/frame/blur?preview=...&strength=... and appends a cache-busting timestamp.
  const openBlurredPreview = async (frame, idx, strength = 12) => {
    try {
      const source = (frame && (frame.preview_path || frame.preview)) || (typeof frame === 'string' ? frame : null);
      if (!source) return setToast({ open: true, severity: 'warning', message: 'No preview path available to generate blurred image' });
      const encoded = encodePreviewPath(source);
      const url = `/api/frame/blur?preview=${encoded}&strength=${encodeURIComponent(String(strength))}&_=${Date.now()}`;
      // Prefetch to ensure server returns OK before opening modal (helps surface errors)
      try {
        const r = await fetch(url, { method: 'GET', cache: 'no-store' });
        if (!r.ok) {
          const t = await r.text();
          console.warn('Server returned non-OK for /api/frame/blur:', r.status, t);
          // fallback: if we have an already-stored blur result, show that
          if (blurResults && blurResults[idx]) {
            openPreview(blurResults[idx]);
            return;
          }
          return setToast({ open: true, severity: 'error', message: `Failed to generate blurred preview (status ${r.status})` });
        }
      } catch (e) {
        console.warn('Error fetching /api/frame/blur:', e);
        if (blurResults && blurResults[idx]) { openPreview(blurResults[idx]); return; }
        return setToast({ open: true, severity: 'error', message: 'Failed to contact server to generate blurred preview' });
      }
      // If prefetch succeeded, open modal pointing at the endpoint URL (with cache-buster)
      setPreviewSrc(url);
      setPreviewOpen(true);
    } catch (e) {
      setToast({ open: true, severity: 'error', message: `Preview failed: ${e.message || e}` });
    }
  };

  const copyToClipboard = (text) => {
    if (!text) return setToast({ open: true, severity: 'warning', message: 'No URL to copy' });
    try { navigator.clipboard.writeText(text); setToast({ open: true, severity: 'success', message: 'Frame URL copied' }); }
    catch (e) { setToast({ open: true, severity: 'error', message: 'Copy failed' }); }
  };

  const downloadFile = async (url, suggestedName) => {
    return downloadFileInternal(url, suggestedName, false);
  };

  // Internal helper with guard to avoid retry loops when attempting image->MP4 conversion
  const downloadFileInternal = async (url, suggestedName, triedImageConversion = false) => {
    if (!url) return setToast({ open: true, severity: 'warning', message: 'No file URL' });
    try {
      setToast({ open: true, severity: 'info', message: 'Preparing download...' });
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const contentType = (resp.headers.get('content-type') || '').toLowerCase();
      const urlIsImageExt = /\.(png|jpe?g|bmp|gif)(\?|$)/i.test(String(url));
      const isImageResp = contentType.includes('image') || urlIsImageExt;
      // If server returned an image or the URL looks like an image, try to convert to MP4 first
      if (isImageResp) {
        if (triedImageConversion) {
          setToast({ open: true, severity: 'error', message: 'Image download blocked; unable to produce MP4.' });
          return;
        }
        try {
          // Prefer server-side MP4 generation: instruct backend to produce MP4 from the blur image
          // Use video id 0 as a safe default if caller doesn't have one
          const candidateUrl = url.startsWith('/') ? window.location.origin + url : url;
          const convPayload = { preview_path: candidateUrl, frame: { blur_result: candidateUrl }, timestamp: 0 };
          const convResp = await fetch(`/api/videos/${savedVideoId || '0'}/apply-frame-blur-to-video`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(convPayload) });
          if (convResp.ok) {
            const j = await convResp.json();
            // find a video URL in the response
            const candidate = findVideoInJson(j) || (j && (j.result_path || j.result || j.url || j.path));
            if (candidate) {
              // Download the produced MP4
              await downloadFileInternal(candidate.startsWith('/') ? window.location.origin + candidate : candidate, suggestedName, true);
              return;
            }
          }
        } catch (e) {
          console.warn('Image->MP4 conversion failed', e);
        }
        setToast({ open: true, severity: 'error', message: 'Server returned an image; could not produce MP4.' });
        return;
      }
      // If the endpoint returned JSON (likely a wrapper with a real file URL), try to follow it
      if (contentType.includes('application/json')) {
        const j = await resp.json();

        // deep search for any candidate URL or data URI in the JSON
        const urlRegex = /^(https?:\/\/|\/|blob:|data:)/i;
        const extRegex = /\.(mp4|webm|mov|mkv|avi|flv)(\?|$)/i;

        const findCandidate = (obj) => {
          if (!obj) return null;
          if (typeof obj === 'string') {
            const s = obj.trim();
            if (s.startsWith('data:video')) return s;
            if (urlRegex.test(s) || extRegex.test(s)) return s;
            return null;
          }
          if (Array.isArray(obj)) {
            for (const it of obj) {
              const f = findCandidate(it);
              if (f) return f;
            }
            return null;
          }
          if (typeof obj === 'object') {
            // check common fields first
            const common = ['result_path', 'result', 'url', 'path', 'download_url', 'file_url', 'file', 'output'];
            for (const k of common) {
              if (obj[k] && typeof obj[k] === 'string') {
                const s = obj[k].trim();
                if (s.startsWith('data:video') || urlRegex.test(s) || extRegex.test(s)) return s;
              }
            }
            for (const key of Object.keys(obj)) {
              const f = findCandidate(obj[key]);
              if (f) return f;
            }
          }
          return null;
        };

        const candidate = findCandidate(j);
        if (candidate) {
          // handle data URI directly
          if (candidate.startsWith('data:')) {
            // If it's a video data URI, allow direct download. If it's an image data URI,
            // attempt server-side conversion to MP4 via the embedded helper so users get a video.
            if (candidate.startsWith('data:video')) {
              const a = document.createElement('a');
              a.href = candidate;
              a.download = suggestedName || 'download';
              document.body.appendChild(a);
              a.click();
              a.remove();
              setToast({ open: true, severity: 'success', message: 'Download started' });
              return;
            }
            // image data URI -> request server to convert and return an MP4
            try {
              const convPayload = { blur_image_data: candidate, preview_path: undefined, timestamp: 0 };
              const convResp = await fetch(`/api/videos/${savedVideoId || '0'}/apply-frame-blur-embedded`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(convPayload) });
              if (convResp.ok) {
                const j = await convResp.json();
                const vc = findVideoInJson(j) || (j && (j.result_path || j.result || j.url || j.path));
                if (vc) {
                  const finalUrl = vc.startsWith('/') ? window.location.origin + vc : vc;
                  await downloadFileInternal(finalUrl, suggestedName, true);
                  return;
                }
              }
            } catch (e) {
              console.warn('Conversion of data URI image to MP4 failed', e);
            }
            setToast({ open: true, severity: 'error', message: 'Image data URI could not be converted to MP4.' });
            return;
          }

          // normalize relative paths to absolute
          let candidateUrl = candidate;
          if (candidateUrl.startsWith('/')) candidateUrl = window.location.origin + candidateUrl;

          // If candidate points to an image, try server conversion first
          const isImageCandidate = /\.(png|jpe?g|bmp|gif)(\?|$)/i.test(String(candidateUrl));
          if (isImageCandidate) {
            // attempt to produce MP4 via backend before downloading the image
            try {
              const convPayload = { preview_path: candidateUrl, frame: { blur_result: candidateUrl }, timestamp: 0 };
              const convResp = await fetch(`/api/videos/${savedVideoId || '0'}/apply-frame-blur-to-video`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(convPayload) });
              if (convResp.ok) {
                const j2 = await convResp.json();
                const vc = findVideoInJson(j2) || (j2 && (j2.result_path || j2.result));
                if (vc) {
                  const finalUrl = vc.startsWith('/') ? window.location.origin + vc : vc;
                  await downloadFileInternal(finalUrl, suggestedName, true);
                  return;
                }
              }
            } catch (e) {
              console.warn('Conversion of candidate image to MP4 failed', e);
            }
            setToast({ open: true, severity: 'error', message: 'Server returned an image; could not produce MP4.' });
            return;
          }

          // fetch the actual file
          const fileResp = await fetch(candidateUrl);
          if (!fileResp.ok) throw new Error(`HTTP ${fileResp.status} when fetching file`);
          const blob = await fileResp.blob();
          // determine filename
          let filename = suggestedName;
          const cd = fileResp.headers.get('content-disposition');
          if (!filename && cd) {
            const m = /filename\*=UTF-8''(.+)$/.exec(cd) || /filename="?([^";]+)"?/.exec(cd);
            if (m) filename = decodeURIComponent(m[1]);
          }
          if (!filename) filename = candidateUrl.split('/').pop();
          const blobUrl = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = blobUrl;
          a.download = filename || 'download';
          document.body.appendChild(a);
          a.click();
          a.remove();
          URL.revokeObjectURL(blobUrl);
          setToast({ open: true, severity: 'success', message: 'Download started' });
          return;
        }

        setToast({ open: true, severity: 'error', message: 'Server returned JSON instead of a file; no download URL found.' });
        return;
      }

      const blob = await resp.blob();
      // determine filename
      let filename = suggestedName;
      const cd = resp.headers.get('content-disposition');
      if (!filename && cd) {
        const m = /filename\*=UTF-8''(.+)$/.exec(cd) || /filename="?([^";]+)"?/.exec(cd);
        if (m) filename = decodeURIComponent(m[1]);
      }
      if (!filename) filename = url.split('/').pop();
      const blobUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = filename || 'download';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(blobUrl);
      setToast({ open: true, severity: 'success', message: 'Download started' });
    } catch (e) {
      setToast({ open: true, severity: 'error', message: `Download failed: ${e.message || e}` });
    }
  };

  const toggleSelect = (idx) => {
    setSelectedFrames(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  const exportSelected = async () => {
    // gather selected frames
    const selectedIdxs = Object.keys(selectedFrames).filter(k => selectedFrames[k]).map(k => Number(k));
    if (!selectedIdxs.length) return setToast({ open: true, severity: 'warning', message: 'No frames selected' });

    // ensure we have a saved video id
    let vid = savedVideoId;
    if (!vid) {
      const newVid = await saveReport();
      vid = newVid || savedVideoId;
    }
    if (!vid) return setToast({ open: true, severity: 'warning', message: 'Cannot export: no video id available' });

    // build payload: list of frame indexes and timestamps
    const framesPayload = selectedIdxs.map(i => {
      const f = frames[i];
      return {
        frame_index: f.frame_index ?? i,
        timestamp: Number(f.timestamp || f.ts || 0) || 0,
        preview_path: f.preview_path || f.preview || null,
        // pass any already-applied blur result URL for that frame
        blur_result: blurResults[i] || null,
        bbox: f.bbox || f.box || f.blur_boxes || null,
        options: { strength: f.blur_strength || 12 }
      };
    });

    const source = frames.find(f => f.source_path || f.source || f.video_path)?.source_path || null;
    const payload = { action: 'blur_frames', frames: framesPayload, source_path: source };

    try {
      setExportSelectedLoading(true);
      setToast({ open: true, severity: 'info', message: 'Queuing export job...' });
      const resp = await fetch(`/api/videos/${vid}/apply`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!resp.ok) {
        const txt = await resp.text(); throw new Error(txt || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      const applied_id = data.applied_id;
      if (!applied_id) {
        setToast({ open: true, severity: 'info', message: 'Job queued (no id returned).' });
        return;
      }

      // poll for result
      const poll = async (attempts = 60, delay = 2000) => {
        for (let i = 0; i < attempts; i++) {
          try {
            const r = await fetch(`/api/videos/${vid}/applied/${applied_id}`);
            if (r.ok) {
              const j = await r.json();
              const videoCandidate = findVideoInJson(j);
              if (videoCandidate) {
                setExportSelectedResult(videoCandidate);
                setToast({ open: true, severity: 'success', message: 'Blurred video ready' });
                return videoCandidate;
              }
              if (j && j.applied && (j.applied.result_path || j.applied.result)) {
                const rp = j.applied.result_path || j.applied.result;
                setExportSelectedResult(rp);
                setToast({ open: true, severity: 'success', message: 'Blurred video ready' });
                return rp;
              }
            }
          } catch (e) {
            // ignore
          }
          await new Promise(res => setTimeout(res, delay));
        }
        throw new Error('Timed out waiting for blurred video');
      };

      await poll();
    } catch (e) {
      setToast({ open: true, severity: 'error', message: `Export failed: ${e.message || e}` });
    } finally {
      setExportSelectedLoading(false);
    }
  };

  const blipText = (frame) => {
    return frame?.rag_decision?.explanation?.evidence?.visual_result?.blip ||
           frame?.blip_description?.description ||
           frame?.blip ||
           'N/A';
  };

  const ocrText = (frame) => {
    return frame?.rag_decision?.explanation?.evidence?.visual_result?.ocr ||
           frame?.ocr_text?.text ||
           frame?.ocr ||
           'N/A';
  };

  // Derive category and confidence robustly and decide if a frame is a violation.
  const getCategory = (frame) => {
    const raw = (frame?.visual_analysis?.category || frame?.category || frame?.rag_decision?.label || 'N/A');
    if (!raw) return 'N/A';
    const norm = String(raw).trim().toLowerCase();
    // Mapping to friendlier/safer labels
    const map = {
      'artistic_adult_content': 'artistic',
      'artistic-adult-content': 'artistic',
      'misinformation': 'potential_misinformation'
    };
    if (map[norm]) return map[norm];
    // replace underscores/dashes with spaces for display
    return norm.replace(/[_-]+/g, ' ');
  };

  const getConfidence = (frame) => {
    const c = frame?.visual_analysis?.confidence;
    if (c !== undefined && c !== null && !isNaN(Number(c))) return Number(c);
    if (frame?.confidence !== undefined && frame?.confidence !== null && !isNaN(Number(frame.confidence))) return Number(frame.confidence);
    return NaN;
  };

  const computeIsViolation = (frame) => {
    // Honor explicit flags if present
    if (frame?.__isViolation) return true;
    if (frame?.combined_violation) return true;

    // RAG decisions from backend (if present) should take precedence
    const rd = frame?.rag_decision;
    if (rd) {
      const rdStr = String(rd.decision || rd.label || rd.status || '').toLowerCase();
      if (rdStr.includes('violation') || rdStr.includes('flag') || rdStr.includes('remove')) return true;
    }

    const category = String(getCategory(frame)).toLowerCase();
    const conf = getConfidence(frame);

    // If category explicitly mentions 'safe', treat as safe
    if (category.includes('safe')) return false;

    // Keywords that generally indicate problematic content (excluding mapped categories handled below)
    const violationKeywords = ['sexual', 'adult', 'violence', 'violent', 'gore', 'graphic', 'terror', 'hate', 'weapon', 'profan', 'suggest'];
    for (const kw of violationKeywords) {
      if (category.includes(kw)) {
        if (isNaN(conf)) return true;
        return conf >= 0.3; // if model is somewhat confident, mark violation
      }
    }

    // Special-case mapped categories so they don't auto-flag noisily
    if (category === 'artistic') {
      // treat artistic content as safe unless explicitly flagged by backend
      return false;
    }
    if (category === 'potential_misinformation' || category.includes('misinform')) {
      // only flag misinformation when model is very confident or rag_decision flags it
      if (!isNaN(conf) && conf >= 0.9) return true;
      return false;
    }

    // Generic rule: anything not labeled safe and with decent confidence -> violation
    if (!category || category === 'n/a') return false;
    if (!category.includes('safe') && !isNaN(conf) && conf >= 0.5) return true;

    return false;
  };

  const encodePreviewPath = (rawPath) => {
    if (!rawPath) return rawPath;
    try {
      // encode only the filename portion to preserve leading slashes and directories
      const idx = rawPath.lastIndexOf('/');
      if (idx === -1) return encodeURI(rawPath);
      const dir = rawPath.slice(0, idx + 1);
      const filename = rawPath.slice(idx + 1);
      return dir + encodeURIComponent(filename);
    } catch (e) {
      return encodeURI(rawPath);
    }
  };

  // Search a JSON object for a candidate video URL (mp4/webm/mov/etc) or data:video; returns first match or null
  const findVideoInJson = (obj) => {
    if (!obj) return null;
    const urlRegex = /^(https?:\/\/|\/|blob:|data:)/i;
    const videoExt = /\.(mp4|webm|mov|mkv|avi|flv)(\?|$)/i;
    if (typeof obj === 'string') {
      const s = obj.trim();
      if (s.startsWith('data:video')) return s;
      if (urlRegex.test(s) && videoExt.test(s)) return s;
      return null;
    }
    if (Array.isArray(obj)) {
      for (const it of obj) {
        const f = findVideoInJson(it);
        if (f) return f;
      }
      return null;
    }
    if (typeof obj === 'object') {
      const common = ['video_path', 'result_path', 'result', 'url', 'path', 'download_url', 'file_url', 'output'];
      for (const k of common) {
        if (obj[k] && typeof obj[k] === 'string') {
          const s = obj[k].trim();
          if (s.startsWith('data:video')) return s;
          if (urlRegex.test(s) && videoExt.test(s)) return s;
        }
      }
      for (const key of Object.keys(obj)) {
        const f = findVideoInJson(obj[key]);
        if (f) return f;
      }
    }
    return null;
  };

  // Add helper to synchronously request an MP4 with the blur overlay and download it
  async function createBlurredVideoAndDownload(videoId, frame, options = {}) {
    try {
      const payload = {
        // provide either a source video path or a preview image path if available
        preview_path: frame.preview_path || frame.preview || undefined,
        source_path: frame.source_path || undefined,
        frame: {
          blur_result: frame.blur_result,
          preview_path: frame.preview_path || frame.preview || undefined,
          bbox: frame.bbox || undefined,
          options: frame.options || options || undefined
        },
        timestamp: (frame.timestamp !== undefined) ? frame.timestamp : (options.timestamp || 0)
      };

      const resp = await fetch(`/api/videos/${videoId}/apply-frame-blur-to-video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        // Preserve existing UX: log and throw to allow fallback
        const txt = await resp.text();
        console.error('apply-frame-blur-to-video failed', resp.status, txt);
        throw new Error(`Server returned ${resp.status}`);
      }

      const data = await resp.json();
      if (data && data.status === 'ok' && data.result_path) {
        // Use the existing downloader helper to download the produced MP4
        await downloadFile(data.result_path);
        return { status: 'ok', result_path: data.result_path };
      }
      console.error('Unexpected response from apply-frame-blur-to-video', data);
      throw new Error('Unexpected response from server');
    } catch (err) {
      console.error('createBlurredVideoAndDownload error', err);
      throw err;
    }
  }

  // Modify the existing export handler to call the synchronous MP4 endpoint when downloadAfter is true
  // We'll replace or wrap the existing handleExportBlurredVideo function if present. Insert a safe wrapper that
  // prefers the new immediate-download flow when requested and falls back to the queued apply endpoint.

  async function handleExportBlurredVideo(videoId, frame, downloadAfter = false) {
    // If the caller specifically requested an immediate download, try the synchronous endpoint
    if (downloadAfter) {
      try {
        // Attempt to create MP4 on-demand and download it
        await createBlurredVideoAndDownload(videoId, frame);
        return;
      } catch (err) {
        // If sync generation failed, fall back to queued apply (existing behavior)
        console.warn('Immediate MP4 generation failed, falling back to queued export:', err);
      }
    }

    // Build payload similar to previous behavior so background worker can process it
    const payload = {
      frame: {
        frame_index: frame.frame_index || frame.frame || 0,
        preview_path: frame.preview_path || frame.preview || undefined,
        blur_result: frame.blur_result || undefined,
        bbox: frame.bbox || undefined,
        options: frame.options || {}
      },
      options: {
        download_after: !!downloadAfter
      }
    };

    const q = await fetch(`/api/videos/${videoId}/apply`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const queued = await q.json();
    if (queued && queued.applied_id) {
      // Poll for completion
      const appliedId = queued.applied_id;
      const poll = async () => {
        for (let i = 0; i < 60; i++) {
          await new Promise(r => setTimeout(r, 1000));
          const r = await fetch(`/api/videos/${videoId}/applied/${appliedId}`);
          if (!r.ok) continue;
          const d = await r.json();
          if (d && d.applied && d.applied.result_path) {
            const rp = d.applied.result_path;
            // If the result is an image, attempt to generate an MP4 from it before downloading
            const isImage = /\.(png|jpe?g|bmp|gif)(\?|$)/i.test(String(rp));
            if (downloadAfter) {
              if (isImage && videoId) {
                try {
                  // request server to produce MP4 from the blur image
                  const payload = { preview_path: rp, frame: { blur_result: rp }, timestamp: frame.timestamp || 0 };
                  const resp = await fetch(`/api/videos/${videoId}/apply-frame-blur-to-video`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                  if (resp.ok) {
                    const j = await resp.json();
                    if (j && j.status === 'ok' && j.result_path) {
                      await downloadFile(j.result_path);
                      return d.applied;
                    }
                  }
                } catch (e) {
                  console.warn('Failed to generate MP4 from image result; falling back to direct download', e);
                }
              }
              // default: download whatever the server returned (expected MP4)
              await downloadFile(rp);
            }
            return d.applied;
          }
        }
        console.warn('Timed out waiting for applied action result');
        return null;
      };
      return await poll();
    }
    return null;
  }

  if (!frames || frames.length === 0) {
    return (
      <Box sx={{ p: 4 }}>
        <Button variant="outlined" onClick={onBack} sx={{ mb: 2 }}>← Back</Button>
        <Typography>No frames found.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 16, pt: 3 }}>
      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mb: 2 }}>
        <Button variant="outlined" onClick={onBack}>← Back</Button>
        <Button variant="contained" color="primary" onClick={saveReport}>Save Report</Button>
        <Button variant="contained" color="secondary" onClick={exportSelected} disabled={exportSelectedLoading} sx={{ ml: 1 }}>{exportSelectedLoading ? 'Exporting...' : 'Blur selected frames & Export Video'}</Button>
  {exportSelectedResult && <Button variant="outlined" sx={{ ml: 1 }} onClick={async () => {
    try {
      const vid = savedVideoId || null;
      // If exportSelectedResult looks like an image, request MP4 generation first
      const isImage = /\.(png|jpe?g|bmp|gif)(\?|$)/i.test(String(exportSelectedResult));
      if (isImage && vid) {
        // construct a minimal payload to create MP4 from the blurred image
        const payload = { preview_path: exportSelectedResult, frame: { blur_result: exportSelectedResult }, timestamp: 0 };
        try {
          const resp = await fetch(`/api/videos/${vid}/apply-frame-blur-to-video`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
          if (resp.ok) {
            const j = await resp.json();
            if (j && j.status === 'ok' && j.result_path) {
              await downloadFile(j.result_path);
              return;
            }
          }
        } catch (e) {
          console.warn('Failed to generate MP4 from image for download', e);
        }
      }
      // default: try to download the candidate directly (if it's an mp4 it will work)
      await downloadFile(exportSelectedResult);
    } catch (e) {
      setToast({ open: true, severity: 'error', message: `Download failed: ${e.message || e}` });
    }
  }}>Download Blurred Video</Button>}
        {savedVideoId && <Typography variant="body2" sx={{ ml: 2 }}>Saved video_id: {savedVideoId}</Typography>}
      </Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Frames: {frames.length}</Typography>
      </Box>

      {frames.map((frame, idx) => (
        <Card key={`${frame.frame_index ?? idx}-${idx}`} sx={{ display: 'flex', mb: 2, alignItems: 'flex-start', p: 1 }}>
          <Box sx={{ width: 160, p: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
            <Checkbox checked={!!selectedFrames[idx]} onChange={() => toggleSelect(idx)} />
                {frame.preview_path ? (
              <img
                src={encodePreviewPath(frame.preview_path)}
                alt={`Frame ${idx + 1}`}
                style={{ width: 160, height: 90, objectFit: 'cover', borderRadius: 6, cursor: 'pointer', boxShadow: '0 1px 3px rgba(0,0,0,0.15)' }}
                onClick={() => (blurResults && blurResults[idx]) ? openBlurredPreview({ preview_path: blurResults[idx] }, idx) : openPreview(encodePreviewPath(frame.preview_path))}
                onError={(e) => { e.currentTarget.onerror = null; e.currentTarget.src = '/static/img/placeholder_frame.png'; }}
              />
            ) : (
              <Box sx={{ width: 160, height: 90, background: '#f0f0f0', borderRadius: 1 }} />
            )}
          </Box>

          <CardContent sx={{ flex: 1 }}>
            {/* Header row: Frame number, timestamp, VIOLATION chip, category, confidence */}
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Box>
                <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                  Frame {frame.frame_index !== undefined ? frame.frame_index + 1 : idx + 1}
                  <Typography component="span" sx={{ ml: 1, color: 'text.secondary', fontSize: 12 }}> {formatTs(frame.timestamp)}</Typography>
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', mt: 0.5 }}>
                        {(() => {
                          const isV = computeIsViolation(frame);
                          return <Chip label={isV ? 'VIOLATION' : 'SAFE'} color={isV ? 'error' : 'success'} size="small" />;
                        })()}
                        <Typography variant="caption">{getCategory(frame)}</Typography>
                        <Typography variant="caption" sx={{ ml: 2 }}>Conf: {(() => { const c = getConfidence(frame); return !isNaN(c) ? c.toFixed(2) : 'N/A'; })()}</Typography>
                </Box>
              </Box>
            </Box>

            <Divider sx={{ my: 1 }} />

            {/* Details lines matching your screenshot */}
            <Typography variant="body2" sx={{ mb: 0.5 }}><strong>Category:</strong> {computeIsViolation(frame) ? 'BAD CONTENT' : getCategory(frame)}</Typography>
            <Typography variant="body2" sx={{ mb: 0.5 }}><strong>Confidence:</strong> {(() => { const c = getConfidence(frame); return !isNaN(c) ? c.toFixed(2) : 'N/A'; })()}</Typography>
            <Typography variant="body2" sx={{ mb: 0.5 }}><strong>BLIP Description:</strong> {blipText(frame)}</Typography>
            <Typography variant="body2" sx={{ mb: 0.5 }}><strong>OCR Text:</strong> {ocrText(frame)}</Typography>
            <Typography variant="body2" sx={{ mb: 0.5 }}><strong>Preview Path:</strong> {frame.preview_path || 'N/A'}</Typography>
            <Typography variant="body2" sx={{ mb: 0.5 }}><strong>Frame Index:</strong> {frame.frame_index !== undefined ? frame.frame_index : idx}</Typography>

            <Box sx={{ mt: 1, display: 'flex', gap: 1 }}>
              {frame.preview_path && <Button size="small" variant="outlined" onClick={() => (blurResults && blurResults[idx]) ? openBlurredPreview({ preview_path: blurResults[idx] }, idx) : openPreview(frame.preview_path)}>Preview</Button>}
              {frame.preview_path && <Button size="small" variant="contained" onClick={() => copyToClipboard(frame.preview_path)}>Copy URL</Button>}
              {frame.preview_path && <Button size="small" variant="outlined" color="warning" onClick={() => handleBlur(frame, idx)} disabled={!!blurLoading[idx]}>{blurLoading[idx] ? 'Blurring...' : 'Blur'}</Button>}
              {blurResults[idx] && <Button size="small" variant="contained" color="secondary" onClick={() => openBlurredPreview({ preview_path: blurResults[idx] }, idx)}>View Blurred</Button>}
              <Box>
                <Button size="small" variant="outlined" color="primary" onClick={() => openExportConfirm(frame, idx)} disabled={!!exportLoading[idx]}>
                  {exportLoading[idx] ? 'Exporting...' : 'Export Blurred Video'}
                </Button>
                {exportResults[idx] && (
                  <Button size="small" variant="contained" sx={{ ml: 1 }} onClick={async () => {
                    try {
                      const candidate = exportResults[idx];
                      const isImage = /\.(png|jpe?g|bmp|gif)(\?|$)/i.test(String(candidate));
                      const vid = savedVideoId || null;
                      if (isImage && vid) {
                        const payload = { preview_path: candidate, frame: { blur_result: candidate }, timestamp: frame.timestamp || 0 };
                        try {
                          const resp = await fetch(`/api/videos/${vid}/apply-frame-blur-to-video`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                          if (resp.ok) {
                            const j = await resp.json();
                            if (j && j.status === 'ok' && j.result_path) {
                              await downloadFile(j.result_path);
                              return;
                            }
                          }
                        } catch (e) {
                          console.warn('Failed to generate MP4 from image for download', e);
                        }
                      }
                      await downloadFile(candidate);
                    } catch (e) {
                      setToast({ open: true, severity: 'error', message: `Download failed: ${e.message || e}` });
                    }
                  }}>Download</Button>
                )}
              </Box>
            </Box>
          </CardContent>
        </Card>
      ))}

      {/* Preview dialog */}
      <Dialog open={previewOpen} onClose={closePreview} maxWidth="lg">
        <DialogTitle>
          Frame preview
          <IconButton aria-label="close" onClick={closePreview} sx={{ position: 'absolute', right: 8, top: 8 }}>
            <CloseIcon />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers>
          {previewSrc ? <img src={previewSrc} alt="preview" style={{ maxWidth: '100%', maxHeight: '70vh' }} /> : <Typography>No preview available</Typography>}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => copyToClipboard(previewSrc)}>Copy URL</Button>
          <Button onClick={closePreview}>Close</Button>
        </DialogActions>
      </Dialog>

      {/* Export confirmation dialog */}
      <Dialog open={!!exportConfirmOpen} onClose={closeExportConfirm}>
        <DialogTitle>Confirm export</DialogTitle>
        <DialogContent>
          <Typography>Export a short blurred video around this frame? This will queue a background job and may take a few moments.</Typography>
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption">Video id: {exportPending?.frame?.video_id || exportPending?.frame?.videoId || savedVideoId || 'N/A'}</Typography>
            <Typography variant="caption" sx={{ display: 'block' }}>Timestamp: {exportPending?.frame ? formatTs(exportPending.frame.timestamp) : 'N/A'}</Typography>
            <Box sx={{ mt: 1 }}>
              <FormControlLabel
                control={<Checkbox checked={exportDownloadAfter} onChange={(e) => setExportDownloadAfter(e.target.checked)} />}
                label="Download when ready"
              />
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={closeExportConfirm}>Cancel</Button>
          <Button onClick={async () => {
            if (exportPending) {
              const { frame, idx } = exportPending;
              closeExportConfirm();
              const vid = frame.video_id || frame.videoId || frame.video || savedVideoId || null;
              await handleExportBlurredVideo(vid, frame, exportDownloadAfter);
            }
          }} variant="contained" color="primary">Confirm</Button>
        </DialogActions>
      </Dialog>

      <Snackbar open={toast.open} autoHideDuration={3000} onClose={() => setToast(t => ({ ...t, open: false }))}>
        <Alert severity={toast.severity || 'info'} onClose={() => setToast(t => ({ ...t, open: false }))}>{toast.message}</Alert>
      </Snackbar>
    </Box>
  );
};

export default FrameDetailsPage;
