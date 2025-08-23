import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';
import AudiotrackIcon from '@mui/icons-material/Audiotrack';

export default function ComprehensiveReportDisplay({ report }) {
    if (!report) return null;



    // Helper styles
    const headerChipStyle = { fontWeight: 'bold', fontSize: 15, px: 2, py: 0.5, borderRadius: 2 };
    const flaggedStyle = { color: '#fff', background: '#e53935', borderRadius: 2, px: 1, py: 0.5, fontWeight: 'bold', display: 'inline-block' };

    // Parse model output and fallback
    const parsed = report.analysis_json ? report : null;
    const fields = parsed && parsed.analysis_json ? parsed.analysis_json : {};
    const fallback = report.parsedReport || {};

    // Statistics (robust: use backend frames array if present)
    // Use true model output for compliance and violations
    const framesArr = Array.isArray(fields.frame_details)
        ? fields.frame_details
        : Array.isArray(fields.detailed_frames)
            ? fields.detailed_frames
            : [];
    const totalFrames = framesArr.length;
    // Count all frames, not just those above a confidence threshold
    const safeFrames = framesArr.filter(f => {
        const cat = f.visual_analysis?.category || f.category || '';
        return String(cat).toLowerCase() === 'safe_content';
    }).length;
    const violationFrames = framesArr.filter(f => {
        const cat = f.visual_analysis?.category || f.category || '';
        return String(cat).toLowerCase() !== 'safe_content' || f.visual_analysis?.violation_detected === true;
    }).length;
    const violationCount = violationFrames;
    const complianceRate = totalFrames > 0 ? Math.round((safeFrames / totalFrames) * 100) : 0;

    // Content extraction
    const transcript = fields.audio_analysis?.transcript || fields.transcript || fallback.transcript || '';
    const ocrText = fields.ocr_results?.text || fields.text_in_video?.text || fallback.ocr_text || '';
    const ocrItems = fields.ocr_results?.extracted_items || fields.text_in_video?.extracted_items || fallback.ocr_items || [];
    const blipDescriptions = Array.isArray(framesArr)
        ? framesArr.map(f => f.blip_description?.description).filter(d => d && d.trim() && d !== 'No BLIP description available.')
        : (fields.frame_analysis?.blip_descriptions || fallback.blip_descriptions || []);
    const copyrightCheck = report.copyright_check || null;

    // Scores and issues: use true model output fields if present
    const textScore = fields.text_analysis?.score ?? fields.frame_analysis?.text_score ?? fields.overall_scores?.text ?? 'N/A';
    const audioScore = fields.audio_analysis?.score ?? fields.frame_analysis?.audio_score ?? fields.overall_scores?.audio ?? 'N/A';
    const imageScore = fields.frame_analysis?.average_confidence ? Math.round(fields.frame_analysis.average_confidence * 100) : (fields.frame_analysis?.image_score ?? fields.overall_scores?.image ?? 'N/A');
    const textIssues = fields.text_analysis?.issues ?? report.text_issues ?? fields.text_issues ?? 'None';
    const audioIssues = fields.audio_analysis?.issues ?? report.audio_issues ?? fields.audio_issues ?? 'None';
    const imageIssues = fields.frame_analysis?.issues ?? report.image_issues ?? fields.image_issues ?? 'None';

    // Transcript safety logic
    const badWords = fields.audio_analysis?.bad_words || [];
    const badPatterns = fields.audio_analysis?.bad_patterns || [];
    const isInappropriate = badWords.length > 0 || badPatterns.length > 0;
    let transcriptSafetyStatus = '';
    if (transcript === '') {
        transcriptSafetyStatus = 'No transcript available';
    } else if (isInappropriate) {
        if (badWords.length > 0) {
            transcriptSafetyStatus = 'Flagged: Inappropriate language detected';
        } else if (badPatterns.length > 0) {
            transcriptSafetyStatus = 'Close: Censored or partial inappropriate language detected';
        } else {
            transcriptSafetyStatus = 'Flagged: Inappropriate language detected';
        }
    } else {
        transcriptSafetyStatus = 'Safe: No inappropriate language detected';
    }

    // RAG guideline summary logic
    const ragExplanations = (fields.rag_explanations || report.rag_explanations || []);
    const isCompliant = complianceRate > 50;
    let guidelineSummary = '';
    let guidelineReasons = [];
    // Helper to get unique, relevant reasons
    function getUniqueReasons(explanations, compliant) {
        const seen = new Set();
        return explanations
            .map(exp => {
                let reason = '';
                if (exp.policy && exp.policy.description) {
                    reason = exp.policy.description;
                } else if (exp.explanation) {
                    reason = exp.explanation;
                } else if (typeof exp === 'string') {
                    reason = exp;
                } else {
                    reason = JSON.stringify(exp);
                }
                return reason.trim();
            })
            .filter(r => r && !seen.has(r) && r.toLowerCase() !== 'safe content' && seen.add(r))
            .slice(0, 5); // Limit to 5 reasons for clarity
    }

    if (isCompliant) {
        guidelineSummary = 'This video follows YouTube guidelines because:';
        // Show enhanced RAG explanations (scraped policy)
        // Deduplicate and personalize RAG policy explanations
        let ragDetails = [];
        if (Array.isArray(ragExplanations) && ragExplanations.length > 0) {
            // If any explanation is 'Safe Content', show only that
            const safeExp = ragExplanations.find(exp => exp.policy && exp.policy.category === 'Safe Content');
            if (safeExp) {
                ragDetails = [`- [${safeExp.policy.category}] ${safeExp.policy.description}`];
            } else {
                const seenPolicies = new Set();
                ragDetails = ragExplanations
                    .map((exp) => {
                        if (exp.policy && exp.policy.description) {
                            const key = `${exp.policy.category || ''}|${exp.policy.description}`;
                            if (seenPolicies.has(key)) return null;
                            seenPolicies.add(key);
                            let line = `- [${exp.policy.category || 'Policy'}] ${exp.policy.description}`;
                            if (exp.examples && exp.examples.length > 0) {
                                line += ` (Examples: ${exp.examples.slice(0,2).join('; ')})`;
                            }
                            return line;
                        }
                        if (exp.explanation) {
                            return `- ${exp.explanation}`;
                        }
                        return null;
                    })
                    .filter(Boolean);
            }
        }
        // Get BLIP descriptions for more content, deduplicated
        const blipDescs = Array.isArray(framesArr)
            ? framesArr.map(f => f.blip_description?.description).filter(d => d && d.trim() && d !== 'No BLIP description available.')
            : [];
        const uniqueBlipDescs = [...new Set(blipDescs)].slice(0, 3); // Limit to 3 for clarity
        guidelineReasons = [];
        if (ragDetails.length > 0) {
            guidelineReasons = ragDetails;
        } else {
            guidelineReasons = ['- No violations detected.'];
        }
        if (uniqueBlipDescs.length > 0) {
            guidelineReasons.push('Frame Descriptions (AI visual summary):');
            uniqueBlipDescs.forEach(desc => guidelineReasons.push(`- ${desc}`));
        }
    } else {
        guidelineSummary = 'This video does not follow YouTube guidelines because:';
        guidelineReasons = ragExplanations.length > 0
            ? getUniqueReasons(ragExplanations, false)
            : ['- Violations detected, but no detailed policy explanation available.'];
    }

    return (
        <Box sx={{ width: '100vw', minHeight: '100vh', maxHeight: '100vh', background: '#f7f8fa', display: 'flex', flexDirection: 'column', alignItems: 'center', p: 2, overflowY: 'auto' }}>
            {/* Guideline summary section */}
            <Box sx={{ width: '100%', maxWidth: 1100, mb: 2, p: 2, background: isCompliant ? '#e8f5e9' : '#ffebee', borderRadius: 3, boxShadow: 2 }}>
                <Typography variant="h6" sx={{ fontWeight: 'bold', color: isCompliant ? '#43a047' : '#e53935', mb: 1 }}>{guidelineSummary}</Typography>
                {guidelineReasons.map((reason, idx) => (
                    <Typography key={idx} variant="body2" sx={{ mb: 0.5 }}>{reason}</Typography>
                ))}
            </Box>
            {/* Header chips for source/analysis */}
            <Box sx={{ width: '100%', maxWidth: 1100, display: 'flex', flexDirection: 'row', gap: 2, mb: 1, justifyContent: 'center' }}>
                <Chip label="Source: Model Output" sx={{ ...headerChipStyle, background: '#1976d2', color: '#fff' }} />
                <Chip label="Analysis: AI Model" sx={{ ...headerChipStyle, background: '#43a047', color: '#fff' }} />
            </Box>
            {/* Top summary cards */}
            <Box sx={{ width: '100%', maxWidth: 1100, display: 'flex', flexDirection: 'row', gap: 2, mb: 2, justifyContent: 'center' }}>
                <Card sx={{ flex: 1, boxShadow: 2, borderRadius: 3, p: 2 }}>
                    <CardContent>
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#1976d2', mb: 1 }}>Compliance Rate</Typography>
                        <Typography variant="h6" sx={{ color: complianceRate > 50 ? '#43a047' : '#e53935', fontWeight: 'bold', mt: 1 }}>{complianceRate}%</Typography>
                        <Chip label={complianceRate > 50 ? 'Compliant' : 'Non-Compliant'} color={complianceRate > 50 ? 'success' : 'error'} sx={{ mt: 1, fontWeight: 'bold' }} />
                    </CardContent>
                </Card>
                <Card sx={{ flex: 1, boxShadow: 2, borderRadius: 3, p: 2 }}>
                    <CardContent>
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#1976d2', mb: 1 }}>Violations</Typography>
                        <Typography variant="h6" sx={{ color: '#e53935', fontWeight: 'bold', mt: 1 }}>{violationCount}</Typography>
                        <Chip label="Violations" color="error" sx={{ mt: 1, fontWeight: 'bold' }} />
                    </CardContent>
                </Card>
                <Card sx={{ flex: 1, boxShadow: 2, borderRadius: 3, p: 2 }}>
                    <CardContent>
                        <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#1976d2', mb: 1 }}>Scores</Typography>
                        <Typography variant="body2">Text: <b>{textScore || 'N/A'}%</b></Typography>
                        <Typography variant="body2">Audio: <b>{audioScore || 'N/A'}%</b></Typography>
                        <Typography variant="body2">Image: <b>{imageScore || 'N/A'}%</b></Typography>
                    </CardContent>
                </Card>
            </Box>
            {/* Main content: left and right columns */}
            <Box sx={{ width: '100%', maxWidth: 1100, display: 'flex', flexDirection: 'row', gap: 2, alignItems: 'stretch', justifyContent: 'center' }}>
                {/* Left: Transcript Safety & Extracted Content */}
                <Box sx={{ flex: 1, minWidth: 320, background: '#f5faff', borderRadius: 3, boxShadow: 2, p: 2, overflow: 'auto', height: '100%' }}>
                    <Card sx={{ boxShadow: 1, borderRadius: 2, mb: 2 }}>
                        <CardContent>
                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                                <AudiotrackIcon color="info" sx={{ fontSize: 28, mr: 1 }} />
                                <Typography variant="h6" sx={{ fontWeight: 'bold' }}>Transcript Safety (Model Output)</Typography>
                            </Box>
                            {transcriptSafetyStatus.startsWith('Flagged') ? (
                                <span style={flaggedStyle}>{transcriptSafetyStatus}</span>
                            ) : (
                                <Chip label={transcriptSafetyStatus} color={transcriptSafetyStatus.startsWith('Safe') ? 'success' : transcriptSafetyStatus.startsWith('Close') ? 'warning' : transcriptSafetyStatus.startsWith('No') ? 'warning' : 'error'} sx={{ fontWeight: 'bold', fontSize: 15, mb: 1 }} />
                            )}
                            {isInappropriate && (badWords.length > 0 || badPatterns.length > 0) && (
                                <Typography variant="body2" sx={{ color: transcriptSafetyStatus.startsWith('Close') ? 'warning.main' : 'error.main', mt: 1 }}>
                                    {badWords.length > 0 && `Bad words: ${badWords.join(', ')}`}
                                    {badPatterns.length > 0 && `Close matches or censored forms detected.`}
                                </Typography>
                            )}
                        </CardContent>
                    </Card>
                    <Card sx={{ boxShadow: 1, borderRadius: 2, mb: 2 }}>
                        <CardContent>
                            <Typography variant="h6" sx={{ fontWeight: 'bold', mb: 1 }}>Extracted Content (Model Output)</Typography>
                            <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>Audio Transcript:</Typography>
                            <Typography variant="body2" sx={{ fontStyle: 'italic', mb: 1 }}>{transcript || 'No transcript available.'}</Typography>
                            <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>OCR Text:</Typography>
                            <Typography variant="body2" sx={{ fontStyle: 'italic', mb: 1 }}>{ocrText || 'No OCR text.'}</Typography>
                            <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>OCR Items:</Typography>
                            {ocrItems.length > 0 ? (
                                <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                    {ocrItems.map((item, idx) => (
                                        <Chip key={idx} label={item} color="info" />
                                    ))}
                                </Box>
                            ) : (
                                <Typography variant="body2" color="success.main">None detected.</Typography>
                            )}
                        </CardContent>
                    </Card>
                </Box>
                {/* Right: Model Analysis Report */}
                <Box sx={{ flex: 2, minWidth: 400, background: '#f8f8ff', borderRadius: 3, boxShadow: 2, p: 2, overflow: 'auto', height: '100%' }}>
                    <Typography variant="h6" sx={{ fontWeight: 'bold', mb: 2 }}>Model Analysis Report</Typography>
                    {/* General Info */}
                    <Card sx={{ mb: 2, boxShadow: 1 }}>
                        <CardContent>
                            <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>General Info (Model Output)</Typography>
                            <Typography variant="body2">Duration: <b>{fields.video_info?.duration ?? fields.video_info?.duration_formatted ?? 'N/A'}</b></Typography>
                            <Typography variant="body2">Resolution: <b>{fields.video_info?.resolution ?? 'N/A'}</b></Typography>
                            <Typography variant="body2">FPS: <b>{fields.video_info?.fps ?? 'N/A'}</b></Typography>
                            <Typography variant="body2">Processing Time: <b>{fields.video_info?.analysis_time ?? fields.processing_time ?? 'N/A'}</b></Typography>
                        </CardContent>
                    </Card>
                    {/* Text Analysis */}
                    <Card sx={{ mb: 2, boxShadow: 1 }}>
                        <CardContent>
                            <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>Text Analysis (Model Output)</Typography>
                            <Typography variant="body2">Score: <b>{textScore || 'N/A'}%</b></Typography>
                            <Typography variant="body2">Issues: {textIssues && textIssues.includes('major_violations') ? <span style={flaggedStyle}>{textIssues}</span> : <b>{textIssues || 'None'}</b>}</Typography>
                        </CardContent>
                    </Card>
                    {/* Audio Analysis */}
                    <Card sx={{ mb: 2, boxShadow: 1 }}>
                        <CardContent>
                            <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>Audio Analysis (Model Output)</Typography>
                            <Typography variant="body2">Score: <b>{audioScore || 'N/A'}%</b></Typography>
                            <Typography variant="body2">Issues: <b>{audioIssues || 'None'}</b></Typography>
                        </CardContent>
                    </Card>
                    {/* Frame/Image Analysis */}
                    <Card sx={{ mb: 2, boxShadow: 1 }}>
                        <CardContent>
                            <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>Frame/Image Analysis (Model Output)</Typography>
                            <Typography variant="body2">Score: <b>{imageScore || 'N/A'}%</b></Typography>
                            <Typography variant="body2">Issues: <b>{imageIssues || 'None'}</b></Typography>
                        </CardContent>
                    </Card>
                    {/* Copyright Analysis */}
                    <Card sx={{ mb: 2, boxShadow: 1 }}>
                        <CardContent>
                            <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>Copyright Analysis</Typography>
                            <Typography variant="body2">Status: {copyrightCheck?.status ? <span style={flaggedStyle}>{copyrightCheck.status}</span> : <b>{fields.copyright_status || 'N/A'}</b>}</Typography>
                            <Typography variant="body2">Details: <b>{copyrightCheck?.details || fields.copyright_details || 'No details available.'}</b></Typography>
                        </CardContent>
                    </Card>
                    {/* Overall Compliance */}
                    <Card sx={{ mb: 2, boxShadow: 1 }}>
                        <CardContent>
                            <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>Overall Compliance (Model Output)</Typography>
                            <Typography variant="body2">Status: <b>{fields.overall || 'N/A'}</b></Typography>
                            <Typography variant="body2">Scores: <b>Text: {fields.overall_scores?.text || 'N/A'}% | Audio: {fields.overall_scores?.audio || 'N/A'}% | Image: {fields.overall_scores?.image || 'N/A'}%</b></Typography>
                        </CardContent>
                    </Card>
                </Box>
            </Box>
        </Box>
    );
}