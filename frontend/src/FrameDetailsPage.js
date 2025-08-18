import React from 'react';
import { Card, CardContent, Typography, Box, Divider, Button } from '@mui/material';

const FrameDetailsPage = ({ frames, onBack }) => {
    // Only show frames that do not follow compliance and have confidence > 0.6
    const badFrames = frames.filter(frame => {
        const confidence = frame.visual_analysis?.confidence !== undefined ? frame.visual_analysis.confidence : frame.confidence;
        // Non-compliant: violation_detected true or action_required present
        const nonCompliant = (frame.visual_analysis?.violation_detected || frame.combined_violation || frame.action_required);
        return nonCompliant && confidence > 0.6;
    });
    if (!badFrames || badFrames.length === 0) return <Typography>No non-compliant frames with confidence above 0.6 found.</Typography>;
    const handleBack = () => {
        if (onBack) {
            onBack();
        }
    };
    return (
        <Box sx={{ p: 4 }}>
            <Button variant="outlined" color="primary" sx={{ mb: 2 }} onClick={handleBack}>
                ← Back
            </Button>
            <Typography variant="h4" color="primary" sx={{ mb: 3, fontWeight: 'bold' }}>
                Detailed Frame Report
            </Typography>
            {badFrames.map((frame, idx) => (
                <Card key={idx} sx={{ mb: 2, boxShadow: 2 }}>
                    <CardContent>
                        <Typography variant="h6" sx={{ fontWeight: 'bold' }}>Frame {frame.frame_index !== undefined ? frame.frame_index + 1 : idx + 1}</Typography>
                        <Typography variant="body2">Timestamp: <b>{frame.timestamp !== undefined ? frame.timestamp : 'N/A'}</b></Typography>
                        {frame.preview_path && <img src={frame.preview_path} alt={`Frame ${idx + 1}`} style={{ maxWidth: 320, borderRadius: 8, marginBottom: 8 }} />}
                        <Divider sx={{ my: 1 }} />
                        <Typography variant="body2">Category: <b>{frame.visual_analysis?.category || frame.category || 'N/A'}</b></Typography>
                        <Typography variant="body2">Confidence: <b>{frame.visual_analysis?.confidence !== undefined ? frame.visual_analysis.confidence : frame.confidence || 'N/A'}</b></Typography>
                        <Typography variant="body2">Violation Detected: <b>{frame.visual_analysis?.violation_detected !== undefined ? (frame.visual_analysis.violation_detected ? 'Yes' : 'No') : (frame.combined_violation ? 'Yes' : 'No')}</b></Typography>
                        <Divider sx={{ my: 1 }} />
                        {frame.blip_description?.description && frame.blip_description.description.trim() !== '' && (
                            <Typography variant="body2">BLIP Description: <b>{frame.blip_description.description}</b></Typography>
                        )}
                        {(!frame.blip_description?.description || frame.blip_description.description.trim() === '') && (
                            <Typography variant="body2">BLIP Description: <b>N/A</b></Typography>
                        )}
                        {frame.ocr_text?.text && frame.ocr_text.text.trim() !== '' && (
                            <Typography variant="body2">OCR Text: <b>{frame.ocr_text.text}</b></Typography>
                        )}
                        {(!frame.ocr_text?.text || frame.ocr_text.text.trim() === '') && (
                            <Typography variant="body2">OCR Text: <b>N/A</b></Typography>
                        )}
                        {frame.ocr_text?.extracted_items && frame.ocr_text.extracted_items.length > 0 && (
                            <Typography variant="body2">OCR Items: <b>{frame.ocr_text.extracted_items.map((item, i) => item.text).join(', ')}</b></Typography>
                        )}
                        {/* Policy, Examples, Action Required from RAG if available */}
                        {frame.policy && frame.policy.description && (
                            <Typography variant="body2">Policy: <b>{frame.policy.description}</b></Typography>
                        )}
                        {frame.examples && frame.examples.length > 0 && (
                            <Typography variant="body2">Examples: <b>{frame.examples.join(', ')}</b></Typography>
                        )}
                        {frame.action_required && (
                            <Typography variant="body2">Action Required: <b>{frame.action_required}</b></Typography>
                        )}
                        <Divider sx={{ my: 1 }} />
                        <Typography variant="body2">Preview Path: <b>{frame.preview_path || 'N/A'}</b></Typography>
                        <Typography variant="body2">Frame Index: <b>{frame.frame_index !== undefined ? frame.frame_index : idx}</b></Typography>
                        {/* Add more fields as needed for full model analysis */}
                    </CardContent>
                </Card>
            ))}
        </Box>
    );
};

export default FrameDetailsPage;
