import React from 'react';
import { Card, CardContent, Typography, Box, Chip, Divider } from '@mui/material';

function ComprehensiveReportDisplay({ report }) {
    if (!report) return null;

    // Try to use parsed fields if available
    const parsed = report.analysis_json || report.full_report ? report : null;
    const fields = parsed && parsed.analysis_json ? parsed.analysis_json : {};

    // Fallback to parsedReport if available
    const fallback = report.parsedReport || {};

    // Helper to display a section
    const Section = ({ title, children }) => (
        <Box sx={{ mb: 2 }}>
            <Typography variant="h6" sx={{ fontWeight: 'bold', mb: 1 }}>{title}</Typography>
            {children}
            <Divider sx={{ my: 1 }} />
        </Box>
    );

    return (
        <Card sx={{ mt: 4, boxShadow: 3, borderRadius: 4 }}>
            <CardContent>
                <Typography variant="h5" sx={{ fontWeight: 'bold', mb: 2 }} color="primary">
                    Model Analysis Report
                </Typography>
                <Section title="General Info">
                    <Typography variant="body2">Duration: <b>{fields.video_info?.duration || fallback.duration || 'N/A'}</b></Typography>
                    <Typography variant="body2">Resolution: <b>{fields.video_info?.resolution || fallback.resolution || 'N/A'}</b></Typography>
                    <Typography variant="body2">FPS: <b>{fields.video_info?.fps || fallback.fps || 'N/A'}</b></Typography>
                    <Typography variant="body2">Processing Time: <b>{fields.video_info?.analysis_time || fallback.processing_time || 'N/A'}</b></Typography>
                </Section>
                <Section title="Text Analysis">
                    <Typography variant="body2">Score: <b>{fields.frame_analysis?.average_confidence ? (fields.frame_analysis.average_confidence * 100).toFixed(1) : fallback.text_score || 'N/A'}%</b></Typography>
                    <Typography variant="body2">Status: <b>{fields.overall_assessment?.status || fallback.text_status || 'N/A'}</b></Typography>
                    <Typography variant="body2">Issues: <b>{fields.recommendations ? fields.recommendations.join(', ') : (fallback.text_issues || 'None')}</b></Typography>
                </Section>
                <Section title="Audio Analysis">
                    <Typography variant="body2">Score: <b>{fields.overall_assessment?.audio_compliance ? fields.overall_assessment.audio_compliance.toFixed(1) : fallback.audio_score || 'N/A'}%</b></Typography>
                    <Typography variant="body2">Status: <b>{fields.overall_assessment?.status || fallback.audio_status || 'N/A'}</b></Typography>
                    <Typography variant="body2">Issues: <b>{fields.audio_analysis?.policy_flags ? Object.keys(fields.audio_analysis.policy_flags).join(', ') : (fallback.audio_issues || 'None')}</b></Typography>
                </Section>
                <Section title="Frame/Image Analysis">
                    <Typography variant="body2">Score: <b>{fields.overall_assessment?.visual_compliance ? fields.overall_assessment.visual_compliance.toFixed(1) : fallback.image_score || 'N/A'}%</b></Typography>
                    <Typography variant="body2">Status: <b>{fields.overall_assessment?.status || fallback.image_status || 'N/A'}</b></Typography>
                    <Typography variant="body2">Issues: <b>{fields.frame_analysis?.violation_categories ? fields.frame_analysis.violation_categories.join(', ') : (fallback.image_issues || 'None')}</b></Typography>
                    <Typography variant="body2">Frames Analyzed: <b>{fields.frame_analysis?.total_frames || fallback.frames_analyzed || 'N/A'}</b></Typography>
                </Section>
                <Section title="Overall Compliance">
                    <Typography variant="body2">Status: <b>{fields.overall_assessment?.status || fallback.overall || 'N/A'}</b></Typography>
                    <Typography variant="body2">Text Score: <b>{fields.overall_assessment?.visual_compliance ? fields.overall_assessment.visual_compliance.toFixed(1) : fallback.overall_scores?.text || 'N/A'}%</b></Typography>
                    <Typography variant="body2">Audio Score: <b>{fields.overall_assessment?.audio_compliance ? fields.overall_assessment.audio_compliance.toFixed(1) : fallback.overall_scores?.audio || 'N/A'}%</b></Typography>
                    <Typography variant="body2">Image Score: <b>{fields.overall_assessment?.overall_compliance ? fields.overall_assessment.overall_compliance.toFixed(1) : fallback.overall_scores?.image || 'N/A'}%</b></Typography>
                </Section>
                {report.full_report && (
                    <Section title="Raw Report">
                        <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace', background: '#f5f5f5', p: 2, borderRadius: 2 }}>
                            {report.full_report}
                        </Typography>
                    </Section>
                )}
            </CardContent>
        </Card>
    );
}

export default ComprehensiveReportDisplay;
