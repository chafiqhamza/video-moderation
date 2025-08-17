import React from 'react';
import { Card, CardContent, Typography, Box, Divider, Button } from '@mui/material';

const FrameDetailsPage = ({ frames }) => {
    if (!frames || frames.length === 0) return <Typography>No frame details available.</Typography>;
    const handleBack = () => {
        if (typeof window !== 'undefined' && window.__react_app_back) {
            window.__react_app_back();
        } else if (typeof window !== 'undefined' && window.history.length > 1) {
            window.history.back();
        } else {
            window.location.href = '/';
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
            {frames.map((frame, idx) => (
                <Card key={idx} sx={{ mb: 2, boxShadow: 2 }}>
                    <CardContent>
                        <Typography variant="h6" sx={{ fontWeight: 'bold' }}>Frame {frame.frame || idx + 1}</Typography>
                        <Typography variant="body2">Timestamp: <b>{frame.timestamp || 'N/A'}</b></Typography>
                        <Typography variant="body2">Category: <b>{frame.category || 'N/A'}</b></Typography>
                        <Typography variant="body2">Confidence: <b>{frame.confidence || 'N/A'}</b></Typography>
                        <Typography variant="body2">Reasoning: <b>{frame.reasoning || 'N/A'}</b></Typography>
                        {/* Add more details as needed */}
                    </CardContent>
                </Card>
            ))}
        </Box>
    );
};

export default FrameDetailsPage;
