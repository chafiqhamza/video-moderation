import React from 'react';
import { Box, Card, CardContent, Typography, Button, Alert } from '@mui/material';
import CloudUpload from '@mui/icons-material/CloudUpload';

/**
 * VideoUpload component handles file selection, drag & drop, and upload button.
 * Props:
 * - selectedFile: File object
 * - onFileChange: function to handle file selection
 * - onAnalyze: function to trigger analysis
 * - loading: boolean
 * - error: string
 * - frameSettings: object
 * - connectionStatus: string
 */
function VideoUpload({ selectedFile, onFileChange, onAnalyze, loading, error, frameSettings, connectionStatus }) {
    const formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i]);
    };

    return (
        <Card sx={{ width: '100%', maxWidth: 320, mx: 'auto', p: 2, boxShadow: '0 2px 8px rgba(33,203,243,0.08)', borderRadius: 3, background: '#fff' }}>
            <CardContent>
                <Typography variant="subtitle1" align="center" sx={{ mb: 2, fontWeight: 'bold', color: '#1976d2', letterSpacing: 1 }}>
                    Full Video Moderation (Upload)
                </Typography>
                {/* Display chosen frame settings */}
                <Box sx={{ mb: 1, p: 1, background: '#e3f2fd', borderRadius: 1, fontSize: '0.95rem', color: '#1976d2' }}>
                    <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>Current Frame Extraction Settings:</Typography>
                    <ul style={{ margin: 0, paddingLeft: 20 }}>
                        <li>Frames: <b>{frameSettings.frameCount}</b></li>
                        <li>Interval: <b>{frameSettings.frameInterval}</b> sec</li>
                        <li>Resolution: <b>{frameSettings.frameResolution}</b></li>
                        <li>Format: <b>{frameSettings.frameFormat}</b></li>
                        <li>Start: <b>{frameSettings.frameStart}</b> sec</li>
                        <li>End: <b>{frameSettings.frameEnd !== null ? frameSettings.frameEnd : 'Full video'}</b></li>
                        <li>Sampling: <b>{frameSettings.frameSampling}</b></li>
                    </ul>
                </Box>
                <Box
                    sx={{
                        border: '2px dashed #1976d2',
                        borderRadius: 2,
                        p: 2,
                        mb: 2,
                        textAlign: 'center',
                        background: '#e3f2fd',
                        cursor: 'pointer',
                        transition: '0.2s',
                        ':hover': { background: '#bbdefb', borderColor: '#1565c0' },
                    }}
                    onClick={() => document.getElementById('video-upload').click()}
                    onDragOver={e => e.preventDefault()}
                    onDrop={e => {
                        e.preventDefault();
                        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                            onFileChange({ target: { files: e.dataTransfer.files } });
                        }
                    }}
                >
                    <input
                        accept="video/*"
                        style={{ display: 'none' }}
                        id="video-upload"
                        type="file"
                        onChange={onFileChange}
                    />
                    {selectedFile ? (
                        <Typography variant="body2" color="primary" sx={{ fontWeight: 'bold', fontSize: '1rem' }}>
                            {selectedFile.name} ({formatFileSize(selectedFile.size)})
                        </Typography>
                    ) : (
                        <Typography variant="body2" color="text.secondary" sx={{ fontSize: '1rem' }}>
                            Drag & Drop a video here, or click to upload
                        </Typography>
                    )}
                </Box>
                <Button
                    variant="contained"
                    color="primary"
                    fullWidth
                    sx={{ fontWeight: 'bold', py: 1, fontSize: '1rem', borderRadius: 2, boxShadow: 1, background: 'linear-gradient(90deg,#1976d2 60%,#21cbf3 100%)', ':hover': { background: 'linear-gradient(90deg,#1565c0 60%,#039be5 100%)', transform: 'scale(1.01)' } }}
                    onClick={onAnalyze}
                    disabled={loading || !selectedFile || connectionStatus !== 'connected'}
                >
                    {loading ? 'Uploading...' : 'UPLOAD VIDEO'}
                </Button>
                {error && (
                    <Alert severity="error" sx={{ mt: 3 }}>
                        {error}
                    </Alert>
                )}
            </CardContent>
        </Card>
    );
}

export default VideoUpload;
