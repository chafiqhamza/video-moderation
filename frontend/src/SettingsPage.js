import React, { useState } from 'react';
import { Container, Typography, Box, TextField, Button, Paper, Divider } from '@mui/material';

const DEFAULT_FRAME_SETTINGS = {
    frameCount: 1,
    frameInterval: 0.5,
    frameResolution: 'auto',
    frameFormat: 'jpg',
    frameStart: 0,
    frameEnd: null,
    frameSampling: 'interval',
};

function SettingsPage({ settings, onSave }) {
    const [localSettings, setLocalSettings] = useState(settings);

    function handleDefault() {
        setLocalSettings(DEFAULT_FRAME_SETTINGS);
    }

    const handleChange = (field, value) => {
        setLocalSettings(prev => ({ ...prev, [field]: value }));
    };

    const handleSave = () => {
        if (onSave) onSave(localSettings);
    };

    return (
        <Container maxWidth="sm" sx={{ mt: 6, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Paper sx={{ p: 5, borderRadius: 4, boxShadow: 4, width: '100%', background: 'linear-gradient(120deg, #e3f2fd 60%, #fffde7 100%)' }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    <Button variant="outlined" color="secondary" sx={{ mr: 2, fontWeight: 'bold', borderRadius: 3 }} onClick={() => onSave && onSave(localSettings)}>
                        ← Back
                    </Button>
                    <Typography variant="h4" sx={{ fontWeight: 'bold', color: '#1976d2' }}>
                        Frame Extraction Settings
                    </Typography>
                </Box>
                <Divider sx={{ mb: 3 }} />
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                    <Button variant="contained" color="primary" sx={{ mb: 2, fontWeight: 'bold', borderRadius: 2 }} onClick={handleDefault}>
                        Use Normal Analysis (1 frame every 0.5 seconds)
                    </Button>
                    <TextField
                        label="Frame Extraction Mode"
                        select
                        SelectProps={{ native: true }}
                        value={localSettings.frameSampling}
                        onChange={e => handleChange('frameSampling', e.target.value)}
                        fullWidth
                        helperText="Choose 'Interval' to extract every N seconds, or 'Count' to extract a fixed number of frames."
                    >
                        <option value="interval">Interval (every N seconds)</option>
                        <option value="count">Count (fixed number of frames)</option>
                        <option value="random">Random</option>
                        <option value="keyframes">Keyframes</option>
                    </TextField>
                    <TextField
                        label="Number of Frames to Analyze"
                        type="number"
                        value={localSettings.frameCount}
                        onChange={e => setLocalSettings({ ...localSettings, frameCount: Number(e.target.value) })}
                        sx={{ mb: 2 }}
                        fullWidth
                        helperText="If mode is 'Count', this is the number of frames to extract."
                    />
                    <TextField
                        label="Interval Between Frames (seconds)"
                        type="number"
                        value={localSettings.frameInterval}
                        onChange={e => setLocalSettings({ ...localSettings, frameInterval: Number(e.target.value) })}
                        sx={{ mb: 2 }}
                        fullWidth
                        helperText="If mode is 'Interval', this is the time gap between each extracted frame."
                    />
                    <TextField
                        label="Frame Resolution (e.g. 720p, 1080p, auto)"
                        value={localSettings.frameResolution}
                        onChange={e => handleChange('frameResolution', e.target.value)}
                        fullWidth
                        helperText="Resolution for extracted frames. Use 'auto' for original."
                    />
                    <TextField
                        label="Frame Format (jpg, png, webp)"
                        value={localSettings.frameFormat}
                        onChange={e => handleChange('frameFormat', e.target.value)}
                        fullWidth
                        helperText="Image format for saved frames."
                    />
                    <TextField
                        label="Start Time (seconds)"
                        type="number"
                        value={localSettings.frameStart}
                        onChange={e => handleChange('frameStart', Math.max(0, parseFloat(e.target.value) || 0))}
                        inputProps={{ min: 0, step: 0.1 }}
                        fullWidth
                        helperText="Time in seconds to start frame extraction."
                    />
                    <TextField
                        label="End Time (seconds, blank for full video)"
                        type="number"
                        value={localSettings.frameEnd || ''}
                        onChange={e => handleChange('frameEnd', e.target.value ? Math.max(0, parseFloat(e.target.value)) : null)}
                        inputProps={{ min: 0, step: 0.1 }}
                        fullWidth
                        helperText="Time in seconds to end frame extraction. Leave blank for full video."
                    />
                    <TextField
                        label="Frame Sampling Method"
                        select
                        SelectProps={{ native: true }}
                        value={localSettings.frameSampling}
                        onChange={e => handleChange('frameSampling', e.target.value)}
                        fullWidth
                        helperText="How frames are selected: interval, random, or keyframes."
                    >
                        <option value="interval">Interval (default)</option>
                        <option value="random">Random</option>
                        <option value="keyframes">Keyframes</option>
                    </TextField>
                </Box>
                <Button variant="contained" color="success" sx={{ mt: 4, borderRadius: 3, fontWeight: 'bold', px: 4 }} onClick={handleSave}>
                    Save Settings
                </Button>
            </Paper>
        </Container>
    );
}

export default SettingsPage;
