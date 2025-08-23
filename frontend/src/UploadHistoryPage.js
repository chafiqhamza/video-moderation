import React, { useState, useEffect } from 'react';
import { Container, Typography, Card, CardContent, Box, Chip, Button, TextField, Grid } from '@mui/material';
import CheckCircle from '@mui/icons-material/CheckCircle';
import Warning from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';

const API_BASE_URL = 'http://localhost:8000';

function UploadHistoryPage() {
    const [history, setHistory] = useState([]);
    const [dateFilter, setDateFilter] = useState('');
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        fetchHistory();
    }, [dateFilter]);

    const fetchHistory = async () => {
        setLoading(true);
        try {
            let url = `${API_BASE_URL}/videos`;
            if (dateFilter) {
                url += `?date=${dateFilter}`;
            }
            const response = await fetch(url);
            if (!response.ok) throw new Error('Failed to fetch history');
            const data = await response.json();
            setHistory(data);
        } catch (err) {
            setHistory([]);
        } finally {
            setLoading(false);
        }
    };

    const handleBack = () => {
        if (typeof window !== 'undefined' && window.__react_app_back) {
            window.__react_app_back();
        } else if (typeof window !== 'undefined' && window.history.length > 1) {
            window.history.back();
        } else {
            // fallback: reload or set location
            window.location.href = '/';
        }
    };
    return (
        <Container maxWidth="xl" sx={{ mt: 6, mb: 6 }}>
            <Button variant="outlined" color="primary" sx={{ mb: 2 }} onClick={handleBack}>
                ← Back
            </Button>
            <Typography variant="h4" sx={{ fontWeight: 'bold', mb: 4, color: '#1976d2' }}>
                📜 Upload History
            </Typography>
            <Box sx={{ mb: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
                <TextField
                    label="Filter by Date (YYYY-MM-DD)"
                    type="date"
                    value={dateFilter}
                    onChange={e => setDateFilter(e.target.value)}
                    InputLabelProps={{ shrink: true }}
                />
                <Button variant="contained" color="primary" onClick={fetchHistory} sx={{ fontWeight: 'bold', borderRadius: 2 }}>
                    Filter
                </Button>
                <Button variant="outlined" color="secondary" onClick={() => setDateFilter('')} sx={{ fontWeight: 'bold', borderRadius: 2 }}>
                    Clear
                </Button>
            </Box>
            {loading ? (
                <Typography>Loading...</Typography>
            ) : history.length === 0 ? (
                <Typography>No uploads found for this date.</Typography>
            ) : (
                <Grid container spacing={3}>
                    {history.map((item) => {
                        let parsed = {};
                        try {
                            parsed = JSON.parse(item.report);
                        } catch {
                            parsed = { full_report: item.report };
                        }
                        const status = parsed.analysis_json?.overall_assessment?.status || parsed.overall || 'Unknown';
                        const compliance = parsed.analysis_json?.overall_assessment?.overall_compliance || parsed.image_score || 'N/A';
                        const violations = parsed.analysis_json?.frame_analysis?.violation_categories?.join(', ') || parsed.image_issues || 'None';
                        const frames = parsed.analysis_json?.frame_analysis?.total_frames || parsed.frames_analyzed || 'N/A';
                        const audioIssues = parsed.analysis_json?.audio_analysis?.policy_flags ? Object.keys(parsed.analysis_json.audio_analysis.policy_flags).join(', ') : parsed.audio_issues || 'None';
                        let color = 'default';
                        let icon = null;
                        if (status === 'compliant' || status === 'Safe' || status === 'conforme') { color = 'success'; icon = <CheckCircle />; }
                        else if (status === 'Violation' || status === 'non_conforme' || status === 'major_violations') { color = 'error'; icon = <ErrorIcon />; }
                        else if (status === 'Suggestive' || status === 'attention' || status === 'minor_violations') { color = 'warning'; icon = <Warning />; }
                        return (
                            <Grid item xs={12} sm={6} md={4} lg={3} key={item.id}>
                                <Card sx={{ backgroundColor: '#f9fbe7', boxShadow: 2, borderRadius: 3 }}>
                                    <CardContent>
                                        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                                            <Box>
                                                <Typography variant="subtitle1"><b>Filename:</b> {item.filename}</Typography>
                                                <Typography variant="body2" color="text.secondary"><b>Uploaded:</b> {item.upload_time}</Typography>
                                            </Box>
                                            <Chip icon={icon} label={status} color={color} variant="filled" sx={{ fontWeight: 'bold', fontSize: '1rem' }} />
                                        </Box>
                                        <Typography variant="body2"><b>Compliance:</b> {compliance}%</Typography>
                                        <Typography variant="body2"><b>Violations:</b> {violations}</Typography>
                                        <Typography variant="body2"><b>Frames Analyzed:</b> {frames}</Typography>
                                        <Typography variant="body2"><b>Audio Issues:</b> {audioIssues}</Typography>
                                    </CardContent>
                                </Card>
                            </Grid>
                        );
                    })}
                </Grid>
            )}
        </Container>
    );
}

export default UploadHistoryPage;
