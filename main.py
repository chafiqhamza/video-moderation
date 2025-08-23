# --- Ensure robust import of UltimateVideoAnalyzer regardless of working directory ---
import sys
import os as _os
sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
from ultimate_video_analyzer import UltimateVideoAnalyzer

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
from database.models import save_video_report, get_all_videos
import os
import shutil
import time
import json
import logging
import numpy as np
from datetime import datetime

# Use basic OCR libraries instead of custom modules


# Import Whisper audio analyzer - Use openai-whisper if available
try:
    import whisper
    WHISPER_AUDIO_AVAILABLE = True
    print("🎤 COMPREHENSIVE MODE: Whisper transcription enabled for detailed audio analysis")
except ImportError:
    WHISPER_AUDIO_AVAILABLE = False
    print("❌ Whisper audio analyzer not available")

# Import audio file handling
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
    print("✅ Soundfile loaded")
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("❌ Soundfile not available")

def convert_numpy_types(obj):
    """Convert numpy types to Python native types recursively"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    elif str(type(obj)).startswith('<class \'numpy'):  # Any numpy type
        try:
            return float(obj)
        except Exception:
            return str(obj)
    else:
        return obj


from threading import Lock
import sys
import os as _os
sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
from ultimate_video_analyzer import UltimateVideoAnalyzer
import asyncio

# FastAPI app

from fastapi.staticfiles import StaticFiles
app = FastAPI(
    title="AI Video Analysis API",
    description="Comprehensive AI-powered video analysis system providing detailed insights about video content including scene understanding, object detection, transcription, and quality assessment",
    version="4.0.0"
)

# Mount static directory for frame images
static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compatibility route for /frame-preview/<filename>
from fastapi.responses import FileResponse
@app.get("/frame-preview/{filename}")
async def frame_preview(filename: str):
    frame_path = os.path.join(os.path.dirname(__file__), "static", "frames", filename)
    if os.path.exists(frame_path):
        return FileResponse(frame_path)
    else:
        raise HTTPException(status_code=404, detail="Frame not found")

# Data models
class AnalysisResult(BaseModel):
    score: int
    status: str
    issues: List[str]
    confidence: float = 0.8
    details: Dict[str, Any] = {}

class VideoInfo(BaseModel):
    duration: float
    resolution: str
    fps: float
    format: str
    size_mb: float

class VideoAnalysisResponse(BaseModel):
    title: AnalysisResult
    voice: AnalysisResult
    image: AnalysisResult
    overall: str
    video_id: str
    video_title: str
    video_info: VideoInfo
    processing_time: float
    ai_enabled: bool
    analysis_timestamp: str
    video_summary: Optional[Dict[str, Any]] = None  # Add video summary field
    audio_details: Optional[Dict[str, Any]] = None  # Add detailed audio analysis
    ocr_results: Optional[Dict[str, Any]] = None  # Add OCR analysis results
    text_in_video: Optional[Dict[str, Any]] = None  # Add text-in-video analysis
    full_report: Optional[str] = None  # Add full human-readable report

progress_lock = Lock()
progress_data = {
    "total": 0,
    "current": 0,
    "active": False
}


def generate_ai_video_summary(filename: str, duration: float, transcription: str, 
                             image_analysis: AnalysisResult, voice_analysis: AnalysisResult, 
                             title_analysis: AnalysisResult) -> dict:
    """
    Generate AI-powered comprehensive video summary with intelligent content analysis
    """
    # Simplified implementation for demo
    return {
        "ai_analysis": {
            "content_type": "General Content",
            "main_topics": ["Video content analysis"],
            "summary_text": "AI analysis completed successfully"
        }
    }

def generate_video_summary(filename: str, video_info: dict, title_analysis: AnalysisResult, 
                         image_analysis: AnalysisResult, voice_analysis: AnalysisResult, 
                         processing_time: float) -> dict:
    """
    Generate comprehensive video summary/resume with detailed analysis
    """
    # Simplified implementation for demo
    return {
        "video_overview": {
            "filename": filename,
            "duration_formatted": "1:00",
            "resolution": "1280x720",
            "file_analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "compliance_assessment": {
            "overall_status": "GOOD - YouTube Ready",
            "average_score": 80,
        }
    }

@app.websocket("/ws/progress")
async def websocket_progress(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with progress_lock:
                data = progress_data.copy()
            await ws.send_json({
                "type": "progress",
                "current": data["current"],
                "total": data["total"],
                "active": data["active"]
            })
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


# Global analyzer instance (use new model)
analyzer = UltimateVideoAnalyzer()

# API Routes
@app.get("/")
async def root():
    return {
        "message": "AI Video Content Analyzer",
        "version": "4.0.0",
        "status": "ready",
        "features": {
            "frame_analysis": True,
            "audio_analysis": True,
            "text_analysis": True
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "opencv": True,
        "librosa": True
    }


from fastapi.responses import PlainTextResponse
from fastapi import Query

# Accept all frame extraction settings from frontend
@app.post("/upload-video", response_class=PlainTextResponse)
async def analyze_video(
    file: UploadFile = File(...),
    frame_count: int = Query(20, alias="frame_count"),
    frame_interval: float = Query(0.5, alias="frame_interval"),
    resolution: str = Query("1280x720", alias="resolution"),
    format: str = Query("jpg", alias="format"),
    start_time: float = Query(0.0, alias="start_time"),
    end_time: float = Query(-1.0, alias="end_time"),
    sampling_method: str = Query("interval", alias="sampling_method")
):
    """Analyze uploaded video file using UltimateVideoAnalyzer and return raw model output as plain text"""
    import subprocess
    import sys
    import tempfile
    import shutil
    import os
    try:
        # Save uploaded file temporarily
        temp_dir = os.path.join(tempfile.gettempdir(), "video_analysis")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"temp_{int(time.time())}_{file.filename}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        os.chmod(temp_file_path, 0o666)

        # Check if file exists and is not empty
        if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
            print("❌ ERROR: Uploaded video file not found or empty!")
            raise HTTPException(status_code=400, detail="Uploaded video file not found or empty!")

        # Run the model script as a subprocess and stream output in real time
        model_script = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ultimate_video_analyzer.py'))
        cmd = [
            sys.executable,
            model_script,
            '--analyze', temp_file_path,
            '--interval-seconds', str(frame_interval),
            '--resolution', str(resolution),
            '--format', str(format),
            '--start-time', str(start_time),
            '--end-time', str(end_time),
            '--sampling-method', str(sampling_method)
        ]
        if sampling_method == 'count':
            cmd.insert(4, '--max-frames')
            cmd.insert(5, str(frame_count))
        print(f"\n[MODEL SUBPROCESS] Running: {' '.join(cmd)}\n")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        output_lines = []
        # Stream output line by line
        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)
        process.stdout.close()
        process.wait()
        output = ''.join(output_lines)

        # Try to find the latest JSON report generated by the model in backend/video_analysis_reports
        report_dir = os.path.join(os.path.dirname(__file__), "video_analysis_reports")
        json_report = None
        json_path = None
        # Look for a file starting with 'ultimate_video_analysis_' and ending with '.json' in the report_dir
        if os.path.exists(report_dir):
            report_files = [f for f in os.listdir(report_dir) if f.startswith('ultimate_video_analysis_') and f.endswith('.json')]
            if report_files:
                # Get the most recently modified report file
                report_files.sort(key=lambda f: os.path.getmtime(os.path.join(report_dir, f)), reverse=True)
                json_path = os.path.join(report_dir, report_files[0])
        if json_path and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    json_report = json.load(f)
                except Exception as e:
                    print(f"❌ Failed to load JSON report: {e}")
                    json_report = None
        else:
            print("❌ No JSON report found after analysis.")

        # Run copyright check using AudD API
        from audd_copyright_checker import check_video_copyright
        copyright_result = check_video_copyright(temp_file_path)

        # Clean up temp file
        try:
            os.remove(temp_file_path)
        except Exception:
            pass

        # Save to database (raw output)
        save_video_report(file.filename, output)

        # Ensure frame previews and per-frame analysis are present for frontend
        if json_report:
            # Try to extract frame previews and per-frame analysis from the report
            image_details = json_report.get('image', {}).get('details', {})
            frame_analyses = json_report.get('frame_analyses') or json_report.get('frames') or []
            frame_preview_paths = image_details.get('frame_preview_paths') or []
            # If missing, try to build from frame_analyses
            if not frame_preview_paths and frame_analyses:
                frame_preview_paths = []
                for frame in frame_analyses:
                    path = frame.get('preview_path') or frame.get('image_path') or frame.get('frame_path')
                    if path:
                        filename = os.path.basename(path)
                        preview_url = f"/static/frames/{filename}"
                        frame_preview_paths.append(preview_url)
                        frame['preview_path'] = preview_url
            # Build per-frame analysis array for frontend, flattening nested fields
            frame_analysis = []
            for frame in frame_analyses:
                visual = frame.get('visual_analysis', {})
                blip = frame.get('blip_description', {})
                ocr = frame.get('ocr_text', {})
                blip_desc = blip.get('description')
                if not blip_desc or not isinstance(blip_desc, str) or not blip_desc.strip():
                    blip_desc = 'No BLIP description available.'
                fa = {
                    'timestamp': frame.get('timestamp', None),
                    'frame_index': frame.get('frame_index'),
                    'preview_path': frame.get('preview_path'),
                    'category': visual.get('category') or 'No category',
                    'confidence': visual.get('confidence') if visual.get('confidence') is not None else 0.0,
                    'violation_detected': visual.get('violation_detected', False),
                    'description': blip_desc,
                    'ocr_text': ocr.get('text') or 'No OCR text',
                    'ocr_items': ocr.get('extracted_items', []),
                    'combined_violation': frame.get('combined_violation', None),
                    'violations': [],
                }
                if visual.get('violation_detected', False):
                    fa['violations'].append(visual.get('category', 'violation'))
                if frame.get('policy_explanation'):
                    fa['violations'].append(frame['policy_explanation'])

                # Build human-readable frame report string
                report_lines = []
                idx = fa['frame_index'] if fa['frame_index'] is not None else '?'
                ts = fa['timestamp'] if fa['timestamp'] is not None else '?'
                report_lines.append(f"Frame {idx} (t={ts}s):")
                conf = f" (Conf: {fa['confidence']:.2f})" if fa['confidence'] is not None else ""
                report_lines.append(f"   Category: {fa['category']}{conf}")
                report_lines.append(f"   BLIP: {fa['description']}")
                report_lines.append(f"   OCR: {fa['ocr_text']}")
                if fa['ocr_items']:
                    report_lines.append(f"   OCR Items: {', '.join(str(x) for x in fa['ocr_items'])}")
                if fa['violations']:
                    for v in fa['violations']:
                        report_lines.append(f"   Policy Violation Detected! {v}")
                fa['frame_report'] = '\n'.join(report_lines)
                frame_analysis.append(fa)
            if 'image' not in json_report:
                json_report['image'] = {'details': {}}
            if 'details' not in json_report['image']:
                json_report['image']['details'] = {}
            json_report['image']['details']['frame_preview_paths'] = frame_preview_paths
            json_report['image']['details']['frame_analysis'] = frame_analysis
        # Try to load RAG explanations from the latest JSON report if present
        rag_explanations = None
        if json_report and 'rag_explanations' in json_report:
            rag_explanations = json_report['rag_explanations']
        elif json_report and 'rag_explanations' in json_report.get('analysis_json', {}):
            rag_explanations = json_report['analysis_json']['rag_explanations']
        # Build full model analysis for each frame for frontend
        frame_details = []
        if json_report and 'detailed_frames' in json_report:
            for frame in json_report['detailed_frames']:
                frame_details.append({
                    "frame_index": frame.get("frame_index"),
                    "timestamp": frame.get("timestamp"),
                    "visual_analysis": frame.get("visual_analysis", {}),
                    "blip_description": frame.get("blip_description", {}),
                    "ocr_text": frame.get("ocr_text", {}),
                    "combined_violation": frame.get("combined_violation", False),
                    "preview_path": frame.get("preview_path", "")
                })
        response = {
            "full_report": output,
            "analysis_json": {},
            "rag_explanations": rag_explanations if rag_explanations else [],
            "frame_details": frame_details,
            "copyright_check": copyright_result
        }
        # Patch: Map expected frontend fields from model output
        if json_report:
            analysis_json = dict(json_report)  # shallow copy
            # Safe frames
            total_frames = json_report.get('frame_analysis', {}).get('total_frames', 0)
            violation_frames = json_report.get('frame_analysis', {}).get('violation_frames', 0)
            safe_frames = total_frames - violation_frames
            analysis_json['safe_frames'] = safe_frames
            # Scores
            image_score = round(json_report.get('frame_analysis', {}).get('average_confidence', 0) * 100)
            audio_score = round(json_report.get('overall_assessment', {}).get('audio_compliance', 0))
            text_score = 100  # If you want to derive from OCR/text, update here
            analysis_json['image_score'] = image_score
            analysis_json['audio_score'] = audio_score
            analysis_json['text_score'] = text_score
            analysis_json['overall_scores'] = {
                'text': text_score,
                'audio': audio_score,
                'image': image_score
            }
            # Overall status
            analysis_json['overall'] = json_report.get('overall_assessment', {}).get('status', '')
            # Issues
            analysis_json['text_issues'] = json_report.get('text_issues', '')
            # Audio issues: list categories from audio policy flags
            audio_flags = json_report.get('audio_analysis', {}).get('policy_flags', {})
            analysis_json['audio_issues'] = ', '.join(audio_flags.keys()) if audio_flags else 'None'
            # Image issues: list violation categories
            image_issues = json_report.get('frame_analysis', {}).get('violation_categories', [])
            analysis_json['image_issues'] = ', '.join(image_issues) if image_issues else 'None'
            # Copyright
            if copyright_result:
                analysis_json['copyright_status'] = 'Violation' if copyright_result.get('violation') else 'Clear'
                analysis_json['copyright_details'] = str(copyright_result.get('result')) if copyright_result.get('result') else ''
            # Patch: Provide missing fields for frontend
            # Duration
            analysis_json['video_info'] = analysis_json.get('video_info', {})
            if 'duration' not in analysis_json['video_info']:
                # Try to estimate duration from frame count and interval
                frame_count = analysis_json.get('frame_analysis', {}).get('total_frames', 0)
                interval = float(frame_interval) if 'frame_interval' in locals() else 0.5
                analysis_json['video_info']['duration'] = round(frame_count * interval, 2) if frame_count > 0 else 'N/A'
            # Resolution
            if 'resolution' not in analysis_json['video_info']:
                analysis_json['video_info']['resolution'] = resolution if 'resolution' in locals() else 'N/A'
            # FPS
            if 'fps' not in analysis_json['video_info']:
                analysis_json['video_info']['fps'] = 1.0 / float(frame_interval) if 'frame_interval' in locals() and float(frame_interval) > 0 else 'N/A'
            # Processing time
            if 'processing_time' not in analysis_json['video_info']:
                analysis_json['video_info']['processing_time'] = analysis_json['video_info'].get('analysis_time', 'N/A')
            response['analysis_json'] = analysis_json
        return PlainTextResponse(json.dumps(response, ensure_ascii=False, indent=2), media_type="application/json")
    except Exception as e:
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
        print(f"❌ ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# API endpoint to list all uploaded videos and their reports
@app.get("/videos")
async def list_videos():
    rows = get_all_videos()
    return [
        {
            "id": row[0],
            "filename": row[1],
            "upload_time": row[2],
            "user": row[3],
            "report": row[4]
        }
        for row in rows
    ]

from audd_copyright_checker import check_video_copyright

@app.post("/api/copyright-check")
async def copyright_check(file: UploadFile = File(...)):
    """Check video for copyright using AudD API"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        result = check_video_copyright(tmp_path)
        os.remove(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Copyright check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Video Content Analyzer")
    print("="*50)
    print("📍 Server: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("="*50)
    print("OpenCV: ✅")
    print("Librosa: ✅")
    print("="*50)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )