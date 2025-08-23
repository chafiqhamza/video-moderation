#!/usr/bin/env python3
from rag_content_supervision import ContentModerationRAG
"""
🎬 ULTIMATE VIDEO POLICY ANALYZER
Complete multi-modal video analysis using your 97.48% accuracy model + BLIP + Whisper + OCR + RAG
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import tempfile
import os
from typing import Dict, List, Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time


try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper not available - audio analysis disabled")
    # Removed the invalid line that caused errors
    # results = await analyzer.analyze_video_comprehensive(video_path, frames_per_second=args.interval_seconds, max_frames=args.max_frames)


try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("⚠️ BLIP not available - frame descriptions disabled")

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ EasyOCR not available - text extraction disabled")

from PIL import Image
from torchvision import transforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltimateVideoAnalyzer:
    """Ultimate multi-modal video policy analyzer"""
    
    def __init__(self, model_path="ULTIMATE_youtube_policy_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🚀 INITIALIZING ULTIMATE VIDEO ANALYZER")
        logger.info(f"📱 Device: {self.device}")
        
        # Load your trained ultimate model
        self.visual_model, self.class_names = self._load_visual_model(model_path)
        
        # Initialize multi-modal components
        self.blip_model = self._load_blip_model()
        self.whisper_model = self._load_whisper_model()
        self.ocr_reader = self._load_ocr_model()
        
        # Policy knowledge base
        self.policy_knowledge = self._initialize_policy_kb()
        
        logger.info("✅ ULTIMATE VIDEO ANALYZER READY!")
        self._log_capabilities()
    
    def _load_visual_model(self, model_path: str) -> Tuple[Optional[nn.Module], List[str]]:
        """Load your 97.48% accuracy ultimate model"""
        try:
            from ultimate_dataset_trainer import UltimateModel
            if not Path(model_path).exists():
                logger.error(f"❌ Model not found: {model_path}")
                return None, []
            checkpoint = torch.load(model_path, map_location=self.device)
            class_names = checkpoint.get('class_names', [])
            model = UltimateModel(num_classes=len(class_names)).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model.eval()
            accuracy = checkpoint.get('best_val_accuracy', 0)
            logger.info(f"✅ Visual model loaded: {accuracy:.2f}% accuracy, {len(class_names)} classes")
            return model, class_names
        except Exception as e:
            logger.error(f"❌ Error loading visual model: {e}")
            return None, []
    
    def _load_blip_model(self):
        """Load BLIP for frame descriptions"""
        if not BLIP_AVAILABLE:
            return None
        
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            model.to(self.device)
            
            logger.info("✅ BLIP model loaded")
            return {"processor": processor, "model": model}
        except Exception as e:
            logger.error(f"❌ Error loading BLIP: {e}")
            return None
    
    def _load_whisper_model(self):
        """Load Whisper for audio analysis"""
        if not WHISPER_AVAILABLE:
            return None
        
        try:
            model = whisper.load_model("base")
            logger.info("✅ Whisper model loaded")
            return model
        except Exception as e:
            logger.error(f"❌ Error loading Whisper: {e}")
            return None
    
    def _load_ocr_model(self):
        """Load OCR for text extraction"""
        if not OCR_AVAILABLE:
            return None
        
        try:
            reader = easyocr.Reader(['en'])
            logger.info("✅ OCR model loaded")
            return reader
        except Exception as e:
            logger.error(f"❌ Error loading OCR: {e}")
            return None
    
    def _initialize_policy_kb(self):
        """Initialize policy knowledge base"""
        return {
            "violation_keywords": {
                "violence": ["weapon", "gun", "knife", "blood", "fight", "attack", "kill", "murder", "war"],
                "hate_speech": ["hate", "racist", "nazi", "supremacist", "discrimination", "slur"],
                "dangerous_activities": ["suicide", "self-harm", "dangerous", "stunt", "risk", "harmful"],
                "misinformation": ["fake", "conspiracy", "hoax", "false", "misleading", "debunked"],
                "harassment": ["bully", "harass", "threaten", "intimidate", "stalk", "abuse"],
                "spam_scam": ["scam", "fraud", "fake", "phishing", "spam", "scheme"]
            },
            "severity_levels": {
                "critical": 0.95,
                "high": 0.85,
                "medium": 0.70,
                "low": 0.50
            }
        }
    
    def _log_capabilities(self):
        """Log available capabilities"""
        capabilities = []
        if self.visual_model: capabilities.append("🎯 Visual Policy Detection (97.48% accuracy)")
        if self.blip_model: capabilities.append("🤖 BLIP Frame Descriptions")
        if self.whisper_model: capabilities.append("🎵 Whisper Audio Analysis")
        if self.ocr_reader: capabilities.append("📝 OCR Text Extraction")
        capabilities.append("🧠 RAG Policy Knowledge")
        capabilities.append("🎬 Video Frame Processing")
        
        logger.info(f"🎯 CAPABILITIES: {len(capabilities)} modules loaded")
        for cap in capabilities:
            logger.info(f"   {cap}")
    
    def extract_video_frames(self, video_path: str, interval_seconds: float = 1.0, max_frames: int = None,
                            resolution: str = "1280x720", frame_format: str = "jpg",
                            start_time: float = 0.0, end_time: float = -1.0, sampling_method: str = "interval") -> List[Tuple[float, np.ndarray]]:
        """Extract frames from video using all user settings"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            # Calculate start/end frames
            start_frame = int(start_time * fps) if start_time > 0 else 0
            end_frame = int(end_time * fps) if end_time > 0 else total_frames
            logger.info(f"🎬 Extracting frames: method={sampling_method}, interval={interval_seconds}s, max={max_frames}, resolution={resolution}, format={frame_format}, start={start_time}, end={end_time}")
            frames = []
            # Parse resolution
            if resolution.lower() == 'auto':
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                try:
                    width, height = map(int, resolution.lower().split('x'))
                except Exception:
                    width, height = 1280, 720
            # Support both interval and count extraction modes
            frames = []
            actual_end = duration if end_time < 0 else end_time
            if sampling_method == "count" and max_frames is not None and max_frames > 0:
                # Evenly spaced timestamps for exactly max_frames
                if max_frames == 1:
                    timestamps = [start_time]
                else:
                    timestamps = [start_time + i * (actual_end - start_time) / (max_frames - 1) for i in range(max_frames)]
                for ts in timestamps:
                    frame_idx = int(ts * fps)
                    if frame_idx < start_frame or frame_idx >= end_frame:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.resize(frame, (width, height))
                    frames.append((ts, frame))
            else:
                # Interval mode (default)
                timestamps = np.arange(start_time, actual_end, interval_seconds)
                for ts in timestamps:
                    frame_idx = int(ts * fps)
                    if frame_idx < start_frame or frame_idx >= end_frame:
                        continue
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.resize(frame, (width, height))
                    frames.append((ts, frame))
            cap.release()
            logger.info(f"✅ Extracted {len(frames)} frames")
            return frames
        except Exception as e:
            logger.error(f"❌ Error extracting frames: {e}")
            return []
    
    def analyze_frame_visual(self, frame: np.ndarray) -> Dict:
        """Analyze frame with your 97.48% accuracy model"""
        if not self.visual_model:
            return {"error": "Visual model not available"}
        
        try:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Preprocess
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                output = self.visual_model(image_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            category = self.class_names[predicted.item()]
            confidence_score = float(confidence.item())
            
            return {
                "category": category,
                "confidence": confidence_score,
                "violation_detected": category != "safe_content",
                "all_probabilities": {
                    self.class_names[i]: float(probabilities[0][i].item()) 
                    for i in range(len(self.class_names))
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Visual analysis error: {e}")
            return {"error": str(e)}
    
    def analyze_frame_blip(self, frame: np.ndarray) -> Dict:
        """Generate BLIP description for frame"""
        if not self.blip_model:
            return {"description": "BLIP not available"}
        
        try:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Generate description
            inputs = self.blip_model["processor"](image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model["model"].generate(**inputs, max_length=50)
                description = self.blip_model["processor"].decode(out[0], skip_special_tokens=True)
            
            return {
                "description": description,
                "blip_available": True
            }
            
        except Exception as e:
            logger.error(f"❌ BLIP analysis error: {e}")
            return {"description": f"BLIP error: {e}", "blip_available": False}
    
    def analyze_frame_ocr(self, frame: np.ndarray) -> Dict:
        """Extract text from frame using OCR"""
        if not self.ocr_reader:
            return {"text": "", "ocr_available": False}
        
        try:
            # OCR analysis
            results = self.ocr_reader.readtext(frame)
            
            # Extract text and confidence
            extracted_texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence
                    extracted_texts.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox
                    })
            
            full_text = " ".join([item["text"] for item in extracted_texts])
            
            return {
                "text": full_text,
                "extracted_items": extracted_texts,
                "ocr_available": True
            }
            
        except Exception as e:
            logger.error(f"❌ OCR analysis error: {e}")
            return {"text": "", "ocr_available": False, "error": str(e)}
    
    def analyze_audio_whisper(self, video_path: str) -> Dict:
        """Analyze audio with Whisper"""
        if not self.whisper_model:
            logger.error("❌ Whisper model not available.")
            return {"transcript": "Whisper not available", "whisper_available": False}
        try:
            logger.info("🎵 Analyzing audio with Whisper...")
            
            # Transcribe audio
            result = self.whisper_model.transcribe(video_path)
            
            transcript = result.get("text", "")
            language = result.get("language", "unknown")
            
            # Analyze transcript for policy violations
            policy_flags = self._analyze_text_for_violations(transcript)
            logger.info(f"Whisper transcript: {transcript}")
            logger.info(f"Whisper policy flags: {policy_flags}")
            
            return {
                "transcript": transcript,
                "language": language,
                "policy_flags": policy_flags,
                "whisper_available": True,
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            logger.error(f"❌ Whisper analysis error: {e}")
            return {"transcript": "", "whisper_available": False, "error": str(e)}
    
    def _analyze_text_for_violations(self, text: str) -> Dict:
        """Analyze text for policy violations using keyword matching and bad word detection"""
        if not text:
            return {"safe": {"keywords_found": [], "severity": "none"}}

        text_lower = text.lower()
        violations = {}

        # Comprehensive bad word list
        bad_words = [
            "badword1", "badword2", "hate", "violence", "explicit", "offensive", "racist", "sexist", "drugs", "weapon", "kill", "shoot", "bomb", "terror", "abuse", "adult", "nsfw", "profanity", "curse", "swear", "misinformation", "suggestive", "inappropriate"
        ]
        found_bad_words = [word for word in bad_words if word in text_lower]
        if found_bad_words:
            violations["bad_words"] = {
                "keywords_found": found_bad_words,
                "severity": "high" if len(found_bad_words) > 2 else "medium"
            }

        for category, keywords in self.policy_knowledge["violation_keywords"].items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                violations[category] = {
                    "keywords_found": found_keywords,
                    "severity": "high" if len(found_keywords) > 2 else "medium"
                }

        # If no violations or bad words, mark as safe
        if not violations:
            violations["safe"] = {"keywords_found": [], "severity": "none"}

        return violations
    
    async def analyze_video_comprehensive(self, video_path: str, frames_per_second: float = 1.0, max_frames: int = None,
                                         resolution: str = None, frame_format: str = None,
                                         start_time: float = None, end_time: float = None, sampling_method: str = None) -> Dict:
        """Comprehensive multi-modal video analysis"""
        analysis_start_time = time.time()
        logger.info(f"🎬 Starting comprehensive analysis: {Path(video_path).name}")
        # Use the correct start_time and end_time from arguments, not analysis_start_time
        frames = self.extract_video_frames(
            video_path,
            interval_seconds=frames_per_second,
            max_frames=max_frames,
            resolution=resolution,
            frame_format=frame_format,
            start_time=start_time,
            end_time=end_time,
            sampling_method=sampling_method
        )

        if not frames:
            return {"error": "No frames extracted"}

        # Prepare directory for frame images in the correct static folder (backend/static/frames)
        static_dir = Path(__file__).parent / "static" / "frames"
        static_dir.mkdir(parents=True, exist_ok=True)
        video_stem = Path(video_path).stem

        # Parallel audio analysis
        audio_task = asyncio.create_task(asyncio.to_thread(self.analyze_audio_whisper, video_path))

        # Analyze frames and save images
        frame_analyses = []
        frame_preview_paths = []
        frame_img_paths = []
        logger.info(f"🖼️ Analyzing {len(frames)} frames...")

        for i, (timestamp, frame) in enumerate(frames):
            logger.info(f"   📊 Processing frame {i+1}/{len(frames)} (t={timestamp:.1f}s)")

            # Save frame as image
            frame_img_path = static_dir / f"{video_stem}_frame_{i+1}.jpg"
            try:
                import cv2
                cv2.imwrite(str(frame_img_path), frame)
                # For frontend: use relative path from static
                preview_url = f"/static/frames/{video_stem}_frame_{i+1}.jpg"
                frame_preview_paths.append(preview_url)
                frame_img_paths.append(str(frame_img_path))
            except Exception as e:
                logger.error(f"❌ Error saving frame image: {e}")
                frame_preview_paths.append("")

            # Multi-modal frame analysis
            visual_result = self.analyze_frame_visual(frame)
            blip_result = self.analyze_frame_blip(frame)
            ocr_result = self.analyze_frame_ocr(frame)

            # Combine results
            frame_analysis = {
                "timestamp": timestamp,
                "frame_index": i,
                "visual_analysis": visual_result,
                "blip_description": blip_result,
                "ocr_text": ocr_result,
                "combined_violation": visual_result.get("violation_detected", False),
                "preview_path": preview_url
            }

            frame_analyses.append(frame_analysis)

        # Wait for audio analysis with error handling
        logger.info("🎵 Waiting for audio analysis...")
        audio_result = None
        try:
            audio_result = await audio_task
        except asyncio.CancelledError:
            logger.error("❌ Audio analysis task was cancelled. Proceeding with empty audio result.")
            audio_result = {"transcript": "", "whisper_available": False, "policy_flags": {}}
        except Exception as e:
            logger.error(f"❌ Audio analysis task failed: {e}. Proceeding with empty audio result.")
            audio_result = {"transcript": "", "whisper_available": False, "policy_flags": {}}

        # Comprehensive analysis
        analysis_time = time.time() - analysis_start_time
        comprehensive_result = self._generate_comprehensive_assessment(
            frame_analyses, audio_result, video_path, analysis_time
        )
        # Ensure audio_analysis is always present
        if "audio_analysis" not in comprehensive_result:
            comprehensive_result["audio_analysis"] = audio_result
        # Add frame preview paths to report for frontend
        if "image" not in comprehensive_result:
            comprehensive_result["image"] = {"details": {}}
        if "details" not in comprehensive_result["image"]:
            comprehensive_result["image"]["details"] = {}
        comprehensive_result["image"]["details"]["frame_preview_paths"] = frame_preview_paths
        comprehensive_result["image"]["details"]["frame_analysis"] = frame_analyses
        # Defensive: ensure text_issues is always a string for frontend compatibility
        text_issues = comprehensive_result.get("text_issues", "")
        if isinstance(text_issues, list):
            comprehensive_result["text_issues"] = ", ".join(str(x) for x in text_issues)
        elif not isinstance(text_issues, str):
            comprehensive_result["text_issues"] = str(text_issues)
        # Do NOT delete frame images immediately; keep them for frontend access
        # You can implement scheduled cleanup or a manual endpoint for deletion if needed
        logger.info(f"✅ Comprehensive analysis complete ({analysis_time:.1f}s)")

        return comprehensive_result
    
    def _generate_comprehensive_assessment(self, frame_analyses: List, audio_result: Dict, video_path: str, analysis_time: float) -> Dict:
        """Generate final comprehensive assessment"""
        
        # Aggregate frame results
        total_frames = len(frame_analyses)
        violation_frames = sum(1 for f in frame_analyses if f["combined_violation"])
        
        # Calculate confidence scores
        visual_confidences = [f["visual_analysis"].get("confidence", 0) for f in frame_analyses if "visual_analysis" in f]
        avg_confidence = np.mean(visual_confidences) if visual_confidences else 0
        
        # Collect all categories
        all_categories = []
        for frame in frame_analyses:
            if "visual_analysis" in frame and "category" in frame["visual_analysis"]:
                all_categories.append(frame["visual_analysis"]["category"])
        
        violation_categories = list(set([cat for cat in all_categories if cat != "safe_content"]))
        
        # Audio violations
        audio_violations = audio_result.get("policy_flags", {})
        
        # Overall compliance calculation
        visual_compliance = (total_frames - violation_frames) / total_frames * 100 if total_frames > 0 else 100
        audio_compliance = 90 if not audio_violations else max(50, 90 - len(audio_violations) * 20)
        overall_compliance = (visual_compliance + audio_compliance) / 2
        
        # Status determination
        if overall_compliance >= 95:
            status = "compliant"
        elif overall_compliance >= 80:
            status = "minor_violations"
        elif overall_compliance >= 60:
            status = "moderate_violations"
        else:
            status = "major_violations"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violation_categories, audio_violations, overall_compliance)
        
        # Collect text issues from OCR results in frames
        text_issues = []
        for frame in frame_analyses:
            ocr = frame.get("ocr_text", {})
            text = ocr.get("text", "")
            if text:
                text_issues.append(text)
        # Optionally, join as a string for frontend compatibility
        text_issues_str = ", ".join(text_issues) if text_issues else ""

        return {
            "video_info": {
                "path": str(video_path),
                "name": Path(video_path).name,
                "analysis_time": analysis_time,
                "timestamp": datetime.now().isoformat()
            },
            "frame_analysis": {
                "total_frames": total_frames,
                "violation_frames": violation_frames,
                "average_confidence": avg_confidence,
                "violation_categories": violation_categories
            },
            "audio_analysis": audio_result,
            "overall_assessment": {
                "overall_compliance": overall_compliance,
                "visual_compliance": visual_compliance,
                "audio_compliance": audio_compliance,
                "status": status,
                "total_violations": violation_frames + len(audio_violations),
                "violation_categories": violation_categories + list(audio_violations.keys())
            },
            "detailed_frames": frame_analyses,
            "recommendations": recommendations,
            "multi_modal_capabilities": {
                "visual_model_accuracy": 97.48,
                "blip_available": self.blip_model is not None,
                "whisper_available": self.whisper_model is not None,
                "ocr_available": self.ocr_reader is not None
            },
            "text_issues": text_issues_str
        }
    
    def _generate_recommendations(self, visual_violations: List, audio_violations: Dict, compliance: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if compliance >= 95:
            recommendations.append("✅ Content is compliant with YouTube policies")
        
        if visual_violations:
            recommendations.append(f"⚠️ Visual policy violations detected: {', '.join(visual_violations)}")
            recommendations.append("🔍 Consider reviewing flagged video segments")
        
        if audio_violations:
            recommendations.append(f"🎵 Audio policy violations detected: {', '.join(audio_violations.keys())}")
            recommendations.append("📝 Review transcript for policy-violating language")
        
        if compliance < 80:
            recommendations.append("🚨 Multiple violations detected - content may need significant editing")
        elif compliance < 95:
            recommendations.append("⚡ Minor violations detected - targeted edits recommended")
        
        if not visual_violations and not audio_violations:
            recommendations.append("🎉 No violations detected - content ready for upload")
        
        return recommendations
    
    def save_analysis_report(self, analysis_result: Dict) -> str:
        """Save comprehensive analysis report (fix NumPy serialization and sanitize filename)"""
        import numpy as np
        import re
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)
        report_path = None
        try:
            # Defensive: handle missing video_info or name
            video_name = None
            if "video_info" in analysis_result and "name" in analysis_result["video_info"]:
                video_name = Path(analysis_result["video_info"]["name"]).stem
            else:
                # Fallback: use timestamp only
                video_name = "unknown_video"
            video_name_safe = re.sub(r'[^A-Za-z0-9_-]', '_', str(video_name))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            reports_dir = Path(__file__).parent / "video_analysis_reports"
            reports_dir.mkdir(exist_ok=True)
            report_path = reports_dir / f"ultimate_video_analysis_{video_name_safe}_{timestamp}.json"
            logger.info(f"Attempting to save report to: {report_path}")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False, default=convert)
            logger.info(f"📄 Analysis report saved: {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"❌ Error saving report to {report_path}: {e}")
            return ""

import sys
import argparse
def safe_print(text):
    print(text.encode('ascii', 'ignore').decode('ascii'))

def get_args_and_vars():
    parser = argparse.ArgumentParser(description="Ultimate Video Analyzer")
    parser.add_argument('--analyze', type=str, help='Path to video file to analyze')
    parser.add_argument('--max-frames', type=int, required=False, help='Maximum number of frames to analyze (required for count mode, optional for interval mode)')
    parser.add_argument('--interval-seconds', type=float, required=True, help='Interval between frames in seconds')
    parser.add_argument('--resolution', type=str, required=True, help='Frame resolution (e.g., 1280x720, auto)')
    parser.add_argument('--format', type=str, required=True, help='Frame image format (e.g., jpg, png)')
    parser.add_argument('--start-time', type=float, required=True, help='Start time for frame extraction (seconds)')
    parser.add_argument('--end-time', type=float, required=True, help='End time for frame extraction (seconds, -1 for end of video)')
    parser.add_argument('--sampling-method', type=str, required=True, help='Frame sampling method (interval, random, etc.)')
    args = parser.parse_args()

    analyzer = UltimateVideoAnalyzer()
    if not analyzer.visual_model:
        safe_print("❌ Visual model not loaded. Please check your model file.")
        sys.exit(1)
    if args.analyze:
        video_path = args.analyze
    else:
        video_path = input("\nEnter video path to analyze: ").strip()
    if not video_path or not Path(video_path).exists():
        safe_print("❌ Video file not found!")
        sys.exit(1)
    return analyzer, video_path, args

async def main(analyzer, video_path, args):
    """Demo of ultimate video analysis"""
    safe_print("🎬 ULTIMATE VIDEO POLICY ANALYZER")
    safe_print("="*70)
    safe_print("🚀 Multi-modal analysis with 97.48% accuracy model + BLIP + Whisper + OCR")
    safe_print("="*70)

    safe_print(f"\n🎬 Starting ultimate analysis of: {Path(video_path).name}")
    safe_print("📊 This will analyze:")
    safe_print("   🎯 Visual content with 97.48% accuracy model")
    safe_print("   🤖 BLIP frame descriptions")
    safe_print("   🎵 Whisper audio transcription")
    safe_print("   📝 OCR text extraction")
    safe_print("   🧠 Multi-modal policy assessment")
    
    # Perform analysis
    start_time = time.time()
    results = await analyzer.analyze_video_comprehensive(
        video_path,
        frames_per_second=args.interval_seconds,
        max_frames=args.max_frames,
        resolution=args.resolution,
        frame_format=args.format,
        start_time=args.start_time,
        end_time=args.end_time,
        sampling_method=args.sampling_method
    )
    total_time = time.time() - start_time
    safe_print("="*50)

    overall = results.get("overall_assessment", {})
    safe_print(f"🎯 Overall Compliance: {overall.get('overall_compliance', 0):.1f}%")
    safe_print(f"📋 Status: {overall.get('status', 'unknown')}")
    safe_print(f"🚨 Total Violations: {overall.get('total_violations', 0)}")

    if overall.get('violation_categories'):
        safe_print(f"🔍 Violation Categories: {', '.join(overall.get('violation_categories', []))}")

    # Audio analysis
    audio = results.get("audio_analysis", {})
    safe_print(f"\n🎵 AUDIO ANALYSIS:")
    if audio.get("whisper_available"):
        transcript = audio.get('transcript', '')
        if transcript:
            safe_print(f"   📝 Full Transcript: {transcript}")
        else:
            safe_print("   📝 Transcript: (none detected)")
        policy_flags = audio.get("policy_flags", {})
        if policy_flags:
            safe_print(f"   🚨 Detected Bad Words / Policy Violations:")
            for category, info in policy_flags.items():
                safe_print(f"      - {category}: {', '.join(info.get('keywords_found', []))} (Severity: {info.get('severity', '')})")
        else:
            safe_print("   ✅ No bad words or policy violations detected in audio.")
    else:
        safe_print("   ❌ Whisper audio analysis not available or failed.")

    # Recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        safe_print(f"\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            safe_print(f"   {rec}")




    # Show frame-by-frame details
    safe_print("\n🖼️ FRAME-BY-FRAME DETAILS:")
    detailed_frames = results.get("detailed_frames", [])
    rag = ContentModerationRAG(model_path="ULTIMATE_youtube_policy_model.pth")
    rag.policy_db_path = "video-moderation-second/database/content_moderation_rag.db"
    violation_explanations = []
    # Always use only positive policies if video is compliant

    # Patch: If ALL frames are classified as 'safe_content', always show only positive policies
    all_safe = all(
        frame.get("visual_analysis", {}).get("category", "") == "safe_content"
        for frame in detailed_frames if "visual_analysis" in frame
    )
    if all_safe:
        positive_categories = [
            "Safe Content",
            "Educational Content",
            "Entertainment Content",
            "News Content",
            "Tutorials & How-To",
            "Community & Family-Friendly",
            "Artistic & Creative Expression",
            "Positive Social Impact"
        ]
        for cat in positive_categories:
            policies = rag.retrieve_relevant_policies(cat, top_k=1)
            for p in policies:
                if p["category"].lower() == cat.lower() and p["policy_info"].get("description"):
                    violation_explanations.append({
                        "type": "safe",
                        "category": cat,
                        "policy": p["policy_info"]
                    })
        if not violation_explanations:
            violation_explanations = [{
                "type": "safe",
                "category": "Safe Content",
                "policy": {
                    "description": "No violations detected. This video complies with all YouTube policies.",
                    "examples": ["educational", "entertainment", "news", "tutorials"]
                }
            }]
    else:
        # Use compliance status as fallback
        overall_status = results.get("overall_assessment", {}).get("status", "unknown")
        if overall_status == "compliant":
            positive_categories = [
                "Safe Content",
                "Educational Content",
                "Entertainment Content",
                "News Content",
                "Tutorials & How-To",
                "Community & Family-Friendly",
                "Artistic & Creative Expression",
                "Positive Social Impact"
            ]
            for cat in positive_categories:
                policies = rag.retrieve_relevant_policies(cat, top_k=1)
                for p in policies:
                    if p["category"].lower() == cat.lower() and p["policy_info"].get("description"):
                        violation_explanations.append({
                            "type": "safe",
                            "category": cat,
                            "policy": p["policy_info"]
                        })
            if not violation_explanations:
                violation_explanations = [{
                    "type": "safe",
                    "category": "Safe Content",
                    "policy": {
                        "description": "No violations detected. This video complies with all YouTube policies.",
                        "examples": ["educational", "entertainment", "news", "tutorials"]
                    }
                }]
        else:
            # Non-compliant: show violations as before
            detected_categories = set()
            for frame in detailed_frames:
                visual = frame.get("visual_analysis", {})
                if visual.get("violation_detected", False):
                    cat = visual.get("category", None)
                    if cat and cat.lower() != "safe_content":
                        detected_categories.add(cat)
            audio = results.get("audio_analysis", {})
            if audio.get("policy_flags"):
                for polcat in audio["policy_flags"].keys():
                    if polcat and polcat.lower() != "safe_content":
                        detected_categories.add(polcat)
            for cat in detected_categories:
                if cat.lower() not in [
                    "safe content", "educational content", "entertainment content", "news content",
                    "tutorials & how-to", "community & family-friendly", "artistic & creative expression", "positive social impact"
                ]:
                    policies = rag.retrieve_relevant_policies(cat, top_k=3)
                    for policy in policies:
                        policy_info = policy["policy_info"] if policy else {}
                        # Build a context-matched summary for each violation
                        summary_lines = []
                        summary_lines.append(f"Category: {cat}")
                        if policy_info.get('description'):
                            summary_lines.append(f"Description: {policy_info['description']}")
                        if policy_info.get('examples'):
                            summary_lines.append(f"Examples: {', '.join(policy_info['examples'])}")
                        if policy_info.get('severity_indicators'):
                            summary_lines.append(f"Severity: {', '.join(policy_info['severity_indicators'])}")
                        if policy_info.get('context_matters'):
                            summary_lines.append(f"Context Factors: {', '.join(policy_info['context_matters'])}")
                        if policy_info.get('action_required'):
                            summary_lines.append(f"Action Required: {policy_info['action_required']}")
                        if policy.get('source'):
                            summary_lines.append(f"Source: {policy['source']}")
                        summary = '\n'.join(summary_lines)
                        explanation = {
                            "type": "violation",
                            "category": cat,
                            "policy": policy_info,
                            "description": policy_info.get('description', ''),
                            "examples": policy_info.get('examples', []),
                            "severity_indicators": policy_info.get('severity_indicators', []),
                            "context_factors": policy_info.get('context_matters', []),
                            "action_required": policy_info.get('action_required', ''),
                            "source": policy.get('source', ''),
                            "summary": summary
                        }
                        violation_explanations.append(explanation)
            # Audio violation explanations
            if audio.get("policy_flags"):
                for polcat, polinfo in audio["policy_flags"].items():
                    policies = rag.retrieve_relevant_policies(polcat, top_k=1)
                    policy_info = policies[0]["policy_info"] if policies else {}
                    violation_explanations.append({
                        "type": "audio",
                        "category": polcat,
                        "keywords": polinfo.get("keywords_found", []),
                        "severity": polinfo.get("severity", ""),
                        "transcript": audio.get("transcript", "")[:100],
                        "policy": policy_info,
                        "action_required": policy_info.get('action_required', ''),
                        "examples": policy_info.get('examples', []),
                        "severity_indicators": policy_info.get('severity_indicators', []),
                        "context_factors": policy_info.get('context_matters', [])
                    })

    # Print detailed explanations
    if violation_explanations:
        safe_print("\n🔎 DETAILED POLICY EXPLANATIONS (RAG):")
        for v in violation_explanations:
            if v["type"] == "safe":
                safe_print(f"- {v['category']}: {v['policy'].get('description', '')}")
                safe_print(f"    Examples: {', '.join(v['policy'].get('examples', []))}")
            elif v["type"] == "violation":
                safe_print(f"- Violation: {v['category']}")
                safe_print(f"    📜 Policy: {v['policy'].get('description', '')}")
                safe_print(f"    Examples: {', '.join(v['policy'].get('examples', []))}")
                safe_print(f"    Action Required: {v['policy'].get('action_required', '')}")
            elif v["type"] == "audio":
                safe_print(f"- Audio: {v['category']} detected in transcript (severity: {v['severity']})")
                if v['keywords']:
                    safe_print(f"    Keywords: {', '.join(v['keywords'])}")
                if v['transcript']:
                    safe_print(f"    Transcript excerpt: {v['transcript']}...")
                if v['policy']:
                    safe_print(f"    📜 Policy: {v['policy'].get('description', '')}")
                    safe_print(f"    Examples: {', '.join(v['policy'].get('examples', []))}")
                    safe_print(f"    Action Required: {v['policy'].get('action_required', '')}")


    # Always include frame details for frontend
    results['rag_explanations'] = violation_explanations
    # Patch: ensure frame details are present and mapped
    results['frame_details'] = []
    for frame in detailed_frames:
        results['frame_details'].append({
            "frame_index": frame.get("frame_index"),
            "timestamp": frame.get("timestamp"),
            "visual_analysis": frame.get("visual_analysis", {}),
            "blip_description": frame.get("blip_description", {}),
            "ocr_text": frame.get("ocr_text", {}),
            "combined_violation": frame.get("combined_violation", False),
            "preview_path": frame.get("preview_path", "")
        })

    report_path = analyzer.save_analysis_report(results)
    if report_path and Path(report_path).exists():
        safe_print(f"\n📄 Detailed report saved: {report_path}")
    else:
        safe_print("❌ No JSON report found after analysis. Check for errors above.")
        if not report_path:
            safe_print("❌ Report path was empty. Possible error in save_analysis_report.")
        else:
            safe_print(f"❌ Report path returned: {report_path} but file does not exist.")

    safe_print("\n🎉 Ultimate video analysis complete!")

if __name__ == "__main__":
    import asyncio
    analyzer, video_path, args = get_args_and_vars()
    asyncio.run(main(analyzer, video_path, args))
