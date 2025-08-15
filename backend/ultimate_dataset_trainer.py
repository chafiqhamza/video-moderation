from typing import Dict, List, Tuple, Optional, Any

import logging
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torchvision.models as models

class UltimateModel(nn.Module):
    def __init__(self, num_classes: int = 7):
        super(UltimateModel, self).__init__()
        """EXACT architecture matching your saved checkpoint"""
        self.backbone = models.efficientnet_b0(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),                    # Layer 0
            nn.Linear(in_features, 512),                        # Layer 1
            nn.BatchNorm1d(512),                                # Layer 2
            nn.SiLU(inplace=True),                              # Layer 3
            nn.Dropout(p=0.2),                                  # Layer 4
            nn.Linear(512, 256),                                # Layer 5
            nn.BatchNorm1d(256),                                # Layer 6
            nn.SiLU(inplace=True),                              # Layer 7
            nn.Dropout(p=0.2),                                  # Layer 8
            nn.Linear(256, num_classes)                         # Layer 9 (output)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
        for param in self.backbone.features[:5].parameters():
            param.requires_grad = False
            
        # Replace classifier with custom head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class UltimateVideoAnalyzer:
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize the video analyzer with all components.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.models = {}
        
        try:
            # Load visual model
            self.models['visual'] = self._load_visual_model(model_path)
            logger.info(f"✅ Visual model loaded: {self.models['visual'].__class__.__name__}")
            
            # Initialize other components
            self._initialize_blip()
            self._initialize_whisper()
            self._initialize_ocr()
            self._initialize_rag()
            
            logger.info("✅ ULTIMATE VIDEO ANALYZER READY!")
            logger.info("🎯 CAPABILITIES: 6 modules loaded")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            raise

    def _load_visual_model(self, model_path: str) -> nn.Module:
        """Load the visual classification model, handling checkpoint mismatches."""
        try:
            model = UltimateModel(num_classes=7).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            model_state = model.state_dict()
            # Log missing and unexpected keys
            missing_keys = [k for k in model_state if k not in state_dict]
            unexpected_keys = [k for k in state_dict if k not in model_state]
            size_mismatches = [k for k in model_state if k in state_dict and model_state[k].size() != state_dict[k].size()]
            if missing_keys:
                logger.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
            if size_mismatches:
                logger.warning(f"Size mismatches: {size_mismatches}")
            # Filter out mismatched keys
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state and v.size() == model_state[k].size()}
            model.load_state_dict(filtered_state_dict, strict=False)
            model.eval()
            logger.info("✅ Visual model loaded with filtered checkpoint.")
            return model
        except Exception as e:
            logger.error(f"❌ Error loading visual model: {str(e)}")
            raise

    def _initialize_blip(self):
        """Initialize BLIP model for image captioning."""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            self.models['blip_processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.models['blip_model'] = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
            
            logger.info("✅ BLIP model loaded")
            
        except ImportError:
            logger.warning("❌ BLIP not available - install transformers package")
        except Exception as e:
            logger.error(f"❌ BLIP initialization failed: {str(e)}")

    def _initialize_whisper(self):
        """Initialize Whisper model for speech recognition."""
        try:
            import whisper
            self.models['whisper'] = whisper.load_model("base").to(self.device)
            logger.info("✅ Whisper model loaded")
            
        except ImportError:
            logger.warning("❌ Whisper not available - install openai-whisper package")
        except Exception as e:
            logger.error(f"❌ Whisper initialization failed: {str(e)}")

    def _initialize_ocr(self):
        """Initialize OCR model for text extraction."""
        try:
            import easyocr
            self.models['ocr'] = easyocr.Reader(['en'])
            logger.info("✅ OCR model loaded")
            
        except ImportError:
            logger.warning("❌ EasyOCR not available - install easyocr package")
        except Exception as e:
            logger.error(f"❌ OCR initialization failed: {str(e)}")

    def _initialize_rag(self):
        """Initialize RAG system for policy knowledge."""
        try:
            # Placeholder for RAG implementation
            self.models['rag'] = {"policy_db": "content_moderation_rag.db"}
            logger.info("✅ RAG Policy Knowledge loaded")
            
        except Exception as e:
            logger.error(f"❌ RAG initialization failed: {str(e)}")

    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Analyze a video file for policy violations.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing analysis results
        """
        results = {
            "compliance_score": 0,
            "violations": [],
            "frames": [],
            "audio": {},
            "recommendations": []
        }
        
        try:
            # Extract frames
            frames = self._extract_frames(video_path)
            
            # Process each frame
            for i, frame in enumerate(frames):
                frame_result = self._analyze_frame(frame)
                results["frames"].append(frame_result)
                
                if frame_result.get("violation"):
                    results["violations"].append(frame_result["violation"])
            
            # Analyze audio
            if "whisper" in self.models:
                results["audio"] = self._analyze_audio(video_path)
            
            # Calculate overall compliance score
            results["compliance_score"] = self._calculate_compliance_score(results)
            
            # Generate recommendations
            results["recommendations"] = self._generate_recommendations(results)
            
        except Exception as e:
            logger.error(f"❌ Video analysis failed: {str(e)}")
            raise
            
        return results

    def _extract_frames(self, video_path: str, fps: float = 0.5, max_frames: int = 15) -> list:
        """Extract frames from video at specified interval."""
        import cv2
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_step = int(frame_rate / fps)
        
        for i in range(0, frame_count, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret and len(frames) < max_frames:
                frames.append(frame)
                
        cap.release()
        return frames

    def _analyze_frame(self, frame) -> Dict[str, Any]:
        """Analyze a single video frame."""
        import torchvision.transforms as transforms
        from PIL import Image
        
        result = {
            "visual_class": None,
            "confidence": 0,
            "caption": "",
            "text": [],
            "violation": None,
            "rag_explanation": None
        }
        
        try:
            # Preprocess image
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            img = Image.fromarray(frame)
            img_tensor = transform(img).unsqueeze(0).to(self.device)
            
            # Visual classification
            with torch.no_grad():
                outputs = self.models['visual'](img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                
                result["visual_class"] = pred.item()
                result["confidence"] = conf.item()
                
                # Ajout explication RAG pour tous les cas
                if pred.item() == 0:  # safe_content
                    result["rag_explanation"] = {
                        "category": "safe_content",
                        "explanation": "Aucune violation détectée. Ce contenu est conforme à la politique de modération.",
                        "policy_reference": "Règle générale de conformité"
                    }
                else:
                    result["violation"] = {
                        "type": self._get_class_name(pred.item()),
                        "confidence": conf.item()
                    }
                    result["rag_explanation"] = {
                        "category": self._get_class_name(pred.item()),
                        "explanation": f"Violation détectée : {self._get_class_name(pred.item())}",
                        "policy_reference": "Référence à la règle spécifique"
                    }
            
            # Generate caption
            if 'blip_model' in self.models:
                inputs = self.models['blip_processor'](img, return_tensors="pt").to(self.device)
                out = self.models['blip_model'].generate(**inputs)
                result["caption"] = self.models['blip_processor'].decode(out[0], skip_special_tokens=True)
            
            # Extract text
            if 'ocr' in self.models:
                result["text"] = self.models['ocr'].readtext(frame, detail=0)
                
        except Exception as e:
            logger.error(f"Frame analysis error: {str(e)}")
            
        return result

    def _analyze_audio(self, video_path: str) -> Dict[str, Any]:
        """Analyze audio track for policy violations."""
        import tempfile
        import subprocess
        result = {
            "transcript": "",
            "violations": []
        }
        try:
            # Extract audio using ffmpeg (must be installed)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmp_audio_path = tmpfile.name
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp_audio_path
            ]
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Transcribe
            transcript = self.models['whisper'].transcribe(tmp_audio_path)
            result["transcript"] = transcript["text"]
            # Simple keyword-based violation detection
            violation_keywords = ["hate", "kill", "violence", "attack"]
            for word in violation_keywords:
                if word in transcript["text"].lower():
                    result["violations"].append({
                        "type": "hate_speech" if word == "hate" else "violence",
                        "context": transcript["text"],
                        "confidence": 0.7  # Placeholder
                    })
            # Clean up temp file
            import os
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
        except Exception as e:
            logger.error(f"Audio analysis error: {str(e)}")
        return result

    def _calculate_compliance_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall compliance score (0-100)."""
        total_frames = len(results["frames"])
        if total_frames == 0:
            return 0
            
        violation_frames = len(results["violations"])
        audio_violations = len(results["audio"].get("violations", []))
        
        safe_frames = total_frames - violation_frames
        score = (safe_frames / total_frames) * 100
        
        # Penalize for audio violations
        if audio_violations > 0:
            score *= 0.7  # 30% penalty
            
        return round(score, 2)

    def _generate_recommendations(self, results: Dict[str, Any]) -> list:
        """Generate content moderation recommendations."""
        recs = []
        
        if results["compliance_score"] < 70:
            recs.append("🚨 Multiple violations detected - content may need significant editing")
        elif results["compliance_score"] < 90:
            recs.append("⚠️ Minor violations detected - targeted edits recommended")
            
        if any(v["type"] == "hate_speech" for v in results["violations"]):
            recs.append("🎵 Hate speech detected - immediate review required")
            
        if any(v["type"] == "violence" for v in results["violations"]):
            recs.append("🖼️ Violent content detected - age restriction recommended")
            
        return recs if recs else ["✅ Content appears compliant with policies"]

    def _get_class_name(self, class_idx: int) -> str:
        """Map class index to human-readable name."""
        classes = [
            "safe_content",
            "suggestive_content",
            "artistic_adult_content",
            "misinformation",
            "violence",
            "hate_speech",
            "dangerous_activities"
        ]
        return classes[class_idx] if 0 <= class_idx < len(classes) else "unknown"

if __name__ == "__main__":
    analyzer = UltimateVideoAnalyzer(
        model_path="ULTIMATE_youtube_policy_model.pth",
        device="cuda"
    )
    
    # Example usage
    results = analyzer.analyze_video("test_video.mp4")
    print(f"Compliance Score: {results['compliance_score']}%")
    print(f"Violations: {len(results['violations'])}")