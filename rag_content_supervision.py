#!/usr/bin/env python3
"""
🧠 RAG-ENHANCED CONTENT SUPERVISION SYSTEM
Combines your trained model with RAG for intelligent, contextual content moderation
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pathlib import Path
import json
import sqlite3
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentModerationRAG:
    """RAG-enhanced content moderation system"""
    
    def __init__(self, model_path="ULTIMATE_youtube_policy_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load your trained visual model
        self.visual_model = self._load_visual_model(model_path)
        
        # Initialize RAG components
        self.embeddings_model = self._load_embeddings_model()
        self.knowledge_base = self._initialize_knowledge_base()
        self.policy_db = self._initialize_policy_database()
        
        logger.info("🧠 RAG-Enhanced Content Moderation System Ready!")
    
    def _load_visual_model(self, model_path):
        """Load your trained ultimate model"""
        try:
            from ultimate_dataset_trainer import UltimateModel
            
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                
                self.class_names = checkpoint['class_names']
                model = UltimateModel(len(self.class_names))
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                
                logger.info(f"✅ Visual model loaded: {len(self.class_names)} classes")
                return model
            else:
                logger.error(f"❌ Model not found: {model_path}")
                return None
        except Exception as e:
            logger.error(f"❌ Error loading visual model: {e}")
            return None
    
    def _load_embeddings_model(self):
        """Load sentence embeddings model for RAG"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embeddings model loaded")
            return model
        except ImportError:
            logger.warning("⚠️ sentence-transformers not installed, using basic embeddings")
            return None
    
    def _initialize_knowledge_base(self):
        """Initialize comprehensive policy knowledge base"""
        knowledge_base = {
            "youtube_policies": {
                "violence": {
                    "description": "Content depicting violence, weapons, or graphic imagery",
                    "examples": ["fighting", "weapons", "blood", "gore", "combat"],
                    "severity_indicators": ["realistic weapons", "explicit violence", "gore"],
                    "context_matters": ["gaming violence vs real violence", "educational vs gratuitous"],
                    "action_required": "human_review_required"
                },
                "hate_speech": {
                    "description": "Content promoting hatred against protected groups",
                    "examples": ["discriminatory symbols", "hateful rhetoric", "supremacist content"],
                    "severity_indicators": ["nazi symbols", "racial slurs", "incitement"],
                    "context_matters": ["historical education vs promotion", "criticism vs hatred"],
                    "action_required": "immediate_removal"
                },
                "dangerous_activities": {
                    "description": "Content showing dangerous or harmful activities",
                    "examples": ["extreme stunts", "self-harm", "dangerous challenges"],
                    "severity_indicators": ["life-threatening", "copyable dangers", "no safety warnings"],
                    "context_matters": ["professional vs amateur", "safety precautions shown"],
                    "action_required": "age_restriction_or_removal"
                },
                "misinformation": {
                    "description": "False or misleading information",
                    "examples": ["fake news", "conspiracy theories", "medical misinformation"],
                    "severity_indicators": ["health misinformation", "election fraud claims"],
                    "context_matters": ["satire vs serious claims", "opinion vs fact"],
                    "action_required": "fact_check_label"
                },
                "artistic_adult_content": {
                    "description": "Artistic or educational adult content",
                    "examples": ["classical art nudity", "educational anatomy", "artistic expression"],
                    "severity_indicators": ["sexual intent", "focus on genitals", "pornographic"],
                    "context_matters": ["artistic merit", "educational value", "target audience"],
                    "action_required": "age_restriction"
                },
                "suggestive_content": {
                    "description": "Sexually suggestive content",
                    "examples": ["provocative poses", "sexual themes", "suggestive clothing"],
                    "severity_indicators": ["sexual focus", "provocative dancing", "sexual scenarios"],
                    "context_matters": ["artistic vs explicit", "age appropriateness"],
                    "action_required": "age_restriction"
                },
                "safe_content": {
                    "description": "Content that complies with all policies",
                    "examples": ["educational", "entertainment", "news", "tutorials"],
                    "severity_indicators": [],
                    "context_matters": [],
                    "action_required": "none"
                }
            },
            "contextual_factors": [
                "audience_age_group",
                "educational_value", 
                "artistic_merit",
                "news_worthiness",
                "satirical_intent",
                "historical_significance",
                "cultural_context",
                "safety_warnings_present"
            ],
            "escalation_criteria": {
                "immediate_removal": ["illegal_content", "terrorism", "child_exploitation"],
                "human_review": ["borderline_cases", "context_dependent", "appeal_worthy"],
                "automated_action": ["clear_violations", "high_confidence_predictions"],
                "age_restriction": ["mature_themes", "mild_violations", "educational_adult_content"]
            }
        }
        
        logger.info("✅ Knowledge base initialized")
        return knowledge_base
    
    def _initialize_policy_database(self):
        """Initialize SQLite database for policy cases"""
        db_path = r"d:/video-moderation-second/video-moderation-second/database/content_moderation_rag.db"
        conn = sqlite3.connect(db_path)
        
        # Create tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS moderation_cases (
                id INTEGER PRIMARY KEY,
                content_hash TEXT UNIQUE,
                visual_prediction TEXT,
                rag_context TEXT,
                final_decision TEXT,
                confidence REAL,
                human_reviewed BOOLEAN,
                timestamp DATETIME,
                metadata TEXT
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS policy_embeddings (
                id INTEGER PRIMARY KEY,
                policy_text TEXT,
                embedding BLOB,
                category TEXT,
                importance REAL
            )
        ''')
        
        conn.commit()
        logger.info(f"✅ Policy database initialized: {db_path}")
        return conn
    
    def embed_text(self, text: str) -> np.ndarray:
        """Create embeddings for text"""
        if self.embeddings_model:
            return self.embeddings_model.encode([text])[0]
        else:
            # Simple fallback embedding
            return np.random.rand(384).astype(np.float32)
    
    def retrieve_relevant_policies(self, content_description: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant policies using RAG (static KB + dynamic scraped DB)"""
        query_embedding = self.embed_text(content_description)
        relevant_policies = []

        # Search static knowledge base
        for category, policy_info in self.knowledge_base["youtube_policies"].items():
            policy_text = f"{policy_info['description']} {' '.join(policy_info['examples'])}"
            policy_embedding = self.embed_text(policy_text)
            similarity = np.dot(query_embedding, policy_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(policy_embedding)
            )
            relevant_policies.append({
                "category": category,
                "similarity": similarity,
                "policy_info": policy_info,
                "source": "static"
            })

        # Search dynamic scraped policy DB
        try:
            cursor = self.policy_db.cursor()
            cursor.execute("SELECT id, policy_text, embedding, category, description, examples FROM policy_embeddings")
            rows = cursor.fetchall()
            for row in rows:
                policy_text = row[1]
                embedding = np.frombuffer(row[2], dtype=np.float32)
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                relevant_policies.append({
                    "category": row[3],
                    "similarity": similarity,
                    "policy_info": {
                        "description": row[4],
                        "examples": json.loads(row[5])
                    },
                    "source": "scraped"
                })
        except Exception as e:
            logger.error(f"Error searching scraped policy DB: {e}")

        # Sort by similarity and return top_k
        relevant_policies.sort(key=lambda x: x["similarity"], reverse=True)
        return relevant_policies[:top_k]
    
    def analyze_content_with_rag(self, image_path: str, additional_context: str = "") -> Dict:
        """Comprehensive content analysis using visual model + RAG"""
        try:
            # Step 1: Visual analysis with your trained model
            visual_result = self._analyze_visual_content(image_path)
            
            # Step 2: Generate content description for RAG
            content_description = self._generate_content_description(image_path, visual_result, additional_context)
            
            # Step 3: Retrieve relevant policies
            relevant_policies = self.retrieve_relevant_policies(content_description)
            
            # Step 4: RAG-enhanced decision making
            rag_decision = self._make_rag_decision(visual_result, relevant_policies, content_description)
            
            # Step 5: Store case in database
            case_id = self._store_moderation_case(image_path, visual_result, rag_decision)
            
            return {
                "case_id": case_id,
                "visual_analysis": visual_result,
                "content_description": content_description,
                "relevant_policies": relevant_policies,
                "rag_decision": rag_decision,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing content: {e}")
            return {"error": str(e)}
    
    def _analyze_visual_content(self, image_path: str) -> Dict:
        """Analyze image with your trained model"""
        if not self.visual_model:
            return {"error": "Visual model not loaded"}
        
        try:
            # Load and preprocess image
            from torchvision import transforms
            
            image = Image.open(image_path).convert('RGB')
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
            confidence_score = confidence.item()
            
            return {
                "predicted_category": category,
                "confidence": confidence_score,
                "all_probabilities": {
                    self.class_names[i]: probabilities[0][i].item() 
                    for i in range(len(self.class_names))
                },
                "violation_detected": category != "safe_content"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_content_description(self, image_path: str, visual_result: Dict, additional_context: str) -> str:
        """Generate detailed content description for RAG"""
        description_parts = []
        
        # Visual analysis results
        if "predicted_category" in visual_result:
            description_parts.append(f"Visual analysis detected: {visual_result['predicted_category']}")
            description_parts.append(f"Confidence: {visual_result['confidence']:.2f}")
        
        # Additional context if provided
        if additional_context:
            description_parts.append(f"Additional context: {additional_context}")
        
        # Image metadata analysis
        try:
            img = Image.open(image_path)
            description_parts.append(f"Image dimensions: {img.size}")
            
            # Basic image analysis
            img_array = np.array(img)
            avg_brightness = np.mean(img_array)
            description_parts.append(f"Average brightness: {avg_brightness:.1f}")
            
        except Exception as e:
            description_parts.append(f"Image metadata error: {e}")
        
        return " | ".join(description_parts)
    
    def _make_rag_decision(self, visual_result: Dict, relevant_policies: List[Dict], content_description: str) -> Dict:
        """Make intelligent decision using RAG context"""
        if not visual_result or "predicted_category" in visual_result and visual_result["predicted_category"] == "safe_content":
            return {
                "decision": "approved",
                "action": "none",
                "reasoning": "Content classified as safe by visual model",
                "human_review_needed": False,
                "confidence": visual_result.get("confidence", 0.0)
            }
        
        predicted_category = visual_result.get("predicted_category", "unknown")
        confidence = visual_result.get("confidence", 0.0)
        
        # Find most relevant policy
        most_relevant_policy = relevant_policies[0] if relevant_policies else None
        
        if not most_relevant_policy:
            return {
                "decision": "flagged",
                "action": "human_review",
                "reasoning": "No relevant policy found",
                "human_review_needed": True,
                "confidence": 0.0
            }
        
        policy_info = most_relevant_policy["policy_info"]
        
        # RAG-enhanced decision logic
        if confidence > 0.9:
            # High confidence - follow policy directly
            action = policy_info.get("action_required", "human_review")
            decision = "violation_detected" if action != "none" else "approved"
        elif confidence > 0.7:
            # Medium confidence - consider context
            if "context_matters" in policy_info and policy_info["context_matters"]:
                action = "human_review"
                decision = "needs_context_review"
            else:
                action = policy_info.get("action_required", "human_review")
                decision = "probable_violation"
        else:
            # Low confidence - human review needed
            action = "human_review"
            decision = "uncertain"
        
        return {
            "decision": decision,
            "action": action,
            "reasoning": f"Based on {predicted_category} classification and {most_relevant_policy['category']} policy",
            "relevant_policy": most_relevant_policy["category"],
            "policy_similarity": most_relevant_policy["similarity"],
            "human_review_needed": action == "human_review",
            "confidence": confidence,
            "severity_indicators": policy_info.get("severity_indicators", []),
            "context_factors": policy_info.get("context_matters", [])
        }
    
    def _store_moderation_case(self, image_path: str, visual_result: Dict, rag_decision: Dict) -> str:
        """Store case in database for learning"""
        try:
            # Create content hash
            with open(image_path, 'rb') as f:
                content_hash = hashlib.md5(f.read()).hexdigest()
            
            # Store in database
            self.policy_db.execute('''
                INSERT OR REPLACE INTO moderation_cases 
                (content_hash, visual_prediction, rag_context, final_decision, confidence, human_reviewed, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                content_hash,
                json.dumps(visual_result),
                json.dumps(rag_decision),
                rag_decision.get("decision", "unknown"),
                rag_decision.get("confidence", 0.0),
                False,
                datetime.now().isoformat(),
                json.dumps({"image_path": str(image_path)})
            ))
            
            self.policy_db.commit()
            return content_hash
            
        except Exception as e:
            logger.error(f"❌ Error storing case: {e}")
            return "unknown"
    
    def batch_analyze_video(self, video_path: str, frames_per_second: float = 1.0) -> Dict:
        """Analyze entire video with RAG-enhanced supervision"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / frames_per_second)
            
            frame_results = []
            frame_count = 0
            
            logger.info(f"🎬 Analyzing video: {video_path}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Save frame temporarily
                    temp_frame_path = f"temp_frame_{frame_count}.jpg"
                    cv2.imwrite(temp_frame_path, frame)
                    
                    # Analyze frame with RAG
                    frame_result = self.analyze_content_with_rag(
                        temp_frame_path, 
                        f"Video frame at {frame_count/fps:.1f}s"
                    )
                    frame_result["timestamp"] = frame_count / fps
                    frame_results.append(frame_result)
                    
                    # Clean up temp file
                    Path(temp_frame_path).unlink(missing_ok=True)
                
                frame_count += 1
            
            cap.release()
            
            # Aggregate results
            total_violations = sum(1 for r in frame_results if r.get("rag_decision", {}).get("decision") != "approved")
            avg_confidence = np.mean([r.get("rag_decision", {}).get("confidence", 0) for r in frame_results])
            
            return {
                "video_path": video_path,
                "total_frames_analyzed": len(frame_results),
                "violations_detected": total_violations,
                "average_confidence": avg_confidence,
                "frame_results": frame_results,
                "overall_decision": "violation_detected" if total_violations > 0 else "approved",
                "rag_enhanced": True
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing video: {e}")
            return {"error": str(e)}
    
    def get_moderation_stats(self) -> Dict:
        """Get comprehensive moderation statistics"""
        try:
            cursor = self.policy_db.cursor()
            
            # Total cases
            cursor.execute("SELECT COUNT(*) FROM moderation_cases")
            total_cases = cursor.fetchone()[0]
            
            # Decisions breakdown
            cursor.execute("SELECT final_decision, COUNT(*) FROM moderation_cases GROUP BY final_decision")
            decisions = dict(cursor.fetchall())
            
            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM moderation_cases")
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            return {
                "total_cases": total_cases,
                "decisions_breakdown": decisions,
                "average_confidence": avg_confidence,
                "rag_enhanced": True,
                "knowledge_base_policies": len(self.knowledge_base["youtube_policies"])
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            return {"error": str(e)}

def main():
    """Demo of RAG-enhanced content moderation"""
    print("🧠 RAG-ENHANCED CONTENT SUPERVISION SYSTEM")
    print("="*60)
    
    # Initialize RAG system
    rag_system = ContentModerationRAG()
    
    if not rag_system.visual_model:
        print("❌ Visual model not loaded. Please train the model first.")
        return
    
    print("✅ RAG system ready!")
    print("\n🎯 CAPABILITIES:")
    print("   🔍 Visual content analysis with your trained model")
    print("   🧠 RAG-powered policy reasoning")
    print("   📚 Comprehensive YouTube policy knowledge base")
    print("   🎬 Video analysis with temporal context")
    print("   📊 Case storage and learning")
    
    # Example usage
    test_image = input("\nEnter path to test image (or press Enter to skip): ").strip()
    
    if test_image and Path(test_image).exists():
        print(f"\n🔍 Analyzing: {test_image}")
        result = rag_system.analyze_content_with_rag(test_image)
        
        print("\n📊 RAG-ENHANCED ANALYSIS RESULTS:")
        print(f"   🎯 Decision: {result.get('rag_decision', {}).get('decision', 'unknown')}")
        print(f"   ⚡ Action: {result.get('rag_decision', {}).get('action', 'unknown')}")
        print(f"   🧠 Reasoning: {result.get('rag_decision', {}).get('reasoning', 'N/A')}")
        print(f"   📈 Confidence: {result.get('rag_decision', {}).get('confidence', 0):.2f}")
        print(f"   👥 Human Review: {'Yes' if result.get('rag_decision', {}).get('human_review_needed') else 'No'}")
    
    # Show stats
    stats = rag_system.get_moderation_stats()
    print(f"\n📈 SYSTEM STATISTICS:")
    print(f"   📊 Total cases processed: {stats.get('total_cases', 0)}")
    print(f"   🧠 Knowledge base policies: {stats.get('knowledge_base_policies', 0)}")
    print(f"   ⚡ RAG-enhanced: {stats.get('rag_enhanced', False)}")

if __name__ == "__main__":
    main()
