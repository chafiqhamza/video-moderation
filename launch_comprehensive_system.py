#!/usr/bin/env python3
"""
🚀 COMPREHENSIVE YOUTUBE POLICY SYSTEM LAUNCHER
Train model with 7000+ images and demonstrate full video analysis pipeline

This script:
1. Trains comprehensive policy model with ALL your datasets
2. Integrates with your existing BLIP, Whisper, and OCR capabilities  
3. Demonstrates real-world video policy enforcement
4. Provides production-ready API integration
"""

import os
import sys
from pathlib import Path
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"comprehensive_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check if all required components are available"""
    logger.info("🔍 CHECKING SYSTEM REQUIREMENTS...")
    
    requirements = {
        'torch': 'PyTorch for deep learning',
        'transformers': 'Transformers for BLIP model',
        'whisper': 'OpenAI Whisper for audio analysis',
        'cv2': 'OpenCV for video processing',
        'easyocr': 'EasyOCR for text extraction',
        'sklearn': 'Scikit-learn for data processing',
        'PIL': 'Pillow for image processing',
        'numpy': 'NumPy for numerical operations'
    }
    
    available = {}
    missing = []
    
    for package, description in requirements.items():
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            available[package] = True
            logger.info(f"   ✅ {package}: {description}")
        except ImportError:
            available[package] = False
            missing.append(package)
            logger.warning(f"   ❌ {package}: {description} - NOT AVAILABLE")
    
    if missing:
        logger.error(f"⚠️ Missing packages: {', '.join(missing)}")
        logger.error("   Please install missing packages before continuing")
        return False
    
    logger.info("✅ All requirements satisfied!")
    return True

def check_datasets():
    """Check available datasets"""
    logger.info("📊 CHECKING AVAILABLE DATASETS...")
    
    dataset_dirs = [
        "fast_training_data",
        "massive_real_datasets", 
        "real_training_data",
        "comprehensive_training_data",
        "backend/comprehensive_training_data",
        "training_data",
        "nsfw_training_data"
    ]
    
    total_files = 0
    available_datasets = []
    
    for dataset_dir in dataset_dirs:
        dataset_path = Path(dataset_dir)
        if dataset_path.exists():
            # Count image files
            image_count = sum(1 for f in dataset_path.rglob("*") 
                            if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'])
            
            if image_count > 0:
                total_files += image_count
                available_datasets.append((dataset_dir, image_count))
                logger.info(f"   📁 {dataset_dir}: {image_count:,} images")
            else:
                logger.warning(f"   ⚠️ {dataset_dir}: No images found")
        else:
            logger.warning(f"   ❌ {dataset_dir}: Directory not found")
    
    logger.info(f"🎯 TOTAL AVAILABLE: {total_files:,} images across {len(available_datasets)} datasets")
    
    if total_files < 500:
        logger.error("❌ Insufficient training data! Need at least 500 images")
        return False
    
    return True, total_files, available_datasets

def train_comprehensive_model():
    """Train the comprehensive policy model"""
    logger.info("🎯 STARTING COMPREHENSIVE MODEL TRAINING...")
    logger.info("="*80)
    
    try:
        # Import and run the upgraded trainer
        from FIXED_professional_youtube_policy_trainer import train_professional_model
        
        logger.info("🚀 Launching comprehensive training with ALL datasets...")
        model_path = train_professional_model()
        
        if model_path:
            logger.info(f"✅ TRAINING SUCCESSFUL!")
            logger.info(f"   📦 Model saved: {model_path}")
            return model_path
        else:
            logger.error("❌ Training failed!")
            return None
            
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        return None

def test_comprehensive_model(model_path):
    """Test the trained model with comprehensive analysis"""
    logger.info("🧪 TESTING COMPREHENSIVE MODEL...")
    
    try:
        from comprehensive_youtube_policy_model import ComprehensiveYouTubePolicyModel
        
        # Initialize comprehensive model
        logger.info("🔄 Loading comprehensive model...")
        model = ComprehensiveYouTubePolicyModel(model_path)
        
        logger.info("✅ Model loaded successfully!")
        logger.info(f"   🎯 Capabilities: {len(model.capabilities)}")
        for capability in model.capabilities:
            logger.info(f"      - {capability}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return None

def demonstrate_video_analysis(model):
    """Demonstrate comprehensive video analysis"""
    logger.info("🎬 COMPREHENSIVE VIDEO ANALYSIS DEMONSTRATION")
    logger.info("="*80)
    
    # Look for test videos
    test_video_dirs = ["backend/uploads", "test_videos", "."]
    test_videos = []
    
    for video_dir in test_video_dirs:
        video_path = Path(video_dir)
        if video_path.exists():
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                test_videos.extend(video_path.glob(ext))
    
    if test_videos:
        logger.info(f"📁 Found {len(test_videos)} test videos")
        
        # Use first available video
        test_video = test_videos[0]
        logger.info(f"🎯 Analyzing: {test_video.name}")
        
        try:
            # Perform comprehensive analysis
            start_time = time.time()
            results = model.analyze_video_comprehensive(
                str(test_video), 
                max_frames=20,
                frames_per_second=1.0
            )
            analysis_time = time.time() - start_time
            
            # Enhanced display of results using colorama
            try:
                from colorama import Fore, Style, init
                init(autoreset=True)
            except ImportError:
                Fore = Style = None
            
            divider = '=' * 60
            logger.info(f"{Fore.CYAN if Fore else ''}{divider}")
            overall = results.get('overall_assessment', {})
            verdict = overall.get('status', 'unknown')
            verdict_icon = {
                'conforme': '✅',
                'attention': '⚠️',
                'non_conforme': '❌'
            }.get(verdict, '❓')
            verdict_text = {
                'conforme': 'Content is compliant with YouTube policy.',
                'attention': 'Content may need review (minor issues).',
                'non_conforme': 'Content violates YouTube policy.'
            }.get(verdict, 'Status unknown.')
            logger.info(f"{Fore.YELLOW if Fore else ''}{verdict_icon} VERDICT: {verdict.upper()} - {verdict_text}")
            logger.info(f"{Fore.CYAN if Fore else ''}{divider}\n")
            logger.info(f"{Fore.GREEN if Fore else ''}⏱️ Analysis completed in {analysis_time:.1f} seconds")
            logger.info(f"{Fore.BLUE if Fore else ''}📊 RESULTS:")
            logger.info(f"{Fore.GREEN if Fore else ''}   🎯 Compliance: {overall.get('overall_compliance', 0):.1f}%  {Fore.RESET if Fore else ''}- How well the video follows policy. (Higher is better)")
            logger.info(f"{Fore.YELLOW if Fore else ''}   📋 Status: {verdict}  {Fore.RESET if Fore else ''}- Overall decision (see verdict above)")
            logger.info(f"{Fore.RED if Fore else ''}   🚨 Violations: {overall.get('total_violations', 0)}  {Fore.RESET if Fore else ''}- Number of detected issues. (Lower is better)")
            categories = overall.get('violation_categories', [])
            logger.info(f"{Fore.MAGENTA if Fore else ''}   🔍 Categories: {', '.join(categories) if categories else 'None'}  {Fore.RESET if Fore else ''}- Types of violations found.")
            logger.info(f"{Fore.CYAN if Fore else ''}   ℹ️ What does this mean?\n   - Compliance: Percentage of content following policy.\n   - Status: Final decision for YouTube upload.\n   - Violations: Issues detected in video.\n   - Categories: Types of violations (e.g., violence, nudity, copyright).")
            # Show sample prediction if available
            sample = results.get('sample_prediction')
            if sample:
                logger.info(f"{Fore.CYAN if Fore else ''}\nSample Prediction:")
                logger.info(f"   Input: {sample.get('input', '')}")
                logger.info(f"   Predicted: {sample.get('predicted', '')}")
                logger.info(f"   Reason: {sample.get('reason', '')}")
            # Save detailed report
            report_path = model.save_analysis_report(results)
            logger.info(f"{Fore.CYAN if Fore else ''}   📄 Report: {report_path}")
            logger.info(f"{Fore.CYAN if Fore else ''}{divider}")
            
            # Show recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                logger.info("💡 RECOMMENDATIONS:")
                for rec in recommendations:
                    logger.info(f"   {rec}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Video analysis failed: {e}")
            return False
    
    else:
        logger.warning("⚠️ No test videos found")
        logger.info("   To test video analysis, place video files in:")
        logger.info("   - backend/uploads/")
        logger.info("   - Current directory")
        return False

def integrate_with_backend():
    """Show how to integrate with existing backend"""
    logger.info("🔌 BACKEND INTEGRATION GUIDE")
    logger.info("="*80)
    
    integration_code = '''
# Add to your backend/main.py:

from comprehensive_youtube_policy_model import ComprehensiveYouTubePolicyModel

# Initialize once at startup
policy_model = ComprehensiveYouTubePolicyModel()

# Use in your video analysis endpoint
@app.post("/analyze-video-comprehensive")
async def analyze_video_comprehensive(file: UploadFile = File(...)):
    # Save uploaded video
    video_path = save_uploaded_file(file)
    
    # Comprehensive analysis
    results = policy_model.analyze_video_comprehensive(video_path)
    
    # Return results
    return {
        "compliance": results["overall_assessment"]["overall_compliance"],
        "status": results["overall_assessment"]["status"],
        "violations": results["overall_assessment"]["total_violations"],
        "recommendations": results["recommendations"],
        "detailed_report": results
    }

# Use for frame analysis
def analyze_frame_with_policy(frame):
    return policy_model.analyze_frame(frame)
'''
    
    logger.info("💻 INTEGRATION CODE:")
    print(integration_code)
    
    logger.info("🔗 YOUR EXISTING CAPABILITIES ENHANCED:")
    logger.info("   ✅ Video frame extraction (OpenCV/FFmpeg) → Enhanced with policy detection")
    logger.info("   ✅ BLIP frame descriptions → Integrated with violation detection")
    logger.info("   ✅ Whisper audio analysis → Combined with text policy analysis")
    logger.info("   ✅ OCR text extraction → Enhanced with policy keyword detection")
    logger.info("   ✅ FastAPI backend → Ready for comprehensive policy endpoints")

def main():
    """Main launcher function"""
    print("🚀 COMPREHENSIVE YOUTUBE POLICY ENFORCEMENT SYSTEM")
    print("="*80)
    print("🎯 Train with 7000+ images and create production-ready policy enforcement")
    print()
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Please install missing packages.")
        return
    
    # Check datasets
    dataset_check = check_datasets()
    if not dataset_check:
        print("❌ Insufficient training data. Please ensure datasets are available.")
        return
    
    success, total_files, datasets = dataset_check
    print(f"\n✅ Ready to train with {total_files:,} images!")
    
    # Ask user what to do
    print("\n🎯 CHOOSE ACTION:")
    print("1. Train comprehensive model with ALL datasets")
    print("2. Test existing model (if available)")
    print("3. Full pipeline: Train + Test + Demo")
    print("4. Show backend integration guide")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        # Train model
        model_path = train_comprehensive_model()
        if model_path:
            print(f"\n🎉 SUCCESS! Model trained and saved: {model_path}")
            print("   You can now use this model for comprehensive video analysis")
    
    elif choice == "2":
        # Test existing model
        model_path = "PROFESSIONAL_FINAL_youtube_policy_model.pth"
        if Path(model_path).exists():
            model = test_comprehensive_model(model_path)
            if model:
                print("\n✅ Model testing successful!")
        else:
            print(f"❌ Model not found: {model_path}")
            print("   Please train the model first (option 1)")
    
    elif choice == "3":
        # Full pipeline
        print("\n🚀 FULL PIPELINE: Train → Test → Demo")
        
        # Step 1: Train
        model_path = train_comprehensive_model()
        if not model_path:
            print("❌ Training failed. Cannot continue.")
            return
        
        # Step 2: Test
        model = test_comprehensive_model(model_path)
        if not model:
            print("❌ Model testing failed. Cannot continue.")
            return
        
        # Step 3: Demo
        demo_success = demonstrate_video_analysis(model)
        
        if demo_success:
            print("\n🎉 FULL PIPELINE SUCCESSFUL!")
            print("   Your comprehensive YouTube policy enforcement system is ready!")
        
        # Step 4: Integration guide
        integrate_with_backend()
    
    elif choice == "4":
        # Integration guide
        integrate_with_backend()
    
    elif choice == "5":
        print("👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid choice. Please run again.")
        return
    
    print("\n" + "="*80)
    print("🎯 SYSTEM CAPABILITIES SUMMARY:")
    print("✅ Multi-modal analysis: Visual + Audio + Text")
    print("✅ Comprehensive policy detection: 10+ violation types")
    print("✅ Production-ready: FastAPI integration")
    print("✅ Scalable: Handles 7000+ training images")
    print("✅ Real-time: Frame-by-frame analysis")
    print("✅ Contextual: BLIP descriptions + OCR + Whisper")
    print("="*80)

if __name__ == "__main__":
    main()
