#!/usr/bin/env python3
"""
🧹 FINAL PROJECT CLEANUP - Remove ALL problematic files
Fixes import errors and removes any remaining unnecessary files
"""

import os
import shutil
import glob
from pathlib import Path

def final_cleanup():
    """Perform final comprehensive cleanup"""
    print("🔥 FINAL COMPREHENSIVE CLEANUP")
    print("="*60)
    
    base_dir = Path("c:/Users/PC/Desktop/ProjetStage")
    removed_count = 0
    
    # Find and remove ALL problematic files
    problematic_files = [
        "backend/main_clean.py",
        "backend/main_backup.py", 
        "backend/main_working.py",
        "backend/training_data_manager.py",
        "backend/genuine_ai_analyzer_new.py",
        "backend/audio_fix_simple_new.py",
        "backend/cleanup_project.py"
    ]
    
    # Additional patterns to remove
    remove_patterns = [
        "**/*_backup.py",
        "**/*_old.py", 
        "**/*_new.py",
        "**/*_clean.py",
        "**/*_working.py",
        "**/test_*.py",
        "**/demo_*.py",
        "**/quick_*.py",
        "**/__pycache__",
        "**/frame_sample_*.jpg",
        "**/test_*.jpg"
    ]
    
    print("🗑️ Removing specific problematic files...")
    for file_path in problematic_files:
        full_path = base_dir / file_path
        if full_path.exists():
            try:
                if full_path.is_file():
                    full_path.unlink()
                    print(f"   ❌ Removed: {file_path}")
                    removed_count += 1
            except Exception as e:
                print(f"   ⚠️ Could not remove {file_path}: {e}")
    
    print("🧹 Removing files by patterns...")
    for pattern in remove_patterns:
        for file_path in base_dir.rglob(pattern):
            # Keep essential files
            if any(essential in str(file_path) for essential in [
                'youtube_policy_classifier.py',
                'comprehensive_realistic_training.py',
                'test_simple_api.py'
            ]):
                continue
                
            try:
                if file_path.is_file():
                    file_path.unlink()
                    print(f"   ❌ Removed: {file_path.relative_to(base_dir)}")
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    print(f"   ❌ Removed dir: {file_path.relative_to(base_dir)}")
                removed_count += 1
            except Exception as e:
                print(f"   ⚠️ Could not remove {file_path.name}: {e}")
    
    # Remove VS Code tasks.json references to deleted files
    vscode_tasks = base_dir / ".vscode" / "tasks.json"
    if vscode_tasks.exists():
        try:
            content = vscode_tasks.read_text()
            if "main_clean.py" in content:
                # Replace with main.py
                content = content.replace("main_clean.py", "main.py")
                vscode_tasks.write_text(content)
                print("   ✅ Fixed VS Code tasks.json")
        except Exception as e:
            print(f"   ⚠️ Could not fix tasks.json: {e}")
    
    print(f"\n✅ FINAL CLEANUP COMPLETE!")
    print(f"📊 Total items removed: {removed_count}")
    
    # Show final clean structure
    print("\n📁 FINAL CLEAN PROJECT STRUCTURE:")
    show_final_structure(base_dir)

def show_final_structure(base_dir):
    """Show the final cleaned structure"""
    print("\nROOT:")
    for item in sorted(base_dir.iterdir()):
        if item.name.startswith('.'):
            continue
        if item.is_dir():
            print(f"   📁 {item.name}/")
        else:
            print(f"   📄 {item.name}")
    
    print("\nBACKEND:")
    backend_dir = base_dir / "backend"
    if backend_dir.exists():
        for item in sorted(backend_dir.iterdir()):
            if item.name.startswith('.'):
                continue
            if item.is_dir():
                print(f"   📁 backend/{item.name}/")
            else:
                print(f"   📄 backend/{item.name}")

def create_final_requirements():
    """Create clean requirements.txt without problematic imports"""
    requirements_content = """# Core web framework
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6

# AI and ML
torch==2.1.0
torchvision==0.16.0
transformers==4.35.0
pillow==10.0.1
opencv-python==4.8.1.78
numpy==1.24.3

# Audio processing
whisper==1.1.10
librosa==0.10.1
soundfile==0.12.1

# API and utilities
requests==2.31.0
python-jose==3.3.0
passlib==1.7.4
python-dotenv==1.0.0
pydantic==2.4.2
typing-extensions==4.8.0

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
"""
    
    backend_dir = Path("c:/Users/PC/Desktop/ProjetStage/backend")
    req_file = backend_dir / "requirements.txt"
    
    with open(req_file, 'w') as f:
        f.write(requirements_content)
    
    print("📄 Created clean requirements.txt")

if __name__ == "__main__":
    final_cleanup()
    create_final_requirements()
