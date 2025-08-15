
"""
Improved Content Analyzer - Replaces hardcoded model responses
"""

import json
import re
from pathlib import Path

class ImprovedContentAnalyzer:
    def __init__(self):
        """Initialize improved analyzer"""
        self.rules = self.load_rules()
        
    def load_rules(self):
        """Load analysis rules"""
        try:
            with open("smart_analysis_rules.json", "r") as f:
                return json.load(f)
        except:
            return {}
    
    def analyze_frame_content(self, description, transcript=""):
        """Analyze frame content with smart rules"""
        text = f"{description} {transcript}".lower()
        
        # Base compliance score
        compliance = 80
        violations = 0
        confidence = 0.85
        reasoning = "Content analyzed with smart rules"
        
        # God of War specific analysis
        if any(keyword in text for keyword in self.rules.get("gaming_content_analysis", {}).get("god_of_war", {}).get("keywords", [])):
            compliance = 70
            reasoning = "M-rated gaming content (God of War) - Age verification recommended"
            violations = 0
            confidence = 0.90
        
        # Rick and Morty specific analysis
        elif any(keyword in text for keyword in self.rules.get("gaming_content_analysis", {}).get("rick_and_morty", {}).get("keywords", [])):
            compliance = 75
            reasoning = "Adult animation content - Age-appropriate context"
            violations = 0
            confidence = 0.88
        
        # General gaming context
        elif any(keyword in text for keyword in ["game", "gaming", "character", "virtual"]):
            if any(weapon in text for weapon in ["weapon", "sword", "axe", "gun"]):
                compliance = 78
                reasoning = "Gaming content with fictional weapons - Acceptable in context"
            else:
                compliance = 82
                reasoning = "Gaming content detected - Generally compliant"
            violations = 0
            confidence = 0.85
        
        # Violence detection
        elif any(violence in text for violence in ["violence", "blood", "kill", "death"]):
            if any(context in text for context in ["game", "gaming", "virtual", "fictional"]):
                compliance = 75
                reasoning = "Violence in gaming context - Fictional content"
                violations = 0
            else:
                compliance = 45
                reasoning = "Violence detected without clear context"
                violations = 1
            confidence = 0.80
        
        # Weapons without context
        elif any(weapon in text for weapon in ["weapon", "gun", "knife", "axe"]) and not any(context in text for context in ["game", "gaming", "virtual"]):
            compliance = 50
            reasoning = "Weapons detected without clear gaming context"
            violations = 1
            confidence = 0.75
        
        # Educational content
        elif any(edu in text for edu in ["tutorial", "learn", "education", "lesson"]):
            compliance = 85
            reasoning = "Educational content - Generally policy compliant"
            violations = 0
            confidence = 0.90
        
        # Inappropriate language adjustment
        if "inappropriate language detected" in text:
            compliance -= 5
            reasoning += " (with mild language concerns)"
        
        return {
            "compliance_score": max(0, min(100, compliance)),
            "violations_detected": violations,
            "confidence": confidence,
            "reasoning": reasoning,
            "analysis_method": "smart_rules"
        }

# Global analyzer instance
smart_analyzer = ImprovedContentAnalyzer()

def analyze_content_smart(description, transcript=""):
    """Smart content analysis function"""
    return smart_analyzer.analyze_frame_content(description, transcript)
