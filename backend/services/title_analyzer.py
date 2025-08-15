"""
Service d'analyse de titre
Utilise l'API OpenAI pour analyser la conformité du titre
"""

import openai
from typing import Dict, List
import os
import re

class TitleAnalyzer:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        if self.openai_key:
            openai.api_key = self.openai_key
    
    async def analyze_title(self, title: str) -> Dict:
        """
        Analyser le titre d'une vidéo pour détecter :
        - Contenu inapproprié
        - Clickbait excessif
        - Langage offensant
        - Conformité aux règles
        """
        try:
            # Règles de base sans IA
            score = 100
            issues = []
            
            # Vérifications de base
            if self._contains_inappropriate_words(title):
                score -= 30
                issues.append("Contenu potentiellement inapproprié détecté")
            
            if self._is_excessive_clickbait(title):
                score -= 20
                issues.append("Titre clickbait excessif")
            
            if self._contains_caps_spam(title):
                score -= 15
                issues.append("Utilisation excessive de majuscules")
            
            if len(title) > 100:
                score -= 10
                issues.append("Titre trop long")
            
            if not issues:
                issues.append("Titre conforme aux règles")
            
            # Déterminer le statut
            if score >= 80:
                status = "conforme"
            elif score >= 60:
                status = "attention"
            else:
                status = "non-conforme"
            
            return {
                "score": max(0, score),
                "status": status,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "score": 0,
                "status": "erreur",
                "issues": [f"Erreur d'analyse: {str(e)}"]
            }
    
    def _contains_inappropriate_words(self, title: str) -> bool:
        """Vérifier la présence de mots inappropriés"""
        inappropriate_words = [
            "haine", "violence", "discrimination", "insulte",
            "pornographie", "drogue", "arme", "terrorism"
        ]
        title_lower = title.lower()
        return any(word in title_lower for word in inappropriate_words)
    
    def _is_excessive_clickbait(self, title: str) -> bool:
        """Détecter le clickbait excessif"""
        clickbait_patterns = [
            r"vous ne croirez jamais",
            r"incroyable.*!{3,}",
            r"choquant.*!{3,}",
            r"!{5,}",
            r".*\?\?\?+",
            r"secret.*révélé"
        ]
        
        title_lower = title.lower()
        return any(re.search(pattern, title_lower) for pattern in clickbait_patterns)
    
    def _contains_caps_spam(self, title: str) -> bool:
        """Détecter l'utilisation excessive de majuscules"""
        caps_count = sum(1 for c in title if c.isupper())
        total_letters = sum(1 for c in title if c.isalpha())
        
        if total_letters == 0:
            return False
        
        caps_ratio = caps_count / total_letters
        return caps_ratio > 0.6  # Plus de 60% en majuscules
