# Ultimate Video Content Moderation System – Detailed Architecture

## Introduction
Ce document détaille l’architecture complète du projet, avec le rôle et l’interaction de chaque script et composant.

---

## Structure des dossiers principaux

- `backend/` : Scripts d’analyse, modèles, base de données, rapports.
- `frontend/` : Interface utilisateur pour visualiser les résultats.
- `scripts/` : Utilitaires pour le téléchargement et la préparation des datasets.
- `coco_data/` : Données COCO pour l’entraînement.

---

## Scripts et leur rôle

### 1. **backend/ultimate_dataset_trainer.py**
- Définition du modèle principal (`UltimateModel` – EfficientNet-B0).
- Pipeline d’analyse vidéo : extraction des frames, classification, analyse audio, OCR, BLIP.
- Intégration du RAG : récupération des règles depuis la base `content_moderation_rag.db`.
- Calcul du score de conformité et génération des recommandations.
- Génération du rapport d’analyse.

### 2. **backend/ultimate_video_analyzer.py**
- Analyse vidéo avancée : gestion des modules BLIP, Whisper, EasyOCR, RAG.
- Chargement des modèles et initialisation des composants.
- Peut être appelé par d’autres scripts pour l’analyse.

### 3. **backend/services/image_analyzer.py**
- Fonctions utilitaires pour l’analyse d’images (prétraitement, classification, etc.).

### 4. **backend/services/title_analyzer.py**
- Analyse des titres de vidéos (détection de clickbait, etc.).

### 5. **backend/services/voice_analyzer.py**
- Analyse de la voix/audio (détection de mots clés, transcription, etc.).

### 6. **backend/services/youtube_processor.py**
- Traitement spécifique des vidéos YouTube (extraction, formatage, etc.).

### 7. **backend/rag_content_supervision.py**
- Logique avancée pour la supervision RAG : recherche de règles, génération d’explications.

### 8. **backend/comprehensive_realistic_training.py**
- Script d’entraînement avancé pour le modèle sur différents datasets.

### 9. **backend/video_analysis_reports/**
- Dossier contenant les rapports JSON générés après chaque analyse vidéo.

### 10. **backend/content_moderation_rag.db**
- Base de données SQLite contenant les règles et politiques de modération utilisées par le RAG.

### 11. **scripts/download_coco_data.py**
- Téléchargement et extraction des fichiers COCO (images et annotations) avec barre de progression.

### 12. **scripts/coco_safe_content_dataset.py**
- Création d’un dataset COCO filtré pour le contenu « safe ».
- Utilisé pour l’entraînement/fine-tuning du modèle.

---

## Interaction des composants

1. **L’utilisateur lance une analyse vidéo via le frontend ou un script backend.**
2. **`ultimate_dataset_trainer.py`** extrait les frames et l’audio, puis :
   - Analyse chaque frame avec `UltimateModel`.
   - Génère une description avec BLIP (`backend/ultimate_video_analyzer.py`).
   - Extrait le texte avec EasyOCR.
   - Transcrit l’audio avec Whisper.
   - Recherche des violations dans l’audio avec `voice_analyzer.py`.
3. **Pour chaque frame/audio :**
   - Si une violation est détectée, le module RAG (`rag_content_supervision.py`) interroge la base `content_moderation_rag.db` pour récupérer la règle la plus pertinente.
   - Même pour le contenu « safe », une explication RAG est générée.
4. **Le score de conformité est calculé et des recommandations sont générées.**
5. **Un rapport JSON est créé dans `video_analysis_reports/`.**
6. **Le frontend lit ce rapport et affiche chaque champ (catégorie, explication RAG, score, recommandations, etc.) dans l’interface utilisateur.**

---

## Exemple de flux de données

```
[frontend/src/App_Clean.js]  <---  [backend/video_analysis_reports/rapport.json]
        ^
        |
[backend/ultimate_dataset_trainer.py]
        |
        |---> [backend/ultimate_video_analyzer.py]
        |---> [backend/services/image_analyzer.py]
        |---> [backend/services/voice_analyzer.py]
        |---> [backend/services/title_analyzer.py]
        |---> [backend/rag_content_supervision.py]
        |---> [backend/content_moderation_rag.db]
        |
        |---> [scripts/coco_safe_content_dataset.py]
        |---> [scripts/download_coco_data.py]
        |
        |---> [coco_data/]
```

---

## Résumé

- Chaque script a un rôle précis dans la chaîne d’analyse et de modération.
- Les modèles (EfficientNet, BLIP, Whisper, EasyOCR) interagissent pour extraire et analyser le contenu.
- Le RAG utilise la base de données pour fournir des explications et recommandations dynamiques.
- Le frontend affiche tout le résultat de façon claire et détaillée.

---

## Extensibilité

- Tu peux ajouter/modifier des règles dans la base sans toucher au code.
- Les modèles peuvent être améliorés ou remplacés facilement.
- L’architecture est modulaire et adaptée à différents types de contenus ou politiques.
