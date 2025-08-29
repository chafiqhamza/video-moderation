import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Path to your RAG database
DB_PATH = r"D:\video-moderation-second\video-moderation-second\database\content_moderation_rag.db"

# YouTube-style policy entries matching visual model categories
POLICIES = [
    {
        "category": "suggestive_content",
        "policy_text": "Sexually suggestive content: Sexually suggestive content, including sexual focus, provocative dancing, or sexual scenarios is not allowed on YouTube. This includes content that depicts sexualized behavior meant to be sexually gratifying. YouTube requires age restriction for such content. Context matters: artistic vs explicit, age appropriateness.",
        "description": "Sexually suggestive content, including sexual focus, provocative dancing, or sexual scenarios. YouTube requires age restriction for such content. Context matters: artistic vs explicit, age appropriateness.",
        "examples": ["Provocative dancing that is focused on the dancer's genitals, buttocks, or breasts", "Sexual scenarios or role-playing", "Content that depicts someone in a sexualized manner", "Suggestive poses or clothing"],
        "severity_indicators": ["sexual focus", "provocative dancing", "sexual scenarios"],
        "context_matters": ["artistic vs explicit", "age appropriateness"],
        "action_required": "age_restriction",
        "importance": 1.0
    },
    {
        "category": "misinformation",
        "policy_text": "Misinformation: Certain types of misleading or deceptive content with serious risk of egregious harm are not allowed on YouTube. This includes certain types of misinformation that can cause real-world harm, certain types of technically manipulated content, or content interfering with democratic processes.",
        "description": "False or misleading information that can cause real-world harm. This includes health misinformation, election fraud claims, and other deceptive content.",
        "examples": ["False claims about health or medical treatments", "Election fraud claims", "Conspiracy theories that can cause harm", "Technically manipulated content"],
        "severity_indicators": ["health misinformation", "election fraud claims"],
        "context_matters": ["satire vs serious claims", "opinion vs fact"],
        "action_required": "fact_check_label",
        "importance": 1.0
    },
    {
        "category": "artistic_adult_content",
        "policy_text": "Artistic or educational adult content: Artistic or educational adult content may be allowed on YouTube if it has artistic merit or educational value. However, content that focuses on sexual intent, genitals, or is pornographic in nature will be age-restricted or removed.",
        "description": "Artistic or educational adult content. Context matters: artistic merit, educational value, target audience. Action Required: age_restriction.",
        "examples": ["Classical art nudity", "Educational anatomy", "Artistic expression with adult themes", "Documentary content with mature themes"],
        "severity_indicators": ["sexual intent", "focus on genitals", "pornographic"],
        "context_matters": ["artistic merit", "educational value", "target audience"],
        "action_required": "age_restriction",
        "importance": 0.8
    },
    {
        "category": "safe_content",
        "policy_text": "Safe content: Content that complies with all YouTube policies and promotes a positive experience. This includes educational content, entertainment, news, tutorials, and other appropriate material.",
        "description": "Content that complies with all YouTube policies and promotes a positive experience.",
        "examples": ["Educational content", "Entertainment", "News", "Tutorials", "Community & family-friendly content"],
        "severity_indicators": [],
        "context_matters": [],
        "action_required": "none",
        "importance": 0.5
    },
    {
        "category": "violence",
        "policy_text": "Violent or graphic content: Violent or gory content intended to shock or disgust viewers, or content encouraging others to commit violent acts, are not allowed on YouTube.",
        "description": "Content depicting violence, weapons, or graphic imagery that can shock or disgust viewers.",
        "examples": ["Graphic violence", "Weapon usage", "Gory content", "Content encouraging violence"],
        "severity_indicators": ["realistic weapons", "explicit violence", "gore"],
        "context_matters": ["gaming violence vs real violence", "educational vs gratuitous"],
        "action_required": "human_review_required",
        "importance": 1.0
    },
    {
        "category": "hate_speech",
        "policy_text": "Hate speech: Content that promotes hatred against protected groups is not allowed on YouTube. This includes discriminatory symbols, hateful rhetoric, and supremacist content.",
        "description": "Content promoting hatred against protected groups based on race, religion, gender, or other protected characteristics.",
        "examples": ["Racial slurs", "Hateful rhetoric", "Supremacist content", "Discriminatory symbols"],
        "severity_indicators": ["nazi symbols", "racial slurs", "incitement"],
        "context_matters": ["historical education vs promotion", "criticism vs hatred"],
        "action_required": "immediate_removal",
        "importance": 1.0
    }
]

def create_embeddings_model():
    """Create sentence transformer model for embeddings"""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except:
        print("Warning: sentence-transformers not available, skipping embeddings")
        return None

def upsert_policy(conn, policy_data, embeddings_model=None):
    cursor = conn.cursor()
    
    # Check if category exists
    cursor.execute("SELECT COUNT(*) FROM policy_embeddings WHERE category = ?", (policy_data["category"],))
    exists = cursor.fetchone()[0]
    
    # Create embedding if model available
    embedding_blob = None
    if embeddings_model:
        embedding = embeddings_model.encode([policy_data["policy_text"]])[0]
        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
    
    examples_json = json.dumps(policy_data["examples"], ensure_ascii=False)
    
    # Create enhanced description that includes all policy details
    enhanced_description = {
        "description": policy_data["description"],
        "severity_indicators": policy_data["severity_indicators"],
        "context_matters": policy_data["context_matters"],
        "action_required": policy_data["action_required"]
    }
    description_json = json.dumps(enhanced_description, ensure_ascii=False)
    
    if exists:
        if embedding_blob:
            cursor.execute("""
                UPDATE policy_embeddings 
                SET policy_text = ?, embedding = ?, description = ?, examples = ?, importance = ?
                WHERE category = ?
            """, (policy_data["policy_text"], embedding_blob, description_json, 
                  examples_json, policy_data["importance"], policy_data["category"]))
        else:
            cursor.execute("""
                UPDATE policy_embeddings 
                SET policy_text = ?, description = ?, examples = ?, importance = ?
                WHERE category = ?
            """, (policy_data["policy_text"], description_json, 
                  examples_json, policy_data["importance"], policy_data["category"]))
    else:
        cursor.execute("""
            INSERT INTO policy_embeddings 
            (policy_text, embedding, category, importance, description, examples)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (policy_data["policy_text"], embedding_blob, policy_data["category"], 
              policy_data["importance"], description_json, examples_json))
    
    conn.commit()

if __name__ == "__main__":
    # Create embeddings model
    embeddings_model = create_embeddings_model()
    
    conn = sqlite3.connect(DB_PATH)
    
    # Ensure the policy_embeddings table exists with correct schema
    conn.execute('''
        CREATE TABLE IF NOT EXISTS policy_embeddings (
            id INTEGER PRIMARY KEY,
            policy_text TEXT,
            embedding BLOB,
            category TEXT,
            importance REAL,
            description TEXT,
            examples TEXT
        )
    ''')
    
    for policy in POLICIES:
        upsert_policy(conn, policy, embeddings_model)
    
    conn.close()
    print("✅ Database updated with YouTube-style policy explanations for visual model categories.")
