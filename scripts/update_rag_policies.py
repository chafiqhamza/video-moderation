import sqlite3
import json

# Path to your RAG database
DB_PATH = r"D:\video-moderation-second\video-moderation-second\database\content_moderation_rag.db"

# Example YouTube-style policy entries to insert/update
POLICIES = [
    {
        "category": "suggestive_content",
        "policy_info": {
            "description": "Sexually suggestive content, including sexual focus, provocative dancing, or sexual scenarios. YouTube requires age restriction for such content. Context matters: artistic vs explicit, age appropriateness.",
            "examples": ["provocative dancing", "sexual scenarios", "explicit focus"],
            "severity_indicators": ["sexual focus", "provocative dancing", "sexual scenarios"],
            "context_matters": ["artistic vs explicit", "age appropriateness"],
            "action_required": "age_restriction"
        }
    },
    {
        "category": "dangerous_activities",
        "policy_info": {
            "description": "Content showing dangerous or harmful activities, especially life-threatening stunts or copyable dangers, must be age-restricted or removed. Context: professional vs amateur, safety precautions shown.",
            "examples": ["life-threatening stunts", "copyable dangers", "no safety warnings"],
            "severity_indicators": ["life-threatening", "copyable dangers", "no safety warnings"],
            "context_matters": ["professional vs amateur", "safety precautions shown"],
            "action_required": "age_restriction_or_removal"
        }
    },
    {
        "category": "safe_content",
        "policy_info": {
            "description": "Content that complies with all YouTube policies and promotes a positive experience.",
            "examples": ["educational", "entertainment", "news", "tutorials"],
            "severity_indicators": [],
            "context_matters": [],
            "action_required": "none"
        }
    }
    # Add more categories as needed
]

def upsert_policy(conn, category, policy_info):
    cursor = conn.cursor()
    # Check if category exists
    cursor.execute("SELECT COUNT(*) FROM policies WHERE category = ?", (category,))
    exists = cursor.fetchone()[0]
    policy_json = json.dumps(policy_info, ensure_ascii=False)
    if exists:
        cursor.execute("UPDATE policies SET policy_info = ? WHERE category = ?", (policy_json, category))
    else:
        cursor.execute("INSERT INTO policies (category, policy_info) VALUES (?, ?)", (category, policy_json))
    conn.commit()

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    for entry in POLICIES:
        upsert_policy(conn, entry["category"], entry["policy_info"])
    conn.close()
    print("✅ Database updated with YouTube-style policy explanations.")
