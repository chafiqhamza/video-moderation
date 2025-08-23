
# --- Enhanced Scraper for Specific Violation Sections ---
import requests
from bs4 import BeautifulSoup
import sqlite3
import json
from sentence_transformers import SentenceTransformer

DB_PATH = r"d:/video-moderation-second/video-moderation-second/database/content_moderation_rag.db"

VIOLATION_PAGES = [
    ("Violent or graphic content", "https://support.google.com/youtube/answer/2802008?hl=en"),
    ("Hate speech", "https://support.google.com/youtube/answer/2801939?hl=en"),
    ("Child safety", "https://support.google.com/youtube/answer/2801999?hl=en"),
    ("Nudity & Sexual Content", "https://support.google.com/youtube/answer/2802002?hl=en"),
    ("Harassment & cyberbullying", "https://support.google.com/youtube/answer/2802268?hl=en"),
    ("Spam, deceptive practices & scams", "https://support.google.com/youtube/answer/2801973?hl=en"),
    ("Misinformation", "https://support.google.com/youtube/answer/10834785?hl=en"),
    ("Suicide, self-harm, and eating disorders", "https://support.google.com/youtube/answer/2802245?hl=en"),
    # Add positive guideline categories for safe/good content
    ("Safe Content", None),
    ("Educational Content", None),
    ("Entertainment Content", None),
    ("News Content", None),
    ("Tutorials & How-To", None),
    ("Community & Family-Friendly", None),
    ("Artistic & Creative Expression", None),
    ("Positive Social Impact", None),
]

def extract_violation_section(title, url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    desc = ""
    examples = []
    # Get first non-navigation paragraph as description
    for p in soup.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if txt and not any(nav in txt for nav in ["Help Center", "Privacy Policy", "YouTube Premium", "Monetize", "Create & grow"]):
            desc = txt
            break
    # Find all <ul> or <ol> for examples, but only those near headings or with violation keywords
    for ul in soup.find_all(["ul", "ol"]):
        parent = ul.find_parent()
        # Only include lists that are near headings or have violation keywords
        heading = None
        for tag in ["h2", "h3", "h4"]:
            heading = ul.find_previous(tag)
            if heading: break
        if heading and any(k in heading.get_text().lower() for k in ["example", "not allowed", "don't post", "prohibited"]):
            for li in ul.find_all("li"):
                txt = li.get_text(" ", strip=True)
                # Filter out navigation links and unrelated items
                if txt and not any(nav in txt for nav in ["Help Center", "Privacy Policy", "YouTube Premium", "Monetize", "Create & grow"]):
                    examples.append(txt)
    # Fallback: grab all <li> with violation keywords
    if not examples:
        for li in soup.find_all("li"):
            txt = li.get_text(" ", strip=True)
            if txt and any(k in txt.lower() for k in ["not allowed", "don't post", "prohibited", "violate", "harm", "abuse", "violence", "sexual", "scam", "hate", "self-harm", "suicide"]):
                examples.append(txt)
    return {
        "category": title,
        "description": desc,
        "examples": examples[:10]  # Limit to 10 examples
    }

def scrape_violation_policies():
    sections = []
    for title, url in VIOLATION_PAGES:
        print(f"Scraping {title}...")
        if url:
            try:
                sec = extract_violation_section(title, url)
                sections.append(sec)
            except Exception as e:
                print(f"Failed to scrape {title}: {e}")
        else:
            # Add positive guideline section manually
            if title == "Safe Content":
                sec = {
                    "category": title,
                    "description": "Content that complies with all YouTube policies and promotes a positive experience.",
                    "examples": ["educational", "entertainment", "news", "tutorials", "family-friendly", "community-building", "positive social impact"]
                }
            elif title == "Educational Content":
                sec = {
                    "category": title,
                    "description": "Videos that teach, inform, or explain topics in a clear and accurate way.",
                    "examples": ["science lessons", "history documentaries", "how-to guides", "language learning"]
                }
            elif title == "Entertainment Content":
                sec = {
                    "category": title,
                    "description": "Content designed to amuse, engage, or entertain viewers in a safe and positive manner.",
                    "examples": ["comedy", "music", "gaming", "vlogs", "family shows"]
                }
            elif title == "News Content":
                sec = {
                    "category": title,
                    "description": "Videos that report on current events, issues, or stories in a factual and responsible way.",
                    "examples": ["news reports", "interviews", "public service announcements"]
                }
            elif title == "Tutorials & How-To":
                sec = {
                    "category": title,
                    "description": "Step-by-step guides and instructional videos that help viewers learn new skills.",
                    "examples": ["cooking tutorials", "tech how-tos", "DIY projects", "art lessons"]
                }
            elif title == "Community & Family-Friendly":
                sec = {
                    "category": title,
                    "description": "Content that is suitable for all ages and fosters a sense of community and inclusivity.",
                    "examples": ["family vlogs", "community events", "positive challenges", "inclusive activities"]
                }
            elif title == "Artistic & Creative Expression":
                sec = {
                    "category": title,
                    "description": "Videos that showcase creativity, art, and self-expression in a positive and respectful way.",
                    "examples": ["painting", "music performances", "dance", "film-making"]
                }
            elif title == "Positive Social Impact":
                sec = {
                    "category": title,
                    "description": "Content that inspires, educates, or helps others and contributes to a better society.",
                    "examples": ["charity work", "mental health awareness", "environmental education", "motivational talks"]
                }
            else:
                sec = {
                    "category": title,
                    "description": "Content that is positive and safe.",
                    "examples": []
                }
            sections.append(sec)
    return sections

def store_violation_sections(sections):
    # Always use the correct DB path for backend
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS policy_embeddings (
        id INTEGER PRIMARY KEY,
        policy_text TEXT,
        embedding BLOB,
        category TEXT,
        importance REAL,
        description TEXT,
        examples TEXT
    )''')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for sec in sections:
        text = f"{sec['category']}: {sec['description']}\nExamples: {', '.join(sec['examples'])}"
        embedding = model.encode([text])[0].tobytes()
        # Upsert: avoid duplicate categories
        conn.execute('''INSERT OR REPLACE INTO policy_embeddings (id, policy_text, embedding, category, importance, description, examples) VALUES ((SELECT id FROM policy_embeddings WHERE category=?),?,?,?,?,?,?)''',
            (sec['category'], text, embedding, sec['category'], 1.0, sec['description'], json.dumps(sec['examples'])))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape YouTube policies and store in DB")
    parser.add_argument('--db_path', type=str, default=DB_PATH, help="Path to SQLite DB")
    args = parser.parse_args()

    print("Scraping YouTube violation policies...")
    sections = scrape_violation_policies()
    print(f"Found {len(sections)} violation sections.")
    # Always update the specified DB path with both violation and positive policies
    DB_PATH = args.db_path
    store_violation_sections(sections)
    print("Stored violation policies and embeddings in DB.")
