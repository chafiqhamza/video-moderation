import sqlite3
import os


DB_PATH = os.path.join(os.path.dirname(__file__), 'content_moderation.db')

def save_video_report(filename, report, user=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    upload_time = datetime.now().isoformat()
    c.execute('''
        INSERT INTO videos (filename, upload_time, user, report)
        VALUES (?, ?, ?, ?)
    ''', (filename, upload_time, user, report))
    conn.commit()
    conn.close()

def get_all_videos():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, filename, upload_time, user, report FROM videos')
    rows = c.fetchall()
    conn.close()
    return rows

from datetime import datetime

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_time TEXT NOT NULL,
            user TEXT,
            report TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized: videos table created (or already exists) at {DB_PATH}")

if __name__ == '__main__':
    init_db()
