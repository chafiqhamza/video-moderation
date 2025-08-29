import sqlite3
import os
import json
from datetime import datetime


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


def get_video_report(video_id: int):
    """Return the parsed report stored for a video id, or None if not found.

    The report column stores a JSON string (or raw string); attempt to parse to a dict.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT report FROM videos WHERE id = ?', (video_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    raw = row[0]
    try:
        return json.loads(raw)
    except Exception:
        try:
            return json.loads(str(raw))
        except Exception:
            return None


def save_recommendations(video_id: int, recommendations: list):
    """Persist a JSON blob of recommendations for a given video id.

    This will create the recommendations table if it doesn't exist.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Ensure recommendations table exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            recommendations TEXT NOT NULL,
            FOREIGN KEY(video_id) REFERENCES videos(id)
        )
    ''')
    rec_json = json.dumps(recommendations, ensure_ascii=False)
    created = datetime.now().isoformat()
    c.execute('INSERT INTO recommendations (video_id, created_at, recommendations) VALUES (?, ?, ?)', (video_id, created, rec_json))
    conn.commit()
    conn.close()


def get_recommendations_for_video(video_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, created_at, recommendations FROM recommendations WHERE video_id = ? ORDER BY created_at DESC', (video_id,))
    rows = c.fetchall()
    conn.close()
    # return parsed JSON for convenience
    result = []
    for row in rows:
        try:
            recs = json.loads(row[2])
        except Exception:
            recs = row[2]
        result.append({ 'id': row[0], 'created_at': row[1], 'recommendations': recs })
    return result


def save_applied_actions(video_id: int, applied_payload: dict):
    """Save a record of applied recommendations/actions for a video and return the inserted id."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Ensure applied_actions table exists with extended schema
    c.execute('''
        CREATE TABLE IF NOT EXISTS applied_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL,
            result_path TEXT,
            status TEXT DEFAULT 'pending',
            started_at TEXT,
            finished_at TEXT,
            error_message TEXT
        )
    ''')
    created = datetime.now().isoformat()
    payload_json = json.dumps(applied_payload, ensure_ascii=False)
    # Insert with initial 'pending' status
    c.execute('INSERT INTO applied_actions (video_id, created_at, payload, result_path, status) VALUES (?, ?, ?, ?, ?)', (video_id, created, payload_json, None, 'pending'))
    inserted_id = c.lastrowid
    conn.commit()
    conn.close()
    return inserted_id


def get_applied_action(video_id: int, applied_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, video_id, created_at, payload, result_path, status, started_at, finished_at, error_message FROM applied_actions WHERE video_id = ? AND id = ?', (video_id, applied_id))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    try:
        payload = json.loads(row[3])
    except Exception:
        payload = row[3]
    return {
        'id': row[0],
        'video_id': row[1],
        'created_at': row[2],
        'payload': payload,
        'result_path': row[4],
        'status': row[5],
        'started_at': row[6],
        'finished_at': row[7],
        'error_message': row[8]
    }


def fetch_pending_applied_actions(limit: int = 5):
    """Return a list of pending applied_actions records for background processing."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, video_id, created_at, payload FROM applied_actions WHERE status = ? ORDER BY created_at ASC LIMIT ?', ('pending', limit))
    rows = c.fetchall()
    conn.close()
    result = []
    for r in rows:
        try:
            payload = json.loads(r[3])
        except Exception:
            payload = r[3]
        result.append({'id': r[0], 'video_id': r[1], 'created_at': r[2], 'payload': payload})
    return result


def update_applied_action(applied_id: int, **fields):
    """Update fields for an applied_actions row. Accepts result_path, status, started_at, finished_at, error_message."""
    allowed = {'result_path', 'status', 'started_at', 'finished_at', 'error_message'}
    kv = {k: v for k, v in fields.items() if k in allowed}
    if not kv:
        return False
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    sets = ', '.join([f"{k} = ?" for k in kv.keys()])
    params = list(kv.values())
    params.append(applied_id)
    sql = f'UPDATE applied_actions SET {sets} WHERE id = ?'
    c.execute(sql, params)
    conn.commit()
    conn.close()
    return True


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
    # Also ensure recommendations table exists on init
    c.execute('''
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            recommendations TEXT NOT NULL,
            FOREIGN KEY(video_id) REFERENCES videos(id)
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized: videos & recommendations tables created (or already exist) at {DB_PATH}")


if __name__ == '__main__':
    init_db()
