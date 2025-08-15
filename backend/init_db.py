import sqlite3

# Path to your database file
DB_PATH = '../../data/content_moderation.db'

def create_videos_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user TEXT,
            report TEXT,
            status TEXT,
            result_json TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print('✅ videos table created or already exists.')

if __name__ == '__main__':
    create_videos_table()
