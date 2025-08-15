import sqlite3

DB_PATH = '../../data/content_moderation.db'

ALTERS = [
    "ALTER TABLE videos ADD COLUMN status TEXT;",
    "ALTER TABLE videos ADD COLUMN result_json TEXT;"
]

def migrate_videos_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for alter in ALTERS:
        try:
            cursor.execute(alter)
            print(f"✅ Migration applied: {alter}")
        except sqlite3.OperationalError as e:
            if 'duplicate column name' in str(e):
                print(f"ℹ️ Column already exists: {alter}")
            else:
                print(f"❌ Migration error: {e}")
    conn.commit()
    conn.close()
    print('✅ Database migration complete.')

if __name__ == '__main__':
    migrate_videos_table()
