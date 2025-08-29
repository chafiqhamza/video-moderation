#!/usr/bin/env python3
import sqlite3, json, os, sys
DB = r"D:/video-moderation-second/video-moderation-second/database/content_moderation_rag.db"
OUT = r"D:/video-moderation-second/video-moderation-second/tools/policies_dump.json"
result = {"db": DB, "tables": {}}
try:
    if not os.path.exists(DB):
        print(json.dumps({"error": "DB not found", "path": DB}))
        sys.exit(1)
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    for t in tables:
        try:
            cur.execute(f"PRAGMA table_info('{t}')")
            cols = [c[1] for c in cur.fetchall()]
            # fetch up to 100 rows
            rows = []
            try:
                cur.execute(f"SELECT * FROM '{t}' LIMIT 100")
                raws = cur.fetchall()
                for r in raws:
                    obj = {}
                    for i, v in enumerate(r):
                        col = cols[i] if i < len(cols) else f"col{i}"
                        # try to decode bytes
                        if isinstance(v, (bytes, bytearray)):
                            try:
                                v = v.decode('utf-8', errors='ignore')
                            except:
                                v = str(v)
                        obj[col] = v
                    rows.append(obj)
            except Exception as e:
                rows = ["unable to select rows: " + str(e)]
            result['tables'][t] = {"columns": cols, "rows": rows, "count": len(rows)}
        except Exception as e:
            result['tables'][t] = {"error": str(e)}
    conn.close()
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({"ok": True, "out": OUT}))
except Exception as exc:
    print(json.dumps({"error": str(exc)}))
    sys.exit(2)
