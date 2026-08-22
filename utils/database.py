"""
ScanDatabase
-------------
SQLite persistence layer for scan history.
Stores every scan: image path, predicted class, confidence, timestamp,
and whether it was later flagged by the user as a "new/unknown" pest.
"""

import os
import sqlite3
from datetime import datetime
from kivy.utils import platform

if platform == "android":
    from android.storage import app_storage_path
    DB_DIR = app_storage_path()
else:
    DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage")

os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "fdcs_scans.db")


class ScanDatabase:
    def __init__(self, db_path=DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                group_name TEXT,
                flagged_new INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def add_scan(self, image_path, predicted_label, confidence, group_name, flagged_new=False):
        cur = self.conn.execute(
            """INSERT INTO scans (image_path, predicted_label, confidence, group_name, flagged_new, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (image_path, predicted_label, confidence, group_name, int(flagged_new),
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_all_scans(self, limit=200):
        cur = self.conn.execute(
            "SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)
        )
        return cur.fetchall()

    def get_scan(self, scan_id):
        cur = self.conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,))
        return cur.fetchone()

    def delete_scan(self, scan_id):
        self.conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        self.conn.commit()

    def clear_history(self):
        self.conn.execute("DELETE FROM scans")
        self.conn.commit()

    def close(self):
        self.conn.close()
