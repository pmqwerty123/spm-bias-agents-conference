"""Storage.

Self-contained: no MOSAIC module is imported. The three MOSAIC tables this
experiment writes -- users, posts, follows -- are declared here with the same
columns and types MOSAIC's DatabaseManager uses, so a run still opens cleanly in
MOSAIC's own tooling, and then the four tables the SPM/SW-MSR experiment needs are
added on top.

Declaring the columns rather than importing them keeps this repository free of
third-party source. MOSAIC ships no LICENSE file, so its code is all-rights-
reserved by default despite the paper describing it as open-source; a table layout
is the interoperability contract, not the implementation.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3

# Column-compatible with MOSAIC's DatabaseManager.create_tables for the tables this
# experiment touches. Only the columns actually written are declared; MOSAIC creates
# several more tables (community_notes, note_ratings, spread_metrics, fact_checks,
# agent_memories, moderation_logs) that nothing here uses.
MOSAIC_TABLES = {
    "users": """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            persona TEXT,
            background_labels JSON,
            creation_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            follower_count INTEGER DEFAULT 0,
            total_likes_received INTEGER DEFAULT 0,
            total_shares_received INTEGER DEFAULT 0,
            total_comments_received INTEGER DEFAULT 0,
            influence_score FLOAT DEFAULT 0.0,
            is_influencer BOOLEAN DEFAULT FALSE,
            last_influence_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "posts": """
        CREATE TABLE IF NOT EXISTS posts (
            post_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            author_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            num_likes INTEGER DEFAULT 0,
            num_shares INTEGER DEFAULT 0,
            num_flags INTEGER DEFAULT 0,
            num_comments INTEGER DEFAULT 0,
            original_post_id TEXT,
            is_news BOOLEAN DEFAULT FALSE,
            news_type TEXT,
            status TEXT CHECK(status IN ('active', 'taken_down')),
            takedown_timestamp TIMESTAMP,
            takedown_reason TEXT,
            fact_check_status TEXT,
            fact_checked_at TIMESTAMP,
            FOREIGN KEY (author_id) REFERENCES users(user_id),
            FOREIGN KEY (original_post_id) REFERENCES posts(post_id)
        )
    """,
    "follows": """
        CREATE TABLE IF NOT EXISTS follows (
            follower_id TEXT NOT NULL,
            followed_id TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (follower_id, followed_id),
            FOREIGN KEY (follower_id) REFERENCES users(user_id),
            FOREIGN KEY (followed_id) REFERENCES users(user_id)
        )
    """,
}

EXTRA_TABLES = {
    # The series everything downstream is computed from.
    "agent_beliefs": """
        CREATE TABLE IF NOT EXISTS agent_beliefs (
            user_id TEXT NOT NULL,
            time_step INTEGER NOT NULL,
            q REAL NOT NULL,
            psi REAL NOT NULL,
            stance_text TEXT,
            is_manipulator INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, time_step)
        )
    """,
    # Lets the calibration regression recover kappa, hence s_bar.
    "private_signals": """
        CREATE TABLE IF NOT EXISTS private_signals (
            user_id TEXT NOT NULL,
            time_step INTEGER NOT NULL,
            x INTEGER NOT NULL,
            llr REAL NOT NULL,
            PRIMARY KEY (user_id, time_step)
        )
    """,
    # Ground truth for detector precision/recall, and the record of Cor. 1's phases.
    "attack_log": """
        CREATE TABLE IF NOT EXISTS attack_log (
            user_id TEXT NOT NULL,
            time_step INTEGER NOT NULL,
            policy TEXT NOT NULL,
            ref_q REAL,
            lo REAL,
            hi REAL,
            m_star REAL,
            psi_star REAL,
            phase TEXT,
            PRIMARY KEY (user_id, time_step)
        )
    """,
    # One row per (observer, target, step): every detector variant scored side by side.
    "defense_log": """
        CREATE TABLE IF NOT EXISTS defense_log (
            observer_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            time_step INTEGER NOT NULL,
            te_in REAL,
            te_out REAL,
            te_agg REAL,
            var_stat REAL,
            z_score REAL,
            flagged INTEGER NOT NULL DEFAULT 0,
            severed INTEGER NOT NULL DEFAULT 0,
            trimmed INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (observer_id, target_id, time_step)
        )
    """,
}


def logit(q: float) -> float:
    q = min(max(q, 1e-6), 1 - 1e-6)
    return math.log(q / (1 - q))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Store:
    """Thin data layer. All SQL for the experiment lives here."""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        if os.path.exists(db_path):
            os.remove(db_path)
        self.conn = sqlite3.connect(db_path, timeout=30.0)
        self.conn.execute("PRAGMA foreign_keys = ON")
        for stmt in (*MOSAIC_TABLES.values(), *EXTRA_TABLES.values()):
            self.conn.execute(stmt)
        self.conn.commit()

    # ---------- population ----------

    def add_user(self, user_id: str, persona: str, labels: dict) -> None:
        self.conn.execute(
            "INSERT INTO users (user_id, persona, background_labels) VALUES (?, ?, json(?))",
            (user_id, str(persona), json.dumps(labels)),
        )

    def add_follow(self, follower_id: str, followed_id: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO follows (follower_id, followed_id) VALUES (?, ?)",
            (follower_id, followed_id),
        )

    def neighbours(self, user_id: str) -> list[str]:
        """Who this agent listens to: the accounts it follows."""
        cur = self.conn.execute(
            "SELECT followed_id FROM follows WHERE follower_id = ? ORDER BY followed_id",
            (user_id,),
        )
        return [r[0] for r in cur.fetchall()]

    def all_follows(self) -> list[tuple[str, str]]:
        cur = self.conn.execute("SELECT follower_id, followed_id FROM follows")
        return [(r[0], r[1]) for r in cur.fetchall()]

    # ---------- per-step writes ----------

    def write_belief(self, user_id: str, step: int, q: float, text: str,
                     is_manipulator: bool) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO agent_beliefs "
            "(user_id, time_step, q, psi, stance_text, is_manipulator) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, step, q, logit(q), text, int(is_manipulator)),
        )
        # Mirror the stance into MOSAIC's posts table so the run reads as a MOSAIC run.
        self.conn.execute(
            "INSERT INTO posts (post_id, content, author_id, status) VALUES (?, ?, ?, 'active')",
            (f"post-{user_id}-{step}", text, user_id),
        )

    def write_signal(self, user_id: str, step: int, x: int, llr: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO private_signals (user_id, time_step, x, llr) "
            "VALUES (?, ?, ?, ?)",
            (user_id, step, x, llr),
        )

    def write_attack(self, user_id: str, step: int, policy: str, ref_q: float,
                     lo: float, hi: float, m_star: float, psi_star: float,
                     phase: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO attack_log "
            "(user_id, time_step, policy, ref_q, lo, hi, m_star, psi_star, phase) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, step, policy, ref_q, lo, hi, m_star, psi_star, phase),
        )

    def write_defense(self, observer_id: str, target_id: str, step: int,
                      te_in: float | None, te_out: float | None,
                      var_stat: float | None, z: float | None,
                      flagged: bool, severed: bool, trimmed: bool,
                      te_agg: float | None = None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO defense_log "
            "(observer_id, target_id, time_step, te_in, te_out, te_agg, var_stat, "
            "z_score, flagged, severed, trimmed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (observer_id, target_id, step, te_in, te_out, te_agg, var_stat, z,
             int(flagged), int(severed), int(trimmed)),
        )

    # ---------- reads ----------

    def stances_at(self, step: int) -> dict[str, tuple[float, str]]:
        """{user_id: (q, stance_text)} for one step."""
        cur = self.conn.execute(
            "SELECT user_id, q, stance_text FROM agent_beliefs WHERE time_step = ?",
            (step,),
        )
        return {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    def own_recent_stances(self, user_id: str, before_step: int, limit: int) -> list[str]:
        cur = self.conn.execute(
            "SELECT stance_text FROM agent_beliefs WHERE user_id = ? AND time_step < ? "
            "ORDER BY time_step DESC LIMIT ?",
            (user_id, before_step, limit),
        )
        return [r[0] for r in cur.fetchall()]

    def series(self, user_id: str, lo_step: int, hi_step: int) -> list[float]:
        """Contiguous q series on [lo_step, hi_step]; missing steps are dropped."""
        cur = self.conn.execute(
            "SELECT time_step, q FROM agent_beliefs "
            "WHERE user_id = ? AND time_step BETWEEN ? AND ? ORDER BY time_step",
            (user_id, lo_step, hi_step),
        )
        return [r[1] for r in cur.fetchall()]

    def series_steps(self, user_id: str, lo_step: int,
                     hi_step: int) -> list[tuple[int, float]]:
        """(step, q) pairs on [lo_step, hi_step]. Unlike series(), the step index
        is kept, so several agents' series can be aligned before averaging."""
        cur = self.conn.execute(
            "SELECT time_step, q FROM agent_beliefs "
            "WHERE user_id = ? AND time_step BETWEEN ? AND ? ORDER BY time_step",
            (user_id, lo_step, hi_step),
        )
        return list(cur.fetchall())

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
