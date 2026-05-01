import aiosqlite
import json
from datetime import datetime
from .schemas import ArbitrateResponse
from .config import get_settings


async def init_db() -> None:
    settings = get_settings()
    async with aiosqlite.connect(settings.sqlite_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS arbitrations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                llm_output TEXT NOT NULL,
                original_prompt TEXT,
                overall_score INTEGER NOT NULL,
                confidence TEXT NOT NULL,
                confirmed_issue_count INTEGER NOT NULL,
                dismissed_flag_count INTEGER NOT NULL,
                disagreement_count INTEGER NOT NULL,
                full_result TEXT NOT NULL
            )
        """)
        await db.commit()


async def save_arbitration(
    llm_output: str,
    original_prompt: str | None,
    response: ArbitrateResponse,
) -> None:
    settings = get_settings()
    async with aiosqlite.connect(settings.sqlite_path) as db:
        await db.execute(
            """
            INSERT INTO arbitrations VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                response.arbitration_id,
                datetime.utcnow().isoformat(),
                llm_output,
                original_prompt,
                response.verdict.overall_score,
                response.verdict.confidence.value,
                len(response.verdict.confirmed_issues),
                len(response.verdict.dismissed_flags),
                len(response.disagreements),
                response.model_dump_json(),
            ),
        )
        await db.commit()


async def get_arbitration(arbitration_id: str) -> dict | None:
    settings = get_settings()
    async with aiosqlite.connect(settings.sqlite_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT full_result FROM arbitrations WHERE id = ?",
            (arbitration_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return json.loads(row["full_result"])


async def get_analytics() -> dict:
    settings = get_settings()
    async with aiosqlite.connect(settings.sqlite_path) as db:
        db.row_factory = aiosqlite.Row

        async with db.execute("SELECT COUNT(*) as total FROM arbitrations") as cursor:
            row = await cursor.fetchone()
            total = row["total"]

        async with db.execute("SELECT AVG(overall_score) as avg FROM arbitrations") as cursor:
            row = await cursor.fetchone()
            avg_score = round(row["avg"] or 0, 2)

        async with db.execute(
            "SELECT confidence, COUNT(*) as count FROM arbitrations GROUP BY confidence"
        ) as cursor:
            rows = await cursor.fetchall()
            confidence_dist = {r["confidence"]: r["count"] for r in rows}

        async with db.execute("SELECT AVG(disagreement_count) as avg FROM arbitrations") as cursor:
            row = await cursor.fetchone()
            avg_disagreements = round(row["avg"] or 0, 2)

        return {
            "total_arbitrations": total,
            "average_overall_score": avg_score,
            "confidence_distribution": confidence_dist,
            "average_disagreements_per_arbitration": avg_disagreements,
        }
