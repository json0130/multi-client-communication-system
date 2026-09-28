"""
data/scenario_repo.py
=====================
The 'scenarios' table (migration 014): one row per scenario profile, upserted
at boot so every scenario_id a log row carries has a parent. The YAML in
profiles/ stays the source of truth; this is its mirror.
"""

from __future__ import annotations

from typing import Optional

from data.connection import get_client


def upsert_scenario(scenario_id: str, name: str = "", profile_path: Optional[str] = None) -> bool:
    try:
        (get_client().table("scenarios")
            .upsert({"id": scenario_id, "name": name or scenario_id,
                     "profile_path": profile_path}, on_conflict="id").execute())
        return True
    except Exception as e:
        print(f"[scenario_repo] upsert_scenario error for {scenario_id}: {e}")
        return False
