# database.py
from supabase_client import supabase

class Database:
    def __init__(self):
        self.client = supabase

    def create_user(self, client_id: str, name: str | None = None) -> int:
        """Insert a new user and return its user_id."""
        response = (
            self.client
            .table("users")
            .insert({"client_id": client_id, "name": name})
            .select("user_id")
            .execute()
        )
        # `.data[0]["user_id"]` holds the new ID
        return response.data[0]["user_id"]
    # :contentReference[oaicite:3]{index=3}

    def get_user_by_client_id(self, client_id: str) -> dict | None:
        """Fetch a single user row by client_id."""
        response = (
            self.client
            .table("users")
            .select("*")
            .eq("client_id", client_id)
            .single()
            .execute()
        )
        return response.data  # None if not found
    # :contentReference[oaicite:4]{index=4}

    def add_topic(self, user_id: int, topic_name: str) -> None:
        """Associate a topic with a user."""
        self.client.table("topics").insert({
            "user_id": user_id,
            "topic_name": topic_name
        }).execute()
    # :contentReference[oaicite:5]{index=5}

    def add_log(self, user_id: int, log_type: str, content: dict) -> None:
        """Store an arbitrary JSON log entry."""
        self.client.table("raw_logs").insert({
            "user_id":  user_id,
            "log_type": log_type,
            "content":  content
        }).execute()
    # :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

    def get_user_logs(self, user_id: int) -> list[dict]:
        """Return all logs for a user, newest first."""
        response = (
            self.client
            .table("raw_logs")
            .select("*")
            .eq("user_id", user_id)
            .order("timestamp", {"ascending": False})
            .execute()
        )
        return response.data
