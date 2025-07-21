# database.py
from supabase_client import supabase

class Database:
    def __init__(self):
        self.client = supabase

    def create_user(
        self,
        client_id: str,
        name: str,
        interests: list[str] | None = None,
        health_conditions: list[str] | None = None
    ) -> int:
        """
        Insert a new user row and return its user_id.
        Initializes interests and health_conditions to empty arrays if not provided.
        """
        payload = {
            "client_id": client_id,
            "name": name,
            "interests": interests or [],
            "health_conditions": health_conditions or []
        }
        response = (
            self.client
            .table("users")
            .insert(payload)
            .select("user_id")
            .execute()
        )
        # response.data is a list of inserted rows; grab the first one's user_id
        return response.data[0]["user_id"]
    # insert() example :contentReference[oaicite:0]{index=0}

    def get_user_by_client_id(self, client_id: str) -> dict | None:
        """
        Return the user row matching this client_id, or None if not found.
        """
        response = (
            self.client
            .table("users")
            .select("*")
            .eq("client_id", client_id)
            .single()
            .execute()
        )
        return response.data
    # select() example :contentReference[oaicite:1]{index=1}

    def update_user(
        self,
        user_id: int,
        name: str | None = None,
        interests: list[str] | None = None,
        health_conditions: list[str] | None = None
    ) -> dict:
        """
        Update any subset of name, interests, health_conditions for a user.
        Returns the updated row.
        """
        updates: dict = {}
        if name is not None:
            updates["name"] = name
        if interests is not None:
            updates["interests"] = interests
        if health_conditions is not None:
            updates["health_conditions"] = health_conditions

        response = (
            self.client
            .table("users")
            .update(updates)
            .eq("user_id", user_id)
            .select("*")
            .execute()
        )
        return response.data[0]
    # update() example :contentReference[oaicite:2]{index=2}

    def add_interest(self, user_id: int, interest: str) -> dict:
        """
        Append a single interest (if not already present) and return the updated row.
        """
        user = self.get_user_by_id(user_id)
        current = user.get("interests", [])
        if interest not in current:
            return self.update_user(user_id, interests=current + [interest])
        return user

    def add_health_condition(self, user_id: int, condition: str) -> dict:
        """
        Append a single health_condition (if not already present) and return the updated row.
        """
        user = self.get_user_by_id(user_id)
        current = user.get("health_conditions", [])
        if condition not in current:
            return self.update_user(user_id, health_conditions=current + [condition])
        return user

    def get_user_by_id(self, user_id: int) -> dict | None:
        """
        Fetch a user row by its primary key.
        """
        response = (
            self.client
            .table("users")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        return response.data
