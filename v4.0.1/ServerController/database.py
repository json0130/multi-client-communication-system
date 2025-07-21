# database.py
from typing import Optional, List, Dict, Any
from supabase_client import supabase

class Database:
    def __init__(self):
        self.client = supabase

    def create_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        interests: Optional[List[str]] = None,
        health_conditions: Optional[List[str]] = None
    ) -> int:
        """
        Insert a new user row and return its user_id.
        Initializes interests and health_conditions to empty lists if not provided.
        """
        payload: Dict[str, Any] = {
            "user_id": user_id,
            "name": name,
            "interests": interests or [],
            "health_conditions": health_conditions or []
        }
        resp = (
            self.client
            .table("Users")
            .insert(payload)
            .execute()
        )
        # resp.data is a list of inserted rows; grab the first one's user_id
        return resp.data[0]["user_id"]

    def get_user_by_user_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Return the user row matching this user_id, or None if not found.
        """
        resp = (
            self.client
            .table("Users")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        return resp.data

    def update_user(
        self,
        user_id: int,
        name: Optional[str] = None,
        interests: Optional[List[str]] = None,
        health_conditions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Update any subset of name, interests, health_conditions for a user.
        Returns the updated row.
        """
        updates: Dict[str, Any] = {}
        if name is not None:
            updates["name"] = name
        if interests is not None:
            updates["interests"] = interests
        if health_conditions is not None:
            updates["health_conditions"] = health_conditions

        resp = (
            self.client
            .table("Users")
            .update(updates)
            .eq("user_id", user_id)
            .execute()
        )
        return resp.data[0]

    def add_interest(self, user_id: int, interest: str) -> Dict[str, Any]:
        """
        Append a single interest (if not already present) and return the updated row.
        """
        user = self.get_user_by_id(user_id) or {}
        current: List[str] = user.get("interests", [])
        if interest not in current:
            return self.update_user(user_id, interests=current + [interest])
        return user

    def add_health_condition(self, user_id: int, condition: str) -> Dict[str, Any]:
        """
        Append a single health_condition (if not already present) and return the updated row.
        """
        user = self.get_user_by_id(user_id) or {}
        current: List[str] = user.get("health_conditions", [])
        if condition not in current:
            return self.update_user(user_id, health_conditions=current + [condition])
        return user

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch a user row by its primary key.
        """
        resp = (
            self.client
            .table("Users")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        return resp.data
