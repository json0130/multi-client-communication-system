# database.py
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from supabase_client import SupabaseClient

class Database:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')

        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
            
        self.client = SupabaseClient(url=supabase_url, key=supabase_key)

    def create_user(
        self,
        name: Optional[str] = None,
        interests: Optional[List[str]] = None,
        health_conditions: Optional[List[str]] = None
    ) -> int:
        """
        Insert a new user row and return its user_id (integer).
        Initializes interests and health_conditions to empty lists if not provided.
        """
        payload: Dict[str, Any] = {
            "name": name,
            "interests": interests or [],
            "health_conditions": health_conditions or []
        }
        resp = (
            self.client.supabase
            .table("users")
            .insert(payload)
            .execute()
        )
        return resp.data[0]["user_id"]  # Returns integer user_id

    def get_user_by_user_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Return the user row matching this user_id (integer), or None if not found.
        """
        resp = (
            self.client.supabase
            .table("users")
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
            self.client.supabase
            .table("users")
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
        Fetch a user row by its primary key (integer).
        """
        resp = (
            self.client.supabase
            .table("users")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        return resp.data
    
    def insert_chat_log(
        self,
        user_id: int,  # Now consistently using integer
        message: str,
        response: str
    ) -> int:
        """
        Insert a new chat log for a user
        """
        payload = {
            "user_id": user_id,
            "message": message,
            "response": response
        }
        
        resp = (
            self.client.supabase
            .table("chat_logs")
            .insert(payload)
            .execute()
        )
        return resp.data[0]["id"]
    
    def get_chat_messages(
        self, user_id: int, limit: int = 100
    ) -> List[str]:
        """
        Return the newest <limit> user messages (string only) for topic inference.
        """
        resp = (
            self.client.supabase.table("chat_logs")
            .select("message")
            .eq("user_id", user_id)
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        return [row["message"] for row in resp.data or [] if row.get("message")]
    
    def get_active_robots(self) -> list:
        """
        Fetch all active robots and their roles from the Supabase table.
        Replace 'robots' with your actual table name if it's different!
        """
        try:
            resp = (
                self.client.supabase
                .table("robots")  # <--- Change this if your table name is different
                .select("client_id, role, status")
                .eq("status", "active") # Only get online/active robots
                .execute()
            )
            return resp.data or []
        except Exception as e:
            print(f"❌ Failed to fetch active robots from DB: {e}")
            return []