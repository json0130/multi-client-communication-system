from supabase import create_client
from datetime import datetime
import json
import os

class SupabaseClient:
    def __init__(self, url, key):
        self.supabase = create_client(url, key)
    
    async def register_robot(self, config: dict) -> dict:
        try:
            robot_data = {
                "client_id": config["client_id"],
                "robot_name": config["robot_name"],
                "robot_role": config["robot_role"],
                "modules": config.get("modules", []),
                "hardware_config": config.get("hardware", {}),
                "voice_config": config.get("voice_config", {}),
                "last_connected": datetime.now().isoformat()
            }

            result = self.supabase.table("robots")\
                .upsert(robot_data, on_conflict="client_id")\
                .execute()

            return result.data[0] if result.data else None

        except Exception as e:
            print(f"Error registering robot: {e}")
            return None