from supabase import create_client
from datetime import datetime

class SupabaseClient:
    def __init__(self, url, key):
        self.supabase = create_client(url, key)
    
    def register_robot(self, config: dict) -> dict:
        """
        Register or update a robot in the database
        Returns the registered robot data or None if failed
        """
        try:
            robot_data = {
                "client_id": config["client_id"],
                "robot_name": config["robot_name"],
                "robot_role": config["robot_role"],
                'is_active': True,
                'allowed_tags': config.get('allowed_tags', ['[DEFAULT]']),
                "modules": config.get("modules", []),
                "hardware_config": config.get("hardware", {}),
                "voice_config": config.get("voice_config", {})
                # "last_connected": datetime.now().isoformat()
            }

            result = self.supabase.table("robots")\
                .upsert(robot_data, on_conflict="client_id")\
                .execute()

            if result.data:
                print(f"Robot {config['robot_name']} registered successfully")
                return result.data[0]
            return None

        except Exception as e:
            print(f"Error registering robot: {e}")
            return None