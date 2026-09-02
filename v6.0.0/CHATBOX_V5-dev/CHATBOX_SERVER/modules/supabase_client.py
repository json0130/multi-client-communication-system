# supabase_client.py
from dotenv import load_dotenv
import os
from supabase import create_client, Client

# this will read the .env file in your project root
load_dotenv()

SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
