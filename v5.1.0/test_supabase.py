import os
from dotenv import load_dotenv
from ServerController.supabase_client import SupabaseClient

load_dotenv("ServerController/.env")
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    print("No url or key")
    exit(1)

client = SupabaseClient(url, key)
templates = client.get_templates()
print("Templates:", len(templates), [t.get('id') for t in templates])

if templates:
    test_id = templates[0].get('id')
    print(f"Trying to delete {test_id}")
    res = client.delete_template(test_id)
    print("Delete result:", res)
