import json
from utils.connection import Connection

conn = Connection(ip="172.24.192.51", port=8001, type='client')
conn.send(json.dumps({"command": "record", "content": None}).encode())
fuck