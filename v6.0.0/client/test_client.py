"""
mock_jay.py
===========
Minimal mock robot for local testing.
No hardware. No Docker. Just text chat.

Run: python3 mock_jay.py
"""

import asyncio
import websockets
import json
import re
import requests
import threading
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CONFIG = {
    "client_id":   "jay_mock_001",
    "robot_name":  "Jay",
    "robot_role":  "Your name is Jay, a friendly and helpful robot assistant.",
    "allowed_tags": ["[DEFAULT]", "[WAVE]", "[HAPPY]", "[SAD]", "[CONFUSED]"],
    "modules":     ["gpt"],
    "server_url":  "http://127.0.0.1:5000",
    "ip_address":  "127.0.0.1",
    "ws_port":     8765,
}


class MockJay:

    def __init__(self):
        self._ws = None
        self._loop = asyncio.new_event_loop()
        self._connected = threading.Event()

    def register(self):
        """Tell server: I exist at 127.0.0.1:8765"""
        url = f"{CONFIG['server_url']}/robots/register"
        try:
            resp = requests.post(url, json={
                "client_id":   CONFIG["client_id"],
                "robot_name":  CONFIG["robot_name"],
                "robot_role":  CONFIG["robot_role"],
                "allowed_tags": CONFIG["allowed_tags"],
                "modules":     CONFIG["modules"],
                "ip_address":  CONFIG["ip_address"],
                "ws_port":     CONFIG["ws_port"],
            }, timeout=5)
            if resp.status_code == 200:
                print(f"  Registered with server: {resp.json().get('message', 'OK')}")
            else:
                print(f"  Registration returned {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"  Registration failed: {e}")

    def start(self):
        self.register()

        # Start WS server in background thread
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()
        print(f"  Listening for server on port {CONFIG['ws_port']}...")
        print()
        print("  Trigger connection from another terminal:")
        print(f"    curl -X POST {CONFIG['server_url']}/robots/{CONFIG['client_id']}/connect")
        print()

        # Wait for server to connect, then start chat loop
        self._connected.wait()
        self._chat_loop()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(
            self._handle_connection, "0.0.0.0", CONFIG["ws_port"],
            ping_interval=20, ping_timeout=10
        ):
            await asyncio.Future()

    async def _handle_connection(self, websocket):
        print(f"\n  [Connected] Central server connected!")
        print(f"  Type a message and press Enter.\n")
        self._ws = websocket
        self._connected.set()
        try:
            async for raw in websocket:
                data = json.loads(raw)
                self._on_message(data)
        except websockets.ConnectionClosed:
            print("\n  [Disconnected] Server closed connection.")
        finally:
            self._ws = None
            self._connected.clear()

    def _on_message(self, data: dict):
        event = data.get("event")
        if event == "chat_response":
            response  = data.get("response", "")
            clean     = data.get("clean_text", re.sub(r"\[.*?\]", "", response).strip())
            tag       = data.get("emotion_tag", "")
            print(f"\n  Jay [{tag}]: {clean}")
            print("  You: ", end="", flush=True)
        else:
            logger.debug(f"Unknown event: {event}")

    def _send(self, msg: dict):
        if self._ws is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._ws.send(json.dumps(msg)), self._loop
        ).result(timeout=5)

    def _chat_loop(self):
        print("  You: ", end="", flush=True)
        try:
            while True:
                text = input().strip()
                if text.lower() in ("quit", "exit", "q"):
                    break
                if not text:
                    print("  You: ", end="", flush=True)
                    continue
                if not self._connected.is_set():
                    print("  [Not connected yet]")
                    continue
                self._send({"type": "chat", "message": text})
        except (KeyboardInterrupt, EOFError):
            pass
        print("\n  Goodbye.")


if __name__ == "__main__":
    print("=" * 55)
    print("  Jay — Mock Robot Client")
    print("=" * 55)
    jay = MockJay()
    jay.start()