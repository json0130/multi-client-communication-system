import socket
import logging
import time
import re

logger = logging.getLogger(__name__)

class RoboticsOutputModule:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.enabled = False
        self.conn = None
        self.host = self.config.get('host', 'localhost')
        self.port = self.config.get('port', 7790)
        
    def set_client(self, client):
        """Set reference to main client for communication"""
        # This module doesn't strictly need the client reference, 
        # but it must implement the method to avoid the registration error.
        pass
    
    def initialize(self) -> bool:
        """Required by OutputModule abstract base class"""
        return True

    def process_output(self, command):
        if not self.enabled:
            logger.warning("⚠️ Attempted to send command, but Silbot connection is not enabled.")
            return

        # 1. Extract the actual text content from either a dict or a string
        full_text = ""
        if isinstance(command, dict):
            # Handles the call from BasicClient.process_server_response
            full_text = command.get('text', '')
        elif isinstance(command, str):
            # Handles the call from SimpleConcurrentClient.on_chat_response (the plain string)
            full_text = command
        else:
            logger.warning(f"⚠️ Robotics module received non-text/dict input: {command}")
            return
            
        clean_text = full_text.lower()
        print(f"Robotics Module processing text: {clean_text}")

        # 2. Use Regex to find the first command tag: [ANYTHING_HERE]
        # The pattern r"\[(.*?)\]" captures the text inside the first pair of brackets
        tag_match = re.search(r"\[(.*?)\]", clean_text)
        
        command_to_send = ""

        if tag_match:
            # Extract the content inside the brackets and clean it up (e.g., [think] -> think)
            extracted_tag = tag_match.group(1).strip().lower()
            print(f"✅ Extracted command tag: {extracted_tag}")

            # Now, we map the extracted tag to the simple command word the server expects.
            if "wave" in extracted_tag or "waving" in extracted_tag:
                command_to_send = "wave"
            elif "think" in extracted_tag:
                # Use "think" to match the capitalization in your server code for the gesture name
                command_to_send = "think"
            else:
                # Fallback: If an unknown tag is found, send the tag content directly (e.g., [custom_move] -> custom_move)
                command_to_send = extracted_tag

        # 3. Handle the output:
        # If no explicit action tag was found, send a default command for speaking.
        if not command_to_send:
            logger.info("🤖 No explicit action tag found. Sending default 'think'.")
            command_to_send = "think"

        # Ensure we have a string command to work with
        if not isinstance(command_to_send, str) or not command_to_send:
            logger.warning(f"⚠️ Robotics module generated invalid or empty command: {command_to_send}")
            return
        
        try:
            logger.info(f"🤖 Sending resolved action command: {command_to_send}")
            # Use the extracted string, which can now be encoded
            self.conn.sendall(command_to_send.encode('utf-8')) 
            logger.info(f"✅ Successfully sent command: {command_to_send}")
        except ConnectionResetError:
            logger.error("❌ Connection was reset by Silbot. Attempting to reconnect...")
            self.start()  # Try to reconnect
        except Exception as e:
            logger.error(f"❌ Failed to send command: {e}")
            self.enabled = False  # Disable on error

    def start(self):
        try:
            logger.info(f"🔄 Connecting to Silbot at {self.host}:{self.port}...")
            self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.conn.settimeout(None)
            self.conn.connect((self.host, self.port))
            
            # --- FIX: Send an immediate handshake/ping ---
            handshake_message = "client_connected_ok"
            self.conn.sendall(handshake_message.encode('utf-8'))
            logger.info(f"Handshake sent: {handshake_message}")
            # ---------------------------------------------
            
            time.sleep(0.1)
            
            self.enabled = True
            logger.info(f"✅ Successfully connected to Silbot machine at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Silbot machine: {e}")
            self.enabled = False

    def shutdown(self):
        if self.conn:
            self.conn.close()