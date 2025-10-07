import socket
import logging
import time

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

        # ==========================================================
        # ✅ FIX: Extract the string command from the input object
        # The BasicClient sends a dict, but SimpleConcurrentClient sends a str.
        # We handle both by preferring the 'text' key if it's a dict.
        # ==========================================================
        command_to_send = command
        if isinstance(command, dict):
            # This handles the call from BasicClient.process_server_response
            command_to_send = command.get('text', '')
            print(f"Extracted command from dict: {command_to_send.lower()}")
            if "hello" in command_to_send.lower() or "hi" in command_to_send.lower():
                command_to_send = "Think"
            elif "think" in command_to_send.lower() or "believe" in command_to_send.lower():
                command_to_send = "Think"
        
        # Ensure we have a string command to work with
        if not isinstance(command_to_send, str) or not command_to_send:
            logger.warning(f"⚠️ Robotics module received invalid or empty command: {command_to_send}")
            return
        
        try:
            logger.info(f"🤖 Sending command to Silbot: {command_to_send}")
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