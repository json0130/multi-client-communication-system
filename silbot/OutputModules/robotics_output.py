import socket
import sys
import logging

logger = logging.getLogger(__name__)

class RoboticsOutputModule:
    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.enabled = False
        self.conn = None
        self.host = self.config.get('host', 'localhost')
        self.port = self.config.get('port', 7790)

    def process_output(self, command):
        if not self.enabled:
            logger.warning("⚠️ Attempted to send command, but Silbot connection is not enabled.")
            return

        try:
            logger.info(f"🤖 Sending command to Silbot: {command}")
            self.conn.sendall(command.encode('utf-8'))
            logger.info(f"✅ Successfully sent command: {command}")
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
            
            self.enabled = True
            logger.info(f"✅ Successfully connected to Silbot machine at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Silbot machine: {e}")
            self.enabled = False

    def shutdown(self):
        if self.conn:
            self.conn.close()