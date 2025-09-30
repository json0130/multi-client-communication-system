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

    def start(self):
        try:
            self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.conn.connect((self.host, self.port))
            self.enabled = True
            logger.info(f"✅ Successfully connected to Silbot machine at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Silbot machine: {e}")
            self.enabled = False

    def process_output(self, command):
        if not self.enabled:
            logger.warning("Attempted to send command, but Silbot connection is not enabled.")
            return

        try:
            self.conn.sendall(command.encode('utf-8'))
            logger.info(f"Sent command: {command}")
        except Exception as e:
            logger.error(f"Failed to send command: {e}")

    def shutdown(self):
        if self.conn:
            self.conn.close()