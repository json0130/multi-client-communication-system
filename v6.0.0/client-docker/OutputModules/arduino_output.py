# modules/output/arduino_output.py
import serial
import time
import threading
import os
from typing import Optional, Dict, Any
from client import OutputModule
import logging

# Use the same logger as the main client for consistency
logger = logging.getLogger(__name__)

class ArduinoOutputModule(OutputModule):
    def __init__(self, name: str = "arduino_output", config: Dict[str, Any] = None):
        super().__init__(name, config)
        
        # Arduino settings from the config dictionary
        self.arduino_port = self.config.get('arduino_port', '/dev/ttyUSB0')
        self.baud_rate = self.config.get('arduino_baud', 115200)
        self.timeout = self.config.get('arduino_timeout', 1.0)

        # Connection state
        self.serial_connection = None
        self.connected = False
        self.monitoring = False
        self.monitor_thread = None
        self.last_esp32_message = ""

        # Callbacks (assigned by robot.py)
        self.on_connected = None
        self.on_disconnected = None
        self.on_connection_error = None

    def initialize(self) -> bool:
        """Connect to the serial port"""
        if not os.path.exists(self.arduino_port):
            logger.error(f"❌ CRITICAL ERROR: Port {self.arduino_port} does NOT exist on this Jetson!")
            logger.error("👉 Fix: Run 'ls /dev/ttyUSB*' or 'ls /dev/ttyACM*' to find the real port and update your config.")
            if self.on_connection_error:
                self.on_connection_error(f"Port {self.arduino_port} not found.")
            return False

        try:
            logger.info(f"🔌 Connecting to Arduino on {self.arduino_port} at {self.baud_rate} baud...")
            self.serial_connection = serial.Serial(
                port=self.arduino_port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            
            # --- THE ESP32 LINUX FIX ---
            # Explicitly drop DTR and RTS so the ESP32 isn't held in Bootloader/Reset mode!
            self.serial_connection.setDTR(False)
            self.serial_connection.setRTS(False)
            
            # Flush out any garbage bytes that accumulated while connecting
            self.serial_connection.reset_input_buffer()
            self.serial_connection.reset_output_buffer()
            
            # Give the ESP32 2 seconds to boot up its setup() function
            time.sleep(2) 
            self.connected = True
            
            # Fire the callback so robot.py knows we succeeded
            if self.on_connected:
                self.on_connected()
                
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Arduino: {e}")
            if self.on_connection_error:
                self.on_connection_error(str(e))
            return False

    def start(self) -> bool:
        """Start the background serial monitor"""
        if not self.connected:
            if not self.initialize():
                return False
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_serial, daemon=True)
        self.monitor_thread.start()
        return True

    def stop(self):
        """Cleanly close the serial port"""
        self.monitoring = False
        if self.connected and self.serial_connection:
            self.serial_connection.close()
            self.connected = False
            logger.info("🔌 Arduino serial port closed.")
            if self.on_disconnected:
                self.on_disconnected()

    def process_output(self, data: Any) -> bool:
        """Required by abstract OutputModule, but we use send_command instead"""
        return True

    def is_connected(self) -> bool:
        """Check if the serial port is open"""
        return self.connected

    def send_command(self, command: str) -> bool:
        """Send a string command to the Arduino over Serial"""
        if not self.connected or not self.serial_connection:
            logger.warning("⚠️ Cannot send command, Arduino not connected!")
            return False
        
        try:
            formatted_cmd = f"{command.strip()}\r\n"
            logger.info(f"⚡ Writing to Serial: {command.strip()} (with \\r\\n appended)")
            
            self.serial_connection.write(formatted_cmd.encode('utf-8'))
            self.serial_connection.flush()
            return True
        except Exception as e:
            logger.error(f"❌ Error sending command to Arduino: {e}")
            self.connected = False
            return False

    def _monitor_serial(self):
        """Background thread to read incoming messages from the Arduino"""
        while self.monitoring and self.connected:
            try:
                if self.serial_connection.in_waiting > 0:
                    line = self.serial_connection.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self.last_esp32_message = line
                        logger.debug(f"🤖 Arduino says: {line}")
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"❌ Serial monitor error: {e}")
                self.connected = False
                if self.on_disconnected:
                    self.on_disconnected()
                break
