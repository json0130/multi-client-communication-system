# modules/output/arduino_output.py - COMPLETE AND CORRECTED VERSION
import serial
import time
import threading
import os
from typing import Optional, Callable, Dict, Any, List
import re

class ArduinoOutputModule:
    """
    Complete Arduino output module for emotion display and control.
    This version includes the robust connection logic and all necessary
    interface methods for the main client.
    """

    def __init__(self, client_core, config: Dict[str, Any] = None):
        """
        Initialize Arduino output module
        """
        self.client_core = client_core
        self.config = config or client_core.get_config() if hasattr(client_core, 'get_config') else {}
        self.name = "arduino_output"
        self.enabled = False  # Start as disabled until initialized
        self.client = None

        # Arduino settings
        self.arduino_port = self.config.get('hardware',{}).get('arduino_port', '/dev/ttyUSB0')
        self.baud_rate = self.config.get('hardware', {}).get('arduino_baud', 115200)
        self.timeout = self.config.get('hardware', {}).get('arduino_timeout', 1.0)

        # Connection state
        self.arduino = None
        self.connected = False
        self.monitoring = False
        self.monitor_thread = None

        # Bot emotion mapping
        self.bot_emotion_mapping = {
            "GREETING": "greeting", "WAVE": "wave", "POINT": "point",
            "CONFUSED": "confused", "SHRUG": "shrug", "ANGRY": "angry",
            "SAD": "sad", "SLEEP": "sleep", "DEFAULT": "default",
            "POSE": "pose", "HAPPY": "greeting", "FEAR": "sad",
            "SURPRISE": "confused", "NEUTRAL": "default"
        }

    # --- MODIFIED: SIMPLIFIED AND ROBUST CONNECTION LOGIC ---
    def connect(self, port: Optional[str] = None) -> bool:
        """
        Connect to Arduino with a more robust and simple method.
        """
        if self.connected:
            print("✅ Arduino already connected")
            return True

        connection_port = port or self.arduino_port

        try:
            print(f"🔌 Connecting to Arduino on {connection_port}...")
            self.arduino = serial.Serial(
                port=connection_port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            print("   ... waiting for device to initialize ...")
            time.sleep(2.0)

            self.connected = True
            self.arduino_port = connection_port
            print(f"✅ Arduino connected successfully on {connection_port}")

            self._start_monitoring()
            time.sleep(1.0)
            self.send_emotion('default')
            return True

        except serial.SerialException as e:
            print(f"❌ Arduino connection failed: {e}")
            self._close_serial()
            return False

    def _close_serial(self):
        if self.arduino:
            try: self.arduino.close()
            except: pass
        self.arduino = None
        self.connected = False

    def disconnect(self):
        if not self.connected: return
        print("🔌 Disconnecting from Arduino...")
        self._stop_monitoring()
        self.send_emotion('sleep')
        time.sleep(0.5)
        self._close_serial()
        print("✅ ESP32 disconnected")

    def _start_monitoring(self):
        if self.monitoring: return
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_connection, daemon=True)
        self.monitor_thread.start()
        print("👁️ Arduino connection monitoring started")

    def _stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

    def _monitor_connection(self):
        while self.monitoring and self.connected:
            try:
                if self.arduino and self.arduino.in_waiting > 0:
                    response = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                    if response:
                        print(f"📥 ESP32: {response}") # Added for visibility
                time.sleep(0.1)
            except Exception as e:
                print(f"❌ Connection monitoring error: {e}. Disconnecting.")
                self._close_serial()
                break

    def _send_raw_command(self, command: str) -> bool:
        if not self.connected or not self.arduino:
            print(f"⚠️ Cannot send command '{command}', not connected.")
            return False
        try:
            full_command = command.strip() + '\n'
            print(f"DEBUG >>> arduino_output: Sending raw bytes: {full_command.encode('utf-8')}")
            self.arduino.write(full_command.encode('utf-8'))
            self.arduino.flush()
            return True
        except Exception as e:
            print(f"❌ Error sending command '{command.strip()}': {e}")
            self._close_serial()
            return False

    def send_emotion(self, emotion: str) -> bool:
        emotion_command = emotion.lower().strip()
        print(f"🎭 Sending emotion to ESP32: {emotion} -> {emotion_command}")
        return self._send_raw_command(emotion_command)

    def send_bot_emotion(self, bot_emotion_tag: str) -> bool:
        arduino_command = self.bot_emotion_mapping.get(bot_emotion_tag.upper(), "default")
        print(f"🤖 Sending bot emotion to Arduino: {bot_emotion_tag} -> {arduino_command}")
        return self._send_raw_command(arduino_command)

    # --- RESTORED INTERFACE METHODS FOR THE CLIENT ---

    def initialize(self) -> bool:
        """Initialize and connect to the Arduino."""
        print("🔌 Arduino output module initializing...")
        if self.config.get('features', {}).get('arduino_integration', True):
            self.enabled = self.connect()
            return self.enabled
        else:
            print("   ... Arduino integration disabled in config.")
            self.enabled = False
            return True

    def start(self) -> bool:
        """Start the module (already started on init)."""
        return self.enabled

    def stop(self):
        """Stop the module and disconnect."""
        self.disconnect()

    def cleanup(self):
        """Cleanup resources."""
        print("🧹 Cleaning up Arduino output module...")
        self.stop()

    # Replace your entire process_output function with this final version.
    def process_output(self, data: Any) -> bool:
        """Process output data - extract emotion tags and send to Arduino"""
        if not self.enabled or not self.connected:
            return False
        try:
            text = str(data.get('text', '')) if isinstance(data, dict) else str(data)
        
            emotion_match = re.match(r'^\[([A-Z_]+)\]', text)
            if emotion_match:
                emotion_tag = emotion_match.group(1).upper()
                print(f"🎭 Found emotion tag: {emotion_tag}")

                def delayed_send_and_follow_up():
                    """
                    This function runs in a separate thread.
                    It waits for the TTS event, sends the main gesture, waits, and sends a follow-up.
                    """
                    # --- 1. Wait for the TTS "starting pistol" event ---
                    print("   ... Arduino is waiting for TTS to start ...")
                    if self.client and hasattr(self.client, 'tts_started_event'):
                        # This will pause until the tts_started_event.set() call is made.
                        # The timeout is a safety net in case TTS fails.
                        was_set = self.client.tts_started_event.wait(timeout=5.0)
                        if not was_set:
                            print("   ... Arduino timed out waiting for TTS, proceeding anyway.")
                    else:
                        # Fallback to a simple sleep if the event system isn't there
                        time.sleep(2.0)


                    # --- 2. Send the main gesture ---
                    main_command = self.bot_emotion_mapping.get(emotion_tag, "default")
                    self.send_emotion(main_command)

                threading.Thread(target=delayed_send_and_follow_up, daemon=True).start()
        
            return True
        except Exception as e:
            print(f"❌ Arduino process_output error: {e}")
            return False

    def set_client(self, client):
        self.client = client
