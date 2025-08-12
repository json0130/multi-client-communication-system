# modules/output/arduino_output.py - Complete Arduino Output Module
import serial
import time
import threading
import os
from typing import Optional, Callable, Dict, Any, List
import re

class ArduinoOutputModule:
    """
    Complete Arduino output module for emotion display and control.
    
    Features:
    - Serial communication with Arduino
    - Emotion display commands
    - Auto-detection and connection
    - Reconnection handling
    - Bot emotion tag processing
    - Custom command sending
    """
    
    def __init__(self, client_core, config: Dict[str, Any] = None):
        """
        Initialize Arduino output module
        
        Args:
            client_core: Core client instance
            config: Configuration dictionary
        """
        self.client_core = client_core
        self.config = config or client_core.get_config() if hasattr(client_core, 'get_config') else {}
        
        # Arduino settings
        self.arduino_port = self.config.get('hardware', {}).get('arduino_port', '/dev/ttyUSB0')
        self.baud_rate = self.config.get('hardware', {}).get('arduino_baud', 9600)
        self.timeout = self.config.get('hardware', {}).get('arduino_timeout', 1.0)
        
        # Connection state
        self.arduino = None
        self.connected = False
        self.monitoring = False
        self.monitor_thread = None
        
        # Bot emotion mapping (from your original ChatBox code)
        self.bot_emotion_mapping = {
            "GREETING": "GREETING",
            "WAVE": "WAVE", 
            "POINT": "POINT",
            "CONFUSED": "CONFUSED",
            "SHRUG": "SHRUG",
            "ANGRY": "ANGRY",
            "SAD": "SAD",
            "SLEEP": "SLEEP",
            "DEFAULT": "DEFAULT",
            "POSE": "POSE",
            "HAPPY": "HAPPY",
            "FEAR": "FEAR",
            "SURPRISE": "SURPRISE",
            "NEUTRAL": "NEUTRAL"
        }
        
        # Callbacks
        self.on_connected = None        # Callback() - called when Arduino connects
        self.on_disconnected = None     # Callback() - called when Arduino disconnects
        self.on_command_sent = None     # Callback(command) - called when command is sent
        self.on_response_received = None # Callback(response) - called when response received
        self.on_connection_error = None # Callback(error_msg) - called on connection errors
        
        # Register with core client for automatic emotion responses
        if hasattr(client_core, 'register_callback'):
            self.client_core.register_callback('on_emotion_detected', self._handle_emotion_detected)
            self.client_core.register_callback('on_chat_response', self._handle_chat_response)
        
        # Auto-connect if enabled
        auto_connect = self.config.get('features', {}).get('arduino_integration', True)
        if auto_connect:
            self.connect()
        
        print(f"🔌 Arduino output module initialized")
        print(f"   📍 Port: {self.arduino_port}")
        print(f"   ⚡ Baud rate: {self.baud_rate}")
        print(f"   🔗 Auto-connect: {auto_connect}")
    
    def _handle_emotion_detected(self, emotion: str, confidence: float, distribution: Dict[str, float]):
        """Handle emotion detection from camera - send to Arduino"""
        auto_send = self.config.get('features', {}).get('auto_send_emotions', True)
        if auto_send and confidence > 30:  # Only send high-confidence emotions
            self.send_emotion(emotion)
    
    def _handle_chat_response(self, response: str, detected_emotion: str, confidence: float):
        """Handle chat response - extract and send bot emotion"""
        # Extract emotion tag from response (e.g., [HAPPY] Hello! -> HAPPY)
        match = re.match(r"^\[(.*?)\]", response)
        if match:
            bot_emotion_tag = match.group(1).upper()
            self.send_bot_emotion(bot_emotion_tag)
    
    def connect(self, port: Optional[str] = None) -> bool:
        """
        Connect to Arduino
        
        Args:
            port: Optional port override
            
        Returns:
            bool: True if connection successful
        """
        if self.connected:
            print("✅ Arduino already connected")
            return True
        
        connection_port = port or self.arduino_port
        
        try:
            print(f"🔌 Connecting to Arduino on {connection_port}...")
            
            # Create serial connection
            self.arduino = serial.Serial(
                port=connection_port,
                baudrate=self.baud_rate,
                timeout=self.timeout,
                write_timeout=self.timeout
            )
            
            # Wait for Arduino to initialize
            time.sleep(2.0)
            
            # Test connection
            if self._test_connection():
                self.connected = True
                self.arduino_port = connection_port
                
                print(f"✅ Arduino connected successfully on {connection_port}")
                
                # Start connection monitoring
                self._start_monitoring()
                
                # Call callback
                if self.on_connected:
                    self.on_connected()
                
                # Send greeting emotion
                self.send_emotion('GREETING')
                
                return True
            else:
                print(f"❌ Arduino connection test failed")
                self._close_serial()
                return False
                
        except Exception as e:
            print(f"❌ Arduino connection failed: {e}")
            if self.on_connection_error:
                self.on_connection_error(f"Connection failed: {e}")
            
            self._close_serial()
            return False
    
    def _test_connection(self) -> bool:
        """Test Arduino connection by sending a ping"""
        try:
            # Send test command
            self._send_raw_command("DEFAULT")
            
            # Wait a bit for potential response
            time.sleep(0.5)
            
            # Read any available response
            if self.arduino.in_waiting > 0:
                try:
                    response = self.arduino.readline().decode().strip()
                    print(f"🔧 Arduino response: {response}")
                except:
                    pass
            
            # If we got here without exception, connection is working
            return True
            
        except Exception as e:
            print(f"❌ Connection test error: {e}")
            return False
    
    def _close_serial(self):
        """Close serial connection safely"""
        if self.arduino:
            try:
                self.arduino.close()
            except:
                pass
            self.arduino = None
        self.connected = False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if not self.connected:
            return
        
        print("🔌 Disconnecting from Arduino...")
        
        try:
            # Stop monitoring
            self._stop_monitoring()
            
            # Send sleep command
            self.send_emotion('SLEEP')
            time.sleep(0.5)
            
            # Close connection
            self._close_serial()
            
            print("✅ Arduino disconnected")
            
            # Call callback
            if self.on_disconnected:
                self.on_disconnected()
                
        except Exception as e:
            print(f"❌ Error disconnecting Arduino: {e}")
    
    def _start_monitoring(self):
        """Start connection monitoring thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_connection, daemon=True)
        self.monitor_thread.start()
        print("👁️ Arduino connection monitoring started")
    
    def _stop_monitoring(self):
        """Stop connection monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("👁️ Arduino connection monitoring stopped")
    
    def _monitor_connection(self):
        """Monitor Arduino connection and handle responses"""
        while self.monitoring and self.connected:
            try:
                # Read any incoming responses
                if self.arduino and self.arduino.in_waiting > 0:
                    try:
                        response = self.arduino.readline().decode().strip()
                        if response:
                            print(f"🔧 Arduino: {response}")
                            if self.on_response_received:
                                self.on_response_received(response)
                    except Exception as e:
                        print(f"❌ Error reading Arduino response: {e}")
                
                # Periodic connection test (every 30 seconds)
                if time.time() % 30 < 1.0:
                    if not self._test_connection():
                        print("❌ Arduino connection lost")
                        self.connected = False
                        if self.on_disconnected:
                            self.on_disconnected()
                        break
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Connection monitoring error: {e}")
                if self.on_connection_error:
                    self.on_connection_error(f"Monitoring error: {e}")
                break
    
    def send_emotion(self, emotion: str) -> bool:
        """
        Send emotion command to Arduino
        
        Args:
            emotion: Emotion name (e.g., 'happy', 'sad', 'angry')
            
        Returns:
            bool: True if sent successfully
        """
        if not self.connected or not self.arduino:
            print(f"⚠️ Arduino not connected, cannot send emotion: {emotion}")
            return False
        
        # Convert emotion to uppercase for Arduino
        emotion_command = emotion.upper()
        
        success = self._send_raw_command(emotion_command)
        
        if success:
            print(f"🎭 Sent emotion to Arduino: {emotion} -> {emotion_command}")
            if self.on_command_sent:
                self.on_command_sent(emotion_command)
        else:
            print(f"❌ Failed to send emotion: {emotion}")
        
        return success
    
    def send_bot_emotion(self, bot_emotion_tag: str) -> bool:
        """
        Send bot emotion tag to Arduino (from ChatGPT responses)
        
        Args:
            bot_emotion_tag: Bot emotion tag (e.g., 'GREETING', 'HAPPY')
            
        Returns:
            bool: True if sent successfully
        """
        if not self.connected or not self.arduino:
            print(f"⚠️ Arduino not connected, cannot send bot emotion: {bot_emotion_tag}")
            return False
        
        # Map bot emotion to Arduino command
        arduino_command = self.bot_emotion_mapping.get(bot_emotion_tag.upper(), "DEFAULT")
        
        success = self._send_raw_command(arduino_command)
        
        if success:
            print(f"🤖 Sent bot emotion to Arduino: {bot_emotion_tag} -> {arduino_command}")
            if self.on_command_sent:
                self.on_command_sent(arduino_command)
        else:
            print(f"❌ Failed to send bot emotion: {bot_emotion_tag}")
        
        return success
    
    def send_custom_command(self, command: str) -> bool:
        """
        Send custom command to Arduino
        
        Args:
            command: Custom command string
            
        Returns:
            bool: True if sent successfully
        """
        if not self.connected or not self.arduino:
            print(f"⚠️ Arduino not connected, cannot send command: {command}")
            return False
        
        success = self._send_raw_command(command)
        
        if success:
            print(f"📤 Sent custom command to Arduino: {command}")
            if self.on_command_sent:
                self.on_command_sent(command)
        
        return success
    
    def _send_raw_command(self, command: str) -> bool:
        """
        Send raw command to Arduino
        
        Args:
            command: Command string to send
            
        Returns:
            bool: True if sent successfully
        """
        if not self.connected or not self.arduino:
            return False
        
        try:
            # Ensure command ends with newline
            if not command.endswith('\n'):
                command += '\n'
            
            # Send command
            self.arduino.write(command.encode('utf-8'))
            self.arduino.flush()
            
            return True
            
        except Exception as e:
            print(f"❌ Error sending command '{command.strip()}': {e}")
            if self.on_connection_error:
                self.on_connection_error(f"Send error: {e}")
            
            # Mark as disconnected on send error
            self.connected = False
            return False
    
    def get_available_ports(self) -> List[str]:
        """Get list of available serial ports"""
        try:
            import serial.tools.list_ports
            ports = []
            for port in serial.tools.list_ports.comports():
                ports.append(port.device)
            return ports
        except Exception as e:
            print(f"❌ Error listing ports: {e}")
            return []
    
    def auto_detect_arduino(self) -> Optional[str]:
        """
        Auto-detect Arduino port
        
        Returns:
            str: Arduino port if found, None otherwise
        """
        try:
            import serial.tools.list_ports
            
            print("🔍 Auto-detecting Arduino...")
            
            # Common Arduino USB VID:PID pairs
            arduino_ids = [
                (0x2341, None),   # Arduino.cc
                (0x2A03, None),   # Arduino.org  
                (0x0403, 0x6001), # FTDI
                (0x10C4, 0xEA60), # Silicon Labs
                (0x1A86, 0x7523), # CH340
            ]
            
            for port in serial.tools.list_ports.comports():
                # Check if port matches Arduino VID:PID
                for vid, pid in arduino_ids:
                    if port.vid == vid and (pid is None or port.pid == pid):
                        print(f"🎯 Found potential Arduino: {port.device} ({port.description})")
                        
                        # Test connection
                        if self._test_port(port.device):
                            print(f"✅ Arduino detected on {port.device}")
                            return port.device
            
            print("❌ No Arduino detected")
            return None
            
        except Exception as e:
            print(f"❌ Error during auto-detection: {e}")
            return None
    
    def _test_port(self, port: str) -> bool:
        """Test if port has Arduino"""
        try:
            test_serial = serial.Serial(port, self.baud_rate, timeout=2.0)
            time.sleep(2.0)  # Wait for Arduino reset
            
            # Send test command
            test_serial.write(b"DEFAULT\n")
            test_serial.flush()
            
            # Wait for response (optional)
            time.sleep(0.5)
            
            test_serial.close()
            return True  # If no exception, assume it works
            
        except Exception:
            return False
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to Arduino
        
        Returns:
            bool: True if reconnection successful
        """
        print("🔄 Attempting to reconnect to Arduino...")
        
        # Close existing connection
        if self.arduino:
            self._close_serial()
        
        # Try to reconnect
        success = self.connect()
        
        if success:
            print("✅ Arduino reconnected successfully")
        else:
            print("❌ Arduino reconnection failed")
        
        return success
    
    def test_arduino(self) -> bool:
        """
        Test Arduino functionality
        
        Returns:
            bool: True if test successful
        """
        if not self.connected:
            print("❌ Arduino not connected for testing")
            return False
        
        print("🧪 Testing Arduino functionality...")
        
        # Test different emotions
        test_emotions = ['GREETING', 'HAPPY', 'SAD', 'ANGRY', 'DEFAULT']
        
        for emotion in test_emotions:
            print(f"   Testing emotion: {emotion}")
            if not self.send_emotion(emotion):
                print(f"❌ Test failed on emotion: {emotion}")
                return False
            time.sleep(0.5)
        
        print("✅ Arduino test completed successfully")
        return True
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get Arduino connection information
        
        Returns:
            dict: Connection information
        """
        info = {
            'connected': self.connected,
            'port': self.arduino_port,
            'baud_rate': self.baud_rate,
            'monitoring': self.monitoring
        }
        
        if self.connected and self.arduino:
            try:
                info.update({
                    'is_open': self.arduino.is_open,
                    'in_waiting': self.arduino.in_waiting,
                    'out_waiting': self.arduino.out_waiting if hasattr(self.arduino, 'out_waiting') else 0
                })
            except:
                pass
        
        return info
    
    def get_emotion_mappings(self) -> Dict[str, str]:
        """Get bot emotion to Arduino command mappings"""
        return self.bot_emotion_mapping.copy()
    
    def set_emotion_mapping(self, bot_emotion: str, arduino_command: str):
        """
        Set custom bot emotion to Arduino command mapping
        
        Args:
            bot_emotion: Bot emotion tag
            arduino_command: Arduino command
        """
        self.bot_emotion_mapping[bot_emotion.upper()] = arduino_command.upper()
        print(f"🎭 Emotion mapping set: {bot_emotion} -> {arduino_command}")
    
    def is_connected(self) -> bool:
        """Check if Arduino is connected"""
        return self.connected
    
    def cleanup(self):
        """Cleanup Arduino output module"""
        print("🧹 Cleaning up Arduino output module...")
        
        try:
            # Disconnect Arduino
            self.disconnect()
            
            # Unregister callbacks
            if hasattr(self.client_core, 'unregister_callback'):
                self.client_core.unregister_callback('on_emotion_detected', self._handle_emotion_detected)
                self.client_core.unregister_callback('on_chat_response', self._handle_chat_response)
            
            print("✅ Arduino output module cleaned up")
            
        except Exception as e:
            print(f"❌ Error during Arduino cleanup: {e}")