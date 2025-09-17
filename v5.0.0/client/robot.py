# main.py - Simple config concurrent client
import sys
import os
import logging
from typing import Optional

# --- NEW IMPORTS ---
import serial.tools.list_ports

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from client import BasicClient

# Import all available modules
from InputModules.text_input import TextInputModule
from InputModules.voice_input import VoiceInputModule
#from InputModules.camera_input import CameraInputModule
from InputModules.realsense_input import RealSenseInputModule

from OutputModules.console_output import ConsoleOutputModule
from OutputModules.edge_tts_output import EdgeTTSOutputModule
from OutputModules.tts_output import TTSOutputModule, PyttsxTTSOutputModule
from OutputModules.arduino_output import ArduinoOutputModule

logger = logging.getLogger(__name__)

class SimpleConcurrentClient(BasicClient):
    """
    Client that uses simple config format and applies sensible defaults
    Works with minimal config: robot_name, client_id, server_url, modules
    """
    
    def __init__(self, config_file: str = "client_config.json"):
        super().__init__(config_file)

        def on_chat_response(data):
            print(f"\n>>> DEBUG: 'on_chat_response' HANDLER TRIGGERED! Data received: {data}\n")
            response = data.get('response')
            if response:
                # Use the registered console_output module to display the message
                if 'console_output' in self.output_modules:
                    self.output_modules['console_output'].process_output(response)
                else:
                    # Fallback if the console module isn't running for some reason
                    print(f"\n🤖 Server Response: {response}")
                
                # Optional: Make the robot speak the response
                # if 'edge_tts_output' in self.output_modules:
                #     self.output_modules['edge_tts_output'].process(response)
        
        # Register the handler with the socketio client instance
        # This now correctly uses 'self' because it is inside the __init__ method
        if self.server_connection and self.server_connection.sio:
            print("\n>>> DEBUG: Attempting to register 'on_chat_response' handler...")
            self.server_connection.sio.on('chat_response', on_chat_response)
            print(">>> DEBUG: Handler registration successful.\n")
        else:
            print("\n>>> DEBUG: FAILED to register handler! Server connection or SIO object not found.\n")

        # Finally, call setup_all_modules
        self.setup_all_modules()
    
    def _on_arduino_connected(self): 
        """Called when Arduino connects""" 
        logger.info("✅ Arduino robot connected and ready!") 

    def _on_arduino_disconnected(self): 
        """Called when Arduino disconnects""" 
        logger.warning("⚠️ Arduino robot disconnected") 

    def _on_arduino_error(self, error_msg: str): 
        """Called on Arduino connection errors""" 
        logger.error(f"❌ Arduino error: {error_msg}") 

    def send_robot_emotion(self, emotion: str) -> bool: 
        """Send emotion to Arduino robot""" 
        if hasattr(self, 'arduino_module') and self.arduino_module and self.arduino_module.is_connected(): 
            return self.arduino_module.send_emotion(emotion) 
        logger.warning(f"⚠️ Cannot send emotion '{emotion}' - Arduino not connected") 
        return False 

    def send_robot_command(self, command: str) -> bool: 
        """Send custom command to Arduino robot""" 
        if hasattr(self, 'arduino_module') and self.arduino_module and self.arduino_module.is_connected(): 
            return self.arduino_module.send_custom_command(command) 
        logger.warning(f"⚠️ Cannot send command '{command}' - Arduino not connected") 
        return False

    # --- NEW, MORE ROBUST DETECTION METHOD ---
    def _detect_arduino_port(self) -> Optional[str]:
        """
        Auto-detect the Arduino/ESP32 port by looking for common identifiers.
        Returns the port name if found, otherwise None.
        """
        logger.info("   🔍 Searching for Arduino/ESP32 port...")
        ports = serial.tools.list_ports.comports()
        
        # Common identifiers for ESP32 and Arduino boards (CP210x is on your board)
        known_identifiers = ["CP210x", "CH340", "USB Serial", "Arduino"]
        
        for port in ports:
            # Check description and manufacturer for known identifiers
            for identifier in known_identifiers:
                if (port.description and identifier in port.description) or \
                   (port.manufacturer and identifier in port.manufacturer):
                    logger.info(f"   ✅ Found device '{port.description}' on port {port.device}")
                    return port.device
        
        logger.warning("   ⚠️ Could not auto-detect an Arduino/ESP32 port.")
        return None
    
    def setup_all_modules(self):
        """Setup all modules with sensible defaults based on simple config"""
        
        # === INPUT MODULES ===
        # This block enables the client to use its keyboard
        # --- Text Input is now always enabled ---
        logger.info("⌨️ Setting up text input (always on)...")
        text_input = TextInputModule("text_input")
        self.register_input_module(text_input)
        text_input.start()
        
        if 'speech' in self.config.get('modules', []):
            logger.info("🎤 Setting up voice input...")
            voice_config = self.config.get('voice_config', {
                'sample_rate': 48000,
                'channels': 1,
                'input_device_index': 11,
                'max_record_time': 30
            })
            voice_input = VoiceInputModule("voice_input", voice_config)
            self.register_input_module(voice_input)
            voice_input.start()
        
        if 'emotion' in self.config.get('modules', []):
            logger.info("📹 Setting up camera input...")
            camera_config = self.config.get('camera_config', {
                'camera_index': 0, 'width': 1280, 'height': 720,
                'fps': 15, 'send_fps': 15, 'jpeg_quality': 85
            })
            
            logger.info("   🎯 Attempting camera...")
            realsense_input = RealSenseInputModule("camera_input", camera_config)
            realsense_input.start()
            if not self.register_input_module(realsense_input):
                logger.warning("   📸 RealSense failed, registration incomplete.")
                # You might want to add a fallback to a regular camera here if needed

        # === OUTPUT MODULES ===
        
        logger.info("🖥️ Setting up console output...")
        console_output = ConsoleOutputModule("console_output", self.config.get('console_config', {}))
        self.register_output_module(console_output)
        console_output.start()
        
        logger.info("🎙️ Setting up Edge text-to-speech...")
        edge_config = self.config.get('edge_tts_config', {
            'voice': 'en-US-AriaNeural', 'rate': '+0%', 'pitch': '+0Hz', 'remove_emotion_tags': True
        })
        
        edge_tts = EdgeTTSOutputModule("edge_tts_output", edge_config)
        if self.register_output_module(edge_tts):
            logger.info("   ✅ Using Microsoft Edge TTS")
            edge_tts.start()
        else:
            logger.warning("   ⚠️ Edge TTS failed. You may need to install it.")
            # Fallback can be added here if needed

        # --- ARDUINO SETUP SECTION (MODIFIED) ---
        if self.config.get('features', {}).get('arduino_integration', True):
            logger.info("🔌 Setting up Arduino output...")
            
            # Start with the config file's settings
            arduino_config = self.config.get('arduino_output', {})
            
            # Try to auto-detect the port
            detected_port = self._detect_arduino_port()
            
            # If a port is detected, it overrides the one in the config file.
            if detected_port:
                logger.info(f"   🎯 Using auto-detected port: {detected_port}")
                arduino_config['arduino_port'] = detected_port
            else:
                logger.warning(f"   Fallback to port from config file: {arduino_config.get('arduino_port')}")
            
            # Set sensible defaults if they are missing
            arduino_config.setdefault('arduino_baud', 115200)
            arduino_config.setdefault('auto_connect', True)

            # Important: The module name must match the key used in client.py
            self.arduino_module = ArduinoOutputModule("arduino_output", arduino_config)
            
            self.arduino_module.on_connected = self._on_arduino_connected
            self.arduino_module.on_disconnected = self._on_arduino_disconnected
            # Correcting the callback name based on your file
            self.arduino_module.on_connection_error = self._on_arduino_error
            
            if self.register_output_module(self.arduino_module):
                logger.info("   ✅ Arduino output module registered")
            else:
                logger.warning("   ⚠️ Arduino output module failed to register")
    
    def print_startup_info(self):
        """Print information about what's running"""
        print("\n" + "="*60)
        print("🤖 CHATBOX CLIENT STARTED")
        print("="*60)
        print(f"🏷️  Robot: {self.config.get('robot_name', 'Unknown')}")
        print(f"🆔 Client ID: {self.config.get('client_id', 'Unknown')}")
        print(f"🌐 Server: {self.config.get('server_url', 'Unknown')}")
        print(f"📦 Server Modules: {', '.join(self.config.get('modules', []))}")
        
        print("\n📥 INPUT MODULES:")
        for name, module in self.input_modules.items():
            status = "✅ Running" # Registration now happens in setup_all_modules
            print(f"   • {name.replace('_', ' ').title()}: {status}")
        
        print("\n📤 OUTPUT MODULES:")
        for name, module in self.output_modules.items():
            status = "✅ Running"
            print(f"   • {name.replace('_', ' ').title()}: {status}")
        
        print("\n💡 USAGE:")
        if 'voice_input' in self.input_modules:
            print("   🎤 Empty line + Enter = Voice recording")
        if any('camera' in name for name in self.input_modules):
            print("   📸 Camera automatically sends emotion data")
        
        print("   🛑 Type 'exit' or press Ctrl+C to stop")
        print("="*60)
        print()

def main():
    """Simple main function - no configuration needed!"""
    print("🤖 ChatBox Client System")
    print("📋 Using simple configuration format...")
    
    try:
        client = SimpleConcurrentClient("client_config.json")
        client.print_startup_info()
        
        print("🚀 Starting all modules...")
        client.run()
        return 0
        
    except FileNotFoundError:
        print("❌ Error: client_config.json not found")
        # Provide a helpful template
        return 1
        
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
        return 0
    except Exception as e:
        logger.error(f"❌ A critical error occurred in main: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
