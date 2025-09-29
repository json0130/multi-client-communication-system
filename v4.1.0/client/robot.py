# main.py - Simple config concurrent client
import sys
import os
import logging

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
#from OutputModules.arduino_output import ArduinoOutputModule

logger = logging.getLogger(__name__)
 
class SimpleConcurrentClient(BasicClient):
    """
    Client that uses simple config format and applies sensible defaults
    Works with minimal config: robot_name, client_id, server_url, modules
    """
    
    def __init__(self, config_file: str = "client_config.json"):
        super().__init__(config_file)
        self.setup_all_modules()
    
    def setup_all_modules(self):
        """Setup all modules with sensible defaults based on simple config"""
        
        # === INPUT MODULES ===
        
        # 1. Text Input (always available)
        # logger.info("🔤 Setting up text input...")
        # text_input = TextInputModule("text_input")
        # self.register_input_module(text_input)
        
        # 2. Voice Input (if speech module enabled)
        if 'speech' in self.config.get('modules', []):
            logger.info("🎤 Setting up voice input...")
            # Default voice settings
            voice_config = {
                'sample_rate': 48000,
                'channels': 1,
                'input_device_index': 11,  # Common for USB mics
                'max_record_time': 30
            }
            voice_input = VoiceInputModule("voice_input", voice_config)
            self.register_input_module(voice_input)
        
        # 3. Camera Input (if emotion module enabled)
        if 'emotion' in self.config.get('modules', []):
            logger.info("📹 Setting up camera input...")
            
            # Default camera settings
            camera_config = {
                'camera_index': 0,
                'width': 1280, #was 640
                'height': 720, #was 480
                'fps': 15,	# was 30
                'send_fps': 15,  # Send 1 frame per second to server
                'jpeg_quality': 85
            }
            
            # Try RealSense first (for Jetson setups), fallback to regular camera
            logger.info("   🎯 Attempting camera...")
            realsense_input = RealSenseInputModule("camera_input", camera_config)
            if not self.register_input_module(realsense_input):
                logger.info("   📸 RealSense failed, using regular camera...")
                camera_input = CameraInputModule("camera_input", camera_config)
                self.register_input_module(camera_input)
        
        # === OUTPUT MODULES ===
        
        # 1. Console Output
        logger.info("🖥️ Setting up console output...")
        console_config = {
            'show_timestamps': True,
            'show_response_type': True,
            'prefix': '🤖'
        }
        console_output = ConsoleOutputModule("console_output", console_config)
        self.register_output_module(console_output)
        
        # 2. EDGE TTS Output
        logger.info("🎙️ Setting up Edge text-to-speech...")
        
        # Edge TTS
        edge_config = {
            'voice': 'en-US-AriaNeural',  # Very natural female voice
            # Other good options:
            # 'en-US-JennyNeural' - Natural female
            # 'en-US-GuyNeural' - Natural male  
            # 'en-US-DavisNeural' - Warm male
            'rate': '+0%',
            'pitch': '+0Hz',
            'remove_emotion_tags': True,
        }
        
        edge_tts = EdgeTTSOutputModule("edge_tts_output", edge_config)
        if self.register_output_module(edge_tts):
            logger.info("   ✅ Using Microsoft Edge TTS (plughw:3,0)")
        else:
            logger.info("   🔄 Edge TTS not available, trying espeak...")
            
            # Final fallback to espeak
            espeak_config = {
                'voice': 'en+f2',
                'rate': 155,
                'volume': 100,
                'remove_emotion_tags': True
            }
                
            from OutputModules.tts_output import TTSOutputModule
            tts_output = TTSOutputModule("tts_output", espeak_config)
            self.register_output_module(tts_output)
    
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
            status = "✅ Running" if module.enabled else "❌ Failed"
            print(f"   • {name.replace('_', ' ').title()}: {status}")
        
        print("\n📤 OUTPUT MODULES:")
        for name, module in self.output_modules.items():
            status = "✅ Running" if module.enabled else "❌ Failed"
            print(f"   • {name.replace('_', ' ').title()}: {status}")
        
        print("\n💡 USAGE:")
        if 'text_input' in self.input_modules:
            print("   💬 Type any message + Enter = Text chat")
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
        # Create client with simple config
        client = SimpleConcurrentClient("client_config.json")
        
        # Show what's running
        client.print_startup_info()
        
        # Start everything
        print("🚀 Starting all modules...")
        client.run()
        
        return 0
        
    except FileNotFoundError:
        print("❌ Error: client_config.json not found")
        print("\n📝 Please create client_config.json with:")
        print('''{
    "robot_name": "ChatBox",
    "client_id": "chatbox_jetson_001", 
    "server_url": "http://192.168.1.100:5000",
    "modules": ["gpt", "emotion", "speech"]
}''')
        return 1
        
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

