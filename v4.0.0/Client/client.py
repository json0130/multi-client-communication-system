# client.py - Simple Auto-Operating Client
import json
import time
import threading
from client_core import ClientCore

# Import modules as needed
from InputModules.image_module import ImageInputModule
# from InputModules.speech_module import SpeechInputModule
from InputModules.realsense_module import RealSenseInputModule

from OutputModules.tts_module import TextToSpeechModule
from OutputModules.ardunio_module import ArduinoOutputModule

class RobotClient:
    """
    Simple emotion client that automatically operates based on JSON configuration.
    No interactive mode - just connects and runs.
    """
    
    def __init__(self, config_file="client_config.json"):
        # Load simple configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        print(f"🤖 Starting {self.config['robot_name']}")
        print(f"   🆔 ID: {self.config['client_id']}")
        print(f"   🌐 Server: {self.config['server_url']}")
        print(f"   📦 Modules: {', '.join(self.config['modules'])}")
        
        # Initialize core client
        self.core = ClientCore(config_file)
        
        # Initialize modules
        self.modules = {}
        self.running = False
        
        self._load_modules()
        self._setup_callbacks()
    
    def _load_modules(self):
        """Load modules based on configuration"""
        modules_list = self.config.get('modules', [])
        
        # Load image input if emotion or facial modules enabled
        if 'emotion' in modules_list or 'facial' in modules_list:
            try:
                self.modules['image'] = ImageInputModule(self.core)
                print("📸 Image input module loaded")
            except Exception as e:
                print(f"❌ Failed to load image module: {e}")
        
        # Load speech input if speech module enabled
        if 'speech' in modules_list:
            try:
                self.modules['speech'] = SpeechInputModule(self.core)
                print("🎤 Speech input module loaded")
            except Exception as e:
                print(f"❌ Failed to load speech module: {e}")
        
        # Load TTS output (always load for responses)
        try:
            self.modules['tts'] = TextToSpeechModule(self.core)
            print("🔊 TTS output module loaded")
        except Exception as e:
            print(f"❌ Failed to load TTS module: {e}")
        
        # Load Arduino output (always load for emotion display)
        try:
            self.modules['arduino'] = ArduinoOutputModule(self.core)
            print("🔌 Arduino output module loaded")
        except Exception as e:
            print(f"❌ Failed to load Arduino module: {e}")
    
    def _setup_callbacks(self):
        """Setup automatic behaviors between modules"""
        
        # When emotion is detected from camera -> send to Arduino
        if 'image' in self.modules and 'arduino' in self.modules:
            def on_emotion_detected(emotion, confidence, distribution):
                if confidence > 30:  # Only send high-confidence emotions
                    self.modules['arduino'].send_emotion(emotion)
                    print(f"🎭 Emotion: {emotion} ({confidence:.1f}%) -> Arduino")
            
            self.modules['image'].on_server_result = lambda data: on_emotion_detected(
                data.get('result', {}).get('emotion', 'neutral'),
                data.get('result', {}).get('confidence', 0),
                data.get('result', {}).get('distribution', {})
            )
        
        # When chat response received -> speak it via TTS
        if 'tts' in self.modules:
            def on_chat_response(response, emotion, confidence):
                self.modules['tts'].speak_with_emotion_detection(response)
                print(f"🤖 Speaking: {response}")
            
            self.core.register_callback('on_chat_response', on_chat_response)
        
        # When speech processed -> speak response via TTS
        if 'speech' in self.modules and 'tts' in self.modules:
            def on_speech_response(transcription, response, confidence):
                if response:
                    self.modules['tts'].speak_with_emotion_detection(response)
                print(f"📝 '{transcription}' -> 🤖 '{response}'")
            
            self.modules['speech'].on_speech_response = on_speech_response
    
    def start(self):
        """Start the client and all modules"""
        try:
            # Connect to server
            if not self.core.connect():
                print("❌ Failed to connect to server")
                return False
            
            # Start modules
            self._start_modules()
            
            # Start keep-alive
            self.core.start_keep_alive()
            self.running = True
            
            print("✅ Client started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error starting client: {e}")
            return False
    
    def _start_modules(self):
        """Start all loaded modules"""
        
        # Start image input (camera)
        if 'image' in self.modules:
            image_module = self.modules['image']
            if image_module.initialize_camera():
                image_module.start_capture(send_to_server=True)
                print("📸 Camera started - sending frames to server")
            else:
                print("❌ Failed to start camera")
        
        # Speech input is ready (no need to start, waits for commands)
        if 'speech' in self.modules:
            print("🎤 Speech input ready")
        
        # Start TTS worker
        if 'tts' in self.modules:
            self.modules['tts'].start_speech_worker()
            print("🔊 TTS output ready")
        
        # Arduino connects automatically
        if 'arduino' in self.modules:
            print("🔌 Arduino output ready")
    
    def send_chat_message(self, message):
        """Send a chat message (can be called externally)"""
        return self.core.send_chat_message(message)
    
    def record_and_process_speech(self, duration=5.0):
        """Record speech and process it (can be called externally)"""
        if 'speech' in self.modules:
            return self.modules['speech'].record_and_send_to_server(duration)
        return None
    
    def speak_text(self, text):
        """Speak text via TTS (can be called externally)"""
        if 'tts' in self.modules:
            self.modules['tts'].speak_with_emotion_detection(text)
    
    def send_emotion_to_arduino(self, emotion):
        """Send emotion to Arduino (can be called externally)"""
        if 'arduino' in self.modules:
            return self.modules['arduino'].send_emotion(emotion)
        return False
    
    def run(self):
        """Main run loop - keeps client alive"""
        print("🏃 Client running... Press Ctrl+C to stop")
        
        try:
            while self.running:
                # Just keep running - modules handle everything automatically
                time.sleep(1)
                
                # Optional: Add any periodic tasks here
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping client...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop client and cleanup"""
        self.running = False
        
        print("🧹 Cleaning up modules...")
        
        # Stop core
        self.core.stop_keep_alive()
        
        # Cleanup modules
        for module_name, module in self.modules.items():
            try:
                module.cleanup()
                print(f"✅ {module_name} stopped")
            except Exception as e:
                print(f"❌ Error stopping {module_name}: {e}")
        
        # Disconnect from server
        self.core.disconnect()
        
        print("✅ Client stopped")

def main():
    """Main function"""
    try:
        # Create and start client
        client = RobotClient("client_config.json")
        
        if client.start():
            # Run main loop
            client.run()
        else:
            print("❌ Failed to start client")
            
    except FileNotFoundError:
        print("❌ client_config.json not found")
        print("   Please create the configuration file")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()