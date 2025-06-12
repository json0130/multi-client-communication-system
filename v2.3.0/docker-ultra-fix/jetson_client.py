# jetson_client.py - Main client for Jetson Nano with Arduino integration
import cv2
import requests
import base64
import json
import time
import threading
import os
import re
import subprocess
from realsense_stream import RealSenseStreamer
from arduino_handler import ArduinoHandler

class ColabClient:
    def __init__(self, server_url, api_key="emotion_recognition_key_123"):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.current_emotion = "neutral"
        self.emotion_confidence = 0.0
        self.last_emotion_time = 0
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})
        self.top_emotions = []
        
    def encode_frame(self, frame, quality=75):
        """Convert frame to base64 for transmission"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode('utf-8')
    
    def send_emotion_detection(self, frame):
        """Send frame to Colab for emotion detection"""
        try:
            frame_b64 = self.encode_frame(frame)
            
            response = self.session.post(
                f"{self.server_url}/detect_emotion",
                json={"frame": frame_b64},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.current_emotion = data.get("emotion", "neutral")
                self.emotion_confidence = data.get("confidence", 0.0)
                self.last_emotion_time = time.time()
                
                # New: store top 3 emotions from server
                self.top_emotions = data.get("top_emotions", [])
                
                return True
            else:
                print(f"Emotion detection failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error sending frame for emotion detection: {e}")
            return False
    
    def send_chat_message(self, message):
        """Send chat message to Colab server (no emotion data sent)"""
        try:
            payload = {
                "message": message
            }
            
            response = self.session.post(
                f"{self.server_url}/chat",
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Chat request failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error sending chat message: {e}")
            return None
    
    def send_streaming_frame(self, frame, emotion_overlay=True):
        """Send frame for live streaming"""
        try:
            frame_b64 = self.encode_frame(frame, quality=60)
            
            payload = {
                "frame": frame_b64,
                "emotion": self.current_emotion if emotion_overlay else None,
                "confidence": self.emotion_confidence if emotion_overlay else None,
                "top_emotions": self.top_emotions if emotion_overlay else None 
            }
            
            response = self.session.post(
                f"{self.server_url}/stream_frame",
                json=payload,
                timeout=5
            )
            
            return response.status_code == 200
            
        except Exception as e:
            print(f"Error sending streaming frame: {e}")
            return False

class JetsonEmotionSystem:
    def __init__(self, colab_url, arduino_port=None):
        self.colab_client = ColabClient(colab_url)
        self.arduino_handler = ArduinoHandler(arduino_port)
        self.streamer = None
        self.emotion_thread = None
        self.chat_thread = None
        self.running = False
        
    def remove_emotion_tag(self, text):
        return re.sub(r"^\[(.*?)\]\s*", "", text)

    def speak_with_espeak(self, text, rate=150, volume=100):
        """Use espeak to vocalize text"""

        speech_text = self.remove_emotion_tag(text)

        try:
            subprocess.run(
                ['espeak', f'-s{rate}', f'-a{volume}', speech_text],
                check=True
            )
        except Exception as e:
            print(f"❌ TTS error: {e}")

        
    def emotion_frame_processor(self, original_frame, display_frame):
        """Process frames for emotion detection"""
        # Send every 3rd frame to reduce server load
        if hasattr(self, '_frame_counter'):
            self._frame_counter += 1
        else:
            self._frame_counter = 0
            
        if self._frame_counter % 3 == 0:
            # Send frame for emotion detection in background
            threading.Thread(
                target=self.colab_client.send_emotion_detection,
                args=(original_frame,),
                daemon=True
            ).start()
        
        # Add emotion overlay to display frame
        if self.colab_client.current_emotion != "neutral":
            emotion_text = f"{self.colab_client.current_emotion} ({self.colab_client.emotion_confidence:.1f}%)"
            cv2.putText(display_frame, emotion_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Send frame for streaming (less frequently)
        if self._frame_counter % 2 == 0:
            threading.Thread(
                target=self.colab_client.send_streaming_frame,
                args=(display_frame,),
                daemon=True
            ).start()
        
        return display_frame
    
    def start_camera_system(self):
        """Start the camera and emotion detection"""
        print("Starting camera system...")
        self.streamer = RealSenseStreamer(
            web_port=8080, 
            frame_processor=self.emotion_frame_processor
        )
        
        if not self.streamer.start():
            print("Failed to start camera streamer")
            return False
            
        print("Camera system started successfully")
        return True
    
    def start_chat_interface(self):
        """Start the chat interface"""
        def chat_loop():
            print("\n" + "="*50)
            print("     Distributed Emotion-Aware Chatbot")
            print("="*50)
            
            while self.running:
                try:
                    # Show current detected emotion
                    if self.colab_client.emotion_confidence > 10:
                        print(f"\n🎭 Detected: {self.colab_client.current_emotion} "
                              f"({self.colab_client.emotion_confidence:.1f}%)")
                    
                    user_input = input("\n💬 You: ")
                    if user_input.lower() == 'exit':
                        break
                    
                    # Send to Colab server (message only)
                    print("🔄 Processing...")
                    response_data = self.colab_client.send_chat_message(user_input)
                    
                    if response_data:
                        bot_response = response_data.get("response", "Sorry, no response received")
                        bot_emotion = response_data.get("bot_emotion", "DEFAULT")
                        detected_emotion = response_data.get("detected_emotion", "neutral")
                        emotion_confidence = response_data.get("emotion_confidence", 0.0)
                        
                        print(f"🎭 Your detected emotion: {detected_emotion} ({emotion_confidence:.1f}%)")
                        print(f"🤖 Bot {bot_response}")
                        self.speak_with_espeak(bot_response)
                        
                        # Send bot emotion to Arduino
                        if self.arduino_handler.connected:
                            self.arduino_handler.send_bot_emotion(bot_emotion)
                        else:
                            print("  Arduino not connected - emotion not sent")
                            
                    else:
                        print(" Failed to get response from server")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f" Chat error: {e}")
        
        self.chat_thread = threading.Thread(target=chat_loop, daemon=True)
        self.chat_thread.start()
    
    def start(self):
        """Start the complete system"""
        self.running = True
        
        # Test connection to Colab server
        try:
            response = self.colab_client.session.get(f"{self.colab_client.server_url}/health")
            if response.status_code != 200:
                print(f" Cannot connect to Colab server at {self.colab_client.server_url}")
                return False
        except Exception as e:
            print(f" Server connection failed: {e}")
            return False
        
        print(f" Connected to Colab server: {self.colab_client.server_url}")
        
        # Check Arduino connection
        if self.arduino_handler.connected:
            print(f" Arduino connected on: {self.arduino_handler.serial_port}")
            # Test Arduino with a greeting
            self.arduino_handler.send_emotion("GREETING")
        else:
            print(" Arduino not connected - continuing without Arduino integration")
        
        # Start camera system
        if not self.start_camera_system():
            return False
        
        # Start chat interface
        self.start_chat_interface()
        
        print("💬 Chat interface ready")
        if self.arduino_handler.connected:
            print(f"🛰️  Arduino ready on {self.arduino_handler.serial_port}")
        print("\nPress Ctrl+C to stop...")
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n Shutting down...")
            self.stop()
    
    def stop(self):
        """Stop all systems"""
        self.running = False
        
        if self.streamer:
            self.streamer.stop()
        
        # Close Arduino connection
        if self.arduino_handler:
            self.arduino_handler.close()
        
        print("✅ System stopped")

def main():
    # Load environment variables if available
    from dotenv import load_dotenv
    load_dotenv()
    
    # Configuration
    colab_url = os.getenv("COLAB_SERVER_URL")
    if not colab_url:
        colab_url = input("Enter Colab server URL (with ngrok): ").strip()
    
    if not colab_url.startswith(('http://', 'https://')):
        colab_url = 'https://' + colab_url
    
    # Arduino port configuration
    arduino_port = os.getenv("ARDUINO_PORT")
    if not arduino_port:
        arduino_port = input("Enter Arduino port (press Enter for default /dev/ttyUSB0): ").strip()
        if not arduino_port:
            arduino_port = None  # Will use default
    
    print(f"   Colab Server: {colab_url}")
    print(f"   Arduino Port: {arduino_port or '/dev/ttyUSB0 (default)'}")
    
    # Initialize and start system
    system = JetsonEmotionSystem(colab_url, arduino_port)
    system.start()

if __name__ == "__main__":
    main()
