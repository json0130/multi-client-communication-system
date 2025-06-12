# jetson_client.py - Local Network Client
import cv2
import requests
import base64
import json
import time
import threading
import os
import re
import subprocess
import socketio
import concurrent.futures
from threading import Lock
import numpy as np
from realsense_stream import RealSenseStreamer
from arduino_handler import ArduinoHandler
import socket

def get_local_ip():
    """Get the local IP address of this Jetson"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def discover_server_on_network():
    """Try to discover the emotion server on the local network"""
    jetson_ip = get_local_ip()
    print(f"Jetson IP: {jetson_ip}")
    
    # Extract network prefix (assumes /24 subnet)
    network_prefix = '.'.join(jetson_ip.split('.')[:-1]) + '.'
    
    print(f"Scanning network {network_prefix}0/24 for emotion server...")
    
    # Common ports to check
    test_ports = [5000, 8000, 8080, 3000]
    
    # Common server IPs to try first (based on your provided info)
    priority_ips = [
        "172.24.8.81",  # Your dedicated machine WiFi IP
        "130.216.238.6"  # Your Jetson ethernet IP
    ]
    
    # Try priority IPs first
    for ip in priority_ips:
        for port in test_ports:
            try:
                url = f"http://{ip}:{port}/health"
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'healthy':
                        print(f"Found emotion server at {ip}:{port}")
                        return f"http://{ip}:{port}"
            except:
                continue
    
    print("Could not discover emotion server automatically")
    return None

# Optimised Frame Processor Class
class OptimizedFrameProcessor:
    def __init__(self, ws_client):
        self.ws_client = ws_client
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.encoding_lock = Lock()
        self.last_emotion_frame_time = 0
        self.last_stream_frame_time = 0
        self._frame_counter = 0
       
    def __call__(self, original_frame, display_frame):
        """Make the class callable to work as frame_processor"""
        return self.process_frame(original_frame, display_frame)
       
    def process_frame(self, original_frame, display_frame):
        # Optimised frame processor with async encoding
        current_time = time.time()
        self._frame_counter += 1
       
        # Send emotion detection frame (1 FPS)
        if current_time - self.last_emotion_frame_time >= 1.0:
            self.last_emotion_frame_time = current_time
            # Encode and send in background - NO RESIZING for emotion detection
            self.executor.submit(self._send_emotion_frame_async, original_frame)
       
        # Add emotion overlay
        if self.ws_client.current_emotion != "neutral" and self.ws_client.emotion_confidence > 10:
            emotion_text = f"{self.ws_client.current_emotion} ({self.ws_client.emotion_confidence:.1f}%)"
            # Add text overlay on display frame
            cv2.putText(display_frame, emotion_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
       
        # Send streaming frame (10 FPS)
        if current_time - self.last_stream_frame_time >= 0.1:  # 10 FPS
            self.last_stream_frame_time = current_time
            # Only resize streaming frame for bandwidth optimization
            self.executor.submit(self._send_stream_frame_async, display_frame)
       
        return display_frame
   
    def _send_emotion_frame_async(self, frame):
        # Async emotion frame sending - NO RESIZING
        try:
            # Don't resize! Send full resolution for face detection
            frame_b64 = self._encode_frame_optimized(frame, quality=85)
            if self.ws_client.connected:
                self.ws_client.sio.emit('emotion_frame', {'frame': frame_b64})
        except Exception as e:
            print(f"Error sending emotion frame: {e}")
   
    def _send_stream_frame_async(self, frame):
        # Async stream frame sending - already resized
        try:
            frame_b64 = self._encode_frame_optimized(frame, quality=70)
            data = {
                'frame': frame_b64,
                'emotion': self.ws_client.current_emotion,
                'confidence': self.ws_client.emotion_confidence
            }
            if self.ws_client.connected:
                self.ws_client.sio.emit('stream_frame', data)
        except Exception as e:
            print(f"Error sending stream frame: {e}")
   
    def _encode_frame_optimized(self, frame, quality=75):
        #  Optimised frame encoding
        with self.encoding_lock:
            try:
                # Try turbojpeg for faster encoding if available
                import turbojpeg
                if not hasattr(self, 'jpeg_encoder'):
                    self.jpeg_encoder = turbojpeg.TurboJPEG()
                buffer = self.jpeg_encoder.encode(frame, quality=quality)
            except ImportError:
                # Fallback to OpenCV
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                buffer = buffer.tobytes()
           
            return base64.b64encode(buffer).decode('utf-8')
   
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=False)

class LocalWebSocketClient:
    """WebSocket client for real-time communication with local server"""
   
    def __init__(self, server_url, api_key="emotion_recognition_key_123"):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.current_emotion = "neutral"
        self.emotion_confidence = 0.0
        self.last_emotion_time = 0
       
        # Initialize Socket.IO client with optimizations
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=5,
            reconnection_delay=1,
            reconnection_delay_max=5,
            logger=False,
            engineio_logger=False
        )
        self.connected = False
       
        # Setup event handlers
        self.setup_handlers()
   
    def setup_handlers(self):
        """Setup WebSocket event handlers"""
       
        @self.sio.on('connect')
        def on_connect():
            print("Connected to local WebSocket server")
            self.connected = True
           
        @self.sio.on('disconnect')
        def on_disconnect():
            print("Disconnected from local WebSocket server")
            self.connected = False
           
        @self.sio.on('connected')
        def on_connected(data):
            print(f"Server acknowledgment: {data.get('status')}")
            server_ip = data.get('server_ip', 'unknown')
            print(f"Server IP confirmed: {server_ip}")
           
        @self.sio.on('emotion_result')
        def on_emotion_result(data):
            """Handle emotion detection results"""
            self.current_emotion = data.get('emotion', 'neutral')
            self.emotion_confidence = data.get('confidence', 0.0)
            self.last_emotion_time = time.time()
           
        @self.sio.on('error')
        def on_error(data):
            print(f"Server error: {data.get('message')}")
   
    def connect(self):
        """Connect to local WebSocket server"""
        try:
            print(f"Connecting to local WebSocket server: {self.server_url}")
           
            self.sio.connect(
                self.server_url,
                headers={'Authorization': f'Bearer {self.api_key}'},
                transports=['websocket'],
                wait_timeout=10
            )
           
            time.sleep(1)
            return self.connected
           
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            return False
   
    def encode_frame(self, frame, quality=75):
        """Convert frame to base64 for transmission"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode('utf-8')
   
    def send_emotion_frame(self, frame):
        """Send frame for emotion detection via WebSocket"""
        if not self.connected:
            return False
           
        try:
            frame_b64 = self.encode_frame(frame)
            self.sio.emit('emotion_frame', {'frame': frame_b64})
            return True
        except Exception as e:
            print(f"Error sending emotion frame: {e}")
            return False
   
    def send_streaming_frame(self, frame, emotion=None, confidence=None):
        """Send frame for streaming via WebSocket"""
        if not self.connected:
            return False
           
        try:
            frame_b64 = self.encode_frame(frame, quality=60)
            data = {
                'frame': frame_b64,
                'emotion': emotion,
                'confidence': confidence
            }
            self.sio.emit('stream_frame', data)
            return True
        except Exception as e:
            print(f"Error sending stream frame: {e}")
            return False
   
    def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.connected:
            self.sio.disconnect()
            print("Disconnected from local WebSocket server")

class LocalHTTPClient:
    """HTTP client for chat messages to local server"""
   
    def __init__(self, server_url, api_key="emotion_recognition_key_123"):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})
   
    def send_chat_message(self, message):
        """Send chat message to local server"""
        try:
            payload = {"message": message}
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

class JetsonEmotionSystem:
    def __init__(self, server_url, arduino_port=None):
        self.ws_client = LocalWebSocketClient(server_url)
        self.http_client = LocalHTTPClient(server_url)
        self.arduino_handler = ArduinoHandler(arduino_port)
        self.streamer = None
        self.chat_thread = None
        self.running = False
        self.frame_processor = None
        self.server_url = server_url
       
    def remove_emotion_tag(self, text):
        """Remove emotion tag from response text"""
        return re.sub(r"^\[(.*?)\]\s*", "", text)

    def speak_with_espeak(self, text, rate=155, volume=100, voice='en+f2', pitch=60, gap=0):
        """Use espeak to vocalize text with voice options"""
        speech_text = self.remove_emotion_tag(text)
        try:
            subprocess.run(
                ['espeak', f'-v{voice}', f'-s{rate}', f'-a{volume}', f'-p{pitch}', f'-g{gap}', speech_text],
                check=True
            )
        except Exception as e:
            print(f"TTS error: {e}")
   
    def start_camera_system(self):
        """Start the camera and emotion detection with optimized processor"""
       
        # Create optimized frame processor
        self.frame_processor = OptimizedFrameProcessor(self.ws_client)
       
        # Initialize RealSense streamer
        self.streamer = RealSenseStreamer(
            web_port=8080,
            frame_processor=self.frame_processor.process_frame
        )
       
        if not self.streamer.start():
            print("Failed to start camera streamer")
            return False
           
        print("Camera system started successfully")
        return True
   
    def start_chat_interface(self):
        """Start the chat interface"""
        def chat_loop():
           
            while self.running:
                try:
                    user_input = input("\n You: ")
                    if user_input.lower() == 'exit':
                        break
                   
                    print("Processing...")

                    response_data = self.http_client.send_chat_message(user_input)
                   
                    if response_data:
                        bot_response = response_data.get("response", "Sorry, no response received")
                        bot_emotion = response_data.get("bot_emotion", "DEFAULT")
                        detected_emotion = response_data.get("detected_emotion", "neutral")
                        emotion_confidence = response_data.get("emotion_confidence", 0.0)
                       
                        print(f"Bot: {bot_response}")
                 
                        if self.arduino_handler.connected:
                            self.arduino_handler.send_bot_emotion(bot_emotion)
                        else:
                            print("🔌 Arduino not connected - emotion not sent")
                       
                        self.speak_with_espeak(bot_response)
                           
                    else:
                        print("Failed to get response from server")
                       
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Chat error: {e}")
       
        self.chat_thread = threading.Thread(target=chat_loop, daemon=True)
        self.chat_thread.start()
   
    def check_server_health(self):
        """Check if local server is healthy"""
        try:
            response = self.http_client.session.get(
                f"{self.http_client.server_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                print(f"Server health check passed")
                print(f"   Server IP: {data.get('server_ip', 'unknown')}")
                print(f"   Model loaded: {data.get('components', {}).get('model_loaded')}")
                print(f"   OpenAI available: {data.get('components', {}).get('openai_available')}")
                return True
            else:
                print(f"Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Cannot connect to server: {e}")
            return False
   
    def start(self):
        """Start the complete system"""
        self.running = True
       
        print("Checking server health...")
        if not self.check_server_health():
            print("Server is not healthy. Please check the local server.")
            return False
       
        print("Connecting to local WebSocket server...")
        if not self.ws_client.connect():
            print("Failed to connect to WebSocket server")
            print("Continuing without real-time features...")
        else:
            print("WebSocket connection established")
       
        if self.arduino_handler.connected:
            print(f"Arduino connected on: {self.arduino_handler.serial_port}")
            self.arduino_handler.send_emotion("GREETING")
        else:
            print("Arduino not connected - continuing without Arduino integration")
       
        if not self.start_camera_system():
            return False
       
        self.start_chat_interface()
       
        print("Chat interface ready")
        if self.ws_client.connected:
            print("WebSocket connected (real-time emotion & streaming)")
        if self.arduino_handler.connected:
            print(f"Arduino ready on {self.arduino_handler.serial_port}")
        print(f"Local camera stream: http://localhost:8080")
        print(f"Server monitor: {self.server_url}/monitor")
        print("\nType 'exit' to quit or press Ctrl+C to stop...")
        print("="*50)
       
        try:
            while self.running:
                time.sleep(1)
                # Try to reconnect if disconnected
                if not self.ws_client.connected and time.time() % 30 == 0:
                    print("WebSocket disconnected, attempting to reconnect...")
                    self.ws_client.connect()
                   
        except KeyboardInterrupt:
            print("\n Shutting down...")
            self.stop()
   
    def stop(self):
        """Stop all systems"""
        self.running = False
       
        if self.frame_processor:
            self.frame_processor.cleanup()
       
        if self.ws_client:
            self.ws_client.disconnect()
       
        if self.streamer:
            self.streamer.stop()
       
        if self.arduino_handler:
            self.arduino_handler.close()
       
        print("System stopped")

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
   
    print("\n" + "="*60)
    print("Jetson Local Emotion Detection System")
    print("="*60)
   
    # Try to get server URL from environment first
    server_url = os.getenv("LOCAL_SERVER_URL")
    
    if not server_url:
        # Try to discover server automatically
        print("Attempting to discover emotion server on local network...")
        server_url = discover_server_on_network()
    
    if not server_url:
        # Manual input as fallback
        server_url = input("Enter local server URL (e.g., http://172.24.8.81:5000): ").strip()
   
    if not server_url.startswith(('http://', 'https://')):
        server_url = 'http://' + server_url
   
    # Test connection before proceeding
    try:
        print(f"Testing connection to {server_url}...")
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            print("Connection test successful!")
        else:
            print(f"Server returned status code: {response.status_code}")
            return
    except Exception as e:
        print(f"Cannot connect to server: {e}")
        print("Please check:")
        print("   1. Server is running on the target machine")
        print("   2. Firewall allows connections on port 5000")
        print("   3. Both devices are on the same network")
        return
   
    arduino_port = os.getenv("ARDUINO_PORT")
    if not arduino_port:
        arduino_port = input("Enter Arduino port (press Enter for default /dev/ttyUSB0): ").strip()
        if not arduino_port:
            arduino_port = None
   
    print("\nConfiguration:")
    print(f"   Local Server: {server_url}")
    print(f"   Arduino Port: {arduino_port or '/dev/ttyUSB0 (default)'}")
    print(f"   WebSocket Enabled: Yes")
    print(f"   Local Stream: http://localhost:8080")
   
    system = JetsonEmotionSystem(server_url, arduino_port)
   
    try:
        system.start()
    except Exception as e:
        print(f"\n Fatal error: {e}")
        system.stop()

if __name__ == "__main__":
    main()