# jetson_client.py - Local Network Client with Voice Recording Support
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
import pyaudio
import wave
import tempfile
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
    print(f"🔍 Jetson IP: {jetson_ip}")
    
    # Extract network prefix (assumes /24 subnet)
    network_prefix = '.'.join(jetson_ip.split('.')[:-1]) + '.'
    
    print(f"🌐 Scanning network {network_prefix}0/24 for emotion server...")
    
    # Common ports to check
    test_ports = [5000, 8000, 8080, 3000]
    
    # Common server IPs to try first (based on your provided info)
    priority_ips = [
        "130.216.238.6",  # Your actual server IP  
        "172.24.8.81"     # Keep as backup
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
                        print(f"✅ Found emotion server at {ip}:{port}")
                        return f"http://{ip}:{port}"
            except:
                continue
    
    print("❌ Could not discover emotion server automatically")
    return None

class VoiceRecorder:
    """Voice recording class optimized for USB microphone - AUTO-DETECTS USB MIC"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Audio configuration - Updated for USB microphone
        self.sample_rate = self.config.get('sample_rate', 48000)  # USB mic native rate
        self.channels = self.config.get('channels', 1)
        self.chunk_size = self.config.get('chunk_size', 1024)
        self.audio_format = pyaudio.paInt16
        self.max_record_time = self.config.get('max_record_time', 30)
        
        # USB microphone detection - THIS IS THE KEY PART
        self.usb_device_index = None
        self.prefer_usb = self.config.get('prefer_usb', True)
        
        # Recording state
        self.is_recording = False
        self.audio_frames = []
        self.audio = None
        self.stream = None
        self.record_thread = None
        
        # Initialize PyAudio and AUTO-FIND USB microphone
        self._initialize_pyaudio()
    
    def _initialize_pyaudio(self):
        """Initialize PyAudio and automatically find USB microphone"""
        try:
            self.audio = pyaudio.PyAudio()
            print("🎤 PyAudio initialized successfully")
            
            # AUTOMATICALLY find and set USB microphone
            self._find_usb_microphone()
            
        except Exception as e:
            print(f"❌ Failed to initialize PyAudio: {e}")
            self.audio = None
    
    def _find_usb_microphone(self):
        """Automatically find and configure USB microphone - IMPROVED DETECTION"""
        if not self.audio:
            return
    
        print("🔍 Auto-detecting USB microphone...")
    
        # Debug: List all devices first
        print("   📋 Available input devices:")
    
        for i in range(self.audio.get_device_count()):
            try:
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    device_name = info['name']
                    print(f"      [{i}] {device_name}")
                
                    # IMPROVED USB MICROPHONE DETECTION
                    # Look for multiple USB microphone indicators
                    name_lower = device_name.lower()
                
                    # Check for USB microphone patterns
                    usb_patterns = [
                        'uacdemov1.0',        # Your specific device
                        'usb audio',          # Generic USB audio
                        'usb',                # Contains USB
                        'microphone',         # Contains microphone
                        'mic',                # Contains mic
                        'hw:2,0'              # Your specific hardware ID
                    ]
                
                    # Also check for device that's NOT tegra (exclude internal audio)
                    is_not_tegra = 'tegra' not in name_lower
                    is_usb_device = any(pattern in name_lower for pattern in usb_patterns)
                
                    # Special case: If device index is 11 and not tegra, it's likely USB
                    is_likely_usb = (i == 11 and is_not_tegra)
                
                    if is_usb_device or is_likely_usb:
                        # AUTOMATICALLY SET USB MICROPHONE AS INPUT DEVICE
                        self.usb_device_index = i
                        self.sample_rate = int(info['defaultSampleRate'])
                    
                        print(f"   🎯 AUTO-SELECTED USB microphone:")
                        print(f"      Device index: {i}")
                        print(f"      Name: {device_name}")
                        print(f"      Sample rate: {self.sample_rate}")
                        print(f"      Detection reason: {'USB pattern match' if is_usb_device else 'Index 11 + non-Tegra'}")
                        print(f"      ✅ Will be used for all recordings")
                        return True
                    
            except Exception as e:
                print(f"      [?] Device {i}: Error getting info - {e}")
                continue
    
        print("   ⚠️ USB microphone not auto-detected")
        print("   🔧 Falling back to device index 11 (based on your test results)")
    
        # Fallback: Force device 11 since your test showed it works
        try:
            info = self.audio.get_device_info_by_index(11)
            if info['maxInputChannels'] > 0:
                self.usb_device_index = 11
                self.sample_rate = int(info['defaultSampleRate'])
                print(f"   🎯 FALLBACK: Using device 11")
                print(f"      Name: {info['name']}")
                print(f"      Sample rate: {self.sample_rate}")
                return True
        except Exception as e:
            print(f"   ❌ Cannot access device 11: {e}")
    
        return False
    
    def start_recording(self, device_index=None):
        """Start recording - AUTOMATICALLY uses USB microphone if detected"""
        if not self.audio:
            print("❌ PyAudio not available")
            return False
        
        if self.is_recording:
            print("⚠️ Already recording")
            return False
        
        # AUTOMATIC DEVICE SELECTION:
        # 1. Use specified device_index if provided
        # 2. Use USB microphone if detected
        # 3. Use default device as fallback
        # recording_device = device_index or self.usb_device_index
        recording_device = device_index or self.config.get('input_device_index', None)
        
        try:
            self.audio_frames = []
            
            # Show which device is being used
            if recording_device is not None:
                device_info = self.audio.get_device_info_by_index(recording_device)
                print(f"🎤 Recording with: {device_info['name']}")
            else:
                print("🎤 Recording with default device")
            
            # Open audio stream with the selected device
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=recording_device,  # THIS IS THE KEY LINE
                frames_per_buffer=self.chunk_size
            )
            
            self.is_recording = True
            print("🔴 Recording started... Press Enter again to stop")
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self._record_audio, daemon=True)
            self.record_thread.start()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            return False
    
    def _record_audio(self):
        """Record audio with real-time feedback"""
        start_time = time.time()
        
        try:
            while self.is_recording:
                if time.time() - start_time > self.max_record_time:
                    print(f"\n⏰ Maximum recording time reached")
                    break
                
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_frames.append(data)
                    
                    # Show recording progress
                    elapsed = int(time.time() - start_time)
                    if elapsed > 0 and elapsed % 2 == 0:  # Every 2 seconds
                        print(f"   🎤 Recording... {elapsed}s", end='\r')
                    
                except Exception as e:
                    print(f"\n❌ Error reading audio: {e}")
                    break
                    
        except Exception as e:
            print(f"\n❌ Recording error: {e}")
        finally:
            self.is_recording = False
    
    def stop_recording(self):
        """Stop recording and return WAV data"""
        if not self.is_recording:
            print("⚠️ Not currently recording")
            return None
        
        self.is_recording = False
        
        if self.record_thread:
            self.record_thread.join(timeout=1.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        if not self.audio_frames:
            print("❌ No audio data recorded")
            return None
        
        try:
            # Create WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                wav_file = wave.open(tmp_file, 'wb')
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b''.join(self.audio_frames))
                wav_file.close()
                
                tmp_file_path = tmp_file.name
            
            # Read WAV file as bytes
            with open(tmp_file_path, 'rb') as f:
                wav_data = f.read()
            
            os.unlink(tmp_file_path)
            
            duration = len(self.audio_frames) * self.chunk_size / self.sample_rate
            print(f"🟢 Recording completed: {duration:.2f}s, {len(wav_data)} bytes")
            
            return wav_data
            
        except Exception as e:
            print(f"❌ Error creating WAV file: {e}")
            return None
    
    def list_audio_devices(self):
        """List available audio devices"""
        if not self.audio:
            return
        
        print("🎙️ Available audio input devices:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                marker = " 🎯 USB MIC" if i == self.usb_device_index else ""
                print(f"   {i}: {info['name']}{marker}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.is_recording:
            self.stop_recording()
        
        if self.audio:
            self.audio.terminate()
            print("🎤 PyAudio cleanup completed")

# Optimised Frame Processor Class (unchanged)
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
            print("🔗 Connected to local WebSocket server")
            self.connected = True
           
        @self.sio.on('disconnect')
        def on_disconnect():
            print("🔌 Disconnected from local WebSocket server")
            self.connected = False
           
        @self.sio.on('connected')
        def on_connected(data):
            print(f"✅ Server acknowledgment: {data.get('status')}")
            server_ip = data.get('server_ip', 'unknown')
            print(f"🌐 Server IP confirmed: {server_ip}")
           
        @self.sio.on('emotion_result')
        def on_emotion_result(data):
            """Handle emotion detection results"""
            self.current_emotion = data.get('emotion', 'neutral')
            self.emotion_confidence = data.get('confidence', 0.0)
            self.last_emotion_time = time.time()
           
        @self.sio.on('error')
        def on_error(data):
            print(f"❌ Server error: {data.get('message')}")
   
    def connect(self):
        """Connect to local WebSocket server"""
        try:
            print(f"🔗 Connecting to local WebSocket server: {self.server_url}")
           
            self.sio.connect(
                self.server_url,
                headers={'Authorization': f'Bearer {self.api_key}'},
                transports=['websocket'],
                wait_timeout=10
            )
           
            time.sleep(1)
            return self.connected
           
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
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
            print(f"❌ Error sending emotion frame: {e}")
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
            print(f"❌ Error sending stream frame: {e}")
            return False
   
    def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.connected:
            self.sio.disconnect()
            print("🔌 Disconnected from local WebSocket server")

class LocalHTTPClient:
    """HTTP client for chat messages to local server"""
   
    def __init__(self, server_url, api_key="emotion_recognition_key_123"):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})
   
    def send_chat_message(self, message):
        """Send text chat message to local server"""
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
                print(f"❌ Chat request failed: {response.status_code}")
                return None
               
        except Exception as e:
            print(f"❌ Error sending chat message: {e}")
            return None
    
    def send_speech_message(self, audio_data):
        """Send speech audio to local server for transcription and chat"""
        try:
            # Encode audio as base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            payload = {"audio": audio_b64}
            response = self.session.post(
                f"{self.server_url}/speech",
                json=payload,
                timeout=30  # Longer timeout for speech processing
            )
           
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Speech request failed: {response.status_code}")
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                        print(f"   Error details: {error_data.get('details', 'Unknown error')}")
                    except:
                        pass
                return None
               
        except Exception as e:
            print(f"❌ Error sending speech message: {e}")
            return None

class JetsonEmotionSystem:
    def __init__(self, server_url, arduino_port=None):
        self.ws_client = LocalWebSocketClient(server_url)
        self.http_client = LocalHTTPClient(server_url)
        self.arduino_handler = ArduinoHandler(arduino_port)
        # self.voice_recorder = VoiceRecorder()
        self.voice_recorder = VoiceRecorder(config={
          'sample_rate': 48000,
          'input_device_index': 11,  # Force USB microphone (confirmed working)
          'debug': True
        })
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
            print(f"🔊 TTS error: {e}")
   
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
            print("❌ Failed to start camera streamer")
            return False
           
        print("✅ Camera system started successfully")
        return True
   
    def start_chat_interface(self):
        """Start the enhanced chat interface with voice support"""
        def chat_loop():
            print("\n💬 Chat Interface Ready!")
            print("   📝 Type a message and press Enter for text chat")
            print("   🎤 Press Enter on empty line to start voice recording")
            print("   🛑 Type 'exit' to quit")
            print("-" * 50)
           
            while self.running:
                try:
                    user_input = input("\n💬 You (text) or 🎤 Enter for voice: ").strip()
                    
                    if user_input.lower() == 'exit':
                        break
                    
                    # Text input
                    elif user_input:
                        print("🔄 Processing text message...")
                        response_data = self.http_client.send_chat_message(user_input)
                        self._handle_response(response_data, "text")
                    
                    # Voice input (empty input)
                    else:
                        self._handle_voice_input()
                       
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Chat error: {e}")
       
        self.chat_thread = threading.Thread(target=chat_loop, daemon=True)
        self.chat_thread.start()
    
    def _handle_voice_input(self):
        """Handle voice recording and processing"""
        if not self.voice_recorder.audio:
            print("❌ Voice recording not available (PyAudio not initialized)")
            return
        
        # Start recording
        if not self.voice_recorder.start_recording():
            return
        
        # Wait for user to press Enter again to stop
        try:
            input()  # Wait for Enter press
        except KeyboardInterrupt:
            pass
        
        # Stop recording and get audio data
        audio_data = self.voice_recorder.stop_recording()
        
        if audio_data:
            print("🔄 Processing speech message...")
            response_data = self.http_client.send_speech_message(audio_data)
            self._handle_response(response_data, "speech")
        else:
            print("❌ No audio recorded")
    
    def _handle_response(self, response_data, input_type):
        """Handle server response for both text and speech"""
        if response_data:
            bot_response = response_data.get("response", "Sorry, no response received")
            bot_emotion = response_data.get("bot_emotion", "DEFAULT")
            detected_emotion = response_data.get("detected_emotion", "neutral")
            emotion_confidence = response_data.get("emotion_confidence", 0.0)
            
            # Print transcription if speech input
            if input_type == "speech":
                transcription = response_data.get("transcription", "")
                speech_confidence = response_data.get("speech_confidence", 0.0)
                print(f"📝 Transcribed: '{transcription}' (confidence: {speech_confidence:.1f}%)")
            
            print(f"🤖 Bot: {bot_response}")
     
            if self.arduino_handler.connected:
                self.arduino_handler.send_bot_emotion(bot_emotion)
            else:
                print("🔌 Arduino not connected - emotion not sent")
           
            self.speak_with_espeak(bot_response)
               
        else:
            print("❌ Failed to get response from server")
   
    def check_server_health(self):
        """Check if local server is healthy"""
        try:
            response = self.http_client.session.get(
                f"{self.http_client.server_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Server health check passed")
                print(f"   🌐 Server IP: {data.get('server_ip', 'unknown')}")
                print(f"   🤖 Model loaded: {data.get('components', {}).get('model_loaded')}")
                print(f"   🌐 OpenAI available: {data.get('components', {}).get('openai_available')}")
                print(f"   🎤 Speech available: {data.get('components', {}).get('speech_available')}")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
   
    def start(self):
        """Start the complete system"""
        self.running = True
       
        print("🏥 Checking server health...")
        if not self.check_server_health():
            print("❌ Server is not healthy. Please check the local server.")
            return False
       
        print("🔗 Connecting to local WebSocket server...")
        if not self.ws_client.connect():
            print("❌ Failed to connect to WebSocket server")
            print("⚠️  Continuing without real-time features...")
        else:
            print("✅ WebSocket connection established")
       
        if self.arduino_handler.connected:
            print(f"🔌 Arduino connected on: {self.arduino_handler.serial_port}")
            self.arduino_handler.send_emotion("GREETING")
        else:
            print("⚠️  Arduino not connected - continuing without Arduino integration")
        
        # List available audio devices
        if self.voice_recorder.audio:
            self.voice_recorder.list_audio_devices()
        else:
            print("⚠️  Voice recording not available")
       
        if not self.start_camera_system():
            return False
       
        self.start_chat_interface()
       
        print("✅ Chat interface ready")
        if self.ws_client.connected:
            print("🌐 WebSocket connected (real-time emotion & streaming)")
        if self.arduino_handler.connected:
            print(f"🔌 Arduino ready on {self.arduino_handler.serial_port}")
        if self.voice_recorder.audio:
            print("🎤 Voice recording ready")
        print(f"📹 Local camera stream: http://localhost:8080")
        print(f"📊 Server monitor: {self.server_url}/monitor")
        print("\nType 'exit' to quit or press Ctrl+C to stop...")
        print("="*50)
       
        try:
            while self.running:
                time.sleep(1)
                # Try to reconnect if disconnected
                if not self.ws_client.connected and time.time() % 30 == 0:
                    print("🔄 WebSocket disconnected, attempting to reconnect...")
                    self.ws_client.connect()
                   
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
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
        
        if self.voice_recorder:
            self.voice_recorder.cleanup()
       
        print("✅ System stopped")

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
   
    print("\n" + "="*60)
    print("  🤖 Jetson Local Emotion Detection System with Voice")
    print("="*60)
   
    # Try to get server URL from environment first
    server_url = os.getenv("LOCAL_SERVER_URL")
    
    if not server_url:
        # Try to discover server automatically
        print("🔍 Attempting to discover emotion server on local network...")
        server_url = discover_server_on_network()
    
    if not server_url:
        # Manual input as fallback
        server_url = input("📝 Enter local server URL (e.g., http://130.216.238.6:5000): ").strip()
   
    if not server_url.startswith(('http://', 'https://')):
        server_url = 'http://' + server_url
   
    # Test connection before proceeding
    try:
        print(f"🧪 Testing connection to {server_url}...")
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Connection test successful!")
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("🔧 Please check:")
        print("   1. Server is running on the target machine")
        print("   2. Firewall allows connections on port 5000")
        print("   3. Both devices are on the same network")
        return
   
    arduino_port = os.getenv("ARDUINO_PORT")
    if not arduino_port:
        arduino_port = input("🔌 Enter Arduino port (press Enter for default /dev/ttyUSB0): ").strip()
        if not arduino_port:
            arduino_port = None
   
    print("\n🔧 Configuration:")
    print(f"   🌐 Local Server: {server_url}")
    print(f"   🔌 Arduino Port: {arduino_port or '/dev/ttyUSB0 (default)'}")
    print(f"   📡 WebSocket Enabled: Yes")
    print(f"   🎤 Voice Support: Yes")
    print(f"   📹 Local Stream: http://localhost:8080")
   
    system = JetsonEmotionSystem(server_url, arduino_port)
   
    try:
        system.start()
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        system.stop()

if __name__ == "__main__":
    main()