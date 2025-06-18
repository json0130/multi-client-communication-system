# server.py - Modular Main Server with Speech-to-Text Support
import os
import time
import socket
import threading
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from dotenv import load_dotenv

# Import necessary libraries for Healthcare
import csv
import re
from itertools import combinations

# Import our modular components
from emotion_processor import EmotionProcessor
from gpt_client import GPTClient
from web_interface import WebInterface
from websocket_handler import WebSocketHandler
from speech_processor import SpeechProcessor

# Configuration
MODEL_PATH = './models/efficientnet_HQRAF_improved_withCon.pth'
API_KEY = "emotion_recognition_key_123"
PORT = 5001
load_dotenv()

# This is for healthcare drug interaction analysis
def extract_drugs(message):
    # Naive token-based extractor (replace with smarter NER later if needed)
    match = re.findall(r"\b([A-Za-z0-9\(\)\-\+\/]+(?: [A-Za-z0-9\(\)\-\+\/]+)?)\b", message)
    common = {'can', 'i', 'take', 'together', 'and', 'with', 'is', 'safe', 'mix', 'combine', 'what', 'about', 'the'}
    return [w.strip() for w in match if w.lower() not in common]

def analyze_drugs_from_message(message):
    drugs = extract_drugs(message)
    interactions = []

    # Lower all for safer matching
    drugs = [d.strip().lower() for d in drugs]

    if len(drugs) == 1:
        # Check for intention to ask about all combinations with 1 drug
        if re.search(r"all.*(interaction|combination|mix)", message.lower()):
            target = drugs[0]
            filepath = os.path.join(os.path.dirname(__file__), 'ddinter_downloads_code_V.csv')
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        d1 = row['Drug_A'].lower()
                        d2 = row['Drug_B'].lower()
                        if target == d1 or target == d2:
                            interactions.append(
                                f"{row['Level']} interaction: {row['Drug_A']} + {row['Drug_B']}"
                            )
            except Exception as e:
                return [], [f"[ERROR] Could not read CSV: {e}"]
            return drugs, interactions

    elif len(drugs) >= 2:
        pairs = combinations(drugs, 2)
        for d1, d2 in pairs:
            result = check_interaction_csv(d1, d2)
            if result:
                interactions.append(result)
        return drugs, interactions

    return drugs, []
# ------------------

def normalize_name(name):
    return name.lower().strip().replace("’", "'").replace("–", "-")

def normalize_name(name):
    return name.lower().strip()

def check_interaction_csv(drug1, drug2, filepath='ddinter_downloads_code_V.csv'):
    drug1 = normalize_name(drug1)
    drug2 = normalize_name(drug2)
    filepath = os.path.join(os.path.dirname(__file__), filepath)

    print(f"🔍 Checking: {drug1} vs {drug2}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                d1 = normalize_name(row['Drug_A'])
                d2 = normalize_name(row['Drug_B'])
                if (drug1 in d1 and drug2 in d2) or (drug1 in d2 and drug2 in d1):
                    print(f"✅ Match found: {row['Drug_A']} + {row['Drug_B']}")
                    return f"{row['Level']} interaction found between {row['Drug_A']} and {row['Drug_B']}."
        print("❌ No match found.")
        return None
    except Exception as e:
        return f"[ERROR] Could not check CSV: {e}"

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address to determine which interface to use
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

class EmotionServer:
    """Main emotion detection server with modular components and speech support"""
    
    def __init__(self):
        # Server configuration
        self.server_ip = get_local_ip()
        self.port = PORT
        self.api_key = API_KEY
        
        # Configuration for components
        self.config = {
            'emotion_processing_interval': 0.1,
            'stream_fps': 30,
            'frame_skip_ratio': 1,
            'emotion_update_threshold': 0.05,
            'emotion_window_size': 5,
            'confidence_threshold': 30.0,
            'emotion_change_threshold': 15.0,
            'server_ip': self.server_ip,
            # Speech processing configuration
            'whisper_model_size': 'base',  # base, small, medium, large
            'whisper_device': 'auto',
            'whisper_compute_type': 'float16',
            'max_audio_length': 30,
            'sample_rate': 16000
        }
        
        # Initialize Flask app and SocketIO
        self.app = Flask(__name__)
        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode='threading',
            logger=False,
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25,
            max_http_buffer_size=2000000,  # Increased for audio files
            transports=['websocket', 'polling'],
            allow_upgrades=True,
            cookie=False
        )
    
        @self.app.route('/check_drug', methods=['POST'])
        def check_drug():
            data = request.json
            drug1 = data.get("drug1", "").strip()
            drug2 = data.get("drug2", "").strip()

            if not drug1 or not drug2:
                return jsonify({"response": "[DEFAULT] Please provide two drug names."})

            interaction = check_interaction_csv(drug1, drug2)

            if interaction:
                prompt = (
                    f"The user asked about the interaction between {drug1} and {drug2}.\n"
                    f"A known interaction exists in our medical database: {interaction}\n"
                    f"Please explain what this interaction means in simple, medically appropriate language. "
                    f"Be specific and repeat the interaction level (e.g., Moderate, Major), "
                    f"and clearly describe the possible consequences and recommended action."
                )
            else:
                prompt = (
                    f"The user asked about taking {drug1} and {drug2} together. "
                    f"No known interaction was found in the database. "
                    f"Please explain this carefully, and remind the user to consult a healthcare professional."
                )

            gpt = GPTClient()
            if not gpt.setup_openai():
                return jsonify({"response": "[DEFAULT] GPT is not available right now."})

            response = gpt.ask_chatgpt_optimized(prompt, "DEFAULT", 0.9)
            return jsonify({"response": response})

        # Initialize components
        self.emotion_processor = EmotionProcessor(MODEL_PATH, self.config)
        self.gpt_client = GPTClient()
        self.speech_processor = SpeechProcessor(self.config)
        self.web_interface = WebInterface(self.config['stream_fps'])
        self.websocket_handler = WebSocketHandler(
            self.socketio, 
            self.emotion_processor, 
            self.gpt_client, 
            self.config
        )
        
        # Setup routes
        self.setup_routes()
        
        # Component status
        self.components_initialized = 0
        self.total_components = 4  # Updated for speech processor
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/', methods=['GET'])
        def root_info():
            """Root endpoint with enhanced server info"""
            emotion, confidence = self.emotion_processor.get_current_emotion()
            return jsonify({
                "message": "Local Real-time Emotion-Aware System Server with Speech Support",
                "status": "running",
                "server_ip": self.server_ip,
                "components": {
                    **self.emotion_processor.get_status(),
                    "openai_available": self.gpt_client.is_available(),
                    "speech_available": self.speech_processor.is_available()
                },
                "optimization": {
                    "stream_fps": self.config['stream_fps'],
                    "emotion_interval": self.config['emotion_processing_interval'],
                    "frame_skip_ratio": self.config['frame_skip_ratio'],
                    "emotion_window_size": self.config['emotion_window_size'],
                    "confidence_threshold": self.config['confidence_threshold']
                },
                "current_emotion": {
                    "emotion": emotion,
                    "confidence": round(confidence, 1),
                    "distribution": self.emotion_processor.get_emotion_distribution()
                },
                "endpoints": {
                    "health": "/health",
                    "stats": "/stats",
                    "chat": "/chat (POST, requires auth)",
                    "speech": "/speech (POST, requires auth)",
                    "websocket": "/socket.io/",
                    "live_stream": "/live_stream",
                    "monitor": "/monitor"
                }
            })

        @self.app.route('/health', methods=['GET'])
        def health():
            """Enhanced health check with detailed emotion info"""
            emotion, confidence = self.emotion_processor.get_current_emotion()
            return jsonify({
                "status": "healthy",
                "server_ip": self.server_ip,
                "components": {
                    **self.emotion_processor.get_status(),
                    "openai_available": self.gpt_client.is_available(),
                    "speech_available": self.speech_processor.is_available()
                },
                "current_emotion": {
                    "emotion": emotion,
                    "confidence": round(confidence, 1),
                    "distribution": self.emotion_processor.get_emotion_distribution(),
                    "window_size": len(self.emotion_processor.emotion_tracker.emotion_history)
                },
                "active_connections": len(self.websocket_handler.connection_timestamps),
                "timestamp": time.time()
            })

        @self.app.route('/chat', methods=['POST'])
        def chat():
            """Enhanced chat endpoint that broadcasts to monitors"""
            try:
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer ') or auth_header.split(' ')[1] != self.api_key:
                    return jsonify({"error": "Authentication required"}), 401

                data = request.json
                message = data.get('message', '')

                if not message:
                    return jsonify({"error": "No message provided"}), 400

                return self._process_chat_message(message, "text")

            except Exception as e:
                print(f"Chat endpoint error: {e}")
                return jsonify({"error": "Internal server error"}), 500

        @self.app.route('/speech', methods=['POST'])
        def speech():
            """New speech endpoint for audio transcription and chat"""
            try:
                auth_header = request.headers.get('Authorization')
                if not auth_header or not auth_header.startswith('Bearer ') or auth_header.split(' ')[1] != self.api_key:
                    return jsonify({"error": "Authentication required"}), 401

                if not self.speech_processor.is_available():
                    return jsonify({"error": "Speech-to-text not available"}), 503

                data = request.json
                audio_b64 = data.get('audio', '')

                if not audio_b64:
                    return jsonify({"error": "No audio data provided"}), 400

                print("🎤 Received speech request")

                # Transcribe audio to text
                success, transcription, speech_confidence = self.speech_processor.transcribe_audio_base64(audio_b64)

                if not success:
                    print(f"❌ Speech transcription failed: {transcription}")
                    return jsonify({
                        "error": "Speech transcription failed",
                        "details": transcription
                    }), 400

                # Check if transcription is empty or too short
                if not transcription or len(transcription.strip()) < 2:
                    print(f"⚠️ Transcription too short or empty: '{transcription}'")
                    return jsonify({
                        "error": "No meaningful speech detected",
                        "details": f"Transcription: '{transcription}'"
                    }), 400

                print(f"📝 Transcribed: '{transcription}' (confidence: {speech_confidence:.1f}%)")

                # Process the transcribed message like a regular chat
                response = self._process_chat_message(transcription, "speech")
                
                # Add speech-specific info to response
                if isinstance(response, tuple):
                    response_data, status_code = response
                    response_data = response_data.get_json()
                else:
                    response_data = response.get_json()
                    status_code = 200

                response_data.update({
                    "transcription": transcription,
                    "speech_confidence": round(speech_confidence, 1),
                    "input_type": "speech"
                })

                return jsonify(response_data), status_code

            except Exception as e:
                print(f"❌ Speech endpoint error: {e}")
                import traceback
                print(f"   Traceback: {traceback.format_exc()}")
                return jsonify({"error": "Internal server error", "details": str(e)}), 500

        @self.app.route('/stats')
        def stats():
            """Get server statistics"""
            emotion, confidence = self.emotion_processor.get_current_emotion()

            return jsonify({
                "server_ip": self.server_ip,
                "model_loaded": self.emotion_processor.model_loaded,
                "openai_available": self.gpt_client.is_available(),
                "speech_available": self.speech_processor.is_available(),
                "cuda_available": torch.cuda.is_available() if 'torch' in globals() else False,
                "device": str(torch.cuda.get_device_name(0)) if 'torch' in globals() and torch.cuda.is_available() else "CPU",
                "current_emotion": emotion,
                "emotion_confidence": confidence,
                "last_update": self.emotion_processor.last_emotion_update,
                "websocket_enabled": True,
                "active_connections": len(self.websocket_handler.connection_timestamps)
            })

        @self.app.route('/monitor')
        def monitor():
            """Serve the monitoring interface"""
            return self.web_interface.get_monitor_html()

        @self.app.route('/live_stream')
        def live_stream():
            """Live video stream endpoint (MJPEG)"""
            return self.web_interface.generate_live_stream(self.websocket_handler.get_latest_frame)

        @self.app.after_request
        def after_request(response):
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            return response
    
    def _process_chat_message(self, message, input_type="text"):
        """Process chat message (from text or speech) and return response"""
        # Get current emotion state
        detected_emotion, emotion_confidence = self.emotion_processor.get_current_emotion()
        emotion_distribution = self.emotion_processor.get_emotion_distribution()

        # Broadcast user message to monitors
        self.websocket_handler.broadcast_chat_message({
            'type': 'user',
            'content': message,
            'emotion': detected_emotion,
            'input_type': input_type,
            'timestamp': time.time()
        })

        # Analyze drug content and find interactions
        drugs, interactions = analyze_drugs_from_message(message)

        if interactions:
            joined = "\n".join(interactions)
            prompt = (
                f"The user asked about the drugs: {', '.join(drugs)}.\n"
                f"Here are the known interactions I found:\n{joined}\n"
                f"Please explain the risks clearly and whether these drugs are safe together."
            )
        else:
            prompt = f"The user asked: {message}. Please provide a medically appropriate response."

        if emotion_confidence > 10:
            print(f"🎭 Using detected emotion: {detected_emotion} ({emotion_confidence:.1f}%)")

        print(f"📤 Sending to GPT: [{detected_emotion}] {prompt}")
        response_text = self.gpt_client.ask_chatgpt_optimized(prompt, detected_emotion, emotion_confidence)
        bot_emotion = self.gpt_client.extract_emotion_tag(response_text)

        print(f"🤖 GPT-4o-mini: {response_text}")

        # Broadcast bot response to monitors
        self.websocket_handler.broadcast_chat_message({
            'type': 'bot',
            'content': response_text,
            'emotion': bot_emotion,
            'timestamp': time.time()
        })

        return jsonify({
            "response": response_text,
            "bot_emotion": bot_emotion,
            "detected_emotion": detected_emotion,
            "confidence": round(emotion_confidence, 1),
            "emotion_distribution": emotion_distribution,
            "input_type": input_type
        })

    def initialize_components(self):
        """Initialize all server components"""
        print("🚀 Initializing Local Real-time Emotion Detection Server with Speech Support...")
        print("="*70)
        print(f"🌐 Server IP: {self.server_ip}")
        print(f"🔌 Port: {self.port}")
        print("="*70)

        self.components_initialized = 0

        # Initialize emotion processor
        print("\n1️⃣ Loading emotion detection components...")
        emotion_success, emotion_total = self.emotion_processor.initialize()
        if emotion_success == emotion_total:
            self.components_initialized += 1
            print("    ✅ Emotion processing initialized successfully")
        else:
            print(f"    ⚠️ Emotion processing partially initialized ({emotion_success}/{emotion_total})")

        # Initialize GPT client
        print("\n2️⃣ Setting up OpenAI...")
        if self.gpt_client.setup_openai():
            self.components_initialized += 1
            print("    ✅ OpenAI setup successful")
        else:
            print("    ❌ OpenAI setup failed")

        # Initialize speech processor
        print("\n3️⃣ Setting up Speech-to-Text...")
        if self.speech_processor.initialize():
            self.components_initialized += 1
            print("    ✅ Speech-to-text setup successful")
        else:
            print("    ❌ Speech-to-text setup failed")

        # Web interface is always available
        self.components_initialized += 1
        print("\n4️⃣ Web interface ready")
        print("    ✅ Web interface initialized")

        print(f"\n✅ {self.components_initialized}/{self.total_components} components initialized")

        print("\n📊 Component Status:")
        print(f"  🤖 Emotion model loaded: {'✅' if self.emotion_processor.model_loaded else '❌'}")
        print(f"  🔄 Transform loaded: {'✅' if self.emotion_processor.transform_loaded else '❌'}")
        print(f"  👤 Face cascade loaded: {'✅' if self.emotion_processor.face_cascade_loaded else '❌'}")
        print(f"  🌐 OpenAI available: {'✅' if self.gpt_client.is_available() else '❌'}")
        print(f"  🎤 Speech-to-text available: {'✅' if self.speech_processor.is_available() else '❌'}")

        return True
    
    def cleanup_resources(self):
        """Cleanup function to prevent resource leaks"""
        try:
            # Clear emotion tracker history
            self.emotion_processor.emotion_tracker.emotion_history.clear()
            self.emotion_processor.emotion_tracker.confidence_history.clear()
            self.emotion_processor.emotion_tracker.emotion_counts.clear()
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    def start(self):
        """Start the emotion detection server"""
        try:
            # Initialize all components
            self.initialize_components()

            print(f"\n🌐 Local Server URLs:")
            print(f"   🏠 Main: http://{self.server_ip}:{self.port}")
            print(f"   🔌 WebSocket: ws://{self.server_ip}:{self.port}/socket.io/")
            print(f"   📹 Live Stream: http://{self.server_ip}:{self.port}/live_stream")
            print(f"   📊 Monitor: http://{self.server_ip}:{self.port}/monitor")
            print(f"   🏥 Health: http://{self.server_ip}:{self.port}/health")
            print(f"   💬 Chat API: http://{self.server_ip}:{self.port}/chat")
            print(f"   🎤 Speech API: http://{self.server_ip}:{self.port}/speech")
            print(f"\n⚙️ Real-time Config: {self.config['stream_fps']}fps stream, {1/self.config['emotion_processing_interval']:.0f}fps emotion detection")

            print(f"\n🔧 Client Configuration:")
            print(f"   Update your Jetson client to use: http://{self.server_ip}:{self.port}")

            print("\n" + "="*70)
            print("🚀 Local emotion server with speech support is ready!")
            print("="*70)

            import atexit
            atexit.register(self.cleanup_resources)

            # Start the server
            self.socketio.run(
                self.app,
                host='0.0.0.0',  # Listen on all interfaces
                port=self.port,
                debug=False,
                allow_unsafe_werkzeug=True,
                use_reloader=False,
                log_output=False
            )

        except KeyboardInterrupt:
            print("\n🛑 Server shutdown")
            self.cleanup_resources()
        except Exception as e:
            print(f"❌ Server error: {e}")
            self.cleanup_resources()

def main():
    """Main server function"""
    import torch  # Import here to make it available for stats
    globals()['torch'] = torch
    
    server = EmotionServer()
    server.start()

if __name__ == "__main__":
    main()