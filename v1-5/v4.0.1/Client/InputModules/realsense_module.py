# modules/input/realsense_input.py - Simplified RealSense Camera Input
import cv2
import base64
import time
import threading
import numpy as np
from typing import Optional, Callable, Dict, Any
import concurrent.futures
from threading import Lock

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("⚠️ pyrealsense2 not available - RealSense camera disabled")

class RealSenseInputModule:
    """
    Simplified RealSense camera input module for emotion detection.
    
    Features:
    - RealSense D400 series camera support
    - Real-time frame capture and sending to server
    - Optimized for emotion detection
    - Local web streaming (optional)
    - Frame rate control
    """
    
    def __init__(self, client_core, config: Dict[str, Any] = None):
        """
        Initialize RealSense input module
        
        Args:
            client_core: Core client instance
            config: Configuration dictionary
        """
        if not REALSENSE_AVAILABLE:
            raise ImportError("pyrealsense2 not available. Install with: pip install pyrealsense2")
        
        self.client_core = client_core
        self.config = config or client_core.get_config()
        
        # Camera settings
        self.camera_settings = self.config.get('camera_settings', {})
        self.send_fps = self.camera_settings.get('send_fps', 10)  # Increased from 1 to 10 FPS
        self.jpeg_quality = self.camera_settings.get('jpeg_quality', 85)
        self.width = self.camera_settings.get('width', 640)
        self.height = self.camera_settings.get('height', 480)
        self.capture_fps = self.camera_settings.get('fps', 30)
        
        # RealSense pipeline
        self.pipeline = None
        self.running = False
        self.capture_thread = None
        self.last_send_time = 0
        self.frame_counter = 0
        
        # Threading
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.encoding_lock = Lock()
        
        # Current frame for other modules to access
        self.latest_frame = None
        self.frame_lock = Lock()
        
        # Web streaming (optional)
        self.web_streaming = self.config.get('features', {}).get('camera_streaming', False)
        self.stream_port = self.config.get('hardware', {}).get('stream_port', 8080)
        self.latest_jpg = None
        
        # Callbacks
        self.on_frame_captured = None       # Callback(frame) - called for each captured frame
        self.on_frame_sent = None          # Callback(frame_data) - called when frame sent to server
        self.on_server_result = None       # Callback(result_data) - called when server responds
        self.on_capture_error = None       # Callback(error_msg) - called on capture errors
        
        # Register with core client for server responses
        self.client_core.register_callback('on_frame_result', self._handle_server_result)
        
        print(f"📸 RealSense input module initialized")
        print(f"   📐 Resolution: {self.width}x{self.height}")
        print(f"   📤 Send rate: {self.send_fps} FPS")
        print(f"   🎬 Capture rate: {self.capture_fps} FPS")
        print(f"   🌐 Web streaming: {self.web_streaming}")
    
    def _handle_server_result(self, data: Dict[str, Any]):
        """Handle processing results from server"""
        if self.on_server_result:
            self.on_server_result(data)
    
    def initialize_camera(self) -> bool:
        """
        Initialize RealSense camera
        
        Returns:
            bool: True if camera initialized successfully
        """
        try:
            print("📹 Initializing RealSense camera...")
            
            # Create pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable color stream
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.capture_fps)
            
            # Start pipeline
            profile = self.pipeline.start(config)
            
            # Get device and optimize settings
            device = profile.get_device()
            color_sensor = device.first_color_sensor()
            
            # Set auto exposure
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            
            # Test capture
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                print("❌ Failed to capture test frame from RealSense")
                self.pipeline.stop()
                self.pipeline = None
                return False
            
            print("✅ RealSense camera initialized successfully")
            print(f"   📐 Actual resolution: {color_frame.get_width()}x{color_frame.get_height()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing RealSense camera: {e}")
            if self.on_capture_error:
                self.on_capture_error(f"Camera initialization error: {e}")
            return False
    
    def start_capture(self, send_to_server: bool = True) -> bool:
        """
        Start camera capture
        
        Args:
            send_to_server: Whether to automatically send frames to server
            
        Returns:
            bool: True if capture started successfully
        """
        if not self.pipeline:
            print("❌ RealSense camera not initialized")
            return False
        
        if self.running:
            print("⚠️ Capture already running")
            return True
        
        print("🎬 Starting RealSense capture...")
        self.running = True
        self.send_to_server = send_to_server
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Start web streaming if enabled
        if self.web_streaming:
            self._start_web_streaming()
        
        print("✅ RealSense capture started")
        if send_to_server:
            print(f"📤 Auto-sending frames to server at {self.send_fps} FPS")
        
        return True
    
    def stop_capture(self):
        """Stop camera capture"""
        if not self.running:
            return
        
        print("🛑 Stopping RealSense capture...")
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None
        
        self.executor.shutdown(wait=False)
        print("✅ RealSense capture stopped")
    
    def _capture_loop(self):
        """Main camera capture loop"""
        print("📹 RealSense capture loop started")
        
        try:
            while self.running and self.pipeline:
                try:
                    # Get frames with timeout
                    frames = self.pipeline.poll_for_frames()
                    if not frames:
                        time.sleep(0.001)
                        continue
                    
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                    
                    # Convert to numpy array
                    frame = np.asanyarray(color_frame.get_data())
                    self.frame_counter += 1
                    
                    # Update latest frame for other modules
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                    
                    # Update JPEG for web streaming
                    if self.web_streaming:
                        self._update_web_frame(frame)
                    
                    # Call frame captured callback
                    if self.on_frame_captured:
                        try:
                            self.on_frame_captured(frame.copy())
                        except Exception as e:
                            print(f"❌ Error in frame captured callback: {e}")
                    
                    # Send frame to server if enabled
                    if self.send_to_server:
                        self._maybe_send_frame_to_server(frame)
                    
                    # Control capture rate
                    time.sleep(1.0 / self.capture_fps)
                    
                except Exception as e:
                    print(f"❌ Error in capture loop iteration: {e}")
                    time.sleep(0.1)
                
        except Exception as e:
            print(f"❌ Error in RealSense capture loop: {e}")
            if self.on_capture_error:
                self.on_capture_error(f"Capture loop error: {e}")
        finally:
            print("📹 RealSense capture loop ended")
    
    def _maybe_send_frame_to_server(self, frame: np.ndarray):
        """Send frame to server based on configured FPS"""
        current_time = time.time()
        
        # Send frame at specified FPS
        if current_time - self.last_send_time >= (1.0 / self.send_fps):
            self.last_send_time = current_time
            # Send frame asynchronously to avoid blocking capture
            self.executor.submit(self._send_frame_to_server_async, frame.copy())
    
    def _send_frame_to_server_async(self, frame: np.ndarray):
        """Send frame to server via WebSocket (async)"""
        try:
            # Encode frame as base64
            frame_b64 = self._encode_frame(frame, self.jpeg_quality)
            
            # Create frame data
            frame_data = {
                'frame': frame_b64,
                'timestamp': time.time(),
                'frame_counter': self.frame_counter,
                'source': 'realsense_camera'
            }
            
            # Send via WebSocket to server
            if self.client_core.client_initialized:
                success = self.client_core.send_websocket_frame(frame_data)
                
                if success and self.on_frame_sent:
                    self.on_frame_sent(frame_data)
            
        except Exception as e:
            print(f"❌ Error sending frame to server: {e}")
            if self.on_capture_error:
                self.on_capture_error(f"Frame sending error: {e}")
    
    def _encode_frame(self, frame: np.ndarray, quality: int = 85) -> str:
        """Encode frame as base64 JPEG"""
        with self.encoding_lock:
            try:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                return base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(f"❌ Error encoding frame: {e}")
                raise
    
    def _update_web_frame(self, frame: np.ndarray):
        """Update frame for web streaming"""
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # Lower quality for streaming
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            self.latest_jpg = buffer.tobytes()
        except Exception as e:
            print(f"❌ Error encoding web frame: {e}")
    
    def _start_web_streaming(self):
        """Start simple web streaming server"""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import threading
            
            class StreamHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    camera = self.server.camera_module
                    
                    if self.path == '/':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        
                        html = f'''
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>RealSense Camera - {camera.client_core.robot_name}</title>
                            <style>
                                body {{ font-family: Arial; text-align: center; margin: 20px; }}
                                img {{ max-width: 100%; border: 2px solid #333; }}
                            </style>
                        </head>
                        <body>
                            <h1>RealSense Camera - {camera.client_core.robot_name}</h1>
                            <img id="stream" src="/frame.jpg" onload="setTimeout(() => {{ this.src='/frame.jpg?t=' + Date.now() }}, 100)">
                        </body>
                        </html>
                        '''
                        self.wfile.write(html.encode())
                    
                    elif self.path.startswith('/frame.jpg'):
                        if camera.latest_jpg:
                            self.send_response(200)
                            self.send_header('Content-type', 'image/jpeg')
                            self.send_header('Cache-Control', 'no-cache')
                            self.end_headers()
                            self.wfile.write(camera.latest_jpg)
                        else:
                            self.send_response(503)
                            self.end_headers()
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                def log_message(self, format, *args):
                    return  # Suppress logs
            
            class StreamServer(HTTPServer):
                def __init__(self, server_address, RequestHandlerClass, camera_module):
                    self.camera_module = camera_module
                    super().__init__(server_address, RequestHandlerClass)
            
            # Start server in background
            def run_server():
                server = StreamServer(('0.0.0.0', self.stream_port), StreamHandler, self)
                print(f"🌐 Web streaming available at http://localhost:{self.stream_port}")
                server.serve_forever()
            
            threading.Thread(target=run_server, daemon=True).start()
            
        except Exception as e:
            print(f"❌ Error starting web streaming: {e}")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame"""
        if not self.pipeline:
            print("❌ RealSense camera not initialized")
            return None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if color_frame:
                return np.asanyarray(color_frame.get_data())
            else:
                print("❌ Failed to capture single frame")
                return None
        except Exception as e:
            print(f"❌ Error capturing single frame: {e}")
            return None
    
    def send_frame_to_server(self, frame: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """Send a specific frame to server for processing"""
        try:
            frame_b64 = self._encode_frame(frame, self.jpeg_quality)
            
            frame_data = {
                'frame': frame_b64,
                'timestamp': time.time(),
                'source': 'manual_realsense'
            }
            
            if metadata:
                frame_data.update(metadata)
            
            success = self.client_core.send_websocket_frame(frame_data)
            
            if success and self.on_frame_sent:
                self.on_frame_sent(frame_data)
            
            return success
            
        except Exception as e:
            print(f"❌ Error sending manual frame: {e}")
            return False
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information and status"""
        return {
            'status': 'initialized' if self.pipeline else 'not_initialized',
            'running': self.running,
            'camera_type': 'realsense',
            'width': self.width,
            'height': self.height,
            'capture_fps': self.capture_fps,
            'send_fps': self.send_fps,
            'frame_counter': self.frame_counter,
            'jpeg_quality': self.jpeg_quality,
            'web_streaming': self.web_streaming,
            'stream_port': self.stream_port if self.web_streaming else None
        }
    
    def cleanup(self):
        """Cleanup RealSense input module resources"""
        self.stop_capture()
        
        # Unregister callbacks
        self.client_core.unregister_callback('on_frame_result', self._handle_server_result)
        
        print("🧹 RealSense input module cleaned up")