# modules/input/image_input.py - Generic Image Input Module
import cv2
import base64
import time
import threading
import numpy as np
from typing import Optional, Callable, Dict, Any, List
import concurrent.futures
from threading import Lock

class ImageInputModule:
    """
    Generic image input module for capturing and sending images to server.
    
    Features:
    - Camera capture from various sources (USB, built-in, IP cameras)
    - Frame preprocessing and optimization
    - WebSocket frame transmission to server
    - Frame rate control and quality settings
    - Multiple camera support
    - Generic image processing pipeline
    
    The server handles all image processing (emotion detection, facial recognition, etc.)
    based on the enabled modules in client configuration.
    """
    
    def __init__(self, client_core, config: Dict[str, Any] = None):
        """
        Initialize image input module
        
        Args:
            client_core: Core client instance
            config: Configuration dictionary
        """
        self.client_core = client_core
        self.config = config or client_core.get_config()
        
        # Camera settings
        self.camera_settings = self.config.get('camera_settings', {})
        self.camera_device = self.config.get('hardware', {}).get('camera_device', 0)
        
        # Processing settings
        self.send_fps = self.camera_settings.get('send_fps', 1)  # How often to send frames to server
        self.jpeg_quality = self.camera_settings.get('jpeg_quality', 85)
        self.width = self.camera_settings.get('width', 640)
        self.height = self.camera_settings.get('height', 480)
        self.capture_fps = self.camera_settings.get('fps', 30)
        
        # State
        self.camera = None
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
        
        # Callbacks
        self.on_frame_captured = None       # Callback(frame) - called for each captured frame
        self.on_frame_sent = None          # Callback(frame_data) - called when frame sent to server
        self.on_server_result = None       # Callback(result_data) - called when server responds
        self.on_capture_error = None       # Callback(error_msg) - called on capture errors
        
        # Register with core client for server responses
        self.client_core.register_callback('on_frame_result', self._handle_server_result)
        
        print(f"📸 Image input module initialized")
        print(f"   📹 Camera device: {self.camera_device}")
        print(f"   📤 Send rate: {self.send_fps} FPS")
        print(f"   📐 Resolution: {self.width}x{self.height}")
        print(f"   🎬 Capture rate: {self.capture_fps} FPS")
    
    def _handle_server_result(self, data: Dict[str, Any]):
        """Handle processing results from server"""
        if self.on_server_result:
            self.on_server_result(data)
    
    def initialize_camera(self, device_id: Optional[int] = None, camera_type: str = "usb") -> bool:
        """
        Initialize camera capture
        
        Args:
            device_id: Camera device ID (optional, uses config default if None)
            camera_type: Type of camera ("usb", "builtin", "ip")
            
        Returns:
            bool: True if camera initialized successfully
        """
        try:
            camera_id = device_id if device_id is not None else self.camera_device
            
            print(f"📹 Initializing {camera_type} camera {camera_id}...")
            
            # Initialize camera based on type
            if camera_type == "ip":
                # For IP cameras, camera_id should be URL
                self.camera = cv2.VideoCapture(str(camera_id))
            else:
                # For USB/builtin cameras
                self.camera = cv2.VideoCapture(camera_id)
            
            if not self.camera.isOpened():
                print(f"❌ Failed to open camera {camera_id}")
                return False
            
            # Set camera properties
            if camera_type != "ip":  # IP cameras may not support these properties
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.camera.set(cv2.CAP_PROP_FPS, self.capture_fps)
            
            # Test capture
            ret, frame = self.camera.read()
            if not ret:
                print(f"❌ Failed to capture test frame from camera {camera_id}")
                self.camera.release()
                self.camera = None
                return False
            
            # Get actual camera properties
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            print(f"✅ Camera initialized successfully")
            print(f"   📐 Actual resolution: {actual_width}x{actual_height}")
            print(f"   🎬 Actual FPS: {actual_fps}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
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
        if not self.camera:
            print("❌ Camera not initialized")
            return False
        
        if self.running:
            print("⚠️ Capture already running")
            return True
        
        print("🎬 Starting camera capture...")
        self.running = True
        self.send_to_server = send_to_server
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        print("✅ Camera capture started")
        if send_to_server:
            print(f"📤 Auto-sending frames to server at {self.send_fps} FPS")
        
        return True
    
    def stop_capture(self):
        """Stop camera capture"""
        if not self.running:
            return
        
        print("🛑 Stopping camera capture...")
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.executor.shutdown(wait=False)
        print("✅ Camera capture stopped")
    
    def _capture_loop(self):
        """Main camera capture loop"""
        print("📹 Camera capture loop started")
        
        try:
            while self.running and self.camera:
                ret, frame = self.camera.read()
                
                if not ret:
                    print("❌ Failed to capture frame")
                    if self.on_capture_error:
                        self.on_capture_error("Failed to capture frame")
                    break
                
                self.frame_counter += 1
                
                # Update latest frame for other modules
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
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
            print(f"❌ Error in capture loop: {e}")
            if self.on_capture_error:
                self.on_capture_error(f"Capture loop error: {e}")
        finally:
            print("📹 Camera capture loop ended")
    
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
                'source': 'image_input_module'
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
        """
        Encode frame as base64 JPEG
        
        Args:
            frame: OpenCV frame
            quality: JPEG quality (1-100)
            
        Returns:
            str: Base64 encoded frame
        """
        with self.encoding_lock:
            try:
                # Try using turbojpeg for faster encoding if available
                try:
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
                
            except Exception as e:
                print(f"❌ Error encoding frame: {e}")
                raise
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame
        
        Returns:
            numpy.ndarray: Latest frame or None if not available
        """
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def capture_single_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from camera
        
        Returns:
            numpy.ndarray: Captured frame or None if failed
        """
        if not self.camera:
            print("❌ Camera not initialized")
            return None
        
        try:
            ret, frame = self.camera.read()
            if ret:
                return frame
            else:
                print("❌ Failed to capture single frame")
                return None
        except Exception as e:
            print(f"❌ Error capturing single frame: {e}")
            return None
    
    def send_frame_to_server(self, frame: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """
        Send a specific frame to server for processing
        
        Args:
            frame: OpenCV frame to process
            metadata: Optional metadata to include with frame
            
        Returns:
            bool: True if sent successfully
        """
        try:
            frame_b64 = self._encode_frame(frame, self.jpeg_quality)
            
            frame_data = {
                'frame': frame_b64,
                'timestamp': time.time(),
                'source': 'manual_send'
            }
            
            # Add metadata if provided
            if metadata:
                frame_data.update(metadata)
            
            success = self.client_core.send_websocket_frame(frame_data)
            
            if success and self.on_frame_sent:
                self.on_frame_sent(frame_data)
            
            return success
            
        except Exception as e:
            print(f"❌ Error sending manual frame: {e}")
            if self.on_capture_error:
                self.on_capture_error(f"Manual frame error: {e}")
            return False
    
    def send_image_file(self, image_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Send an image file to server for processing
        
        Args:
            image_path: Path to image file
            metadata: Optional metadata to include
            
        Returns:
            bool: True if sent successfully
        """
        try:
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"❌ Failed to load image: {image_path}")
                return False
            
            # Add file info to metadata
            file_metadata = {'source_file': image_path}
            if metadata:
                file_metadata.update(metadata)
            
            return self.send_frame_to_server(frame, file_metadata)
            
        except Exception as e:
            print(f"❌ Error sending image file: {e}")
            return False
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get camera information and status
        
        Returns:
            dict: Camera information
        """
        if not self.camera:
            return {'status': 'not_initialized'}
        
        try:
            info = {
                'status': 'initialized' if self.camera.isOpened() else 'error',
                'running': self.running,
                'device_id': self.camera_device,
                'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.camera.get(cv2.CAP_PROP_FPS),
                'capture_fps': self.capture_fps,
                'send_fps': self.send_fps,
                'frame_counter': self.frame_counter,
                'jpeg_quality': self.jpeg_quality,
                'send_to_server': getattr(self, 'send_to_server', False)
            }
            return info
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def set_send_fps(self, fps: float):
        """
        Set frame sending rate to server
        
        Args:
            fps: Frames per second to send to server
        """
        if fps <= 0:
            print("❌ FPS must be positive")
            return
        
        self.send_fps = fps
        print(f"📤 Server send FPS set to: {fps}")
    
    def set_jpeg_quality(self, quality: int):
        """
        Set JPEG encoding quality
        
        Args:
            quality: JPEG quality (1-100)
        """
        if not 1 <= quality <= 100:
            print("❌ JPEG quality must be between 1 and 100")
            return
        
        self.jpeg_quality = quality
        print(f"📸 JPEG quality set to: {quality}")
    
    def set_resolution(self, width: int, height: int):
        """
        Set camera resolution (requires restart)
        
        Args:
            width: Frame width
            height: Frame height
        """
        self.width = width
        self.height = height
        
        if self.camera and self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📐 Resolution set to: {actual_width}x{actual_height}")
    
    def list_available_cameras(self) -> List[Dict[str, Any]]:
        """
        List available camera devices
        
        Returns:
            list: List of available camera devices
        """
        cameras = []
        
        # Test camera indices 0-10
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        cameras.append({
                            'index': i,
                            'width': width,
                            'height': height,
                            'fps': fps,
                            'status': 'available'
                        })
                cap.release()
            except:
                continue
        
        return cameras
    
    def cleanup(self):
        """Cleanup image input module resources"""
        self.stop_capture()
        
        # Unregister callbacks
        self.client_core.unregister_callback('on_frame_result', self._handle_server_result)
        
        print("🧹 Image input module cleaned up")