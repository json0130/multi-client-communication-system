# modules/input/camera_input.py - Generic camera input module using OpenCV
import cv2
import time
import threading
import base64
import logging
from typing import Optional, Dict
from client import InputModule

logger = logging.getLogger(__name__)

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("⚠️ OpenCV not available - camera input disabled")

class CameraInputModule(InputModule):
    """Generic camera input module using OpenCV"""
    
    def __init__(self, name: str = "camera_input", config: Dict = None):
        super().__init__(name, config)
        
        # Camera configuration
        self.camera_index = self.config.get('camera_index', 0)
        self.width = self.config.get('width', 640)
        self.height = self.config.get('height', 480)
        self.fps = self.config.get('fps', 30)
        self.send_fps = self.config.get('send_fps', 1)  # Send to server FPS
        self.jpeg_quality = self.config.get('jpeg_quality', 85)
        
        # Camera object
        self.cap = None
        self.capture_thread = None
        self.last_send_time = 0
        self.stop_event = threading.Event()

        self.cap_lock = threading.Lock()
    
    def initialize(self) -> bool:
        """Initialize camera"""
        if not OPENCV_AVAILABLE:
            logger.error("❌ OpenCV not available")
            return False
        
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"❌ Cannot open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Test capture
            ret, frame = self.cap.read()
            if ret:
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                
                logger.info(f"📸 Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
                logger.info(f"   📡 Sending frames at {self.send_fps} FPS to server")
                return True
            else:
                logger.error("❌ Failed to capture test frame")
                return False
                
        except Exception as e:
            logger.error(f"❌ Camera initialization error: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def start(self) -> bool:
        """Start camera capture thread"""
        if not self.enabled and self.cap:
            self.enabled = True
            self.stop_event.clear()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            logger.info("📸 Camera capture started")
            return True
        return False
    
    def stop(self):
        """Stop camera capture"""
        if self.enabled:
            self.enabled = False
            self.stop_event.set()
            if self.capture_thread:
                self.capture_thread.join(timeout=1)
            if self.cap:
                self.cap.release()
            logger.info("📸 Camera stopped")
    
    def get_data(self) -> Optional[str]:
        """Get current frame as base64 encoded string"""
        with self.cap_lock:
            if not self.cap or not self.cap.isOpened():
                return None
            
            ret, frame = self.cap.read()
            
            if ret and frame is not None and frame.size > 0:
                # Encode frame to base64
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                return frame_b64
            return None
    
    def _capture_loop(self):
        """Main capture loop"""
        logger.info("📹 Camera capture loop started")
        
        while not self.stop_event.is_set() and self.enabled:
            try:
                current_time = time.time()
                
                # Send frame at specified FPS
                if current_time - self.last_send_time >= (1.0 / self.send_fps):
                    frame_data = self.get_data()
                    if frame_data and self.client:
                        self.client.send_to_server('frame', frame_data)
                        self.last_send_time = current_time
                
                time.sleep(1)  # Maintain capture FPS
                
            except Exception as e:
                logger.error(f"❌ Capture error: {e}")
                time.sleep(1)