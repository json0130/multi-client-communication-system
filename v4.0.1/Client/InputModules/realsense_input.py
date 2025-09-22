# modules/input/realsense_input.py - RealSense camera input module
import time
import threading
import base64
import numpy as np
import logging
from typing import Optional, Dict
from client import InputModule

logger = logging.getLogger(__name__)

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    logger.warning("⚠️ RealSense library not available - RealSense input disabled")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("⚠️ OpenCV not available - image encoding disabled")

class RealSenseInputModule(InputModule):
    """RealSense camera input module with depth support"""
    
    def __init__(self, name: str = "realsense_input", config: Dict = None):
        super().__init__(name, config)
        
        # RealSense configuration
        self.width = self.config.get('width', 640)
        self.height = self.config.get('height', 480)
        self.fps = self.config.get('fps', 30)
        self.send_fps = self.config.get('send_fps', 1)  # Send to server FPS
        self.jpeg_quality = self.config.get('jpeg_quality', 85)
        self.enable_depth = self.config.get('enable_depth', False)
        
        # RealSense objects
        self.pipeline = None
        self.config_rs = None
        
        # Threading
        self.capture_thread = None
        self.last_send_time = 0
        self.stop_event = threading.Event()
    
    def initialize(self) -> bool:
        """Initialize RealSense camera"""
        if not REALSENSE_AVAILABLE:
            logger.error("❌ RealSense library not available")
            return False
        
        if not OPENCV_AVAILABLE:
            logger.error("❌ OpenCV not available for image encoding")
            return False
        
        try:
            logger.info("📹 Initializing RealSense camera...")
            
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config_rs = rs.config()
            
            # Configure streams
            self.config_rs.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            if self.enable_depth:
                self.config_rs.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                logger.info("   📏 Depth stream enabled")
            
            # Start pipeline
            profile = self.pipeline.start(self.config_rs)
            
            # Get device info
            device = profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            
            # Test capture
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            
            if color_frame:
                logger.info(f"✅ RealSense camera initialized: {device_name}")
                logger.info(f"   📐 Resolution: {self.width}x{self.height} @ {self.fps}fps")
                logger.info(f"   📡 Sending frames at {self.send_fps} FPS to server")
                return True
            else:
                logger.error("❌ Failed to capture test frame")
                return False
                
        except Exception as e:
            logger.error(f"❌ RealSense initialization error: {e}")
            return False
    
    def start(self) -> bool:
        """Start RealSense capture thread"""
        if not self.enabled and self.pipeline:
            self.enabled = True
            self.stop_event.clear()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            logger.info("📸 RealSense capture started")
            return True
        return False
    
    def stop(self):
        """Stop RealSense capture"""
        if self.enabled:
            self.enabled = False
            self.stop_event.set()
            if self.capture_thread:
                self.capture_thread.join(timeout=1)
            if self.pipeline:
                self.pipeline.stop()
            logger.info("📸 RealSense stopped")
    
    def get_data(self) -> Optional[Dict]:
        """Get current frame data from RealSense"""
        if not self.pipeline:
            return None
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # Get color frame
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Encode color frame to base64
            _, buffer = cv2.imencode('.jpg', color_image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            color_b64 = base64.b64encode(buffer).decode('utf-8')
            
            frame_data = {
                'color': color_b64,
                'timestamp': time.time(),
                'source': 'realsense'
            }
            
            # Add depth data if enabled
            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    # Convert depth to 8-bit for visualization
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_image, alpha=0.03), 
                        cv2.COLORMAP_JET
                    )
                    
                    # Encode depth frame to base64
                    _, depth_buffer = cv2.imencode('.jpg', depth_colormap, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                    depth_b64 = base64.b64encode(depth_buffer).decode('utf-8')
                    
                    frame_data['depth'] = depth_b64
                    frame_data['depth_raw'] = depth_image.tolist()  # Raw depth values
            
            return frame_data
            
        except Exception as e:
            logger.error(f"❌ Error getting RealSense data: {e}")
            return None
    
    def _capture_loop(self):
        """Main RealSense capture loop"""
        logger.info("📹 RealSense capture loop started")
        
        while not self.stop_event.is_set() and self.enabled:
            try:
                current_time = time.time()
                
                # Send frame at specified FPS
                if current_time - self.last_send_time >= (1.0 / self.send_fps):
                    frame_data = self.get_data()
                    if frame_data and self.client:
                        # Send color frame to server
                        self.client.send_to_server('frame', frame_data['color'])
                        self.last_send_time = current_time
                
                time.sleep(1.0 / self.fps)  # Maintain capture FPS
                
            except Exception as e:
                logger.error(f"❌ RealSense capture error: {e}")
                time.sleep(1)