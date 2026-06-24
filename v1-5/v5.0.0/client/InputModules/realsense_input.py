# modules/input/realsense_input.py - FINAL FINAL CORRECTED VERSION
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
    """RealSense camera input module with depth support and efficient frame throttling"""

    def __init__(self, name: str = "realsense_input", config: Dict = None):
        super().__init__(name, config)
        # ... (rest of __init__ is unchanged) ...
        self.width = self.config.get('width', 1280)
        self.height = self.config.get('height', 720)
        self.fps = self.config.get('fps', 15)
        self.send_fps = self.config.get('send_fps', 5)
        self.jpeg_quality = self.config.get('jpeg_quality', 85)
        self.enable_depth = self.config.get('enable_depth', False)
        self.send_interval = 1.0 / self.send_fps
        self.last_send_time = 0
        self.pipeline = None
        self.config_rs = None
        self.capture_thread = None
        self.stop_event = threading.Event()


    def initialize(self) -> bool:
        # ... (this function is unchanged) ...
        if not REALSENSE_AVAILABLE:
            logger.error("❌ RealSense library not available")
            return False
        if not OPENCV_AVAILABLE:
            logger.error("❌ OpenCV not available for image encoding")
            return False
        try:
            logger.info("📹 Initializing RealSense camera...")
            self.pipeline = rs.pipeline()
            self.config_rs = rs.config()
            self.config_rs.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            if self.enable_depth:
                self.config_rs.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                logger.info("   📏 Depth stream enabled")
            profile = self.pipeline.start(self.config_rs)
            device = profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            for _ in range(30): self.pipeline.wait_for_frames()
            logger.info(f"✅ RealSense camera initialized: {device_name}")
            logger.info(f"   📐 Resolution: {self.width}x{self.height} @ {self.fps}fps")
            logger.info(f"   📡 Sending frames at a maximum of {self.send_fps} FPS to server")
            return True
        except Exception as e:
            logger.error(f"❌ RealSense initialization error: {e}")
            return False


    def start(self) -> bool:
        # ... (this function is unchanged) ...
        if not self.enabled and self.pipeline:
            self.enabled = True
            self.stop_event.clear()
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            return True
        return False

    def stop(self):
        # ... (this function is unchanged) ...
        if self.enabled:
            self.enabled = False
            self.stop_event.set()
            if self.capture_thread:
                self.capture_thread.join(timeout=2)
            if self.pipeline:
                self.pipeline.stop()
            logger.info("📸 RealSense stopped")

    # --- THIS IS THE METHOD THAT WAS MISSING ---
    def get_data(self) -> Optional[Dict]:
        """
        This method is required by the InputModule base class contract.
        In this threaded module, the main work is done in _capture_loop,
        so this method doesn't need to do anything.
        """
        return None
    # ---------------------------------------------

    def _get_and_process_frames(self, frames):
        """Helper function to process frames and encode them."""
        try:
            color_frame = frames.get_color_frame()
            if not color_frame: return None
            color_image = np.asanyarray(color_frame.get_data())
            _, buffer = cv2.imencode('.jpg', color_image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            color_b64 = base64.b64encode(buffer).decode('utf-8')
            frame_data = {'color': color_b64, 'timestamp': time.time(), 'source': 'realsense'}
            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    _, depth_buffer = cv2.imencode('.jpg', depth_colormap)
                    depth_b64 = base64.b64encode(depth_buffer).decode('utf-8')
                    frame_data['depth'] = depth_b64
            return frame_data
        except Exception as e:
            logger.error(f"❌ Error processing RealSense frame: {e}")
            return None

    def _capture_loop(self):
        """Main RealSense capture loop with efficient throttling."""
        logger.info("📹 RealSense capture loop started")
        while not self.stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                if not frames: continue
                current_time = time.time()
                if (current_time - self.last_send_time) >= self.send_interval:
                    frame_data = self._get_and_process_frames(frames)
                    if frame_data and self.client:
                        self.client.send_to_server('frame', frame_data)
                        self.last_send_time = current_time
            except Exception as e:
                logger.error(f"❌ RealSense capture loop error: {e}")
                time.sleep(1)
