# Update your web_interface.py with these performance optimizations

import time
import cv2
import numpy as np
from flask import Response

class WebInterface:
    """Web interface for monitoring and live streaming with performance optimizations"""
    
    def __init__(self, stream_fps=30):
        self.stream_fps = stream_fps
        self.client_id = None
        self.enabled_modules = []
        self.monitor_quality = 60
        
        # 🚀 PERFORMANCE: Add frame caching and optimization
        self.last_encoded_frame = None
        self.last_encode_time = 0
        self.last_frame_hash = None
        self.encode_cache_duration = 0.15  # Cache encoded frame for 150ms
        self.consecutive_identical_frames = 0
        self.max_identical_frames = 3  # Send same frame max 3 times
        
        # Performance statistics
        self.frames_encoded = 0
        self.frames_cached = 0
        self.last_stats_time = time.time()
    
    def generate_live_stream(self, get_latest_frame):
        """Generate optimized live video stream with advanced caching"""
        def generate():
            frame_count = 0
            last_frame_time = time.time()
            target_frame_time = 1.0 / self.stream_fps
            
            # 🚀 PERFORMANCE: Optimized encoding parameters
            encode_params = [
                cv2.IMWRITE_JPEG_QUALITY, max(30, min(90, self.monitor_quality)),  # Clamp quality
                cv2.IMWRITE_JPEG_OPTIMIZE, 1,                    # Enable optimization
                cv2.IMWRITE_JPEG_PROGRESSIVE, 0,                 # Disable progressive for speed
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420  # Faster sampling
            ]
            
            print(f"📺 Starting optimized stream for {getattr(self, 'client_id', 'unknown')} - Quality: {self.monitor_quality}%, FPS: {self.stream_fps}")
            
            while True:
                try:
                    frame_start_time = time.time()
                    
                    # Get the latest frame
                    frame = get_latest_frame()
                    
                    if frame is not None:
                        try:
                            # 🚀 PERFORMANCE: Smart frame caching with hash comparison
                            current_time = time.time()
                            frame_bytes = None
                            
                            # Calculate simple frame hash for comparison
                            frame_hash = self._calculate_frame_hash(frame)
                            
                            # Check if we can use cached encoded frame
                            can_use_cache = (
                                self.last_encoded_frame is not None and
                                current_time - self.last_encode_time < self.encode_cache_duration and
                                frame_hash == self.last_frame_hash and
                                self.consecutive_identical_frames < self.max_identical_frames
                            )
                            
                            if can_use_cache:
                                # Use cached frame
                                frame_bytes = self.last_encoded_frame
                                self.consecutive_identical_frames += 1
                                self.frames_cached += 1
                            else:
                                # Encode new frame
                                success, buffer = cv2.imencode('.jpg', frame, encode_params)
                                
                                if success:
                                    frame_bytes = buffer.tobytes()
                                    
                                    # Update cache
                                    self.last_encoded_frame = frame_bytes
                                    self.last_encode_time = current_time
                                    self.last_frame_hash = frame_hash
                                    self.consecutive_identical_frames = 0
                                    self.frames_encoded += 1
                                else:
                                    print(f"❌ Failed to encode frame for {getattr(self, 'client_id', 'unknown')}")
                                    continue
                            
                            if frame_bytes:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n'
                                       b'Cache-Control: no-cache\r\n\r\n' + frame_bytes + b'\r\n')
                                frame_count += 1
                            
                        except Exception as encode_error:
                            print(f"❌ Frame encoding error for {getattr(self, 'client_id', 'unknown')}: {encode_error}")
                            continue
                    
                    else:
                        # 🚀 PERFORMANCE: Cached placeholder generation
                        placeholder_bytes = self._get_cached_placeholder()
                        if placeholder_bytes:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder_bytes + b'\r\n')
                    
                    # 🚀 PERFORMANCE: Adaptive frame rate control
                    frame_process_time = time.time() - frame_start_time
                    sleep_time = max(0, target_frame_time - frame_process_time)
                    
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif frame_process_time > target_frame_time * 2:
                        # If we're running way behind, skip a frame
                        time.sleep(target_frame_time * 0.5)
                    
                    # 🚀 PERFORMANCE: Periodic statistics logging
                    if frame_count % 100 == 0 and frame_count > 0:
                        self._log_performance_stats(frame_count)
                
                except GeneratorExit:
                    print(f"📺 Live stream ended for {getattr(self, 'client_id', 'unknown')} - {frame_count} frames streamed")
                    break
                except Exception as e:
                    print(f"❌ Live stream error for {getattr(self, 'client_id', 'unknown')}: {e}")
                    time.sleep(0.1)

        return Response(generate(), 
                       mimetype='multipart/x-mixed-replace; boundary=frame',
                       headers={
                           'Cache-Control': 'no-cache, no-store, must-revalidate',
                           'Pragma': 'no-cache',
                           'Expires': '0'
                       })
    
    def _calculate_frame_hash(self, frame):
        """Calculate a simple hash of the frame for comparison"""
        try:
            # 🚀 PERFORMANCE: Simple hash using frame mean values
            # This is much faster than a full hash and good enough for our needs
            if frame is not None and frame.size > 0:
                # Sample every 10th pixel for speed
                sampled = frame[::10, ::10]
                return hash(tuple(sampled.mean(axis=(0,1)).astype(int)))
            return 0
        except:
            return 0
    
    def _get_cached_placeholder(self):
        """Get cached placeholder frame bytes"""
        try:
            # Cache placeholder for reuse
            if not hasattr(self, '_cached_placeholder_bytes'):
                placeholder = self._generate_placeholder_frame()
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
                success, buffer = cv2.imencode('.jpg', placeholder, encode_params)
                if success:
                    self._cached_placeholder_bytes = buffer.tobytes()
                else:
                    self._cached_placeholder_bytes = None
            
            return self._cached_placeholder_bytes
        except:
            return None
    
    def _generate_placeholder_frame(self):
        """Generate a placeholder frame when no camera feed is available"""
        try:
            # Create a small placeholder frame (will be resized by monitor settings)
            placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
            placeholder.fill(50)
            
            client_id = getattr(self, 'client_id', 'Unknown Client')
            enabled_modules = getattr(self, 'enabled_modules', [])
            
            # Add text to placeholder
            cv2.putText(placeholder, f"Client: {client_id}", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(placeholder, "No camera feed", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if enabled_modules:
                modules_text = f"Modules: {', '.join(enabled_modules[:2])}"
                cv2.putText(placeholder, modules_text, (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(placeholder, timestamp, (10, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            
            return placeholder
            
        except Exception as e:
            print(f"❌ Error generating placeholder frame: {e}")
            return np.zeros((240, 320, 3), dtype=np.uint8)
    
    def _log_performance_stats(self, frame_count):
        """Log performance statistics"""
        try:
            current_time = time.time()
            time_elapsed = current_time - self.last_stats_time
            
            if time_elapsed > 0:
                fps = frame_count / time_elapsed if time_elapsed > 0 else 0
                cache_rate = (self.frames_cached / max(1, self.frames_encoded + self.frames_cached)) * 100
                
                print(f"📊 {getattr(self, 'client_id', 'unknown')}: {frame_count} frames, "
                      f"{fps:.1f} FPS, {cache_rate:.1f}% cache hit rate")
                
                # Reset counters
                self.frames_encoded = 0
                self.frames_cached = 0
                self.last_stats_time = current_time
        except:
            pass

    def get_monitor_html(self):
        """Generate the monitoring interface HTML"""
        print("⚠️ WARNING: WebInterface.get_monitor_html() was called - this should not happen for individual monitors!")
        return '''<!DOCTYPE html><html><body><h1>Use individual client monitor instead</h1></body></html>'''