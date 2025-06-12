# realsense_stream.py - Optimized version
import pyrealsense2 as rs
import cv2
import numpy as np
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import json
from collections import deque
import os

class RealSenseStreamer:
    def __init__(self, web_port=8080, frame_processor=None):
        self.web_port = web_port
        self.latest_frame = None
        self.latest_jpg = None
        self.metadata = {"emotion": "Unknown", "fps": 0.0}
        self.running = False
        self.server = None
        self.pipeline = None
        self.server_thread = None
        self.stream_thread = None
        self.frame_processor = frame_processor
       
        # OPTIMIZED: Use deque for frame dropping
        self.frame_buffer = deque(maxlen=2)  # Only keep 2 most recent frames
        self.processed_buffer = deque(maxlen=1)  # Only keep latest processed
       
        # Frame skip counter for processing
        self.frame_skip_counter = 0
        self.PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
       
        # Pre-allocate encoding parameters
        self.jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 80]  # Slightly lower quality for speed
       
    def start_pipeline(self):
        """Initialize and start the RealSense camera pipeline"""
        self.pipeline = rs.pipeline()
        config = rs.config()
       
        # Enable streams - use 640x480 for better performance
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
       
        try:
            # Configure pipeline for low latency
            pipeline_profile = self.pipeline.start(config)
           
            # Get device and configure for low latency
            device = pipeline_profile.get_device()
            depth_sensor = device.first_depth_sensor()
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
               
            return True
        except Exception as e:
            print(f"Error starting pipeline: {e}")
            return False
           
    def get_frame(self):
        """Get the latest frame from RealSense camera with frame dropping"""
        if not self.pipeline:
            return None
           
        try:
            # Use poll_for_frames for non-blocking operation
            frames = self.pipeline.poll_for_frames()
            if not frames:
                return None
               
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
               
            # Convert to numpy array
            return np.asanyarray(color_frame.get_data())
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
           
    def capture_frames(self):
        """Optimized frame capture with frame dropping"""
        fps_frame_count = 0
        fps_start_time = time.time()
        last_frame_time = 0
       
        # Minimum time between frames (for 30 FPS)
        min_frame_interval = 1.0 / 30.0
       
        while self.running:
            current_time = time.time()
           
            # Rate limit frame capture
            if current_time - last_frame_time < min_frame_interval:
                time.sleep(0.001)
                continue
               
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.001)
                continue
           
            last_frame_time = current_time
               
            # Update FPS calculation
            fps_frame_count += 1
            elapsed = current_time - fps_start_time
           
            if elapsed >= 1.0:
                fps = fps_frame_count / elapsed
                self.update_metadata("fps", round(fps, 1))
                fps_frame_count = 0
                fps_start_time = current_time
               
            # Store frame with automatic dropping of old frames
            self.latest_frame = frame
            self.frame_buffer.append((frame, current_time))
               
    def process_frames(self):
        """Process frames with frame skipping for better performance"""
        while self.running:
            try:
                # Get latest frame from buffer
                if not self.frame_buffer:
                    time.sleep(0.01)
                    continue
                   
                frame_data = self.frame_buffer[-1]  # Get most recent
                frame, timestamp = frame_data
               
                # Skip frames for processing
                self.frame_skip_counter += 1
                if self.frame_skip_counter % self.PROCESS_EVERY_N_FRAMES != 0:
                    continue
               
                # Create display frame
                display_frame = frame.copy()
               
               
                # Use custom processor if provided
                if self.frame_processor:
                    try:
                        processed = self.frame_processor(frame, display_frame)
                        if processed is not None:
                            display_frame = processed
                    except Exception as e:
                        print(f"Frame processor error: {e}")
               
                # Encode to JPEG in a separate thread for non-blocking
                self.encode_frame_async(display_frame)
               
            except Exception as e:
                print(f"Error processing frame: {e}")
                time.sleep(0.01)
   
    def encode_frame_async(self, frame):
        """Encode frame to JPEG asynchronously"""
        def encode():
            try:
                ret, jpeg = cv2.imencode('.jpg', frame, self.jpeg_params)
                if ret:
                    self.latest_jpg = jpeg.tobytes()
            except Exception as e:
                print(f"Error encoding frame: {e}")
               
        # Run encoding in a separate thread
        threading.Thread(target=encode, daemon=True).start()
               
    def update_metadata(self, key, value):
        """Update metadata for the frame"""
        self.metadata[key] = value

    class StreamingHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            streamer = self.server.streamer
           
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
               
                html = f'''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>RealSense Camera Stream</title>
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <style>
                        body {{
                            font-family: Arial, sans-serif;
                            margin: 0;
                            padding: 20px;
                            text-align: center;
                            background-color: #f0f0f0;
                        }}
                        .container {{
                            max-width: 800px;
                            margin: 0 auto;
                            background: white;
                            padding: 20px;
                            border-radius: 10px;
                            box-shadow: 0 0 10px rgba(0,0,0,0.1);
                        }}
                        img {{
                            max-width: 100%;
                            border: 3px solid #333;
                            border-radius: 5px;
                        }}
                    </style>
                    <script>
                        let lastUpdate = 0;
                        function refreshImage() {{
                            const now = Date.now();
                            if (now - lastUpdate < 33) return; // Limit to 30fps
                            lastUpdate = now;
                           
                            const img = document.getElementById('camera');
                            const newImg = new Image();
                            newImg.onload = function() {{
                                img.src = newImg.src;
                            }};
                            newImg.src = '/frame.jpg?t=' + now;
                        }}
                       
                        function updateStats() {{
                            fetch('/stats')
                            .then(response => response.json())
                            .then(data => {{
                                document.getElementById('emotion').innerText = 'Detected Emotion: ' + data.emotion;
                                document.getElementById('fps').innerText = 'FPS: ' + data.fps;
                            }})
                            .catch(error => console.error('Error:', error));
                        }}
                       
                        // Use requestAnimationFrame for smooth updates
                        function animate() {{
                            refreshImage();
                            requestAnimationFrame(animate);
                        }}
                        animate();
                       
                    </script>
                </head>
                <body>
                    <div class="container">
                        <h1>RealSense Camera Stream</h1>
                        <img id="camera" src="/frame.jpg" alt="Camera Feed">
                    </div>
                </body>
                </html>
                '''
                self.wfile.write(html.encode())
               
            elif self.path.startswith('/frame.jpg'):
                if streamer.latest_jpg is not None:
                    self.send_response(200)
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                    self.send_header('Content-Length', str(len(streamer.latest_jpg)))
                    self.end_headers()
                    self.wfile.write(streamer.latest_jpg)
                else:
                    self.send_response(503)
                    self.end_headers()
                   
            elif self.path == '/stats':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.wfile.write(json.dumps(streamer.metadata).encode())
               
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            return  # Suppress logs

    class StreamingServer(socketserver.ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True
       
        def __init__(self, server_address, RequestHandlerClass, streamer):
            self.streamer = streamer
            super().__init__(server_address, RequestHandlerClass)
           
    def start_server(self):
        """Start the HTTP server for streaming"""
        server_address = ('0.0.0.0', self.web_port)
        self.server = self.StreamingServer(server_address, self.StreamingHandler, self)
        self.server.serve_forever()
       
    def start(self):
        """Start both the camera pipeline and streaming server"""
        if self.running:
            return False
           
        if not self.start_pipeline():
            print("Failed to start RealSense pipeline")
            return False
           
        self.running = True
       
        # Start threads
        threading.Thread(target=self.capture_frames, daemon=True).start()
        threading.Thread(target=self.process_frames, daemon=True).start()
        threading.Thread(target=self.start_server, daemon=True).start()
       
        return True
       
    def stop(self):
        """Stop the streamer and release resources"""
        self.running = False
        time.sleep(0.5)
       
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception as e:
                print(f"Error shutting down server: {e}")
           
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"Error stopping pipeline: {e}")

if __name__ == "__main__":
    streamer = RealSenseStreamer()
    try:
        if streamer.start():
            print("Press Ctrl+C to exit")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        streamer.stop()
